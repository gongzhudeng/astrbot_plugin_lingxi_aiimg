from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

PLANNER_SYSTEM_PROMPT = (
    "You plan image prompt sets. Output JSON only. "
    "No markdown, no code fence, no explanation."
)

_PLANNER_HINT_BALANCE = "规划模型余额不足"


def _friendly_planner_error(exc: Exception) -> str:
    text = str(exc)
    if "Insufficient Balance" in text or "402" in text:
        return f"{_PLANNER_HINT_BALANCE}: {text}"
    if isinstance(exc, asyncio.TimeoutError):
        return f"规划模型响应超时: {text}"
    return text


@dataclass(slots=True)
class PlannedPromptItem:
    title: str
    prompt: str
    variation_focus: list[str]


def build_batch_planning_prompt(*, mode: str, user_prompt: str, count: int) -> str:
    return (
        "你要为一个图片批量任务规划一组彼此不重复、但整体都满足要求的提示词。\n"
        f"模式: {mode}\n"
        f"目标数量: {count}\n"
        "要求:\n"
        "1. 每条都必须符合用户总要求。\n"
        "2. 整组之间不能只是同一句话换个近义词，必须在姿势、角度、表情、构图、动作、服装细节或氛围上形成明确区分。\n"
        "3. 不要输出解释，不要输出 markdown，不要输出代码块，只输出 JSON 数组。\n"
        '4. 每个元素格式必须是 {"title": "...", "prompt": "...", "variation_focus": ["..."]}。\n'
        "5. title 要简短，prompt 要可直接用于图片生成或改图。\n\n"
        f"用户总要求:\n{str(user_prompt or '').strip()}"
    )


def _strip_code_fence(text: str) -> str:
    raw = str(text or "").strip()
    if raw.startswith("```") and raw.endswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return raw


def _normalize_compare_text(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def parse_planned_prompt_items(text: str) -> list[PlannedPromptItem]:
    raw = _strip_code_fence(text)
    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise ValueError("planner output must be a JSON array")

    out: list[PlannedPromptItem] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("each planner item must be an object")
        title = str(item.get("title") or "").strip()
        prompt = str(item.get("prompt") or "").strip()
        variation_focus_raw = item.get("variation_focus") or []
        if isinstance(variation_focus_raw, list):
            variation_focus = [
                str(x).strip() for x in variation_focus_raw if str(x).strip()
            ]
        else:
            variation_focus = []
        out.append(
            PlannedPromptItem(
                title=title,
                prompt=prompt,
                variation_focus=variation_focus,
            )
        )
    return out


def validate_planned_prompt_items(
    items: list[PlannedPromptItem], *, expected_count: int
) -> str | None:
    if len(items) != int(expected_count):
        return f"expected {expected_count} items, got {len(items)}"

    seen_titles: set[str] = set()
    seen_prompts: set[str] = set()
    for idx, item in enumerate(items, start=1):
        if not item.title:
            return f"item {idx} missing title"
        if not item.prompt:
            return f"item {idx} missing prompt"

        normalized_title = _normalize_compare_text(item.title)
        normalized_prompt = _normalize_compare_text(item.prompt)
        if normalized_title in seen_titles:
            return f"item {idx} duplicated title"
        if normalized_prompt in seen_prompts:
            return f"item {idx} duplicated prompt"
        seen_titles.add(normalized_title)
        seen_prompts.add(normalized_prompt)

    return None


async def plan_with_chain(
    planning_prompt: str,
    count: int,
    chain: list[tuple[Any, str]],
    *,
    timeout_seconds: int,
    retries_per_model: int,
    logger,
) -> list[PlannedPromptItem]:
    """按模型链顺序规划批量提示词；单模型失败重试后切换下一个，全链失败抛 RuntimeError。

    Args:
        planning_prompt: build_batch_planning_prompt 的输出。
        count: 目标提示词条数。
        chain: [(provider_obj, provider_id), ...]，按优先级排列。
        timeout_seconds: 单次 text_chat 的等待上限（秒）。
        retries_per_model: 每个模型失败后的额外重试次数（0 = 失败立即切换下一个）。
        logger: 插件 logger。

    Returns:
        通过校验的提示词条目列表。

    Raises:
        RuntimeError: 链为空或全部尝试失败。
    """

    providers = [(provider, str(pid)) for provider, pid in (chain or []) if provider is not None]
    if not providers:
        raise RuntimeError("批量提示词规划失败: 模型链为空，没有可用的规划模型。")

    attempts_per_model = max(0, int(retries_per_model)) + 1
    timeout = max(1, int(timeout_seconds))
    last_error: Exception | None = None

    for chain_index, (provider, provider_id) in enumerate(providers):
        for attempt in range(1, attempts_per_model + 1):
            try:
                llm_response = await asyncio.wait_for(
                    provider.text_chat(
                        prompt=planning_prompt,
                        contexts=[],
                        image_urls=[],
                        func_tool=None,
                        system_prompt=PLANNER_SYSTEM_PROMPT,
                    ),
                    timeout=timeout,
                )
                text = str(getattr(llm_response, "completion_text", "") or "").strip()
                if not text:
                    raise RuntimeError("LLM returned empty planner output")
                items = parse_planned_prompt_items(text)
                validation_error = validate_planned_prompt_items(
                    items, expected_count=count
                )
                if validation_error is not None:
                    raise ValueError(validation_error)
                if attempt > 1 or chain_index > 0:
                    logger.info(
                        "[batch-planner] 规划成功: provider=%s attempt=%s",
                        provider_id,
                        attempt,
                    )
                return items
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "[batch-planner] provider=%s 第%s/%s次尝试失败: %s",
                    provider_id,
                    attempt,
                    attempts_per_model,
                    exc,
                )

    tried = ", ".join(pid for _, pid in providers)
    raise RuntimeError(
        "批量提示词规划失败: "
        f"已依次尝试模型 [{tried}]（每个 {attempts_per_model} 次），"
        f"最后错误: {_friendly_planner_error(last_error) if last_error else '未知错误'}"
    )
