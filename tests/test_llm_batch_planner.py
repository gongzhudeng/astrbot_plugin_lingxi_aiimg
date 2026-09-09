import asyncio
import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "core" / "llm_batch_planner.py"

LOGGER = logging.getLogger("llm-batch-planner-test")


def _load_module():
    spec = importlib.util.spec_from_file_location("llm_batch_planner_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_planned_prompt_items_from_code_fence():
    mod = _load_module()

    items = mod.parse_planned_prompt_items(
        """```json
[
  {"title":"正面微笑","prompt":"prompt-a","variation_focus":["pose","expression"]},
  {"title":"侧身回头","prompt":"prompt-b","variation_focus":["angle"]}
]
```"""
    )

    assert len(items) == 2
    assert items[0].title == "正面微笑"
    assert items[1].prompt == "prompt-b"


def test_validate_planned_prompt_items_rejects_duplicates():
    mod = _load_module()

    items = [
        mod.PlannedPromptItem(title="正面微笑", prompt="same prompt", variation_focus=[]),
        mod.PlannedPromptItem(title="正面微笑", prompt="same prompt", variation_focus=[]),
    ]

    error = mod.validate_planned_prompt_items(items, expected_count=2)

    assert error is not None


# ---------------------------------------------------------------------------
# plan_with_chain
# ---------------------------------------------------------------------------

_BALANCE_ERROR = Exception(
    "Error code: 402 - {'error': {'message': 'Insufficient Balance', "
    "'type': 'unknown_error', 'param': None, 'code': 'invalid_request_error'}}"
)


def _ok_json(count: int = 2, tag: str = "a") -> str:
    payload = [
        {"title": f"t-{tag}-{i}", "prompt": f"p-{tag}-{i}", "variation_focus": []}
        for i in range(count)
    ]
    return json.dumps(payload, ensure_ascii=False)


class _FakeProvider:
    """按行为队列响应 text_chat 的假 Provider。

    行为项含义：
    - str: 作为 completion_text 返回
    - None: 返回空 completion_text
    - Exception 实例: 抛出该异常
    - "hang": 永久挂起（等待被 wait_for 取消）
    """

    def __init__(self, behaviors):
        self._behaviors = list(behaviors)
        self.calls = 0

    async def text_chat(self, **kwargs):
        self.calls += 1
        behavior = self._behaviors.pop(0) if self._behaviors else "raise_last"
        if behavior == "raise_last":
            raise RuntimeError("planner behavior queue exhausted")
        if behavior == "hang":
            await asyncio.sleep(30)
            return SimpleNamespace(completion_text="")
        if behavior is None:
            return SimpleNamespace(completion_text="")
        if isinstance(behavior, Exception):
            raise behavior
        return SimpleNamespace(completion_text=str(behavior))


def _chain(*providers):
    return [(provider, pid) for provider, pid in providers]


def test_plan_with_chain_falls_back_to_next_provider():
    mod = _load_module()
    bad = _FakeProvider([_BALANCE_ERROR, _BALANCE_ERROR, _BALANCE_ERROR])
    good = _FakeProvider([_ok_json(2, tag="b")])

    items = asyncio.run(
        mod.plan_with_chain(
            "plan",
            2,
            _chain((bad, "prov-a"), (good, "prov-b")),
            timeout_seconds=60,
            retries_per_model=0,
            logger=LOGGER,
        )
    )

    assert [item.prompt for item in items] == ["p-b-0", "p-b-1"]
    assert bad.calls == 1
    assert good.calls == 1


def test_plan_with_chain_all_fail_raises_with_tried_ids():
    mod = _load_module()
    a = _FakeProvider([_BALANCE_ERROR])
    b = _FakeProvider([_BALANCE_ERROR])

    try:
        asyncio.run(
            mod.plan_with_chain(
                "plan",
                2,
                _chain((a, "prov-a"), (b, "prov-b")),
                timeout_seconds=60,
                retries_per_model=0,
                logger=LOGGER,
            )
        )
    except RuntimeError as exc:
        message = str(exc)
        assert "prov-a" in message and "prov-b" in message
        assert "余额不足" in message
    else:
        raise AssertionError("expected RuntimeError when the whole chain fails")


def test_plan_with_chain_timeout_switches_to_next_provider():
    mod = _load_module()
    slow = _FakeProvider(["hang"])
    good = _FakeProvider([_ok_json(2, tag="c")])

    items = asyncio.run(
        mod.plan_with_chain(
            "plan",
            2,
            _chain((slow, "slow"), (good, "fast")),
            timeout_seconds=1,
            retries_per_model=0,
            logger=LOGGER,
        )
    )

    assert len(items) == 2
    assert slow.calls == 1
    assert good.calls == 1


def test_plan_with_chain_retries_same_model_before_switching():
    mod = _load_module()
    flaky = _FakeProvider([_BALANCE_ERROR, _BALANCE_ERROR, _ok_json(2, tag="d")])
    spare = _FakeProvider([_ok_json(2, tag="e")])

    items = asyncio.run(
        mod.plan_with_chain(
            "plan",
            2,
            _chain((flaky, "flaky"), (spare, "spare")),
            timeout_seconds=60,
            retries_per_model=2,
            logger=LOGGER,
        )
    )

    assert [item.prompt for item in items] == ["p-d-0", "p-d-1"]
    assert flaky.calls == 3
    assert spare.calls == 0


def test_plan_with_chain_empty_output_counts_as_failure():
    mod = _load_module()
    empty = _FakeProvider([None, None])
    good = _FakeProvider([_ok_json(2, tag="f")])

    items = asyncio.run(
        mod.plan_with_chain(
            "plan",
            2,
            _chain((empty, "empty"), (good, "good")),
            timeout_seconds=60,
            retries_per_model=1,
            logger=LOGGER,
        )
    )

    assert len(items) == 2
    assert empty.calls == 2
    assert good.calls == 1


def test_plan_with_chain_invalid_json_counts_as_failure():
    mod = _load_module()
    broken = _FakeProvider(["not-json-at-all", _ok_json(2, tag="g")])

    items = asyncio.run(
        mod.plan_with_chain(
            "plan",
            2,
            _chain((broken, "broken")),
            timeout_seconds=60,
            retries_per_model=1,
            logger=LOGGER,
        )
    )

    assert [item.title for item in items] == ["t-g-0", "t-g-1"]
    assert broken.calls == 2


def test_plan_with_chain_empty_chain_raises():
    mod = _load_module()

    try:
        asyncio.run(
            mod.plan_with_chain(
                "plan",
                2,
                [],
                timeout_seconds=60,
                retries_per_model=1,
                logger=LOGGER,
            )
        )
    except RuntimeError as exc:
        assert "模型链为空" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for empty chain")
