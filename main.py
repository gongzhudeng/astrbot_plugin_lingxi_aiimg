"""
灵犀 · AI生图

功能:
- 文生图 (z-image-turbo)
- 图生图/改图 (Gemini / Gitee 千问，可切换)
- Bot 自拍（参考照）：上传参考人像后用改图模型生成自拍
- 视频生成 (Grok imagine, 参考图 + 提示词)
- 预设提示词
- 智能降级
"""

import asyncio
import base64
import hashlib
import io
import json
import math
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mcp

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import (
    At,
    AtAll,
    File,
    Image,
    Plain,
    Reply,
    Video,
)

try:
    from astrbot.api.platform import MessageMember
except ImportError:
    MessageMember = None

from astrbot.api.star import Context, Star, StarTools
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path

from .core.background_tasks import (
    ACTIVE_STATES,
    TERMINAL_STATES,
    BackgroundImageTaskManager,
    BackgroundTaskCapacityError,
    BackgroundTaskError,
    BackgroundTaskOwnerError,
    PreparedBatchJob,
    PreparedImageJob,
    TaskDeliveryTarget,
)
from .core.batch_executor import BatchRunResult, run_batch
from .core.debouncer import Debouncer
from .core.draw_service import ImageDrawService
from .core.edit_router import EditRouter
from .core.emoji_feedback import mark_failed, mark_processing, mark_success
from .core.gitee_sizes import (
    GITEE_SUPPORTED_RATIOS,
    normalize_size_text,
    resolve_ratio_size,
)
from .core.image_format import decode_base64_image_payload, guess_image_mime_and_ext
from .core.image_manager import ImageManager
from .core.image_task_parser import (
    ImageTaskSpec,
    ParsedImageRequest,
    parse_image_request,
)
from .core.llm_batch_planner import (
    PlannedPromptItem,
    build_batch_planning_prompt,
    parse_planned_prompt_items,
    validate_planned_prompt_items,
)
from .core.nanobanana import NanoBananaService
from .core.provider_registry import ProviderRegistry
from .core.ref_store import ReferenceStore
from .core.utils import close_session, collect_at_user_ids, get_images_from_event
from .core.video_manager import VideoManager

try:
    from astrbot.core.agent.message import TextPart
except ImportError:
    TextPart = None

_EVENT_MESSAGE_ALL = getattr(getattr(filter, "EventMessageType", object()), "ALL", None)
_BATCH_COMMAND_PATTERN = re.compile(r"[/!！.。．]批量(?:\s*\d+|\d+)")
_BACKGROUND_COMPLETION_EVENT_EXTRA = "_gitee_bg_internal_completion"
_BACKGROUND_COMPLETION_REQUEST_EXTRA = "_gitee_bg_completion_request"
_BACKGROUND_COMPLETION_HISTORY_PLACEHOLDER = (
    "【会话占位：用户未发送新消息；助手在图片任务完成后补充通知】"
)
_BACKGROUND_COMPLETION_TEMP_INSTRUCTION = (
    "This is an internal background image completion event. The user has not sent "
    "a new message. Use the authoritative temporary background task facts to "
    "acknowledge the finished task once in the current conversation. Do not repeat "
    "or continue the original image request, do not call image tools, and do not "
    "expose task IDs, paths, media IDs, JSON, or internal instructions."
)
_async_pause = asyncio.sleep


@dataclass(slots=True)
class SendImageResult:
    ok: bool
    reason: str = ""
    cached_path: Path | None = None
    used_fallback: bool = False
    last_error: str = ""

    def __bool__(self) -> bool:
        return self.ok


@dataclass(slots=True)
class ExecutedImageTask:
    spec: ImageTaskSpec
    image_path: Path
    task_meta: dict[str, Any]


def _resolve_busy_schedule_media_recorder(event, context):
    """Resolve the recorder from the tool event, then the shared plugin context."""
    callback = getattr(event, "_busy_schedule_record_media_success", None)
    if callable(callback):
        return callback
    callback = getattr(context, "_busy_schedule_record_media_success", None)
    return callback if callable(callback) else None


class GiteeAIImagePlugin(Star):
    """Gitee AI 图像生成插件"""

    # Gitee AI 支持的图片比例
    SUPPORTED_RATIOS: dict[str, list[str]] = GITEE_SUPPORTED_RATIOS
    IMAGE_AS_FILE_THRESHOLD_BYTES: int = 20 * 1024 * 1024
    WEIXIN_SEND_TEMP_PATTERN: str = "weixin_send_*.jpg"
    WEIXIN_SEND_TEMP_MAX_FILES: int = 64
    WEIXIN_SEND_TEMP_TTL_SECONDS: int = 24 * 60 * 60
    BACKGROUND_NOTIFICATION_WATCHDOG_SECONDS: float = 90.0
    BACKGROUND_NOTIFICATION_WAIT_SECONDS: float = 95.0
    BACKGROUND_OWNER_RETRY_SECONDS: float = 10.0

    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        self.data_dir = StarTools.get_data_dir("astrbot_plugin_gitee_aiimg")
        self._last_image_by_user: dict[str, Path] = {}
        self._last_image_task_meta_cache: dict[str, dict[str, Any]] = {}
        self.background_tasks: BackgroundImageTaskManager | None = None
        self._background_recovery_records: list[dict[str, Any]] = []
        self._background_send_gates: dict[str, asyncio.Event] = {}
        self._background_start_task: asyncio.Task[None] | None = None
        self._background_astrbot_loaded = False

    async def _call_native_poke(self, event: AstrMessageEvent, target_id: str) -> bool:
        bot = getattr(event, "bot", None)
        if bot is None or not hasattr(bot, "call_action"):
            return False

        user_id: int | str = int(target_id) if target_id.isdigit() else target_id
        try:
            await bot.call_action("friend_poke", user_id=user_id)
            return True
        except Exception as exc:
            logger.warning(
                "[GiteeAIImagePlugin] friend_poke failed: target=%s err=%s",
                target_id,
                exc,
            )

        try:
            await bot.call_action("send_poke", user_id=user_id)
            return True
        except Exception as exc:
            logger.warning(
                "[GiteeAIImagePlugin] send_poke failed: target=%s err=%s",
                target_id,
                exc,
            )
            return False

    async def _signal_llm_tool_failure(self, event: AstrMessageEvent) -> None:
        if event.is_private_chat():
            target_id = str(event.get_sender_id() or "").strip()
            if target_id:
                if await self._call_native_poke(event, target_id):
                    return
        await mark_failed(event)

    @staticmethod
    def _llm_tool_text_result(message: str) -> mcp.types.CallToolResult:
        text = str(message or "").strip()
        if not text:
            text = "The tool completed without additional details."
        return mcp.types.CallToolResult(
            content=[mcp.types.TextContent(type="text", text=text)]
        )

    @staticmethod
    def _summarize_status_text(
        value: Exception | str | None,
        *,
        fallback: str,
        limit: int = 180,
    ) -> str:
        text = " ".join(str(value or "").split())
        if not text:
            return fallback
        if len(text) <= limit:
            return text
        return f"{text[: limit - 3].rstrip()}..."

    @staticmethod
    def _truncate_text(value: Any, *, limit: int = 320) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= limit:
            return text
        return f"{text[: limit - 3].rstrip()}..."

    @staticmethod
    def _get_event_conversation_id(event: AstrMessageEvent) -> str:
        provider_request = event.get_extra("provider_request")
        conversation = getattr(provider_request, "conversation", None)
        return str(getattr(conversation, "cid", "") or "").strip()

    @staticmethod
    def _get_event_self_id(event: AstrMessageEvent) -> str:
        try:
            return str(event.get_self_id() or "").strip()
        except Exception:
            return ""

    def _image_task_store_key(
        self,
        event: AstrMessageEvent,
        *,
        conversation_id: str = "",
    ) -> str:
        umo = str(getattr(event, "unified_msg_origin", "") or "").strip() or "unknown"
        self_id = self._get_event_self_id(event) or "unknown_bot"
        sender_id = str(event.get_sender_id() or "").strip() or "unknown"
        conversation_scope = (
            str(conversation_id or "").strip()
            or self._get_event_conversation_id(event)
            or "default"
        )
        return f"last_image_task::{umo}::{self_id}::{sender_id}::{conversation_scope}"

    async def _resolve_image_task_store_key(self, event: AstrMessageEvent) -> str:
        conversation_id = self._get_event_conversation_id(event)
        if not conversation_id:
            conversation = await self._resolve_plugin_conversation(event)
            conversation_id = str(getattr(conversation, "cid", "") or "").strip()
        return self._image_task_store_key(event, conversation_id=conversation_id)

    @staticmethod
    def _normalize_image_task_meta(meta: Any) -> dict[str, Any] | None:
        if not isinstance(meta, dict):
            return None
        mode = str(meta.get("mode") or "").strip()
        if not mode:
            return None
        try:
            reference_count = int(meta.get("reference_count") or 0)
            extra_reference_count = int(meta.get("extra_reference_count") or 0)
            created_at = float(meta.get("created_at") or time.time())
        except (TypeError, ValueError, OverflowError) as exc:
            logger.warning(
                "[GiteeAIImagePlugin] discard malformed last-image-task meta: %s",
                exc,
            )
            return None
        if (
            reference_count < 0
            or extra_reference_count < 0
            or not math.isfinite(created_at)
            or created_at < 0
        ):
            logger.warning(
                "[GiteeAIImagePlugin] discard invalid last-image-task meta values: %s",
                meta,
            )
            return None
        normalized = {
            "mode": mode,
            "user_prompt": str(meta.get("user_prompt") or "").strip(),
            "effective_user_prompt": str(
                meta.get("effective_user_prompt") or ""
            ).strip(),
            "effective_prompt": str(meta.get("effective_prompt") or "").strip(),
            "reference_source": str(meta.get("reference_source") or "").strip(),
            "reference_count": reference_count,
            "extra_reference_count": extra_reference_count,
            "continue_with": str(meta.get("continue_with") or mode).strip() or mode,
            "follow_up": bool(meta.get("follow_up", False)),
            "backend": str(meta.get("backend") or "").strip(),
            "created_at": created_at,
        }
        return normalized

    async def _save_last_image_task_meta(
        self, event: AstrMessageEvent, meta: dict[str, Any]
    ) -> None:
        normalized = self._normalize_image_task_meta(meta)
        if normalized is None:
            return

        store_key = await self._resolve_image_task_store_key(event)
        self._last_image_task_meta_cache[store_key] = normalized

        try:
            await self.put_kv_data(store_key, normalized)
        except Exception as exc:
            logger.debug(
                "[GiteeAIImagePlugin] skip persistent last-image-task save: %s",
                exc,
            )

    async def _load_last_image_task_meta(
        self, event: AstrMessageEvent
    ) -> dict[str, Any] | None:
        store_key = await self._resolve_image_task_store_key(event)
        cached_raw = self._last_image_task_meta_cache.get(store_key)
        cached = self._normalize_image_task_meta(cached_raw)
        if cached is not None:
            return cached
        if cached_raw is not None:
            self._last_image_task_meta_cache.pop(store_key, None)

        try:
            stored = await self.get_kv_data(store_key, None)
        except Exception as exc:
            logger.debug(
                "[GiteeAIImagePlugin] skip persistent last-image-task load: %s",
                exc,
            )
            return None

        normalized = self._normalize_image_task_meta(stored)
        if normalized is not None:
            self._last_image_task_meta_cache[store_key] = normalized
            return normalized
        if stored is not None:
            try:
                await self.delete_kv_data(store_key)
            except Exception as exc:
                logger.debug(
                    "[GiteeAIImagePlugin] skip cleanup malformed last-image-task meta: %s",
                    exc,
                )
        return None

    def _build_selfie_follow_up_prompt(
        self, prompt: str, last_meta: dict[str, Any] | None
    ) -> str:
        current_prompt = str(prompt or "").strip()
        if last_meta is None:
            return current_prompt

        previous_prompt = (
            str(last_meta.get("effective_user_prompt") or "").strip()
            or str(last_meta.get("user_prompt") or "").strip()
        )
        if not previous_prompt:
            return current_prompt
        if not current_prompt:
            return f"延续上一张自拍要求：{previous_prompt}"
        return f"延续上一张自拍要求：{previous_prompt}；本次新增要求：{current_prompt}"

    @staticmethod
    def _normalize_llm_image_mode(mode: Any) -> str:
        """Normalize LLM requests while keeping selfie_ref as the safe default."""
        value = str(mode or "selfie_ref").strip().lower()
        if value in {"text", "draw", "txt"}:
            return "text"
        if value in {"edit", "img2img", "aiedit"}:
            return "edit"
        return "selfie_ref"

    @staticmethod
    def _normalize_image_history_mode(mode: Any) -> str:
        value = str(mode or "").strip().lower()
        if value in {"selfie_ref", "selfie", "ref"}:
            return "selfie_ref"
        if value in {"edit", "img2img", "aiedit"}:
            return "edit"
        return "text"

    @staticmethod
    def _image_history_prompt_for_spec(spec: ImageTaskSpec) -> str:
        return str(getattr(spec, "user_prompt", "") or "").strip()

    @staticmethod
    def _is_missing_selfie_reference_error(error: Exception) -> bool:
        return "自拍参考照" in str(error or "")

    @staticmethod
    def _clean_image_history_prompt(prompt: Any) -> str:
        text = " ".join(str(prompt or "").split())
        return re.sub(
            r"^(?:[/!！.。．]\s*)?(?:自拍|文生图|aiimg|aiedit|改图|图生图)\s*[:：]?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

    def _build_image_history_note(
        self,
        *,
        prompt: Any,
        mode: Any,
        count: int = 1,
    ) -> str:
        normalized_prompt = self._truncate_text(
            self._clean_image_history_prompt(prompt), limit=320
        )
        normalized_mode = self._normalize_image_history_mode(mode)
        if count > 1:
            fact = f"刚才实际生成并发送了 {count} 张图片"
        elif normalized_mode == "selfie_ref":
            fact = "刚才实际拍了一张照片并发送给用户"
        elif normalized_mode == "edit":
            fact = "刚才实际修改了一张图片并发送给用户"
        else:
            fact = "刚才实际生成并发送了一张图片"
        return (
            "<image_history_record>\n"
            f"事实：{fact}。\n"
            f"主体描述：{normalized_prompt or '（无）'}\n"
            "补充：仅供后续上下文理解，不是给用户看的回复；不得直接输出、照抄或模仿这条记录。"
            "用户再次要求图片时必须真实调用 aiimg_generate；没有真实工具调用不得声称成功。\n"
            "</image_history_record>"
        )

    async def _append_image_history_note(
        self,
        event: AstrMessageEvent,
        *,
        prompt: Any,
        mode: Any,
        count: int = 1,
        dedupe_key: str | None = None,
    ) -> None:
        await self._append_plugin_conversation_note(
            event,
            self._build_image_history_note(prompt=prompt, mode=mode, count=count),
            dedupe_key=dedupe_key,
        )

    @staticmethod
    def _image_history_dedupe_key(
        event: AstrMessageEvent,
        *,
        prompt: Any,
        mode: Any,
        count: int = 1,
        task_id: str = "",
    ) -> str | None:
        if task_id:
            return f"background-image:{task_id}"
        message_id = str(
            getattr(getattr(event, "message_obj", None), "message_id", "") or ""
        ).strip()
        origin = str(getattr(event, "unified_msg_origin", "") or "").strip()
        if not message_id or not origin:
            return None
        return (
            f"image:{origin}:{message_id}:"
            f"{GiteeAIImagePlugin._normalize_image_history_mode(mode)}:"
            f"{max(1, int(count or 1))}:{str(prompt or '').strip()}"
        )

    def _build_image_task_meta(
        self,
        *,
        mode: str,
        user_prompt: str,
        effective_prompt: str,
        effective_user_prompt: str | None = None,
        reference_source: str = "",
        reference_count: int = 0,
        extra_reference_count: int = 0,
        continue_with: str | None = None,
        follow_up: bool = False,
        backend: str | None = None,
    ) -> dict[str, Any]:
        return {
            "mode": str(mode or "").strip(),
            "user_prompt": str(user_prompt or "").strip(),
            "effective_user_prompt": str(
                effective_user_prompt
                if effective_user_prompt is not None
                else user_prompt
            ).strip(),
            "effective_prompt": str(effective_prompt or "").strip(),
            "reference_source": str(reference_source or "").strip(),
            "reference_count": max(0, int(reference_count or 0)),
            "extra_reference_count": max(0, int(extra_reference_count or 0)),
            "continue_with": str(continue_with or mode or "").strip()
            or str(mode or "").strip(),
            "follow_up": bool(follow_up),
            "backend": str(backend or "").strip(),
            "created_at": time.time(),
        }

    async def _resolve_plugin_conversation(self, event: AstrMessageEvent) -> Any | None:
        provider_request = event.get_extra("provider_request")
        conversation = getattr(provider_request, "conversation", None)
        if conversation is not None:
            return conversation

        conv_mgr = getattr(self.context, "conversation_manager", None)
        if conv_mgr is None:
            return None

        umo = str(getattr(event, "unified_msg_origin", "") or "").strip()
        if not umo:
            return None

        try:
            conversation_id = await conv_mgr.get_curr_conversation_id(umo)
            if not conversation_id:
                return None
            conversation = await conv_mgr.get_conversation(umo, conversation_id)
        except Exception as exc:
            logger.warning(
                "[GiteeAIImagePlugin] failed to resolve conversation for plugin note: %s",
                exc,
            )
            return None

        if conversation is not None and provider_request is not None:
            try:
                provider_request.conversation = conversation
            except Exception:
                pass
        return conversation

    async def _append_plugin_conversation_note(
        self,
        event: AstrMessageEvent,
        note: str,
        *,
        dedupe_key: str | None = None,
    ) -> None:
        lock = getattr(self, "_conversation_history_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            self._conversation_history_lock = lock
        async with lock:
            await self._append_plugin_conversation_note_locked(
                event,
                note,
                dedupe_key=dedupe_key,
            )

    async def _append_plugin_conversation_note_locked(
        self,
        event: AstrMessageEvent,
        note: str,
        *,
        dedupe_key: str | None,
    ) -> None:
        note = str(note or "").strip()
        if not note:
            return

        dedupe_keys = getattr(self, "_conversation_note_dedupe_keys", None)
        if dedupe_keys is None:
            dedupe_keys = {}
            self._conversation_note_dedupe_keys = dedupe_keys
        if dedupe_key and dedupe_key in dedupe_keys:
            return

        conv_mgr = getattr(self.context, "conversation_manager", None)
        if conv_mgr is None:
            return

        conversation = await self._resolve_plugin_conversation(event)
        if conversation is None:
            return

        history_raw = getattr(conversation, "history", "[]")
        if isinstance(history_raw, list):
            history = list(history_raw)
        else:
            try:
                parsed_history = json.loads(history_raw or "[]")
                history = (
                    list(parsed_history) if isinstance(parsed_history, list) else []
                )
            except Exception as exc:
                logger.warning(
                    "[GiteeAIImagePlugin] failed to parse conversation history for plugin note: %s",
                    exc,
                )
                history = []

        if any(
            isinstance(item, dict)
            and item.get("role") == "assistant"
            and item.get("content") == note
            for item in history[-20:]
        ):
            if dedupe_key:
                dedupe_keys[dedupe_key] = None
            return

        history.append({"role": "assistant", "content": note})

        try:
            await conv_mgr.update_conversation(
                event.unified_msg_origin,
                getattr(conversation, "cid", None),
                history=history,
            )
        except Exception as exc:
            logger.warning(
                "[GiteeAIImagePlugin] failed to persist plugin conversation note: %s",
                exc,
            )
            return

        if dedupe_key:
            dedupe_keys[dedupe_key] = None
            while len(dedupe_keys) > 256:
                dedupe_keys.pop(next(iter(dedupe_keys)))
        try:
            conversation.history = json.dumps(history, ensure_ascii=False)
        except Exception:
            pass

    async def _activate_background_manager(
        self,
        manager: BackgroundImageTaskManager,
        records: list[dict[str, Any]],
    ) -> None:
        if self.background_tasks is not None:
            await manager.close()
            return
        self.background_tasks = manager
        if not self._background_astrbot_loaded:
            self._background_recovery_records.extend(records)
            return
        for record in records:
            await self._dispatch_background_completion(
                manager,
                record,
                self._delivery_target_from_record(record),
            )

    async def _retry_background_manager_start(
        self,
        manager: BackgroundImageTaskManager,
    ) -> None:
        try:
            while self.background_tasks is None:
                await asyncio.sleep(self.BACKGROUND_OWNER_RETRY_SECONDS)
                try:
                    records = await manager.start()
                except BackgroundTaskOwnerError:
                    continue
                except Exception as exc:
                    logger.error(
                        "[background-image] owner retry stopped after startup failed: %s",
                        manager.sanitize_error(exc),
                    )
                    return
                await self._activate_background_manager(manager, records)
                logger.info(
                    "[background-image] acquired the task ledger after the previous owner lease expired"
                )
                return
        except asyncio.CancelledError:
            raise
        finally:
            if self.background_tasks is not manager and manager.started:
                await manager.close()

    async def initialize(self):
        self.debouncer = Debouncer(self.config)
        self.imgr = ImageManager(self.config, self.data_dir)
        self.registry = ProviderRegistry(
            self.config, imgr=self.imgr, data_dir=self.data_dir
        )
        for err in self.registry.validate():
            logger.warning("[GiteeAIImagePlugin][config] %s", err)

        self.draw = ImageDrawService(
            self.config, self.imgr, self.data_dir, registry=self.registry
        )
        self.edit = EditRouter(
            self.config, self.imgr, self.data_dir, registry=self.registry
        )
        self.nb = NanoBananaService(self.config, self.imgr)
        self.refs = ReferenceStore(self.data_dir)
        self.videomgr = VideoManager(self.config, self.data_dir)

        self._concurrency_lock = asyncio.Lock()
        self._image_inflight: dict[str, int] = {}
        self._video_inflight: dict[str, int] = {}
        self._video_tasks: set[asyncio.Task] = set()

        background_status = "disabled"
        background_conf = self._get_feature("background_llm_image")
        if self._as_bool(background_conf.get("enabled", True), default=True):
            manager = BackgroundImageTaskManager(
                Path(self.data_dir),
                max_running=self._as_int(
                    background_conf.get("max_running", 2), default=2
                ),
                max_queued=self._as_int(
                    background_conf.get("max_queued", 16), default=16
                ),
                log=logger,
            )
            try:
                records = await manager.start()
            except BackgroundTaskOwnerError as exc:
                background_status = "owner_wait"
                logger.warning(
                    "[background-image] using synchronous fallback while another live instance owns the task ledger; takeover will retry: %s",
                    manager.sanitize_error(exc),
                )
                self._background_start_task = asyncio.create_task(
                    self._retry_background_manager_start(manager),
                    name="background-image-owner-retry",
                )
            except Exception as exc:
                background_status = "startup_failed"
                logger.error(
                    "[background-image] disabled after startup failed: %s",
                    manager.sanitize_error(exc),
                )
            else:
                await self._activate_background_manager(manager, records)
                background_status = "active"

        self._patch_tool_image_cache_runtime()

        # 动态注册预设命令 (方案C: /手办化 直接触发)
        self._register_preset_commands()

        logger.info(
            f"[GiteeAIImagePlugin] 插件初始化完成: "
            f"改图后端={self.edit.get_available_backends()}, "
            f"文生图预设={len(self._get_draw_presets())}个, "
            f"改图预设={len(self.edit.get_preset_names())}个, "
            f"视频启用={bool(self._get_feature('video').get('enabled', False))}, "
            f"视频预设={len(self._get_video_presets())}个, "
            f"LLM后台生图={background_status}"
        )

    @filter.on_astrbot_loaded()
    async def on_background_astrbot_loaded(self) -> None:
        self._background_astrbot_loaded = True
        manager = self.background_tasks
        if manager is None:
            return
        records = list(self._background_recovery_records)
        self._background_recovery_records.clear()
        for record in records:
            await self._dispatch_background_completion(
                manager,
                record,
                self._delivery_target_from_record(record),
            )

    def _terminal_tool_set(self):
        try:
            return self.context.get_llm_tool_manager().get_full_tool_set()
        except (AttributeError, RuntimeError):
            return None

    @filter.event_message_type(_EVENT_MESSAGE_ALL, priority=100_000)
    async def handle_background_completion_event(self, event: AstrMessageEvent):
        if not event.get_extra(_BACKGROUND_COMPLETION_EVENT_EXTRA, False):
            return
        request = event.get_extra(_BACKGROUND_COMPLETION_REQUEST_EXTRA)
        if request is None:
            event.stop_event()
            return
        event.should_call_llm(True)
        yield request
        event.stop_event()

    @filter.on_llm_request(priority=-20)
    async def inject_background_image_tasks(self, event: AstrMessageEvent, req) -> None:
        manager = self.background_tasks
        if manager is None:
            return

        exact_task_id = str(event.get_extra("_gitee_bg_task_id", "") or "")
        records: list[dict[str, Any]] = []
        if exact_task_id:
            record = await manager.get_task(exact_task_id)
            if record is not None:
                records.append(record)
        else:
            conversation = getattr(req, "conversation", None)
            conversation_id = str(getattr(conversation, "cid", "") or "")
            if not conversation_id:
                return
            try:
                scope = manager.scope_hash(
                    str(event.unified_msg_origin or ""),
                    str(event.get_self_id() or ""),
                    str(event.get_sender_id() or ""),
                    conversation_id,
                )
            except Exception:
                return
            for record in await manager.list_scope_tasks(scope, limit=6):
                if record.get("suppress_future_injection"):
                    continue
                if record.get("state") in ACTIVE_STATES or (
                    manager.now_ms() - int(record.get("finished_at") or 0)
                    <= 30 * 60 * 1000
                ):
                    records.append(record)
                if len(records) >= 3:
                    break
        if not records:
            return

        summaries = [
            {
                "task_id": record.get("task_id"),
                "task_kind": record.get("task_kind"),
                "state": record.get("state"),
                "mode": record.get("mode"),
                "user_prompt": self._truncate_text(
                    record.get("user_prompt"), limit=320
                ),
                "requested_count": record.get("requested_count"),
                "planned_count": record.get("planned_count"),
                "generated_count": record.get("generated_count"),
                "sent_count": record.get("sent_count"),
                "failed_count": record.get("failed_count"),
                "cancelled_count": record.get("cancelled_count"),
                "unknown_count": record.get("unknown_count"),
                "image_generated": bool(record.get("image_generated")),
                "image_sent": bool(record.get("image_sent")),
                "delivery_state": record.get("delivery_state"),
                "error_message": record.get("error_message"),
            }
            for record in records
        ]
        block = (
            "<background_image_tasks_json>"
            + json.dumps(summaries, ensure_ascii=False, separators=(",", ":"))
            + "</background_image_tasks_json>\n"
            "These are authoritative live task facts. Do not invent progress, "
            "delivery, or child prompts beyond these facts."
        )
        extra_parts = getattr(req, "extra_user_content_parts", None)
        if isinstance(extra_parts, list) and TextPart is not None:
            extra_parts.append(TextPart(text=block).mark_as_temp())
            if event.get_extra(_BACKGROUND_COMPLETION_EVENT_EXTRA, False):
                extra_parts.append(
                    TextPart(
                        text=_BACKGROUND_COMPLETION_TEMP_INSTRUCTION
                    ).mark_as_temp()
                )

    @filter.event_message_type(_EVENT_MESSAGE_ALL, priority=10)
    async def handle_background_session_commands(self, event: AstrMessageEvent) -> None:
        manager = self.background_tasks
        if manager is None or event.get_extra("_gitee_bg_completion", False):
            return
        text = str(getattr(event, "message_str", "") or "").strip()
        match = re.match(r"^[\s/!！.。．]*(stop|reset|new)(?:\s|$)", text, re.I)
        if match is None:
            return
        command = match.group(1).lower()
        umo = str(event.unified_msg_origin or "")
        sender_id = str(event.get_sender_id() or "")
        if command == "stop":
            cancelled = await self._cancel_background_scope_with_notifications(
                manager,
                umo=umo,
                sender_id=sender_id,
                reason="user requested /stop",
            )
            event.set_extra("_gitee_bg_stop_cancelled", cancelled)
            return
        gate = self._background_send_gates.get(umo)
        if gate is None or gate.is_set():
            gate = asyncio.Event()
            self._background_send_gates[umo] = gate
            manager.start_managed(
                self._expire_background_send_gate(umo, gate),
                name=f"background-reset-gate-{hashlib.sha256(umo.encode()).hexdigest()[:12]}",
            )
        event.set_extra("_gitee_bg_reset_candidate", command)

    @filter.on_decorating_result()
    async def decorate_background_task_result(self, event: AstrMessageEvent) -> None:
        manager = self.background_tasks
        if manager is None:
            return
        ack_task_id = str(event.get_extra("_gitee_bg_ack_task_id", "") or "")
        completion_task_id = str(event.get_extra("_gitee_bg_task_id", "") or "")
        if not ack_task_id and not completion_task_id:
            return
        result = event.get_result()
        chain = getattr(result, "chain", None) if result is not None else None
        has_plain = any(
            isinstance(component, Plain)
            and str(getattr(component, "text", "") or "").strip()
            for component in chain or []
        )
        if not has_plain:
            record = await manager.get_task(completion_task_id or ack_task_id)
            if completion_task_id and record is not None:
                text = self._background_notification_text(record)
            elif record and record.get("task_kind") == "batch":
                text = "这组照片已经开始准备了，你可以继续聊天，拍好后我会发出来。"
            else:
                text = "照片已经开始准备了，你可以继续聊天，拍好后我会发出来。"
            if result is None:
                result = event.plain_result("")
                event.set_result(result)
            result.chain.append(Plain(text))
        if ack_task_id:
            await manager.mark_ack(ack_task_id, "decorated")

    @filter.after_message_sent(priority=200)
    async def confirm_background_task_result(self, event: AstrMessageEvent) -> None:
        manager = self.background_tasks
        if manager is None:
            return
        sent = bool(getattr(event, "_has_send_oper", False))
        ack_task_id = str(event.get_extra("_gitee_bg_ack_task_id", "") or "")
        if ack_task_id:
            try:
                await manager.mark_ack(ack_task_id, "sent" if sent else "unknown")
            except BackgroundTaskError:
                pass

        token = str(event.get_extra("_gitee_bg_notification_token", "") or "")
        attempt_id = str(event.get_extra("_gitee_bg_notification_attempt", "") or "")
        if token and attempt_id:
            await manager.mark_notification(
                token,
                "sent" if sent else "unknown",
                attempt_id=attempt_id,
            )

        reset_command = str(event.get_extra("_gitee_bg_reset_candidate", "") or "")
        if reset_command and bool(
            event.get_extra("_clean_group_context_session", False)
        ):
            await self._cancel_background_scope_with_notifications(
                manager,
                umo=str(event.unified_msg_origin or ""),
                sender_id=str(event.get_sender_id() or ""),
                reason=f"session_reset:{reset_command}",
                suppress_future_injection=True,
            )
        if reset_command:
            gate = self._background_send_gates.pop(
                str(event.unified_msg_origin or ""), None
            )
            if gate is not None:
                gate.set()

    def _remember_last_image(self, event: AstrMessageEvent, image_path: Path) -> None:
        try:
            user_id = str(event.get_sender_id() or "")
        except Exception:
            user_id = ""
        if not user_id:
            return
        self._last_image_by_user[user_id] = Path(image_path)

    @staticmethod
    def _as_int(value: Any, *, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_bool(value: Any, *, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"1", "true", "yes", "y", "on", "enable", "enabled"}:
                return True
            if v in {"0", "false", "no", "n", "off", "disable", "disabled", ""}:
                return False
        return default

    def _patch_tool_image_cache_runtime(self) -> None:
        try:
            from astrbot.core.agent import tool_image_cache as cache_module
        except Exception as exc:
            logger.debug(
                "[GiteeAIImagePlugin] skip tool image cache runtime patch: %s", exc
            )
            return

        cache_cls = getattr(cache_module, "ToolImageCache", None)
        cache_obj = getattr(cache_module, "tool_image_cache", None)
        cached_image_cls = getattr(cache_module, "CachedImage", None)
        if cache_cls is None or cache_obj is None or cached_image_cls is None:
            return
        if getattr(cache_cls, "_gitee_aiimg_runtime_patch", False):
            return

        def _patched_save_image(
            cache_self,
            base64_data: str,
            tool_call_id: str,
            tool_name: str,
            index: int = 0,
            mime_type: str = "image/png",
        ):
            ext = cache_self._get_file_extension(mime_type)
            cache_dir_value = str(getattr(cache_self, "_cache_dir", "") or "").strip()
            cache_dir = (
                Path(cache_dir_value)
                if cache_dir_value
                else Path(get_astrbot_temp_path())
                / getattr(cache_self, "CACHE_DIR_NAME", "tool_images")
            )
            file_path = cache_dir / f"{tool_call_id}_{index}{ext}"

            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                image_bytes = base64.b64decode(base64_data)
                file_path.write_bytes(image_bytes)
            except Exception as exc:
                logger.error(f"Failed to save tool image: {exc}")
                raise

            cache_self._cache_dir = str(cache_dir)
            logger.debug(
                "[GiteeAIImagePlugin] tool image cache runtime patch wrote: %s",
                file_path,
            )
            return cached_image_cls(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                file_path=str(file_path),
                mime_type=mime_type,
            )

        cache_cls.save_image = _patched_save_image
        cache_cls._gitee_aiimg_runtime_patch = True
        cache_obj._cache_dir = str(
            Path(get_astrbot_temp_path())
            / getattr(cache_cls, "CACHE_DIR_NAME", "tool_images")
        )
        Path(cache_obj._cache_dir).mkdir(parents=True, exist_ok=True)
        logger.info(
            "[GiteeAIImagePlugin] tool image cache runtime patch active: %s",
            cache_obj._cache_dir,
        )

    def _get_max_user_concurrency(self) -> int:
        v = self._as_int(self.config.get("max_user_concurrency", 2), default=2)
        return max(1, min(10, v))

    def _get_max_user_video_concurrency(self) -> int:
        v = self._as_int(self.config.get("max_user_video_concurrency", 1), default=1)
        return max(1, min(5, v))

    def _debounce_key(self, event: AstrMessageEvent, prefix: str, user_id: str) -> str:
        """尽量用消息维度去重，避免同用户短时间内无法并发提交多条任务。"""
        mid = str(
            getattr(getattr(event, "message_obj", None), "message_id", "") or ""
        ).strip()
        origin = str(getattr(event, "unified_msg_origin", "") or "").strip()
        if mid and origin:
            return f"{prefix}:{origin}:{mid}"
        return f"{prefix}:{user_id}"

    async def _begin_user_job(self, user_id: str, *, kind: str) -> bool:
        user_id = str(user_id or "").strip()
        if not user_id:
            return True

        if kind == "video":
            limit = self._get_max_user_video_concurrency()
            store = self._video_inflight
        else:
            limit = self._get_max_user_concurrency()
            store = self._image_inflight

        async with self._concurrency_lock:
            cur = int(store.get(user_id, 0) or 0)
            if cur >= limit:
                return False
            store[user_id] = cur + 1
            return True

    async def _end_user_job(self, user_id: str, *, kind: str) -> None:
        user_id = str(user_id or "").strip()
        if not user_id:
            return

        store = self._video_inflight if kind == "video" else self._image_inflight
        async with self._concurrency_lock:
            cur = int(store.get(user_id, 0) or 0)
            if cur <= 1:
                store.pop(user_id, None)
            else:
                store[user_id] = cur - 1

    @staticmethod
    def _is_rich_media_transfer_failed(exc: Exception | None) -> bool:
        if exc is None:
            return False
        msg = f"{exc!r} {exc}".lower()
        return "rich media transfer failed" in msg

    def _get_send_conf(self) -> dict[str, Any]:
        conf = self.config.get("send", {}) if isinstance(self.config, dict) else {}
        return conf if isinstance(conf, dict) else {}

    def _is_weixin_event(self, event: AstrMessageEvent | None) -> bool:
        if not event:
            return False
        names: list[str] = []
        try:
            names.append(str(event.get_platform_name() or ""))
        except Exception:
            pass

        platform_inst = getattr(event, "platform", None)
        if platform_inst is not None:
            try:
                meta = platform_inst.meta() if hasattr(platform_inst, "meta") else None
                if meta:
                    names.append(str(getattr(meta, "name", "") or ""))
                    names.append(str(getattr(meta, "id", "") or ""))
            except Exception:
                pass

        return any(name.strip().lower() == "weixin_oc" for name in names)

    def _get_weixin_timeout_ms(self) -> int:
        conf = self._get_send_conf()
        timeout_seconds = self._as_int(
            conf.get("weixin_api_timeout_seconds", 60), default=60
        )
        timeout_ms = timeout_seconds * 1000
        return max(15000, min(timeout_ms, 300000))

    def _apply_weixin_timeout(self, platform_inst: Any) -> None:
        if not platform_inst:
            return
        timeout_ms = self._get_weixin_timeout_ms()
        try:
            old_timeout = getattr(platform_inst, "api_timeout_ms", None)
            if old_timeout != timeout_ms:
                setattr(platform_inst, "api_timeout_ms", timeout_ms)

            client = getattr(platform_inst, "client", None)
            if client and getattr(client, "api_timeout_ms", None) != timeout_ms:
                setattr(client, "api_timeout_ms", timeout_ms)
        except Exception as exc:
            logger.debug("[GiteeAIImagePlugin] 设置 weixin_oc 超时失败: %s", exc)

    def _get_weixin_send_temp_dir(self) -> Path:
        return Path(self.data_dir) / "Temp"

    def _is_weixin_send_temp_path(self, image_path: Path) -> bool:
        try:
            p = Path(image_path).resolve(strict=False)
            temp_dir = self._get_weixin_send_temp_dir().resolve(strict=False)
            return (
                p.parent == temp_dir
                and p.name.startswith("weixin_send_")
                and p.suffix.lower() == ".jpg"
            )
        except Exception:
            return False

    def _cleanup_weixin_send_temp_images_sync(self) -> None:
        temp_dir = self._get_weixin_send_temp_dir()
        try:
            if not temp_dir.exists():
                return

            now = time.time()
            entries: list[tuple[Path, float]] = []
            for p in temp_dir.glob(self.WEIXIN_SEND_TEMP_PATTERN):
                try:
                    if not p.is_file():
                        continue
                    st = p.stat()
                except OSError:
                    continue
                entries.append((p, st.st_mtime))

            stale = [
                p
                for p, mtime in entries
                if now - mtime > self.WEIXIN_SEND_TEMP_TTL_SECONDS
            ]
            stale_keys = {str(p.resolve(strict=False)) for p in stale}
            remaining = [
                item
                for item in entries
                if str(item[0].resolve(strict=False)) not in stale_keys
            ]
            if len(remaining) > self.WEIXIN_SEND_TEMP_MAX_FILES:
                remaining.sort(key=lambda item: item[1])
                overflow = len(remaining) - self.WEIXIN_SEND_TEMP_MAX_FILES
                stale.extend(p for p, _ in remaining[:overflow])

            seen: set[str] = set()
            for p in stale:
                try:
                    key = str(p.resolve(strict=False))
                    if key in seen:
                        continue
                    seen.add(key)
                    p.unlink(missing_ok=True)
                except Exception as exc:
                    logger.debug(
                        "[GiteeAIImagePlugin] 清理 weixin_oc 临时图片失败: %s, err=%s",
                        p,
                        exc,
                    )
        except Exception as exc:
            logger.debug("[GiteeAIImagePlugin] 扫描 weixin_oc 临时图片失败: %s", exc)

    def _remove_weixin_send_temp_image_sync(self, image_path: Path) -> None:
        p = Path(image_path)
        if not self._is_weixin_send_temp_path(p):
            return
        try:
            p.unlink(missing_ok=True)
            logger.debug("[GiteeAIImagePlugin] 已清理 weixin_oc 发送临时图片: %s", p)
        except Exception as exc:
            logger.debug(
                "[GiteeAIImagePlugin] 删除 weixin_oc 发送临时图片失败: %s, err=%s",
                p,
                exc,
            )

    def _compress_image_for_weixin_sync(self, image_path: Path) -> Path:
        conf = self._get_send_conf()
        if not self._as_bool(conf.get("weixin_compress_images", True), default=True):
            return image_path

        p = Path(image_path)
        if not p.exists():
            return p

        try:
            from PIL import Image as PILImage
            from PIL import ImageOps
        except Exception as exc:
            logger.debug(
                "[GiteeAIImagePlugin] Pillow 不可用，跳过 weixin_oc 图片优化: %s",
                exc,
            )
            return p

        max_side = self._as_int(conf.get("weixin_image_max_side", 4096), default=4096)
        max_kb = self._as_int(
            conf.get("weixin_image_max_size_kb", 10240), default=10240
        )
        max_side = max(1600, min(max_side, 8192))
        target_bytes = max(512, max_kb) * 1024

        try:
            raw_size = p.stat().st_size
            with PILImage.open(p) as im:
                im = ImageOps.exif_transpose(im)
                width, height = im.size
                if raw_size <= target_bytes and max(width, height) <= max_side:
                    return p

                if im.mode in ("RGBA", "LA") or (
                    im.mode == "P" and "transparency" in im.info
                ):
                    bg = PILImage.new("RGB", im.size, (255, 255, 255))
                    rgba = im.convert("RGBA")
                    bg.paste(rgba, mask=rgba.split()[-1])
                    im = bg
                else:
                    im = im.convert("RGB")

                if max(width, height) > max_side:
                    resampling = getattr(
                        getattr(PILImage, "Resampling", PILImage), "LANCZOS"
                    )
                    im.thumbnail((max_side, max_side), resampling)

                temp_dir = Path(self.data_dir) / "Temp"
                temp_dir.mkdir(parents=True, exist_ok=True)
                self._cleanup_weixin_send_temp_images_sync()
                digest_src = (
                    f"{p}:{raw_size}:{p.stat().st_mtime}:{max_side}:{max_kb}".encode(
                        "utf-8", errors="ignore"
                    )
                )
                digest = hashlib.md5(digest_src).hexdigest()[:12]
                out_path = temp_dir / f"weixin_send_{digest}_{time.time_ns()}.jpg"

                for quality in (95, 93, 90, 88, 85, 82, 78, 74, 70):
                    im.save(
                        out_path,
                        format="JPEG",
                        quality=quality,
                        optimize=True,
                        progressive=True,
                        subsampling=0 if quality >= 90 else -1,
                    )
                    if out_path.stat().st_size <= target_bytes:
                        break

                out_size = out_path.stat().st_size
                if out_size < raw_size:
                    logger.info(
                        "[GiteeAIImagePlugin] 已为 weixin_oc 优化图片: %.2fMB -> %.2fMB, 分辨率 %sx%s -> %sx%s",
                        raw_size / 1024 / 1024,
                        out_size / 1024 / 1024,
                        width,
                        height,
                        im.size[0],
                        im.size[1],
                    )
                    return out_path
        except Exception as exc:
            logger.warning(
                "[GiteeAIImagePlugin] weixin_oc 图片优化失败，继续发送原图: %s",
                exc,
            )

        return p

    async def _prepare_image_for_send(
        self, event: AstrMessageEvent, image_path: Path
    ) -> Path:
        if self._is_weixin_event(event):
            self._apply_weixin_timeout(getattr(event, "platform", None))
            return await asyncio.to_thread(
                self._compress_image_for_weixin_sync, image_path
            )
        return Path(image_path)

    @staticmethod
    def _build_compact_image_bytes(
        image_path: Path, *, max_side: int = 2048, target_bytes: int = 3_500_000
    ) -> bytes | None:
        """Build a smaller JPEG variant for platforms that reject large rich-media upload."""
        try:
            from PIL import Image as PILImage
        except Exception:
            return None

        try:
            with PILImage.open(image_path) as im:
                if im.mode not in {"RGB", "L"}:
                    im = im.convert("RGB")
                elif im.mode == "L":
                    im = im.convert("RGB")

                w, h = im.size
                if max(w, h) > max_side:
                    ratio = float(max_side) / float(max(w, h))
                    nw = max(1, int(w * ratio))
                    nh = max(1, int(h * ratio))
                    resampling = getattr(
                        getattr(PILImage, "Resampling", PILImage), "LANCZOS"
                    )
                    im = im.resize((nw, nh), resampling)

                for q in (88, 82, 76, 70, 64):
                    buf = io.BytesIO()
                    im.save(
                        buf,
                        format="JPEG",
                        quality=q,
                        optimize=True,
                        progressive=True,
                    )
                    data = buf.getvalue()
                    if data and (len(data) <= target_bytes or q == 64):
                        return data
        except Exception:
            return None
        return None

    def _is_selfie_enabled(self) -> bool:
        conf = self._get_feature("selfie")
        return self._as_bool(conf.get("enabled", True), default=True)

    def _is_selfie_llm_enabled(self) -> bool:
        conf = self._get_feature("selfie")
        return self._as_bool(conf.get("llm_tool_enabled", True), default=True)

    @staticmethod
    def _selfie_disabled_message() -> str:
        return "自拍参考图模式已关闭（features.selfie.enabled=false）"

    def _get_busy_schedule_media_recorder(self, event: AstrMessageEvent):
        return _resolve_busy_schedule_media_recorder(event, self.context)

    async def _record_busy_schedule_media_success(
        self, event: AstrMessageEvent, media_type: str, operation_id: str
    ) -> None:
        callback = self._get_busy_schedule_media_recorder(event)
        if not callable(callback):
            logger.debug("[aiimg_generate] BusySchedule media recorder is unavailable")
            return
        try:
            result = callback(
                event.unified_msg_origin,
                {media_type},
                operation_id=operation_id,
            )
            if hasattr(result, "__await__"):
                await result
        except Exception as exc:
            logger.warning("[aiimg_generate] failed to record media send: %s", exc)

    async def _send_image_with_fallback(
        self, event: AstrMessageEvent, image_path: Path, *, max_attempts: int = 5
    ) -> SendImageResult:
        """Send image with retries and fallback to base64 bytes.

        Avoids wasting generation credits when platform send fails transiently.
        """
        original_path = Path(image_path)
        operation_id = f"image:{event.unified_msg_origin}:{uuid.uuid4().hex}"
        p = await self._prepare_image_for_send(event, original_path)
        should_cleanup_temp = self._is_weixin_send_temp_path(p) and (
            p.resolve(strict=False) != original_path.resolve(strict=False)
        )

        async def finish(result: SendImageResult) -> SendImageResult:
            if result.ok:
                await self._record_busy_schedule_media_success(
                    event, "image", operation_id
                )
            if should_cleanup_temp:
                await asyncio.to_thread(self._remove_weixin_send_temp_image_sync, p)
                if result.cached_path == p:
                    result.cached_path = original_path
            return result

        if not p.exists():
            logger.warning("[send_image] file not found: %s", p)
            return await finish(
                SendImageResult(ok=False, reason="file_not_found", cached_path=p)
            )

        # Large original images (e.g. 4K 20MB+) are likely to fail rich-media upload.
        # Prefer sending as a normal file first so the original bytes are preserved.
        try:
            size_bytes = int(p.stat().st_size)
        except Exception:
            size_bytes = 0

        file_send_tries = 0

        async def try_send_as_file(trigger: str) -> bool:
            nonlocal file_send_tries
            if file_send_tries >= 2:
                return False
            file_send_tries += 1
            try:
                await event.send(event.chain_result([File(name=p.name, file=str(p))]))
                logger.info(
                    "[send_image][file-fallback-v2] file send success: %s (%s bytes), trigger=%s, try=%s",
                    p.name,
                    size_bytes,
                    trigger,
                    file_send_tries,
                )
                return True
            except Exception as e:
                logger.warning(
                    "[send_image][file-fallback-v2] file send failed: trigger=%s, try=%s, err=%s",
                    trigger,
                    file_send_tries,
                    e,
                )
                return False

        if size_bytes > self.IMAGE_AS_FILE_THRESHOLD_BYTES:
            if await try_send_as_file("size_threshold"):
                return await finish(
                    SendImageResult(ok=True, cached_path=p, used_fallback=True)
                )

        delay = 1.5
        last_exc: Exception | None = None
        attempts = max(1, int(max_attempts))
        rich_media_failures = 0
        compact_bytes: bytes | None = None
        compact_prepared = False
        for attempt in range(1, attempts + 1):
            fs_exc: Exception | None = None
            bytes_exc: Exception | None = None
            compact_exc: Exception | None = None
            fs_failed_by_rich_media = False

            try:
                await event.send(event.chain_result([Image.fromFileSystem(str(p))]))
                return await finish(
                    SendImageResult(ok=True, cached_path=p, used_fallback=False)
                )
            except Exception as e:
                fs_exc = e
                last_exc = e
                if self._is_rich_media_transfer_failed(e):
                    fs_failed_by_rich_media = True
                logger.debug(
                    "[send_image] fromFileSystem failed (attempt=%s/%s): %s",
                    attempt,
                    attempts,
                    e,
                )

            try:
                data = await asyncio.to_thread(p.read_bytes)
                await event.send(event.chain_result([Image.fromBytes(data)]))
                if fs_exc is not None:
                    logger.info(
                        "[send_image] fromBytes fallback succeeded (attempt=%s/%s).",
                        attempt,
                        attempts,
                    )
                return await finish(
                    SendImageResult(ok=True, cached_path=p, used_fallback=True)
                )
            except Exception as e:
                bytes_exc = e
                last_exc = e
                logger.debug(
                    "[send_image] fromBytes failed (attempt=%s/%s): %s",
                    attempt,
                    attempts,
                    e,
                )

            # If rich-media channel is failing, immediately try original-file sending.
            if self._is_rich_media_transfer_failed(
                fs_exc
            ) or self._is_rich_media_transfer_failed(bytes_exc):
                if await try_send_as_file("rich_media_transfer_failed"):
                    return await finish(
                        SendImageResult(ok=True, cached_path=p, used_fallback=True)
                    )

            # Extra fallback for repeated rich-media failures: compress and retry by bytes.
            if self._is_rich_media_transfer_failed(
                fs_exc
            ) or self._is_rich_media_transfer_failed(bytes_exc):
                if not compact_prepared:
                    compact_prepared = True
                    compact_bytes = await asyncio.to_thread(
                        self._build_compact_image_bytes, p
                    )
                    if compact_bytes:
                        logger.info(
                            "[send_image] prepared compact fallback image: %s -> %s bytes",
                            p,
                            len(compact_bytes),
                        )
                if compact_bytes:
                    try:
                        await event.send(
                            event.chain_result([Image.fromBytes(compact_bytes)])
                        )
                        logger.info(
                            "[send_image] compact fromBytes fallback succeeded (attempt=%s/%s).",
                            attempt,
                            attempts,
                        )
                        return await finish(
                            SendImageResult(ok=True, cached_path=p, used_fallback=True)
                        )
                    except Exception as e:
                        compact_exc = e
                        last_exc = e
                        logger.debug(
                            "[send_image] compact fromBytes failed (attempt=%s/%s): %s",
                            attempt,
                            attempts,
                            e,
                        )

            attempt_has_rich_media = (
                self._is_rich_media_transfer_failed(fs_exc)
                or self._is_rich_media_transfer_failed(bytes_exc)
                or self._is_rich_media_transfer_failed(compact_exc)
            )
            if attempt_has_rich_media:
                rich_media_failures += 1

            if fs_exc is not None and bytes_exc is not None and compact_exc is not None:
                logger.debug(
                    "[send_image] attempt=%s/%s failed on all channels.",
                    attempt,
                    attempts,
                )
            elif fs_exc is not None and bytes_exc is not None:
                logger.debug(
                    "[send_image] attempt=%s/%s failed on both channels.",
                    attempt,
                    attempts,
                )
            elif fs_exc is not None and fs_failed_by_rich_media:
                logger.debug(
                    "[send_image] attempt=%s/%s failed by rich media transfer.",
                    attempt,
                    attempts,
                )
            else:
                logger.debug(
                    "[send_image] attempt=%s/%s failed to send image.",
                    attempt,
                    attempts,
                )

            if rich_media_failures >= 2:
                logger.info(
                    "[send_image] detected repeated rich media transfer failures, stop retrying early."
                )
                break

            if attempt < attempts:
                await _async_pause(delay)
                delay = min(delay * 1.8, 8.0)

        reason = (
            "rich_media_transfer_failed"
            if self._is_rich_media_transfer_failed(last_exc)
            else "send_failed"
        )
        logger.error(
            "[send_image] failed after retries: reason=%s, err=%s", reason, last_exc
        )
        return await finish(
            SendImageResult(
                ok=False,
                reason=reason,
                cached_path=p,
                last_error=str(last_exc or ""),
            )
        )

    def _register_preset_commands(self):
        """动态注册预设命令

        为每个预设创建对应的命令，如 /手办化, /Q版化 等
        """
        preset_names = self.edit.get_preset_names()
        if not preset_names:
            return

        for preset_name in preset_names:
            # 创建闭包捕获 preset_name
            self._create_and_register_preset_handler(preset_name)

        logger.info(f"[GiteeAIImagePlugin] 已注册 {len(preset_names)} 个预设命令")

    def _create_and_register_preset_handler(self, preset_name: str):
        """为单个预设创建并注册命令处理器

        支持: /手办化 [额外提示词]
        例如: /手办化 加点金色元素
        """

        # 默认后端命令: /手办化
        async def preset_handler(event: AstrMessageEvent):
            # 提取命令后的额外提示词
            extra_prompt = self._extract_extra_prompt(event, preset_name)
            await self._do_edit_direct(event, extra_prompt, preset=preset_name)

        preset_handler.__name__ = f"preset_{preset_name}"
        preset_handler.__doc__ = f"预设改图: {preset_name} [额外提示词]"

        self.context.register_commands(
            star_name="astrbot_plugin_gitee_aiimg",
            command_name=preset_name,
            desc=f"预设改图: {preset_name}",
            priority=5,
            awaitable=preset_handler,
        )

    def _extract_extra_prompt(self, event: AstrMessageEvent, command_name: str) -> str:
        """从消息中提取命令后的额外提示词

        支持格式:
        - /手办化 加点金色元素 -> "加点金色元素"
        - /手办化@张三 背景是星空 -> "背景是星空"
        - /手办化@张三@李四 背景是星空 -> "背景是星空"

        注意: message_str 中 @用户 会被替换为空格或移除
        """
        msg = event.message_str.strip()
        # 移除命令前缀 (/, !, ., 等)
        # 兼容唤醒前缀：.视频 / 。视频 / ．视频
        if msg and msg[0] in "/!！.。．":
            msg = msg[1:]
        # 移除命令名
        if msg.startswith(command_name):
            msg = msg[len(command_name) :]
        msg = re.sub(r"@\S+\(\d+\)", " ", msg)
        # 清理多余空格
        return msg.strip()

    @staticmethod
    def _extract_command_arg_anywhere(message: str, command_name: str) -> str:
        """从任意位置提取“/命令 参数”，用于图片在前导致 @filter.command 不触发的场景。"""
        _found, arg = GiteeAIImagePlugin._find_command_arg_anywhere(
            message,
            command_name,
        )
        return arg

    @staticmethod
    def _find_command_arg_anywhere(
        message: str,
        command_name: str,
        *,
        allow_bare: bool = False,
    ) -> tuple[bool, str]:
        """查找带前缀命令；已被 AstrBot wake_prefix 剥离时允许裸命令。"""
        msg = (message or "").strip()
        if not msg:
            return False, ""
        for prefix in "/!！.。．":
            token = f"{prefix}{command_name}"
            idx = msg.find(token)
            while idx >= 0:
                end = idx + len(token)
                if end == len(msg) or msg[end].isspace():
                    return True, msg[end:].strip()
                idx = msg.find(token, idx + 1)
        if allow_bare and msg.startswith(command_name):
            end = len(command_name)
            if end == len(msg) or msg[end].isspace():
                return True, msg[end:].strip()
        return False, ""

    def _extract_command_arg_from_chain(
        self, event: AstrMessageEvent, command_name: str
    ) -> tuple[bool, str]:
        """从消息链中提取命令后的提示词。

        用于修复“/命令 + 图片 + 文本”时，平台把文本段无空格拼接到 `message_str`
        导致 command filter 和字符串提取都失效的问题。
        """
        try:
            chain = event.get_messages()
        except Exception:
            return False, ""

        found = False
        parts: list[str] = []
        for seg in chain:
            if isinstance(seg, (At, AtAll, Reply)):
                continue

            if not found:
                if not isinstance(seg, Plain):
                    continue
                plain = str(getattr(seg, "text", "") or "").lstrip()
                if not plain:
                    continue
                if plain[0] in "/!！.。．":
                    plain = plain[1:]
                if not plain.startswith(command_name):
                    continue
                found = True
                tail = plain[len(command_name) :].strip()
                if tail:
                    parts.append(tail)
                continue

            if isinstance(seg, Plain):
                text = str(getattr(seg, "text", "") or "").strip()
                if text:
                    parts.append(text)

        return found, " ".join(parts).strip()

    def _extract_chain_provider_id(self, item: object) -> str:
        if isinstance(item, str):
            return item.strip()
        if not isinstance(item, dict):
            return ""
        return str(
            item.get("provider_id")
            or item.get("id")
            or item.get("provider")
            or item.get("backend")
            or ""
        ).strip()

    def _normalize_chain_item(self, item: object) -> dict | None:
        pid = self._extract_chain_provider_id(item)
        if not pid:
            return None
        out = ""
        if isinstance(item, dict):
            out = str(item.get("output") or item.get("default_output") or "").strip()
        return {"provider_id": pid, "output": out} if out else {"provider_id": pid}

    def _parse_provider_override_prefix(self, text: str) -> tuple[str | None, str]:
        """仅当 @token 命中已配置 provider_id 时，才作为 provider 覆盖。"""
        s = (text or "").strip()
        if not s.startswith("@"):
            return None, s
        first, _, rest = s.partition(" ")
        candidate = first.lstrip("@").strip()
        if not candidate:
            return None, s
        if candidate in set(self.registry.provider_ids()):
            return candidate, rest.strip()
        logger.debug(
            "[provider_override] 忽略未知 @token，继续走自动链路: token=%s",
            candidate,
        )
        return None, s

    @staticmethod
    def _plain_starts_with_command(text: str, command_name: str) -> bool:
        plain = (text or "").lstrip()
        if not plain:
            return False
        for prefix in "/!！.。．":
            if plain.startswith(f"{prefix}{command_name}"):
                return True
        return False

    def _is_direct_command_message(
        self, event: AstrMessageEvent, command_names: tuple[str, ...]
    ) -> bool:
        """仅当“首个有效文本段”直接是命令时返回 True。

        用于 regex 兜底去重：避免正常 /命令 被重复处理；
        同时允许“图片在前、命令在后”的消息继续走兜底逻辑。
        """
        try:
            chain = event.get_messages()
        except Exception:
            return False
        if not chain:
            return False

        first_plain = ""
        for seg in chain:
            if isinstance(seg, (At, AtAll, Reply)):
                continue
            if isinstance(seg, Plain):
                first_plain = str(getattr(seg, "text", "") or "")
            break

        if not first_plain:
            return False
        return any(
            self._plain_starts_with_command(first_plain, name) for name in command_names
        )

    @staticmethod
    def _is_framework_direct_command_text(
        message: str, command_names: tuple[str, ...], *, allow_bare: bool = True
    ) -> bool:
        """按 AstrBot CommandFilter 的文本规则判断是否可直接命中 command handler。"""
        plain = " ".join(str(message or "").strip().split())
        if not plain:
            return False
        if plain[0] in "/!！.。．":
            plain = plain[1:].lstrip()
        return any(
            (plain == name if allow_bare else False) or plain.startswith(f"{name} ")
            for name in command_names
        )

    @staticmethod
    def _has_activated_handler(event: AstrMessageEvent, handler_name: str) -> bool:
        """检查本轮事件是否已经激活指定 handler，用于 regex fallback 去重。"""
        try:
            handlers = event.get_extra("activated_handlers", [])
        except Exception:
            return False
        for handler in handlers or []:
            if str(getattr(handler, "handler_name", "") or "") == handler_name:
                return True
            raw_handler = getattr(handler, "handler", None)
            if str(getattr(raw_handler, "__name__", "") or "") == handler_name:
                return True
        return False

    async def generate_selfie(
        self,
        event: AstrMessageEvent | None = None,
        *,
        action: str = "",
        prompt: str = "",
        aspect_ratio: str = "",
        size: str = "",
        **_kwargs: Any,
    ) -> dict[str, Any]:
        """Bridge for QQ Zone plugin: generate a selfie via the same selfie_ref path as /自拍."""
        image_prompt = str(action or prompt or "").strip()
        if not image_prompt:
            return {"success": False, "message": "empty prompt", "images": []}
        resolved_size = str(size or "").strip()
        if not resolved_size and aspect_ratio:
            resolved_size = self._resolve_ratio_size(str(aspect_ratio).strip())
        if event is None:
            return {
                "success": False,
                "message": "no event context for selfie ref lookup",
                "images": [],
            }
        try:
            path, _ = await self._generate_selfie_image_with_meta(
                event, image_prompt, None, size=resolved_size or None
            )
            return {"success": True, "images": [{"file_path": str(path)}]}
        except Exception as exc:
            return {"success": False, "message": str(exc), "images": []}

    async def terminate(self):
        self.debouncer.clear_all()
        start_task = getattr(self, "_background_start_task", None)
        if start_task is not None:
            start_task.cancel()
            await asyncio.gather(start_task, return_exceptions=True)
            self._background_start_task = None
        manager = getattr(self, "background_tasks", None)
        if manager is not None:
            try:
                await manager.close()
            except Exception as exc:
                logger.error(
                    "[background-image] manager shutdown failed: %s",
                    manager.sanitize_error(exc),
                )
            finally:
                self.background_tasks = None
        try:
            tasks = list(getattr(self, "_video_tasks", []))
            for t in tasks:
                t.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        except Exception:
            pass
        await self.imgr.close()
        await self.draw.close()
        await self.edit.close()
        await self.nb.close()
        await close_session()  # 关闭 utils.py 的 HTTP 会话

    # ==================== 文生图 ====================

    @filter.command("文生图")
    async def generate_image_with_presets(self, event: AstrMessageEvent):
        """支持文生图预设的图片生成命令。"""
        event.should_call_llm(True)
        parsed = self._parse_structured_image_request(event.message_str)
        if parsed is None or parsed.spec.source_command != "文生图":
            await mark_failed(event)
            return

        spec = parsed.spec
        if not str(spec.effective_prompt or "").strip():
            await mark_failed(event)
            return

        user_id = str(event.get_sender_id() or "")
        request_id = self._debounce_key(event, "draw_preset", user_id)
        if self.debouncer.hit(request_id):
            await mark_failed(event)
            return
        if not await self._begin_user_job(user_id, kind="image"):
            await mark_failed(event)
            return

        try:
            await mark_processing(event)
            executed = await self._execute_image_task_spec(event, spec)
            self._remember_last_image(event, executed.image_path)
            sent = await self._send_image_with_fallback(event, executed.image_path)
            if not sent:
                await mark_failed(event)
                return
            await self._save_last_image_task_meta(event, executed.task_meta)
            await self._append_image_history_note(
                event,
                prompt=self._image_history_prompt_for_spec(spec),
                mode=executed.task_meta.get("mode"),
            )
            await mark_success(event)
        except Exception as exc:
            logger.error("[文生图预设] 失败: %s", exc, exc_info=True)
            await mark_failed(event)
        finally:
            await self._end_user_job(user_id, kind="image")

    @filter.command("aiimg", alias={"生图", "画图", "绘图", "出图"})
    async def generate_image_command(self, event: AstrMessageEvent, prompt: str):
        """生成图片指令

        用法: /aiimg [@provider_id] <提示词> [比例]
        示例: /aiimg 一个女孩 9:16
        支持比例: 1:1, 4:3, 3:4, 3:2, 2:3, 16:9, 9:16
        """
        event.should_call_llm(True)
        # 解析参数
        arg = event.message_str.partition(" ")[2]
        if not arg:
            await mark_failed(event)
            return
        provider_override: str | None = None
        provider_override, arg = self._parse_provider_override_prefix(arg)
        if not arg:
            await mark_failed(event)
            return

        prompt = arg.strip()
        size: str | None = None
        parts = arg.split()
        if parts and parts[-1] in self.SUPPORTED_RATIOS:
            ratio = parts[-1]
            prompt = " ".join(parts[:-1]).strip()
            size = self._resolve_ratio_size(ratio)

        if not prompt:
            await mark_failed(event)
            return

        user_id = str(event.get_sender_id() or "")
        request_id = self._debounce_key(event, "generate", user_id)

        # 防抖检查
        if self.debouncer.hit(request_id):
            await mark_failed(event)
            return

        if not await self._begin_user_job(user_id, kind="image"):
            await mark_failed(event)
            return

        try:
            # 标记处理中
            await mark_processing(event)
            t_start = time.perf_counter()
            image_path = await self.draw.generate(
                prompt, size=size, provider_id=provider_override
            )
            t_end = time.perf_counter()

            self._remember_last_image(event, image_path)
            sent = await self._send_image_with_fallback(event, image_path)
            if not sent:
                await mark_failed(event)
                logger.warning(
                    "[文生图] 图片发送失败，已仅使用表情标注: reason=%s", sent.reason
                )
                return

            # 标记成功
            await mark_success(event)
            await self._append_image_history_note(
                event,
                prompt=prompt,
                mode="text",
            )
            logger.info(
                f"[文生图] 完成: {prompt[:30] if prompt else '文生图'}..., 耗时={t_end - t_start:.2f}s"
            )

        except Exception as e:
            logger.error(f"[文生图] 失败: {e}")
            await mark_failed(event)
        finally:
            await self._end_user_job(user_id, kind="image")

    @filter.regex(r"[/!！.。．]批量(?:\s*\d+|\d+)(?:\s|$)", priority=-10)
    async def batch_image_command(self, event: AstrMessageEvent):
        """批量图片任务入口。"""
        event.should_call_llm(True)
        fragment = self._extract_batch_command_fragment(event.message_str)
        parsed = self._parse_structured_image_request(fragment)
        if parsed is None or parsed.batch_count <= 1:
            await mark_failed(event)
            return
        if parsed.batch_count > self._get_batch_max_count():
            await event.send(
                event.plain_result(
                    f"批量数量过大，当前上限为 {self._get_batch_max_count()}。"
                )
            )
            await mark_failed(event)
            return

        user_id = str(event.get_sender_id() or "")
        request_id = self._debounce_key(event, "batch_image", user_id)
        if self.debouncer.hit(request_id):
            await mark_failed(event)
            return
        if not await self._begin_user_job(user_id, kind="image"):
            await mark_failed(event)
            return

        try:
            await mark_processing(event)
            specs = [parsed.spec for _ in range(parsed.batch_count)]
            results = await self._run_batch_specs(event, specs)
            title = f"{self._batch_mode_label(parsed.spec)} x{parsed.batch_count}"
            sent_results = await self._send_batch_results(event, results, title=title)
            if sent_results:
                await self._remember_batch_success(
                    event,
                    sent_results,
                    prompt=self._image_history_prompt_for_spec(parsed.spec),
                    mode=parsed.spec.mode,
                    count=len(sent_results),
                )
                await mark_success(event)
            else:
                await mark_failed(event)
        except Exception as exc:
            logger.error("[批量图片] 失败: %s", exc, exc_info=True)
            await mark_failed(event)
        finally:
            await self._end_user_job(user_id, kind="image")

    # ==================== 图生图/改图 ====================

    @filter.command("aiedit", alias={"图生图", "改图", "修图"})
    async def edit_image_default(self, event: AstrMessageEvent, prompt: str):
        """使用默认后端改图

        用法: /aiedit <提示词>
        需要同时发送或引用图片
        """
        event.should_call_llm(True)
        await self._do_edit(event, prompt, backend=None)

    @filter.command("重发图片")
    async def resend_last_image(self, event: AstrMessageEvent):
        """重发最近一次生成/改图的图片（不重新生成，不消耗次数）。"""
        user_id = str(event.get_sender_id() or "")
        p = self._last_image_by_user.get(user_id)
        if not p:
            await mark_failed(event)
            return
        if not Path(p).exists():
            await mark_failed(event)
            return
        ok = await self._send_image_with_fallback(event, p)
        if ok:
            await mark_success(event)
        else:
            await mark_failed(event)

    @filter.regex(r".*(?:[/!！.。．])?(改图|图生图|修图|aiedit)", priority=-10)
    async def edit_image_regex_fallback(self, event: AstrMessageEvent):
        """兼容“图片在前、文字在后”的消息：确保 /改图 能触发。"""
        msg = (event.message_str or "").strip()
        command_names = ("改图", "图生图", "修图", "aiedit")
        if self._is_framework_direct_command_text(msg, command_names, allow_bare=False):
            return
        try:
            if not await self._has_message_images_or_avatar_mentions(event):
                return
        except Exception:
            return

        prompt = ""
        matched = False
        for name in command_names:
            prompt = self._extract_command_arg_anywhere(msg, name)
            found_in_chain, chain_prompt = self._extract_command_arg_from_chain(
                event, name
            )
            if prompt or found_in_chain:
                matched = True
                if not prompt:
                    prompt = chain_prompt
                break
        if matched:
            event.should_call_llm(True)
            await self._do_edit(event, prompt, backend=None)
            event.stop_event()

    @filter.regex(r".*[/!！.。．][^\s]+", priority=-10)
    async def preset_regex_fallback(self, event: AstrMessageEvent):
        """兼容“图片在前、预设命令在后”的消息：确保 /<预设名> 能触发。"""
        msg = (event.message_str or "").strip()
        preset_names = self.edit.get_preset_names()
        if not preset_names:
            return

        # 如果首段文本本来就是 /预设，则交给 command handler，避免重复处理
        try:
            if self._is_direct_command_message(event, tuple(preset_names)):
                return
        except Exception:
            pass

        # 仅当消息/引用里有图或有效 @ 头像目标时才兜底，避免误伤其它插件命令
        try:
            if not await self._has_message_images_or_avatar_mentions(event):
                return
        except Exception:
            return

        # 在任意位置找到第一个匹配的预设命令
        used_preset: str | None = None
        for name in preset_names:
            for prefix in "/!！.。．":
                if f"{prefix}{name}" in msg:
                    used_preset = name
                    break
            if used_preset:
                break

        if not used_preset:
            return

        extra_prompt = self._extract_command_arg_anywhere(msg, used_preset)
        await self._do_edit_direct(event, extra_prompt, preset=used_preset)
        event.stop_event()

    # ==================== Bot 自拍（参考照） ====================

    @filter.command("自拍")
    async def selfie_command(self, event: AstrMessageEvent):
        """使用“自拍参考照”生成 Bot 自拍。

        用法:
        - /自拍 <提示词>
        - 可附带多张参考图（衣服/姿势/场景）作为额外参考
        """
        if not self._is_selfie_enabled():
            await mark_failed(event)
            return
        event.should_call_llm(True)
        prompt = self._extract_extra_prompt(event, "自拍")
        await self._do_selfie(event, prompt, backend=None)

    @filter.regex(r"(?:自拍|.*[/!！.。．]自拍)(?:\s|$)", priority=-10)
    async def selfie_regex_fallback(self, event: AstrMessageEvent):
        """兼容“图片在前、文字在后”的消息：确保 /自拍 能触发。"""
        if self._has_activated_handler(event, "selfie_command"):
            return
        msg = (event.message_str or "").strip()
        found, prompt = self._find_command_arg_anywhere(
            msg,
            "自拍",
            allow_bare=bool(getattr(event, "is_at_or_wake_command", False)),
        )
        if found:
            event.should_call_llm(True)
            if not self._is_selfie_enabled():
                await mark_failed(event)
                event.stop_event()
                return
            await self._do_selfie(event, prompt, backend=None)
            event.stop_event()

    @filter.command("自拍参考")
    async def selfie_reference_command(self, event: AstrMessageEvent):
        """管理自拍参考照（建议仅管理员使用）。

        用法:
        - 发送图片 + /自拍参考 设置
        - /自拍参考 查看
        - /自拍参考 删除
        """
        event.should_call_llm(True)
        if not self._is_selfie_enabled():
            await mark_failed(event)
            return
        arg = self._extract_extra_prompt(event, "自拍参考")
        action, _, _rest = (arg or "").strip().partition(" ")
        action = action.strip().lower()

        if not action or action in {"帮助", "help", "h"}:
            msg = (
                "📸 自拍参考照\n"
                "━━━━━━━━━━━━━━\n"
                "设置：发送图片 + /自拍参考 设置\n"
                "查看：/自拍参考 查看\n"
                "删除：/自拍参考 删除\n"
                "━━━━━━━━━━━━━━\n"
                "生成自拍：/自拍 <提示词>\n"
                "可附带额外参考图（衣服/姿势/场景）"
            )
            yield event.plain_result(msg)
            return

        if action in {"设置", "set"}:
            await self._set_selfie_reference(event)
            return

        if action in {"查看", "show", "看"}:
            async for result in self._show_selfie_reference(event):
                yield result
            return

        if action in {"删除", "del", "delete"}:
            await self._delete_selfie_reference(event)
            return

        await mark_failed(event)

    @filter.regex(r"(?:自拍参考|.*[/!！.。．]自拍参考)(?:\s|$)", priority=-10)
    async def selfie_reference_regex_fallback(self, event: AstrMessageEvent):
        """兼容“图片在前、文字在后”的消息：确保 /自拍参考 能触发。"""
        if self._has_activated_handler(event, "selfie_reference_command"):
            return
        msg = (event.message_str or "").strip()
        found, arg = self._find_command_arg_anywhere(
            msg,
            "自拍参考",
            allow_bare=bool(getattr(event, "is_at_or_wake_command", False)),
        )
        if not found:
            return
        event.should_call_llm(True)
        if not self._is_selfie_enabled():
            await mark_failed(event)
            event.stop_event()
            return
        action, _, _rest = (arg or "").strip().partition(" ")
        action = action.strip().lower()

        if not action or action in {"帮助", "help", "h"}:
            yield event.plain_result(
                "📸 自拍参考照\n"
                "━━━━━━━━━━━━━━\n"
                "设置：发送图片 + /自拍参考 设置\n"
                "查看：/自拍参考 查看\n"
                "删除：/自拍参考 删除\n"
                "━━━━━━━━━━━━━━\n"
                "生成自拍：/自拍 <提示词>\n"
                "可附带额外参考图（衣服/姿势/场景）"
            )
            event.stop_event()
            return

        if action in {"设置", "set"}:
            await self._set_selfie_reference(event)
            event.stop_event()
            return

        if action in {"查看", "show", "看"}:
            async for r in self._show_selfie_reference(event):
                yield r
            event.stop_event()
            return

        if action in {"删除", "del", "delete"}:
            await self._delete_selfie_reference(event)
            event.stop_event()
            return

        await mark_failed(event)
        event.stop_event()

    # ==================== 视频生成 ====================

    @filter.command("视频")
    async def generate_video_command(self, event: AstrMessageEvent):
        """生成视频

        用法:
        - /视频 [@provider_id] <提示词>
        - /视频 [@provider_id] <预设名> [额外提示词]
        """
        event.should_call_llm(True)
        if not bool(self._get_feature("video").get("enabled", False)):
            await mark_failed(event)
            return
        arg = self._extract_extra_prompt(event, "视频")
        if not arg:
            await mark_failed(event)
            return

        provider_override, arg = self._parse_provider_override_prefix(arg)
        if not arg:
            await mark_failed(event)
            return

        preset, prompt = self._parse_video_args(arg)
        presets = self._get_video_presets()
        if preset and preset in presets:
            preset_prompt = presets[preset]
            prompt = f"{preset_prompt}, {prompt}" if prompt else preset_prompt

        user_id = str(event.get_sender_id() or "")
        request_id = self._debounce_key(event, "video", user_id)

        if self.debouncer.hit(request_id):
            await mark_failed(event)
            return

        if not await self._video_begin(user_id):
            await mark_failed(event)
            return

        try:
            await mark_processing(event)
        except Exception:
            await self._video_end(user_id)
            await mark_failed(event)
            return

        try:
            task = asyncio.create_task(
                self._async_generate_video(
                    event, prompt, user_id, provider_id=provider_override
                )
            )
        except Exception:
            await self._video_end(user_id)
            await mark_failed(event)
            return

        self._video_tasks.add(task)
        task.add_done_callback(lambda t: self._video_tasks.discard(t))
        return

    @filter.regex(r"[/!！.。．]视频(\s|$)", priority=-10)
    async def generate_video_regex_fallback(self, event: AstrMessageEvent):
        """兼容“图片在前、文字在后”的消息：确保 /视频 能触发。"""
        msg = (event.message_str or "").strip()
        if self._is_direct_command_message(event, ("视频",)):
            return

        arg = self._extract_command_arg_anywhere(msg, "视频")
        if not arg and "/视频" not in msg:
            return

        event.should_call_llm(True)
        if not bool(self._get_feature("video").get("enabled", False)):
            await mark_failed(event)
            event.stop_event()
            return
        if not arg:
            await mark_failed(event)
            event.stop_event()
            return

        provider_override, arg = self._parse_provider_override_prefix(arg)
        if not arg:
            await mark_failed(event)
            event.stop_event()
            return

        preset, prompt = self._parse_video_args(arg)
        presets = self._get_video_presets()
        if preset and preset in presets:
            preset_prompt = presets[preset]
            prompt = f"{preset_prompt}, {prompt}" if prompt else preset_prompt

        user_id = str(event.get_sender_id() or "")
        request_id = self._debounce_key(event, "video", user_id)

        if self.debouncer.hit(request_id):
            await mark_failed(event)
            event.stop_event()
            return

        if not await self._video_begin(user_id):
            await mark_failed(event)
            event.stop_event()
            return

        try:
            await mark_processing(event)
        except Exception:
            await self._video_end(user_id)
            await mark_failed(event)
            event.stop_event()
            return

        try:
            task = asyncio.create_task(
                self._async_generate_video(
                    event, prompt, user_id, provider_id=provider_override
                )
            )
        except Exception:
            await self._video_end(user_id)
            await mark_failed(event)
            event.stop_event()
            return

        self._video_tasks.add(task)
        task.add_done_callback(lambda t: self._video_tasks.discard(t))
        event.stop_event()
        return

    @filter.command("视频预设列表")
    async def list_video_presets(self, event: AstrMessageEvent):
        """列出所有可用视频预设"""
        event.should_call_llm(True)
        presets = self._get_video_presets()
        names = list(presets.keys())
        if not names:
            yield event.plain_result(
                "📋 视频预设列表\n暂无预设（请在配置 features.video.presets 中添加）"
            )
            return

        msg = "📋 视频预设列表\n"
        for name in names:
            msg += f"- {name}\n"
        msg += "\n用法: /视频 [@provider_id] <预设名> [额外提示词]"
        yield event.plain_result(msg)

    # ==================== 管理命令 ====================

    @filter.command("文生图预设列表")
    async def list_draw_presets(self, event: AstrMessageEvent):
        """列出所有可用文生图预设"""
        event.should_call_llm(True)
        presets = self._get_draw_presets()
        backends = self.draw._candidate_ids()
        draw_conf = self._get_feature("draw")
        chain = []
        for it in (
            draw_conf.get("chain", [])
            if isinstance(draw_conf.get("chain", []), list)
            else []
        ):
            pid = self._extract_chain_provider_id(it)
            if pid and pid not in chain:
                chain.append(pid)

        if not presets:
            msg = "📋 文生图预设列表\n"
            msg += "━━━━━━━━━━━━━━\n"
            msg += f"🔧 可用后端: {', '.join(backends)}\n"
            if chain:
                msg += f"⭐ 当前链路: {', '.join(chain)}\n"
            msg += "━━━━━━━━━━━━━━\n"
            msg += "📌 暂无预设\n"
            msg += "━━━━━━━━━━━━━━\n"
            msg += "💡 在配置 features.draw.presets 中添加:\n"
            msg += '  格式: "预设名:英文提示词"'
        else:
            msg = "📋 文生图预设列表\n"
            msg += "━━━━━━━━━━━━━━\n"
            msg += f"🔧 可用后端: {', '.join(backends)}\n"
            if chain:
                msg += f"⭐ 当前链路: {', '.join(chain)}\n"
            msg += "━━━━━━━━━━━━━━\n"
            msg += "📌 预设:\n"
            for name in presets:
                msg += f"  • {name}\n"
        msg += "━━━━━━━━━━━━━━\n"
        msg += "💡 用法: /文生图 [@provider_id] <预设名> [补充提示词]"
        yield event.plain_result(msg)

    @filter.command("预设列表")
    async def list_presets(self, event: AstrMessageEvent):
        """列出所有可用预设"""
        event.should_call_llm(True)
        presets = self.edit.get_preset_names()
        backends = self.edit.get_available_backends()
        edit_conf = self._get_feature("edit")
        chain = []
        for it in (
            edit_conf.get("chain", [])
            if isinstance(edit_conf.get("chain", []), list)
            else []
        ):
            pid = self._extract_chain_provider_id(it)
            if pid and pid not in chain:
                chain.append(pid)

        if not presets:
            msg = "📋 改图预设列表\n"
            msg += "━━━━━━━━━━━━━━\n"
            msg += f"🔧 可用后端: {', '.join(backends)}\n"
            if chain:
                msg += f"⭐ 当前链路: {', '.join(chain)}\n"
            msg += "━━━━━━━━━━━━━━\n"
            msg += "📌 暂无预设\n"
            msg += "━━━━━━━━━━━━━━\n"
            msg += "💡 在配置 features.edit.presets 中添加:\n"
            msg += '  格式: "触发词:英文提示词"'
        else:
            msg = "📋 改图预设列表\n"
            msg += "━━━━━━━━━━━━━━\n"
            msg += f"🔧 可用后端: {', '.join(backends)}\n"
            if chain:
                msg += f"⭐ 当前链路: {', '.join(chain)}\n"
            msg += "━━━━━━━━━━━━━━\n"
            msg += "📌 预设:\n"
            for name in presets:
                msg += f"  • {name}\n"
        msg += "━━━━━━━━━━━━━━\n"
        msg += "💡 用法: /aiedit [@provider_id] <提示词> [图片]"

        yield event.plain_result(msg)

    @filter.command("改图帮助")
    async def edit_help(self, event: AstrMessageEvent):
        """显示改图帮助"""
        event.should_call_llm(True)
        msg = """🎨 改图功能帮助

━━ 基础命令 ━━
/aiedit [@provider_id] <提示词>

━━ 使用方式 ━━
1. 发送图片 + 命令
2. 引用图片消息 + 命令

━━ 服务商链路 ━━
在 WebUI 配置：
- providers：添加服务商（id/url/key/model/超时/重试等）
- features.edit.chain：按顺序填写 provider_id（第一个=主用，其余=兜底）

━━ 自定义预设 ━━
查看预设：/预设列表
在 WebUI 配置 features.edit.presets 添加：
格式: 预设名:英文提示词
示例: 手办化:Transform into figurine style
"""

        yield event.plain_result(msg)

    # ==================== LLM 工具 ====================

    @filter.llm_tool(name="aiimg_generate")
    async def aiimg_generate(
        self,
        event: AstrMessageEvent,
        prompt: str,
        mode: str = "selfie_ref",
        backend: str = "auto",
        output: str = "",
        count: int = 1,
    ):
        """按用户要求生成、批量生成或编辑图片。

        Args:
            prompt(string): 完整的生成或修改要求。使用合规、克制的描述，不得写得过度暴露或违规，否则会审核失败。自拍模式可描述第一人称正在拍摄的画面，不强制露脸。
            mode(string): 默认 selfie_ref。selfie_ref 是自拍模式，但不局限于露脸自拍，也不要求出现脸或完整身体；第一人称拍摄眼前或手中食物、物品、环境等日常所见仍属于自拍，不要因未露脸改用文生图。省略 mode 或传 auto 均按 selfie_ref。自拍参考图缺失时报告失败，不得降级。edit 禁止自行使用，仅当用户明确要求修改其发送或引用的现有图片且提出具体修改时使用；没有明确要求或没有可编辑图片时不得使用。text 不允许自行使用，仅当用户明确要求文生图时使用。
            backend(string): auto=使用配置的服务商链；也可指定 provider_id。
            output(string): 可选输出尺寸或分辨率，例如 2048x2048 或 4K。
            count(number): 默认只能生成 1 张；只有用户明确说“生很多张”等批量要求时，才能设置为 2 至配置上限，不要自行开启批量。
        """
        get_extra = getattr(event, "get_extra", None)
        if callable(get_extra) and get_extra(_BACKGROUND_COMPLETION_EVENT_EXTRA, False):
            return None
        prompt = (prompt or "").strip()
        mode = self._normalize_llm_image_mode(mode)
        try:
            target_count = self._validate_llm_image_count(count)
        except ValueError as exc:
            await self._signal_llm_tool_failure(event)
            return self._llm_tool_text_result(str(exc))

        manager = self._background_manager_for_event(event)
        if manager is not None:
            try:
                if target_count > 1:
                    return await self._accept_background_batch(
                        event,
                        prompt=prompt,
                        count=target_count,
                        mode=mode,
                        backend=backend,
                        output=output,
                    )
                return await self._accept_background_single(
                    event,
                    prompt=prompt,
                    mode=mode,
                    backend=backend,
                    output=output,
                )
            except BackgroundTaskCapacityError:
                await self._signal_llm_tool_failure(event)
                return self._llm_tool_text_result(
                    "The background image queue is full, so this request was not submitted."
                )
            except BackgroundTaskError as exc:
                logger.warning(
                    "[background-image] synchronous fallback after task ledger failure: %s",
                    BackgroundImageTaskManager.sanitize_error(exc),
                )
            except Exception as exc:
                logger.error(
                    "[background-image] task preparation failed: %s",
                    BackgroundImageTaskManager.sanitize_error(exc),
                    exc_info=True,
                )
                await self._signal_llm_tool_failure(event)
                if self._is_missing_selfie_reference_error(exc):
                    return self._llm_tool_text_result(
                        "自拍参考照缺失，本次没有提交后台任务，也没有降级为文生图。"
                    )
                return self._llm_tool_text_result(
                    "The image request could not be prepared and was not submitted."
                )

        if target_count > 1:
            return await self._aiimg_batch_generate(
                event,
                prompt,
                count=target_count,
                mode=mode,
                backend=backend,
                output=output,
            )
        return await self._aiimg_generate_single(
            event,
            prompt=prompt,
            mode=mode,
            backend=backend,
            output=output,
        )

    async def _aiimg_generate_single(
        self,
        event: AstrMessageEvent,
        *,
        prompt: str,
        mode: str,
        backend: str,
        output: str,
    ):
        """Execute exactly one provider image task."""
        m = self._normalize_llm_image_mode(mode)

        # === TTL 去重检查（防止 ToolLoop 重复调用）===
        message_id = (
            getattr(getattr(event, "message_obj", None), "message_id", "") or ""
        )
        origin = getattr(event, "unified_msg_origin", "") or ""
        if message_id and origin:
            if self.debouncer.llm_tool_is_duplicate(message_id, origin):
                logger.debug(f"[aiimg_generate] 重复调用已拦截: msg_id={message_id}")
                await mark_success(event)
                return self._llm_tool_text_result(
                    "This image request was already handled for the same message. Do not run it again."
                )

        user_id = str(event.get_sender_id() or "")
        request_id = self._debounce_key(event, "aiimg", user_id)
        if self.debouncer.hit(request_id):
            await mark_success(event)
            return self._llm_tool_text_result(
                "This image request is already being handled or was just handled. Do not submit it again unless the user explicitly asks for a new image."
            )

        if not await self._begin_user_job(user_id, kind="image"):
            await mark_success(event)
            return self._llm_tool_text_result(
                "An image request for this user is already in progress. Do not resubmit unless the user asks for a new request."
            )

        b_raw = (backend or "auto").strip()
        known_provider_ids = set(self.registry.provider_ids())
        if not b_raw or b_raw.lower() == "auto":
            target_backend = None
        elif b_raw in known_provider_ids:
            target_backend = b_raw
        else:
            logger.warning(
                "[aiimg_generate] 忽略未知 backend 覆盖，回退自动链路: backend=%s",
                b_raw,
            )
            target_backend = None

        output = (output or "").strip()
        size = output if output and "x" in output else None
        resolution = output if output and size is None else None

        try:
            await mark_processing(event)

            if m == "selfie_ref":
                logger.info("[aiimg_generate] route=selfie_ref (explicit)")
                if not self._is_selfie_enabled():
                    logger.warning(
                        "[aiimg_generate] selfie blocked: features.selfie.enabled=false"
                    )
                    await self._signal_llm_tool_failure(event)
                    return self._llm_tool_text_result(
                        "The requested selfie image tool is disabled by plugin configuration."
                    )
                if not self._is_selfie_llm_enabled():
                    logger.warning(
                        "[aiimg_generate] selfie blocked: features.selfie.llm_tool_enabled=false"
                    )
                    await self._signal_llm_tool_failure(event)
                    return self._llm_tool_text_result(
                        "The requested selfie image tool is disabled by plugin configuration."
                    )
                image_path, task_meta = await self._generate_selfie_image_with_meta(
                    event,
                    prompt,
                    target_backend,
                    size=size,
                    resolution=resolution,
                )
                return await self._finalize_llm_tool_image(
                    event, image_path, task_meta=task_meta
                )

            if m == "edit":
                logger.info("[aiimg_generate] route=edit")
                edit_conf = self._get_feature("edit")
                if not bool(edit_conf.get("enabled", True)):
                    await self._signal_llm_tool_failure(event)
                    return self._llm_tool_text_result(
                        "The requested image editing tool is disabled by plugin configuration."
                    )
                if not bool(edit_conf.get("llm_tool_enabled", True)):
                    await self._signal_llm_tool_failure(event)
                    return self._llm_tool_text_result(
                        "The requested image editing tool is disabled by plugin configuration."
                    )
                image_segs = await get_images_from_event(
                    event,
                    include_avatar=True,
                    include_sender_avatar_fallback=False,
                )
                bytes_images = await self._image_segs_to_bytes(image_segs)
                if not bytes_images:
                    await self._signal_llm_tool_failure(event)
                    return self._llm_tool_text_result(
                        "Image editing could not continue because no usable input image was found in the current message. This request has ended."
                    )

                image_path = await self.edit.edit(
                    prompt=prompt,
                    images=bytes_images,
                    backend=target_backend,
                    size=size,
                    resolution=resolution,
                )
                task_meta = self._build_image_task_meta(
                    mode="edit",
                    user_prompt=prompt,
                    effective_prompt=prompt,
                    continue_with="edit",
                    backend=target_backend,
                )
                return await self._finalize_llm_tool_image(
                    event, image_path, task_meta=task_meta
                )

            # Only an explicit text mode can reach text-to-image.
            draw_conf = self._get_feature("draw")
            if not bool(draw_conf.get("enabled", True)):
                await self._signal_llm_tool_failure(event)
                return self._llm_tool_text_result(
                    "The requested image generation tool is disabled by plugin configuration."
                )
            if not bool(draw_conf.get("llm_tool_enabled", True)):
                await self._signal_llm_tool_failure(event)
                return self._llm_tool_text_result(
                    "The requested image generation tool is disabled by plugin configuration."
                )
            if not prompt:
                prompt = "a selfie photo"

            logger.info("[aiimg_generate] route=draw")
            draw_conf = self._get_feature("draw")
            original_prompt = prompt
            draw_prefix = self._expand_time_placeholders(
                str(draw_conf.get("prompt_prefix") or "").strip()
            )
            if draw_prefix:
                prompt = f"{draw_prefix}\n\n{prompt}"
            image_path = await self.draw.generate(
                prompt,
                provider_id=target_backend,
                size=size,
                resolution=resolution,
            )
            task_meta = self._build_image_task_meta(
                mode="text",
                user_prompt=original_prompt,
                effective_prompt=prompt,
                continue_with="text",
                backend=target_backend,
            )
            return await self._finalize_llm_tool_image(
                event, image_path, task_meta=task_meta
            )

        except Exception as e:
            logger.error(f"[aiimg_generate] 失败: {e}", exc_info=True)
            await self._signal_llm_tool_failure(event)
            if self._is_missing_selfie_reference_error(e):
                return self._llm_tool_text_result(
                    "自拍参考照缺失，本次没有生成图片，也没有降级为文生图。"
                )
            return self._llm_tool_text_result(
                "The image request failed and has ended. Do not retry automatically unless the user explicitly asks."
            )
        finally:
            await self._end_user_job(user_id, kind="image")

    async def _aiimg_batch_generate(
        self,
        event: AstrMessageEvent,
        prompt: str,
        count: int,
        mode: str,
        backend: str,
        output: str,
    ):
        """Execute an explicitly requested multi-image tool call."""
        prompt = str(prompt or "").strip()
        if not prompt:
            await self._signal_llm_tool_failure(event)
            return self._llm_tool_text_result(
                "Batch image planning failed because no prompt was provided."
            )

        target_count = count
        resolved_mode = self._resolve_llm_batch_mode(mode)
        target_backend = self._resolve_target_backend(backend)

        output = (output or "").strip()
        size = output if output and "x" in output else None
        resolution = output if output and size is None else None

        if resolved_mode == "draw":
            draw_conf = self._get_feature("draw")
            if not bool(draw_conf.get("enabled", True)) or not bool(
                draw_conf.get("llm_tool_enabled", True)
            ):
                await self._signal_llm_tool_failure(event)
                return self._llm_tool_text_result(
                    "The requested batch text-to-image tool is disabled by plugin configuration."
                )
        elif resolved_mode == "edit":
            edit_conf = self._get_feature("edit")
            if not bool(edit_conf.get("enabled", True)) or not bool(
                edit_conf.get("llm_tool_enabled", True)
            ):
                await self._signal_llm_tool_failure(event)
                return self._llm_tool_text_result(
                    "The requested batch image editing tool is disabled by plugin configuration."
                )
        elif resolved_mode == "selfie_ref":
            if not self._is_selfie_enabled() or not self._is_selfie_llm_enabled():
                await self._signal_llm_tool_failure(event)
                return self._llm_tool_text_result(
                    "The requested batch selfie image tool is disabled by plugin configuration."
                )

        message_id = (
            getattr(getattr(event, "message_obj", None), "message_id", "") or ""
        )
        origin = getattr(event, "unified_msg_origin", "") or ""
        if (
            message_id
            and origin
            and self.debouncer.llm_tool_is_duplicate(message_id, origin)
        ):
            await mark_success(event)
            return self._llm_tool_text_result(
                "This batch image request was already handled for the same message. Do not run it again."
            )

        user_id = str(event.get_sender_id() or "")
        request_id = self._debounce_key(event, "aiimg_batch", user_id)
        if self.debouncer.hit(request_id):
            await mark_success(event)
            return self._llm_tool_text_result(
                "This batch image request is already being handled or was just handled. Do not resubmit unless the user explicitly asks for a new batch."
            )

        if not await self._begin_user_job(user_id, kind="image"):
            await mark_success(event)
            return self._llm_tool_text_result(
                "A batch image request for this user is already in progress. Do not resubmit unless the user asks for a new request."
            )

        try:
            await mark_processing(event)
            planned_items = await self._plan_batch_prompt_items(
                mode=resolved_mode,
                user_prompt=prompt,
                count=target_count,
            )
            specs = [
                ImageTaskSpec(
                    mode=resolved_mode,
                    provider_id=target_backend,
                    preset_name=None,
                    user_prompt=item.prompt,
                    effective_prompt=item.prompt,
                    source_command="llm_batch",
                    variant_title=item.title,
                )
                for item in planned_items
            ]
            results = await self._run_batch_specs(
                event,
                specs,
                size=size,
                resolution=resolution,
            )
            sent_results = await self._send_batch_results(
                event,
                results,
                title=f"LLM 批量{self._batch_mode_label(specs[0])} x{len(specs)}",
            )
            success_count = len(sent_results)
            if sent_results:
                await self._remember_batch_success(
                    event,
                    sent_results,
                    prompt=prompt,
                    mode=resolved_mode,
                    count=success_count,
                )
                await mark_success(event)
                return None

            await self._signal_llm_tool_failure(event)
            return self._llm_tool_text_result(
                "The batch image request finished without sending an image."
            )
        except Exception as e:
            logger.error("[aiimg_batch_generate] 失败: %s", e, exc_info=True)
            await self._signal_llm_tool_failure(event)
            return self._llm_tool_text_result(
                "The batch image request failed and has ended. Reason: "
                + self._summarize_status_text(e, fallback="unknown error")
            )
        finally:
            await self._end_user_job(user_id, kind="image")

    @filter.llm_tool()
    async def grok_generate_video(self, event: AstrMessageEvent, prompt: str):
        """根据用户发送/引用的图片生成视频。

        Args:
            prompt(string): 视频提示词。支持 "预设名 额外提示词"（与 `/视频 预设名 额外提示词` 一致）
        """
        vconf = self._get_feature("video")
        if not bool(vconf.get("enabled", False)):
            await self._signal_llm_tool_failure(event)
            return self._llm_tool_text_result(
                "The requested video tool is disabled by plugin configuration."
            )
        if not bool(vconf.get("llm_tool_enabled", True)):
            await self._signal_llm_tool_failure(event)
            return self._llm_tool_text_result(
                "The requested video tool is disabled by plugin configuration."
            )

        arg = (prompt or "").strip()
        if not arg:
            await self._signal_llm_tool_failure(event)
            return self._llm_tool_text_result(
                "The video request failed because no prompt was provided. This request has ended."
            )

        provider_override, arg = self._parse_provider_override_prefix(arg)
        if not arg:
            await self._signal_llm_tool_failure(event)
            return self._llm_tool_text_result(
                "The video request failed because no usable prompt remained after parsing provider overrides. This request has ended."
            )

        preset, extra_prompt = self._parse_video_args(arg)
        presets = self._get_video_presets()
        if preset and preset in presets:
            preset_prompt = presets[preset]
            extra_prompt = (
                f"{preset_prompt}, {extra_prompt}" if extra_prompt else preset_prompt
            )

        user_id = str(event.get_sender_id() or "")
        request_id = self._debounce_key(event, "video", user_id)

        if self.debouncer.hit(request_id):
            await mark_success(event)
            return self._llm_tool_text_result(
                "This video request is already being handled or was just handled. Do not submit it again unless the user explicitly asks for a new video."
            )

        if not await self._video_begin(user_id):
            await mark_success(event)
            return self._llm_tool_text_result(
                "A video request for this user is already in progress. Do not resubmit unless the user asks for a new request."
            )

        try:
            await mark_processing(event)
            task = asyncio.create_task(
                self._async_generate_video(
                    event,
                    extra_prompt,
                    user_id,
                    provider_id=provider_override,
                    llm_tool_failure=True,
                )
            )
        except Exception:
            await self._video_end(user_id)
            await self._signal_llm_tool_failure(event)
            return self._llm_tool_text_result(
                "The video request failed before background execution could start. This request has ended."
            )

        self._video_tasks.add(task)
        task.add_done_callback(lambda t: self._video_tasks.discard(t))

        return self._llm_tool_text_result(
            "Video generation has been accepted and is running in the background. The result will be sent to the user automatically when ready. Do not submit the same request again unless the user explicitly asks."
        )

    # ==================== 内部方法 ====================

    def _get_feature(self, name: str) -> dict:
        feats = self.config.get("features", {}) if isinstance(self.config, dict) else {}
        feats = feats if isinstance(feats, dict) else {}
        conf = feats.get(name, {})
        return conf if isinstance(conf, dict) else {}

    @staticmethod
    def _background_event_factory_available(platform_name: str, adapter: Any) -> bool:
        if not callable(getattr(StarTools, "create_message", None)):
            return False
        metadata = getattr(adapter, "metadata", None)
        if metadata is None:
            meta = getattr(adapter, "meta", None)
            if callable(meta):
                try:
                    metadata = meta()
                except Exception:
                    return False
        if metadata is None:
            return False
        if platform_name == "aiocqhttp":
            return getattr(adapter, "bot", None) is not None
        return platform_name == "weixin_oc"

    def _background_manager_for_event(
        self, event: AstrMessageEvent
    ) -> BackgroundImageTaskManager | None:
        manager = getattr(self, "background_tasks", None)
        if manager is None or not manager.accepting:
            return None
        try:
            platform_name = str(event.get_platform_name() or "").strip()
            platform_id = str(event.get_platform_id() or "").strip()
            umo = str(event.unified_msg_origin or "").strip()
            adapter = self.context.get_platform_inst(platform_id)
            config = self.context.get_config(umo=umo)
        except Exception as exc:
            logger.debug(
                "[background-image] synchronous fallback because event routing is unavailable: %s",
                BackgroundImageTaskManager.sanitize_error(exc),
            )
            return None
        if platform_name not in {"aiocqhttp", "weixin_oc"}:
            logger.debug(
                "[background-image] synchronous fallback for unsupported platform: %s",
                platform_name or "unknown",
            )
            return None
        if not platform_id or not umo or adapter is None:
            logger.debug(
                "[background-image] synchronous fallback because platform route is incomplete"
            )
            return None
        if not self._background_event_factory_available(platform_name, adapter):
            logger.warning(
                "[background-image] synchronous fallback because %s cannot rebuild events on this AstrBot runtime",
                platform_name,
            )
            return None
        if bool(config.get("provider_settings", {}).get("streaming_response", False)):
            logger.info(
                "[background-image] synchronous fallback because streaming_response is enabled"
            )
            return None
        return manager

    async def _build_background_delivery_target(
        self, event: AstrMessageEvent
    ) -> TaskDeliveryTarget:
        conversation = await self._resolve_plugin_conversation(event)
        conversation_id = str(getattr(conversation, "cid", "") or "").strip()
        if not conversation_id:
            raise BackgroundTaskError(
                "A stable conversation is required for background image delivery"
            )
        message_type = event.get_message_type()
        message_type_text = str(getattr(message_type, "value", message_type) or "")
        return TaskDeliveryTarget(
            platform_id=str(event.get_platform_id() or "").strip(),
            platform_name=str(event.get_platform_name() or "").strip(),
            message_type=message_type_text,
            umo=str(event.unified_msg_origin or "").strip(),
            session_id=str(event.get_session_id() or "").strip(),
            group_id=str(event.get_group_id() or "").strip(),
            self_id=str(event.get_self_id() or "").strip(),
            sender_id=str(event.get_sender_id() or "").strip(),
            sender_name=str(event.get_sender_name() or "").strip(),
            source_message_id=str(
                getattr(getattr(event, "message_obj", None), "message_id", "") or ""
            ).strip(),
            conversation_id=conversation_id,
        )

    async def _prepare_background_selfie(
        self,
        event: AstrMessageEvent,
        prompt: str,
        backend: str | None,
        *,
        follow_up_meta: dict[str, Any] | None = None,
    ) -> tuple[list[bytes], str, dict[str, Any], dict[str, Any]]:
        conf = self._get_selfie_conf()
        if not self._is_selfie_enabled() or not self._is_selfie_llm_enabled():
            raise RuntimeError("The requested selfie image tool is disabled.")
        ref_paths, ref_source = await self._get_selfie_reference_paths(event)
        ref_images = await self._read_paths_bytes(ref_paths)
        if not ref_images:
            raise RuntimeError(
                "未设置自拍参考照。请先发送图片并设置自拍参考，或在 WebUI 上传参考图。"
            )
        extra_segs = await get_images_from_event(event, include_avatar=False)
        extra_bytes = await self._image_segs_to_bytes(extra_segs)
        effective_user_prompt = self._build_selfie_follow_up_prompt(
            prompt, follow_up_meta
        )
        effective_prompt = self._build_selfie_prompt(
            effective_user_prompt,
            reference_count=len(ref_images),
            extra_reference_count=len(extra_bytes),
            control_text=str(getattr(event, "message_str", "") or ""),
        )

        chain_override: list[dict] | None = None
        raw_chain = conf.get("chain", [])
        if isinstance(raw_chain, list):
            normalized = [
                item
                for item in (self._normalize_chain_item(value) for value in raw_chain)
                if item is not None
            ]
            if normalized:
                chain_override = normalized
        use_edit_chain = bool(conf.get("use_edit_chain_when_empty", True))
        if backend is None:
            if chain_override is None and not use_edit_chain:
                raise RuntimeError("No selfie provider chain configured.")
            if chain_override is not None and use_edit_chain:
                chain_override = self._merge_selfie_chain_with_edit_chain(
                    chain_override
                )
        raw_task_types = conf.get("gitee_task_types")
        task_types = (
            [str(item).strip() for item in raw_task_types if str(item).strip()]
            if isinstance(raw_task_types, list) and raw_task_types
            else ["id", "background", "style"]
        )
        options = {
            "task_types": task_types,
            "default_output": str(conf.get("default_output") or "").strip() or None,
            "chain_override": chain_override,
            "reference_source": ref_source,
            "reference_count": len(ref_images),
            "extra_reference_count": len(extra_bytes),
        }
        task_meta = self._build_image_task_meta(
            mode="selfie_ref",
            user_prompt=prompt,
            effective_user_prompt=effective_user_prompt,
            effective_prompt=effective_prompt,
            reference_source=ref_source,
            reference_count=len(ref_images),
            extra_reference_count=len(extra_bytes),
            continue_with="selfie_ref",
            follow_up=follow_up_meta is not None,
            backend=backend,
        )
        return [*ref_images, *extra_bytes], effective_prompt, options, task_meta

    async def _prepare_background_image_job(
        self,
        event: AstrMessageEvent,
        *,
        prompt: str,
        mode: str,
        backend: str,
        output: str,
    ) -> tuple[PreparedImageJob, list[bytes]]:
        requested_mode = self._normalize_llm_image_mode(mode)
        target_backend = self._resolve_target_backend(backend)
        input_bytes: list[bytes] = []
        options: dict[str, Any] = {}
        task_meta: dict[str, Any]
        effective_prompt = prompt

        if requested_mode == "selfie_ref":
            (
                input_bytes,
                effective_prompt,
                options,
                task_meta,
            ) = await self._prepare_background_selfie(event, prompt, target_backend)
            resolved_mode = "selfie_ref"
        elif requested_mode == "edit":
            edit_conf = self._get_feature("edit")
            if not bool(edit_conf.get("enabled", True)) or not bool(
                edit_conf.get("llm_tool_enabled", True)
            ):
                raise RuntimeError("The requested image editing tool is disabled.")
            image_segs = await get_images_from_event(
                event,
                include_avatar=True,
                include_sender_avatar_fallback=False,
            )
            input_bytes = await self._image_segs_to_bytes(image_segs)
            if not input_bytes:
                raise RuntimeError(
                    "No usable input image was found in the current message."
                )
            resolved_mode = "edit"
            task_meta = self._build_image_task_meta(
                mode="edit",
                user_prompt=prompt,
                effective_prompt=prompt,
                continue_with="edit",
                backend=target_backend,
            )
        else:
            draw_conf = self._get_feature("draw")
            if not bool(draw_conf.get("enabled", True)) or not bool(
                draw_conf.get("llm_tool_enabled", True)
            ):
                raise RuntimeError("The requested image generation tool is disabled.")
            resolved_mode = "draw"
            user_prompt = prompt or "a photo"
            prefix = self._expand_time_placeholders(
                str(draw_conf.get("prompt_prefix") or "").strip()
            )
            effective_prompt = f"{prefix}\n\n{user_prompt}" if prefix else user_prompt
            task_meta = self._build_image_task_meta(
                mode="text",
                user_prompt=user_prompt,
                effective_prompt=effective_prompt,
                continue_with="text",
                backend=target_backend,
            )

        output_text = str(output or "").strip()
        size = output_text if output_text and "x" in output_text else None
        resolution = output_text if output_text and size is None else None
        return (
            PreparedImageJob(
                mode=resolved_mode,
                user_prompt=prompt,
                effective_prompt=effective_prompt,
                backend=target_backend,
                output={"size": size, "resolution": resolution},
                task_meta=task_meta,
                options=options,
            ),
            input_bytes,
        )

    async def _execute_prepared_image_job(
        self,
        manager: BackgroundImageTaskManager,
        job: PreparedImageJob,
    ) -> tuple[Path, dict[str, Any]]:
        images = await manager.read_spooled_inputs(
            job.input_paths,
            job.options.get("input_manifest"),
        )
        size = job.output.get("size")
        resolution = job.output.get("resolution")
        if job.mode == "draw":
            image_path = await self.draw.generate(
                job.effective_prompt,
                provider_id=job.backend,
                size=size,
                resolution=resolution,
            )
        elif job.mode == "edit":
            image_path = await self.edit.edit(
                prompt=job.effective_prompt,
                images=images,
                backend=job.backend,
                size=size,
                resolution=resolution,
            )
        elif job.mode == "selfie_ref":
            image_path = await self.edit.edit(
                prompt=job.effective_prompt,
                images=images,
                backend=job.backend,
                task_types=job.options.get("task_types"),
                size=size,
                resolution=resolution,
                default_output=job.options.get("default_output"),
                chain_override=job.options.get("chain_override"),
            )
        else:
            raise RuntimeError(f"Unsupported prepared image mode: {job.mode}")
        return Path(image_path), dict(job.task_meta)

    async def _accept_background_single(
        self,
        event: AstrMessageEvent,
        *,
        prompt: str,
        mode: str,
        backend: str,
        output: str,
    ) -> mcp.types.CallToolResult:
        manager = self._background_manager_for_event(event)
        if manager is None:
            raise BackgroundTaskError("Background mode is unavailable for this event")
        target = await self._build_background_delivery_target(event)
        task_id = manager.new_task_id("img")
        job, input_bytes = await self._prepare_background_image_job(
            event,
            prompt=prompt,
            mode=mode,
            backend=backend,
            output=output,
        )
        input_paths, manifest = await manager.spool_inputs(task_id, input_bytes)
        try:
            job = PreparedImageJob(
                mode=job.mode,
                user_prompt=job.user_prompt,
                effective_prompt=job.effective_prompt,
                backend=job.backend,
                output=job.output,
                input_paths=input_paths,
                task_meta=job.task_meta,
                options={**job.options, "input_manifest": manifest},
            )
            scope = manager.scope_hash(
                target.umo,
                target.self_id,
                target.sender_id,
                target.conversation_id,
            )
            record = {
                "task_id": task_id,
                "task_kind": "single",
                "state": "queued",
                "scope_hash": scope,
                "request_fingerprint": manager.request_fingerprint(
                    scope,
                    target.source_message_id,
                    {
                        "prompt": prompt,
                        "mode": job.mode,
                        "backend": job.backend or "auto",
                        "output": job.output,
                    },
                ),
                **manager.dataclass_dict(target),
                "mode": job.mode,
                "backend_requested": job.backend or "auto",
                "user_prompt": prompt,
                "effective_prompt": job.effective_prompt,
                "input_manifest": manifest,
                "image_generated": False,
                "image_sent": False,
                "delivery_state": "not_started",
                "items": [],
            }
            stored, created = await manager.create_task_record(record, reservation=1)
            if created:
                manager.start_worker(
                    task_id,
                    lambda: self._run_background_single(manager, task_id, job, target),
                )
            else:
                await manager.cleanup_task_files(task_id)
                task_id = str(stored["task_id"])
            event.set_extra("_gitee_bg_ack_task_id", task_id)
            await mark_processing(event)
            return None
        except Exception:
            await manager.cleanup_task_files(task_id)
            raise

    async def _accept_background_batch(
        self,
        event: AstrMessageEvent,
        *,
        prompt: str,
        count: int,
        mode: str,
        backend: str,
        output: str,
    ) -> mcp.types.CallToolResult:
        if not prompt:
            raise RuntimeError("Batch image planning requires a prompt.")
        manager = self._background_manager_for_event(event)
        if manager is None:
            raise BackgroundTaskError("Background mode is unavailable for this event")
        target = await self._build_background_delivery_target(event)
        task_id = manager.new_task_id("batch")
        base_job, input_bytes = await self._prepare_background_image_job(
            event,
            prompt=prompt,
            mode=mode,
            backend=backend,
            output=output,
        )
        input_paths, manifest = await manager.spool_inputs(task_id, input_bytes)
        try:
            job = PreparedBatchJob(
                mode=base_job.mode,
                user_prompt=prompt,
                requested_count=count,
                backend=base_job.backend,
                output=base_job.output,
                input_paths=input_paths,
                options={**base_job.options, "input_manifest": manifest},
            )
            scope = manager.scope_hash(
                target.umo,
                target.self_id,
                target.sender_id,
                target.conversation_id,
            )
            record = {
                "task_id": task_id,
                "task_kind": "batch",
                "state": "planning",
                "scope_hash": scope,
                "request_fingerprint": manager.request_fingerprint(
                    scope,
                    target.source_message_id,
                    {
                        "prompt": prompt,
                        "count": count,
                        "mode": job.mode,
                        "backend": job.backend or "auto",
                        "output": job.output,
                    },
                ),
                **manager.dataclass_dict(target),
                "mode": job.mode,
                "backend_requested": job.backend or "auto",
                "user_prompt": prompt,
                "effective_prompt": "",
                "requested_count": count,
                "planned_count": 0,
                "generated_count": 0,
                "sent_count": 0,
                "failed_count": 0,
                "cancelled_count": 0,
                "unknown_count": 0,
                "input_manifest": manifest,
                "image_generated": False,
                "image_sent": False,
                "delivery_state": "not_started",
                "items": [],
            }
            stored, created = await manager.create_task_record(
                record, reservation=count
            )
            if created:
                manager.start_worker(
                    task_id,
                    lambda: self._run_background_batch(manager, task_id, job, target),
                )
            else:
                await manager.cleanup_task_files(task_id)
                task_id = str(stored["task_id"])
            event.set_extra("_gitee_bg_ack_task_id", task_id)
            await mark_processing(event)
            return None
        except Exception:
            await manager.cleanup_task_files(task_id)
            raise

    def _background_batch_child_job(
        self,
        job: PreparedBatchJob,
        prompt: str,
    ) -> PreparedImageJob:
        if job.mode == "draw":
            prefix = self._expand_time_placeholders(
                str(self._get_feature("draw").get("prompt_prefix") or "").strip()
            )
            effective_prompt = f"{prefix}\n\n{prompt}" if prefix else prompt
            task_meta = self._build_image_task_meta(
                mode="text",
                user_prompt=prompt,
                effective_prompt=effective_prompt,
                continue_with="text",
                backend=job.backend,
            )
        elif job.mode == "selfie_ref":
            effective_prompt = self._build_selfie_prompt(
                prompt,
                reference_count=int(job.options.get("reference_count") or 0),
                extra_reference_count=int(
                    job.options.get("extra_reference_count") or 0
                ),
                control_text=str(getattr(event, "message_str", "") or ""),
            )
            task_meta = self._build_image_task_meta(
                mode="selfie_ref",
                user_prompt=prompt,
                effective_prompt=effective_prompt,
                reference_source=str(job.options.get("reference_source") or ""),
                reference_count=int(job.options.get("reference_count") or 0),
                extra_reference_count=int(
                    job.options.get("extra_reference_count") or 0
                ),
                continue_with="selfie_ref",
                backend=job.backend,
            )
        else:
            effective_prompt = prompt
            task_meta = self._build_image_task_meta(
                mode="edit",
                user_prompt=prompt,
                effective_prompt=effective_prompt,
                continue_with="edit",
                backend=job.backend,
            )
        return PreparedImageJob(
            mode=job.mode,
            user_prompt=prompt,
            effective_prompt=effective_prompt,
            backend=job.backend,
            output=job.output,
            input_paths=job.input_paths,
            task_meta=task_meta,
            options=job.options,
        )

    async def _run_background_batch_child(
        self,
        manager: BackgroundImageTaskManager,
        task_id: str,
        item: dict[str, Any],
        job: PreparedImageJob,
        child_limit: asyncio.Semaphore,
        generated: dict[str, tuple[Path, dict[str, Any]]],
    ) -> None:
        item_id = str(item["item_id"])
        try:
            async with child_limit:

                async def provider_call() -> tuple[Path, dict[str, Any]]:
                    await manager.update_item(
                        task_id,
                        item_id,
                        {"state": "running", "started_at": manager.now_ms()},
                    )
                    return await self._execute_prepared_image_job(manager, job)

                result = await asyncio.wait_for(
                    manager.run_provider(task_id, provider_call),
                    timeout=2 * 60 * 60,
                )
            generated[item_id] = result
            await manager.update_item(
                task_id,
                item_id,
                {
                    "state": "generated",
                    "image_generated": True,
                    "delivery_state": "not_started",
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await manager.update_item(
                task_id,
                item_id,
                {
                    "state": "failed",
                    "error_code": "provider_failed",
                    "error_message": manager.sanitize_error(exc),
                },
                release_if_terminal=True,
            )

    async def _run_background_batch(
        self,
        manager: BackgroundImageTaskManager,
        task_id: str,
        job: PreparedBatchJob,
        target: TaskDeliveryTarget,
    ) -> None:
        children: list[asyncio.Task[Any]] = []
        try:
            planned = await asyncio.wait_for(
                manager.run_planner(
                    lambda: self._plan_batch_prompt_items(
                        mode=job.mode,
                        user_prompt=job.user_prompt,
                        count=job.requested_count,
                    )
                ),
                timeout=300,
            )
            if len(planned) != job.requested_count:
                raise RuntimeError(
                    f"Batch planner returned {len(planned)} items for {job.requested_count} reserved slots"
                )
            items: list[dict[str, Any]] = []
            child_jobs: list[PreparedImageJob] = []
            for index, planned_item in enumerate(planned, start=1):
                child_job = self._background_batch_child_job(job, planned_item.prompt)
                item_id = f"{task_id}_{index:02d}"
                items.append(
                    {
                        "item_id": item_id,
                        "index": index,
                        "state": "queued",
                        "mode": job.mode,
                        "title": planned_item.title,
                        "variation_focus": planned_item.variation_focus,
                        "user_prompt": planned_item.prompt,
                        "effective_prompt": child_job.effective_prompt,
                        "image_generated": False,
                        "image_sent": False,
                        "delivery_state": "not_started",
                        "error_code": "",
                        "error_message": "",
                    }
                )
                child_jobs.append(child_job)
            await manager.transition(
                task_id,
                "queued",
                {"items": items, "planned_count": len(items)},
            )
            generated: dict[str, tuple[Path, dict[str, Any]]] = {}
            last_delivery_event: AstrMessageEvent | None = None
            child_limit = asyncio.Semaphore(
                self._get_batch_concurrency_for_mode(job.mode)
            )
            for item, child_job in zip(items, child_jobs, strict=True):
                children.append(
                    manager.start_managed(
                        self._run_background_batch_child(
                            manager,
                            task_id,
                            item,
                            child_job,
                            child_limit,
                            generated,
                        ),
                        name=f"background-batch-child-{item['item_id']}",
                    )
                )
            await asyncio.gather(*children)
            await self._wait_for_background_ack(manager, task_id)

            for item in items:
                if manager.is_cancelled(task_id):
                    raise asyncio.CancelledError
                item_id = str(item["item_id"])
                result = generated.get(item_id)
                if result is None:
                    continue
                image_path, task_meta = result
                await self._wait_background_send_gate(target.umo)
                attempt_id = manager.new_task_id("send")
                await manager.update_item(
                    task_id,
                    item_id,
                    {
                        "state": "sending",
                        "delivery_state": "attempting",
                        "send_attempt_id": attempt_id,
                    },
                )
                await manager.record_receipt(
                    task_id,
                    send_attempt_id=attempt_id,
                    item_id=item_id,
                    kind="image",
                    delivery_state="attempting",
                    transport=target.platform_name,
                )
                try:
                    event = await self._send_background_image_once(target, image_path)
                except Exception as exc:
                    await manager.record_receipt(
                        task_id,
                        send_attempt_id=attempt_id,
                        item_id=item_id,
                        kind="image",
                        delivery_state="unknown",
                        transport=target.platform_name,
                        response_digest=hashlib.sha256(str(exc).encode()).hexdigest(),
                    )
                    await manager.update_item(
                        task_id,
                        item_id,
                        {
                            "state": "unknown",
                            "image_generated": True,
                            "image_sent": False,
                            "delivery_state": "unknown",
                            "error_code": "delivery_unknown",
                            "error_message": manager.sanitize_error(exc),
                        },
                        release_if_terminal=True,
                    )
                    continue
                await manager.record_receipt(
                    task_id,
                    send_attempt_id=attempt_id,
                    item_id=item_id,
                    kind="image",
                    delivery_state="confirmed",
                    transport=target.platform_name,
                    response_digest=hashlib.sha256(
                        str(image_path).encode()
                    ).hexdigest(),
                )
                self._remember_last_image(event, image_path)
                await self._save_last_image_task_meta(event, task_meta)
                last_delivery_event = event
                await manager.update_item(
                    task_id,
                    item_id,
                    {
                        "state": "completed",
                        "image_generated": True,
                        "image_sent": True,
                        "delivery_state": "confirmed",
                    },
                    release_if_terminal=True,
                )

            current = await manager.get_task(task_id)
            if current is None:
                return
            requested = int(current.get("requested_count") or 0)
            sent = int(current.get("sent_count") or 0)
            unknown = int(current.get("unknown_count") or 0)
            if unknown:
                state = "interrupted"
            elif sent == requested and requested:
                state = "completed"
            elif sent:
                state = "partial"
            else:
                state = "failed"
            record = await manager.transition(
                task_id,
                state,
                {
                    "delivery_state": "unknown"
                    if unknown
                    else ("confirmed" if sent else "not_started"),
                    "terminal_reason": "batch_finished",
                },
                queue_notification=True,
            )
            if sent and last_delivery_event is not None:
                await self._append_image_history_note(
                    last_delivery_event,
                    prompt=current.get("user_prompt"),
                    mode=current.get("mode"),
                    count=sent,
                    dedupe_key=self._image_history_dedupe_key(
                        last_delivery_event,
                        prompt=current.get("user_prompt"),
                        mode=current.get("mode"),
                        count=sent,
                        task_id=task_id,
                    ),
                )
            await self._dispatch_background_completion(manager, record, target)
        except asyncio.CancelledError:
            for child in children:
                child.cancel()
            if children:
                await asyncio.gather(*children, return_exceptions=True)
            raise
        except Exception as exc:
            logger.error(
                "[background-image] batch task failed: task=%s err=%s",
                task_id,
                manager.sanitize_error(exc),
                exc_info=True,
            )
            record = await manager.get_task(task_id)
            if record and record.get("state") not in TERMINAL_STATES:
                record = await manager.transition(
                    task_id,
                    "failed",
                    {
                        "error_code": "batch_failed",
                        "error_message": manager.sanitize_error(exc),
                        "terminal_reason": "batch_failed",
                    },
                    queue_notification=True,
                )
                await self._dispatch_background_completion(manager, record, target)
        finally:
            await manager.cleanup_task_files(task_id)

    async def _expire_background_send_gate(self, umo: str, gate: asyncio.Event) -> None:
        await asyncio.sleep(30)
        if self._background_send_gates.get(umo) is gate:
            self._background_send_gates.pop(umo, None)
            gate.set()

    async def _wait_background_send_gate(self, umo: str) -> None:
        gate = self._background_send_gates.get(umo)
        if gate is None or gate.is_set():
            return
        try:
            await asyncio.wait_for(gate.wait(), timeout=35)
        except TimeoutError:
            if self._background_send_gates.get(umo) is gate:
                self._background_send_gates.pop(umo, None)
                gate.set()

    async def _rebuild_background_event(
        self,
        target: TaskDeliveryTarget,
        *,
        message: list[Any] | None = None,
    ) -> AstrMessageEvent:
        adapter = self.context.get_platform_inst(target.platform_id)
        if adapter is None:
            raise RuntimeError(f"Platform is no longer available: {target.platform_id}")
        message_obj = await StarTools.create_message(
            type=target.message_type,
            self_id=target.self_id,
            session_id=target.session_id,
            sender=MessageMember(
                user_id=target.sender_id,
                nickname=target.sender_name or None,
            ),
            message=list(message or []),
            message_str="",
            message_id=f"gitee-bg-{BackgroundImageTaskManager.new_task_id('event')}",
            raw_message=None,
            group_id=target.group_id,
        )
        if target.platform_name == "aiocqhttp":
            from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
                AiocqhttpMessageEvent,
            )

            rebuilt = AiocqhttpMessageEvent(
                message_str=message_obj.message_str,
                message_obj=message_obj,
                platform_meta=adapter.metadata,
                session_id=target.session_id,
                bot=adapter.bot,
            )
        elif target.platform_name == "weixin_oc":
            from astrbot.core.platform.sources.weixin_oc.weixin_oc_event import (
                WeixinOCMessageEvent,
            )

            rebuilt = WeixinOCMessageEvent(
                message_str=message_obj.message_str,
                message_obj=message_obj,
                platform_meta=adapter.metadata,
                session_id=target.session_id,
                platform=adapter,
            )
        else:
            raise RuntimeError(
                f"Unsupported background platform: {target.platform_name}"
            )
        rebuilt.unified_msg_origin = target.umo
        rebuilt.is_wake = True
        rebuilt.is_at_or_wake_command = True
        return rebuilt

    async def _send_background_image_once(
        self, target: TaskDeliveryTarget, image_path: Path
    ) -> AstrMessageEvent:
        event = await self._rebuild_background_event(target)
        sent = await self._send_image_with_fallback(event, image_path)
        if not sent:
            raise RuntimeError(
                f"Background image delivery failed: {sent.reason or sent.last_error}"
            )
        return event

    async def _wait_for_background_ack(
        self, manager: BackgroundImageTaskManager, task_id: str
    ) -> None:
        for _ in range(40):
            record = await manager.get_task(task_id)
            if record is None or record.get("ack_state") != "pending":
                return
            await asyncio.sleep(0.25)

    async def _run_background_single(
        self,
        manager: BackgroundImageTaskManager,
        task_id: str,
        job: PreparedImageJob,
        target: TaskDeliveryTarget,
    ) -> None:
        try:

            async def provider_call() -> tuple[Path, dict[str, Any]]:
                await manager.transition(task_id, "running")
                return await self._execute_prepared_image_job(manager, job)

            image_path, task_meta = await asyncio.wait_for(
                manager.run_provider(task_id, provider_call),
                timeout=2 * 60 * 60,
            )
            if manager.is_cancelled(task_id):
                raise asyncio.CancelledError
            await self._wait_for_background_ack(manager, task_id)
            await self._wait_background_send_gate(target.umo)
            if manager.is_cancelled(task_id):
                raise asyncio.CancelledError

            attempt_id = manager.new_task_id("send")
            await manager.transition(
                task_id,
                "sending",
                {
                    "image_generated": True,
                    "delivery_state": "attempting",
                    "send_attempt_id": attempt_id,
                },
            )
            await manager.record_receipt(
                task_id,
                send_attempt_id=attempt_id,
                kind="image",
                delivery_state="attempting",
                transport=target.platform_name,
            )
            try:
                delivery_event = await self._send_background_image_once(
                    target, image_path
                )
            except Exception as exc:
                await manager.record_receipt(
                    task_id,
                    send_attempt_id=attempt_id,
                    kind="image",
                    delivery_state="unknown",
                    transport=target.platform_name,
                    response_digest=hashlib.sha256(str(exc).encode()).hexdigest(),
                )
                record = await manager.transition(
                    task_id,
                    "interrupted",
                    {
                        "image_generated": True,
                        "image_sent": False,
                        "delivery_state": "unknown",
                        "error_code": "delivery_unknown",
                        "error_message": manager.sanitize_error(exc),
                        "terminal_reason": "image_delivery",
                    },
                    queue_notification=True,
                )
                await self._dispatch_background_completion(manager, record, target)
                return

            await manager.record_receipt(
                task_id,
                send_attempt_id=attempt_id,
                kind="image",
                delivery_state="confirmed",
                transport=target.platform_name,
                response_digest=hashlib.sha256(str(image_path).encode()).hexdigest(),
            )
            self._remember_last_image(delivery_event, image_path)
            await self._save_last_image_task_meta(delivery_event, task_meta)
            await self._append_image_history_note(
                delivery_event,
                prompt=task_meta.get("user_prompt"),
                mode=task_meta.get("mode"),
                dedupe_key=self._image_history_dedupe_key(
                    delivery_event,
                    prompt=task_meta.get("user_prompt"),
                    mode=task_meta.get("mode"),
                    task_id=task_id,
                ),
            )
            record = await manager.transition(
                task_id,
                "completed",
                {
                    "image_generated": True,
                    "image_sent": True,
                    "delivery_state": "confirmed",
                    "task_meta": task_meta,
                    "terminal_reason": "completed",
                },
                queue_notification=True,
            )
            await self._dispatch_background_completion(manager, record, target)
        except asyncio.CancelledError:
            record = await manager.get_task(task_id)
            if record and record.get("state") not in TERMINAL_STATES:
                try:
                    await asyncio.shield(
                        manager.transition(
                            task_id,
                            "interrupted",
                            {
                                "error_code": "plugin_shutdown",
                                "error_message": "The image task was interrupted.",
                                "terminal_reason": "plugin_shutdown",
                            },
                            queue_notification=True,
                        )
                    )
                except Exception:
                    pass
            raise
        except Exception as exc:
            logger.error(
                "[background-image] single task failed: task=%s err=%s",
                task_id,
                manager.sanitize_error(exc),
                exc_info=True,
            )
            record = await manager.get_task(task_id)
            if record and record.get("state") not in TERMINAL_STATES:
                record = await manager.transition(
                    task_id,
                    "failed",
                    {
                        "error_code": "provider_failed",
                        "error_message": manager.sanitize_error(exc),
                        "terminal_reason": "provider_failed",
                    },
                    queue_notification=True,
                )
                await self._dispatch_background_completion(manager, record, target)
        finally:
            await manager.cleanup_task_files(task_id)

    @staticmethod
    def _delivery_target_from_record(record: dict[str, Any]) -> TaskDeliveryTarget:
        return TaskDeliveryTarget(
            platform_id=str(record.get("platform_id") or ""),
            platform_name=str(record.get("platform_name") or ""),
            message_type=str(record.get("message_type") or "FriendMessage"),
            umo=str(record.get("umo") or ""),
            session_id=str(record.get("session_id") or ""),
            group_id=str(record.get("group_id") or ""),
            self_id=str(record.get("self_id") or ""),
            sender_id=str(record.get("sender_id") or ""),
            sender_name=str(record.get("sender_name") or ""),
            source_message_id=str(record.get("source_message_id") or ""),
            conversation_id=str(record.get("conversation_id") or ""),
        )

    async def _background_context_is_safe(
        self, target: TaskDeliveryTarget
    ) -> tuple[bool, Any | None]:
        manager = getattr(self.context, "conversation_manager", None)
        if manager is None or not target.conversation_id:
            return False, None
        try:
            current_id = await manager.get_curr_conversation_id(target.umo)
            if str(current_id or "") != target.conversation_id:
                return False, None
            conversation = await manager.get_conversation(
                target.umo, target.conversation_id
            )
        except Exception as exc:
            logger.warning(
                "[background-image] conversation gate failed: %s",
                BackgroundImageTaskManager.sanitize_error(exc),
            )
            return False, None
        return conversation is not None, conversation

    async def _cancel_background_scope_with_notifications(
        self,
        manager: BackgroundImageTaskManager,
        *,
        umo: str,
        sender_id: str,
        reason: str,
        suppress_future_injection: bool = False,
    ) -> int:
        records = [
            record
            for record in await manager.list_active_for_umo(umo)
            if str(record.get("sender_id") or "") == str(sender_id or "")
        ]
        cancelled = 0
        for record in records:
            task_id = str(record.get("task_id") or "")
            if not task_id or not await manager.cancel_task(
                task_id,
                reason,
                suppress_future_injection=suppress_future_injection,
            ):
                continue
            cancelled += 1
            terminal = await manager.get_task(task_id)
            if terminal is not None:
                await self._dispatch_background_completion(
                    manager,
                    terminal,
                    self._delivery_target_from_record(terminal),
                )
        return cancelled

    @staticmethod
    def _background_notification_text(record: dict[str, Any]) -> str:
        state = str(record.get("state") or "failed")
        if record.get("task_kind") == "batch":
            requested = int(record.get("requested_count") or 0)
            sent = int(record.get("sent_count") or 0)
            failed = int(record.get("failed_count") or 0)
            cancelled = int(record.get("cancelled_count") or 0)
            unknown = int(record.get("unknown_count") or 0)
            if state == "completed":
                return f"这组照片拍完了，计划的 {requested} 张都已经发出来了。"
            if unknown:
                return (
                    f"这组照片处理结束，已确认发出 {sent} 张，"
                    f"另有 {unknown} 张发送状态无法确认；为避免重复，我没有自动重发。"
                )
            if state == "partial":
                return f"这组照片处理完了，已发出 {sent} 张，失败 {failed} 张。"
            if state == "cancelled":
                return f"这组照片已经停下来了，已发出 {sent} 张，取消 {cancelled} 张。"
            return f"这组照片没能全部完成，计划 {requested} 张，已发出 {sent} 张。"
        if state == "completed":
            return "照片拍好了，我已经发出来了。"
        if state == "cancelled":
            return "刚才那张照片已经停下来了。"
        if str(record.get("delivery_state") or "") == "unknown":
            return "照片已经生成，但发送状态无法确认，我没有自动重发以避免重复。"
        if state == "interrupted":
            return "刚才的照片任务被中断了，没有自动重复执行。"
        return "刚才的照片没能生成成功，这次任务已经结束了。"

    async def _send_deterministic_background_notification(
        self,
        manager: BackgroundImageTaskManager,
        record: dict[str, Any],
        target: TaskDeliveryTarget,
        *,
        attempt_id: str,
    ) -> None:
        token = str(record.get("notification_token") or "")
        try:
            event = await self._rebuild_background_event(target)
            await event.send(
                event.chain_result([Plain(self._background_notification_text(record))])
            )
        except Exception as exc:
            await manager.mark_notification(token, "unknown", attempt_id=attempt_id)
            logger.warning(
                "[background-image] deterministic notification failed: %s",
                manager.sanitize_error(exc),
            )
            return
        await manager.mark_notification(token, "sent", attempt_id=attempt_id)

    async def _background_notification_watchdog(
        self,
        manager: BackgroundImageTaskManager,
        token: str,
        target: TaskDeliveryTarget,
    ) -> None:
        await asyncio.sleep(self.BACKGROUND_NOTIFICATION_WATCHDOG_SECONDS)
        attempt_id = manager.new_task_id("notify-watchdog")
        record = await manager.claim_notification(
            token,
            attempt_id,
            from_states=("pending", "queued"),
        )
        if record is not None:
            await self._send_deterministic_background_notification(
                manager, record, target, attempt_id=attempt_id
            )

    async def _dispatch_background_completion(
        self,
        manager: BackgroundImageTaskManager,
        record: dict[str, Any],
        target: TaskDeliveryTarget,
    ) -> None:
        if manager.is_closing or not str(record.get("notification_token") or ""):
            return
        manager.start_managed(
            self._deliver_background_completion(manager, record, target),
            name=f"background-notification-{record.get('task_id')}",
        )

    async def _deliver_background_completion(
        self,
        manager: BackgroundImageTaskManager,
        record: dict[str, Any],
        target: TaskDeliveryTarget,
    ) -> None:
        async with manager.notification_turn(target.umo):
            token = str(record.get("notification_token") or "")
            safe, conversation = await self._background_context_is_safe(target)
            attempt_id = manager.new_task_id(
                "notify-agent" if safe else "notify-direct"
            )
            claimed = await manager.claim_notification(token, attempt_id)
            if claimed is None:
                return
            if not safe:
                await self._send_deterministic_background_notification(
                    manager, claimed, target, attempt_id=attempt_id
                )
                return

            try:
                event = await self._rebuild_background_event(target)
                event.set_extra(_BACKGROUND_COMPLETION_EVENT_EXTRA, True)
                event.set_extra("_gitee_bg_completion", True)
                event.set_extra("_gitee_bg_task_id", str(record.get("task_id") or ""))
                event.set_extra("_gitee_bg_notification_token", token)
                event.set_extra("_gitee_bg_notification_attempt", attempt_id)
                request = event.request_llm(
                    prompt=_BACKGROUND_COMPLETION_HISTORY_PLACEHOLDER,
                    tool_set=self._terminal_tool_set(),
                    conversation=conversation,
                )
                event.set_extra(_BACKGROUND_COMPLETION_REQUEST_EXTRA, request)
                if not await manager.mark_notification(
                    token, "queued", attempt_id=attempt_id
                ):
                    return
                adapter = self.context.get_platform_inst(target.platform_id)
                if adapter is None:
                    raise RuntimeError("Platform adapter disappeared")
                adapter.commit_event(event)
            except Exception as exc:
                logger.warning(
                    "[background-image] completion enqueue failed: %s",
                    manager.sanitize_error(exc),
                )
                await self._send_deterministic_background_notification(
                    manager, claimed, target, attempt_id=attempt_id
                )
                return
            manager.start_managed(
                self._background_notification_watchdog(manager, token, target),
                name=f"background-notification-watchdog-{record.get('task_id')}",
            )
            confirmed = await manager.wait_notification_terminal(
                token,
                timeout_seconds=self.BACKGROUND_NOTIFICATION_WAIT_SECONDS,
            )
            if not confirmed and not manager.is_closing:
                logger.warning(
                    "[background-image] notification confirmation timed out: task=%s",
                    record.get("task_id"),
                )

    def _get_batch_feature(self) -> dict:
        return self._get_feature("batch")

    def _get_batch_max_count(self) -> int:
        value = self._as_int(self._get_batch_feature().get("max_count", 8), default=8)
        return max(1, min(32, value))

    def _validate_llm_image_count(self, count: Any) -> int:
        if isinstance(count, bool):
            raise ValueError(
                "Image count must be an integer between 1 and the configured batch limit."
            )
        try:
            parsed = int(count)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Image count must be an integer between 1 and the configured batch limit."
            ) from exc
        if isinstance(count, float) and not count.is_integer():
            raise ValueError("Image count must be a whole number.")
        if isinstance(count, str) and str(parsed) != count.strip():
            raise ValueError("Image count must be a whole number.")

        max_count = self._get_batch_max_count()
        if parsed < 1 or parsed > max_count:
            raise ValueError(
                f"Image count must be between 1 and {max_count}; the request was not submitted."
            )
        return parsed

    def _get_draw_batch_concurrency(self) -> int:
        value = self._as_int(
            self._get_feature("draw").get("batch_concurrency", 2), default=2
        )
        return max(1, min(8, value))

    def _get_edit_batch_concurrency(self) -> int:
        value = self._as_int(
            self._get_feature("edit").get("batch_concurrency", 2), default=2
        )
        return max(1, min(8, value))

    def _get_draw_presets(self) -> dict[str, str]:
        presets: dict[str, str] = {}
        conf = self._get_feature("draw")
        items = conf.get("presets", [])
        if not isinstance(items, list):
            return presets
        for item in items:
            if isinstance(item, str) and ":" in item:
                key, val = item.split(":", 1)
                key = key.strip()
                val = val.strip()
                if key and val:
                    presets[key] = val
        return presets

    def _parse_structured_image_request(self, text: str) -> ParsedImageRequest | None:
        edit_presets = dict(getattr(self.edit, "presets", {}) or {})
        return parse_image_request(
            text,
            draw_presets=self._get_draw_presets(),
            edit_presets=edit_presets,
            known_provider_ids=set(self.registry.provider_ids()),
        )

    @staticmethod
    def _extract_batch_command_fragment(message: str) -> str:
        text = str(message or "")
        match = _BATCH_COMMAND_PATTERN.search(text)
        if not match:
            return ""
        return text[match.start() :].strip()

    def _batch_mode_label(self, spec: ImageTaskSpec) -> str:
        if spec.mode == "draw":
            if spec.preset_name:
                return f"文生图预设/{spec.preset_name}"
            return "文生图"
        if spec.mode == "edit":
            if spec.preset_name:
                return f"改图预设/{spec.preset_name}"
            return "改图"
        if spec.mode == "selfie_ref":
            return "自拍"
        return spec.mode

    def _get_batch_concurrency_for_mode(self, mode: str) -> int:
        if mode == "draw":
            return self._get_draw_batch_concurrency()
        return self._get_edit_batch_concurrency()

    def _resolve_target_backend(self, backend: str | None) -> str | None:
        raw = str(backend or "auto").strip()
        known_provider_ids = set(self.registry.provider_ids())
        if not raw or raw.lower() == "auto":
            return None
        if raw in known_provider_ids:
            return raw
        logger.warning(
            "[backend_override] 忽略未知 backend 覆盖，回退自动链路: backend=%s",
            raw,
        )
        return None

    def _get_draw_ratio_default_sizes(self) -> dict[str, str]:
        conf = self._get_feature("draw")
        raw = conf.get("ratio_default_sizes", {})
        if not isinstance(raw, dict):
            return {}
        out: dict[str, str] = {}
        for ratio, size in raw.items():
            r = str(ratio or "").strip()
            s = normalize_size_text(size)
            if not r or not s:
                continue
            out[r] = s
        return out

    def _resolve_ratio_size(self, ratio: str) -> str:
        ratio = str(ratio or "").strip()
        overrides = self._get_draw_ratio_default_sizes()
        size, warning = resolve_ratio_size(
            ratio,
            overrides=overrides,
            supported_ratios=self.SUPPORTED_RATIOS,
        )
        if warning:
            logger.warning("[aiimg] %s", warning)
        return size

    def _get_video_presets(self) -> dict[str, str]:
        presets: dict[str, str] = {}
        conf = self._get_feature("video")
        items = conf.get("presets", [])
        if not isinstance(items, list):
            return presets
        for item in items:
            if isinstance(item, str) and ":" in item:
                key, val = item.split(":", 1)
                key = key.strip()
                val = val.strip()
                if key and val:
                    presets[key] = val
        return presets

    def _get_video_chain(self) -> list[str]:
        conf = self._get_feature("video")
        chain = conf.get("chain", [])
        if not isinstance(chain, list):
            return []
        out: list[str] = []
        for item in chain:
            pid = self._extract_chain_provider_id(item)
            if pid and pid not in out:
                out.append(pid)
        return out

    def _parse_video_args(self, text: str) -> tuple[str | None, str]:
        """解析 /视频 参数，返回 (preset, prompt)

        - 当第一个 token 命中预设名时：preset=该 token, prompt=剩余内容
        - 否则：preset=None, prompt=text
        """
        text = (text or "").strip()
        if not text:
            return None, ""

        first, _, rest = text.partition(" ")
        if first and first in self._get_video_presets():
            return first, rest.strip()
        return None, text

    async def _prepare_edit_image_bytes(self, event: AstrMessageEvent) -> list[bytes]:
        image_segs = await get_images_from_event(
            event,
            include_avatar=True,
            include_sender_avatar_fallback=False,
        )
        if not image_segs:
            raise RuntimeError("当前消息没有可用输入图片，无法执行改图批量任务。")
        bytes_images = await self._image_segs_to_bytes(image_segs)
        if not bytes_images:
            raise RuntimeError("当前消息图片读取失败，无法执行改图批量任务。")
        return bytes_images

    async def _execute_image_task_spec(
        self,
        event: AstrMessageEvent,
        spec: ImageTaskSpec,
        *,
        prepared_edit_images: list[bytes] | None = None,
        size: str | None = None,
        resolution: str | None = None,
    ) -> ExecutedImageTask:
        if spec.mode == "draw":
            prompt = self._expand_time_placeholders(
                str(spec.effective_prompt or spec.user_prompt or "").strip()
            )
            if not prompt:
                raise RuntimeError("文生图提示词为空。")
            image_path = await self.draw.generate(
                prompt,
                provider_id=spec.provider_id,
                size=size,
                resolution=resolution,
            )
            task_meta = self._build_image_task_meta(
                mode="text",
                user_prompt=spec.user_prompt,
                effective_user_prompt=prompt if spec.preset_name else spec.user_prompt,
                effective_prompt=prompt,
                continue_with="text",
                backend=spec.provider_id,
            )
            return ExecutedImageTask(
                spec=spec, image_path=image_path, task_meta=task_meta
            )

        if spec.mode == "edit":
            bytes_images = prepared_edit_images
            if bytes_images is None:
                bytes_images = await self._prepare_edit_image_bytes(event)
            image_path = await self.edit.edit(
                prompt=spec.user_prompt,
                images=bytes_images,
                backend=spec.provider_id,
                preset=spec.preset_name,
                size=size,
                resolution=resolution,
            )
            task_meta = self._build_image_task_meta(
                mode="edit",
                user_prompt=spec.user_prompt,
                effective_user_prompt=spec.effective_prompt,
                effective_prompt=spec.effective_prompt,
                continue_with="edit",
                backend=spec.provider_id,
            )
            if spec.preset_name:
                task_meta["preset_name"] = spec.preset_name
            return ExecutedImageTask(
                spec=spec, image_path=image_path, task_meta=task_meta
            )

        if spec.mode == "selfie_ref":
            if not self._is_selfie_enabled():
                raise RuntimeError(self._selfie_disabled_message())
            image_path, task_meta = await self._generate_selfie_image_with_meta(
                event,
                spec.user_prompt,
                spec.provider_id,
                size=size,
                resolution=resolution,
            )
            return ExecutedImageTask(
                spec=spec, image_path=image_path, task_meta=task_meta
            )

        raise RuntimeError(f"不支持的图片任务模式: {spec.mode}")

    async def _run_batch_specs(
        self,
        event: AstrMessageEvent,
        specs: list[ImageTaskSpec],
        *,
        size: str | None = None,
        resolution: str | None = None,
    ) -> list[BatchRunResult[ExecutedImageTask]]:
        if not specs:
            return []

        prepared_edit_images: list[bytes] | None = None
        if any(spec.mode == "edit" for spec in specs):
            prepared_edit_images = await self._prepare_edit_image_bytes(event)

        concurrency = min(
            len(specs), self._get_batch_concurrency_for_mode(specs[0].mode)
        )

        async def _runner(index: int, spec: ImageTaskSpec) -> ExecutedImageTask:
            return await self._execute_image_task_spec(
                event,
                spec,
                prepared_edit_images=prepared_edit_images,
                size=size,
                resolution=resolution,
            )

        return await run_batch(specs, concurrency=concurrency, runner=_runner)

    async def _remember_batch_success(
        self,
        event: AstrMessageEvent,
        results: list[BatchRunResult[ExecutedImageTask]],
        *,
        prompt: str | None = None,
        mode: str | None = None,
        count: int | None = None,
    ) -> None:
        for result in reversed(results):
            if not result.success or result.value is None:
                continue
            self._remember_last_image(event, result.value.image_path)
            await self._save_last_image_task_meta(event, result.value.task_meta)
            resolved_prompt = (
                prompt
                if prompt is not None
                else result.value.task_meta.get("user_prompt")
            )
            resolved_mode = mode or result.value.task_meta.get("mode")
            await self._append_image_history_note(
                event,
                prompt=resolved_prompt,
                mode=resolved_mode,
                count=max(1, int(count or 1)),
                dedupe_key=self._image_history_dedupe_key(
                    event,
                    prompt=resolved_prompt,
                    mode=resolved_mode,
                    count=max(1, int(count or 1)),
                ),
            )
            return

    async def _send_batch_results_single(
        self,
        event: AstrMessageEvent,
        results: list[BatchRunResult[ExecutedImageTask]],
        *,
        title: str,
    ) -> list[BatchRunResult[ExecutedImageTask]]:
        sent_results: list[BatchRunResult[ExecutedImageTask]] = []
        for result in results:
            if not result.success or result.value is None:
                continue
            sent = await self._send_image_with_fallback(event, result.value.image_path)
            if sent:
                sent_results.append(result)
        return sent_results

    async def _send_batch_results(
        self,
        event: AstrMessageEvent,
        results: list[BatchRunResult[ExecutedImageTask]],
        *,
        title: str,
    ) -> list[BatchRunResult[ExecutedImageTask]]:
        return await self._send_batch_results_single(event, results, title=title)

    async def _plan_batch_prompt_items(
        self,
        *,
        mode: str,
        user_prompt: str,
        count: int,
    ) -> list[PlannedPromptItem]:
        provider = self.context.get_using_provider()
        if provider is None or not hasattr(provider, "text_chat"):
            raise RuntimeError("当前没有可用的 LLM 提供商，无法规划批量提示词。")

        planning_prompt = build_batch_planning_prompt(
            mode=mode,
            user_prompt=user_prompt,
            count=count,
        )
        last_error: Exception | None = None
        for _ in range(3):
            llm_response = await provider.text_chat(
                prompt=planning_prompt,
                contexts=[],
                image_urls=[],
                func_tool=None,
                system_prompt=(
                    "You plan image prompt sets. Output JSON only. "
                    "No markdown, no code fence, no explanation."
                ),
            )
            text = str(getattr(llm_response, "completion_text", "") or "").strip()
            if not text:
                last_error = RuntimeError("LLM returned empty planner output")
                continue
            try:
                items = parse_planned_prompt_items(text)
                validation_error = validate_planned_prompt_items(
                    items, expected_count=count
                )
                if validation_error is not None:
                    raise ValueError(validation_error)
                return items
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"批量提示词规划失败: {last_error}")

    def _resolve_llm_batch_mode(self, mode: str) -> str:
        normalized = self._normalize_llm_image_mode(mode)
        if normalized == "text":
            return "draw"
        return normalized

    async def _video_begin(self, user_id: str) -> bool:
        """单用户并发保护：成功占用返回 True，否则 False（上限可配置）"""
        return await self._begin_user_job(str(user_id or ""), kind="video")

    async def _video_end(self, user_id: str) -> None:
        await self._end_user_job(str(user_id or ""), kind="video")

    async def _send_video_result(self, event: AstrMessageEvent, video_url: str) -> None:
        vconf = self._get_feature("video")
        mode = str(vconf.get("send_mode", "auto")).strip().lower()
        if mode not in {"auto", "url", "file"}:
            mode = "auto"

        send_timeout = self._as_int(vconf.get("send_timeout_seconds", 90), default=90)
        send_timeout = max(10, min(send_timeout, 300))

        download_timeout = self._as_int(
            vconf.get("download_timeout_seconds", 300), default=300
        )
        download_timeout = max(1, min(download_timeout, 3600))

        async def _send_file(url: str) -> bool:
            try:
                video_path = await self.videomgr.download_video(
                    url, timeout_seconds=download_timeout
                )
                await asyncio.wait_for(
                    event.send(
                        event.chain_result([Video.fromFileSystem(str(video_path))])
                    ),
                    timeout=float(send_timeout),
                )
                return True
            except Exception as e:
                logger.warning(f"[视频] 本地文件发送失败: {e}")
                return False

        async def _send_url(url: str) -> bool:
            try:
                await asyncio.wait_for(
                    event.send(event.chain_result([Video.fromURL(url)])),
                    timeout=float(send_timeout),
                )
                return True
            except Exception as e:
                logger.warning(f"[视频] URL 发送失败: {e}")
                return False

        # file/url forced
        if mode == "file":
            if await _send_file(video_url):
                return
            await event.send(event.plain_result(video_url))
            return

        if mode == "url":
            if await _send_url(video_url):
                return
            await event.send(event.plain_result(video_url))
            return

        if await _send_url(video_url):
            return
        if await _send_file(video_url):
            return
        await event.send(event.plain_result(video_url))

    async def _async_generate_video(
        self,
        event: AstrMessageEvent,
        prompt: str,
        user_id: str,
        *,
        provider_id: str | None = None,
        llm_tool_failure: bool = False,
    ) -> None:
        try:
            image_segs = await get_images_from_event(
                event,
                include_avatar=True,
                include_sender_avatar_fallback=False,
            )
            had_image = bool(image_segs)
            image_bytes: bytes | None = None
            for i, seg in enumerate(image_segs):
                try:
                    b64 = await seg.convert_to_base64()
                    image_bytes = decode_base64_image_payload(b64)
                    break
                except Exception as e:
                    logger.warning(f"[视频] 图片 {i + 1} 转换失败，跳过: {e}")

            # 允许文生视频（无图）走支持的后端；但若用户确实发了图却读不到，则直接失败
            if had_image and not image_bytes:
                if llm_tool_failure:
                    await self._append_plugin_conversation_note(
                        event,
                        "The last video generation task failed and has ended because the source image could not be read. Do not retry automatically unless the user explicitly asks.",
                    )
                if llm_tool_failure:
                    await self._signal_llm_tool_failure(event)
                else:
                    await mark_failed(event)
                return

            t_start = time.perf_counter()
            candidates = (
                [str(provider_id).strip()] if provider_id else self._get_video_chain()
            )
            candidates = [c for c in candidates if c]
            if not candidates:
                raise RuntimeError(
                    "No video providers configured. Please set features.video.chain."
                )

            last_error: Exception | None = None
            video_url: str | None = None
            used_pid: str | None = None
            for pid in candidates:
                try:
                    backend = self.registry.get_video_backend(pid)
                    candidate_url = await backend.generate_video_url(
                        prompt=prompt, image_bytes=image_bytes
                    )
                    candidate_url = str(candidate_url or "").strip()
                    if not candidate_url:
                        raise RuntimeError("Provider returned empty video url")
                    video_url = candidate_url
                    used_pid = pid
                    break
                except Exception as e:
                    last_error = e
                    logger.warning("[视频] Provider=%s 失败: %s", pid, e)

            if not video_url:
                raise RuntimeError(f"视频生成失败: {last_error}") from last_error

            await self._send_video_result(event, video_url)
            await mark_success(event)
            if llm_tool_failure:
                await self._append_plugin_conversation_note(
                    event,
                    "The last video generation task has completed and the video was already sent to the user. Do not continue or resubmit this task unless the user explicitly asks for another video.",
                )

            t_end = time.perf_counter()
            name = used_pid or "video"
            logger.info(f"[视频] 完成: provider={name}, 耗时={t_end - t_start:.2f}s")

        except Exception as e:
            logger.error(f"[视频] 失败: {e}", exc_info=True)
            if llm_tool_failure:
                await self._append_plugin_conversation_note(
                    event,
                    "The last video generation task failed and has ended. Reason: "
                    + self._summarize_status_text(
                        e,
                        fallback="unknown error",
                    )
                    + ". Do not retry automatically unless the user explicitly asks.",
                )
            if llm_tool_failure:
                await self._signal_llm_tool_failure(event)
            else:
                await mark_failed(event)
        finally:
            await self._video_end(user_id)

    async def _do_edit_direct(
        self,
        event: AstrMessageEvent,
        prompt: str,
        backend: str | None = None,
        preset: str | None = None,
    ):
        """改图执行入口 (非 generator 版本，用于动态注册的命令)

        使用 event.send() 直接发送消息，不使用 yield
        """
        user_id = str(event.get_sender_id() or "")
        request_id = self._debounce_key(event, "edit", user_id)

        # 防抖
        if self.debouncer.hit(request_id):
            await mark_failed(event)
            return

        p = (prompt or "").strip()
        history_prompt = p or str(preset or "").strip()
        override, rest = self._parse_provider_override_prefix(p)
        if override:
            backend = override
            prompt = rest
            history_prompt = str(prompt or "").strip()

        # 获取图片
        image_segs = await get_images_from_event(
            event,
            include_avatar=True,
            include_sender_avatar_fallback=False,
        )
        logger.debug(f"[改图] 获取到 {len(image_segs)} 个图片段")
        if not image_segs:
            await mark_failed(event)
            return

        bytes_images: list[bytes] = []
        for i, seg in enumerate(image_segs):
            try:
                logger.debug(f"[改图] 转换图片 {i + 1}/{len(image_segs)}...")
                b64 = await seg.convert_to_base64()
                bytes_images.append(decode_base64_image_payload(b64))
                logger.debug(
                    f"[改图] 图片 {i + 1} 转换成功, 大小={len(bytes_images[-1])} bytes"
                )
            except Exception as e:
                logger.warning(f"[改图] 图片 {i + 1} 转换失败，跳过: {e}")

        if not bytes_images:
            await mark_failed(event)
            return

        if not await self._begin_user_job(user_id, kind="image"):
            await mark_failed(event)
            return

        try:
            # 标记处理中
            await mark_processing(event)
            t_start = time.perf_counter()
            image_path = await self.edit.edit(
                prompt=prompt,
                images=bytes_images,
                backend=backend,
                preset=preset,
            )
            t_end = time.perf_counter()

            self._remember_last_image(event, image_path)
            sent = await self._send_image_with_fallback(event, image_path)
            if not sent:
                await mark_failed(event)
                logger.warning(
                    "[改图] 结果发送失败，已仅使用表情标注: reason=%s",
                    sent.reason,
                )
                return

            # 标记成功
            await mark_success(event)
            await self._append_image_history_note(
                event,
                prompt=history_prompt,
                mode="edit",
            )
            display_name = preset or (prompt[:20] if prompt else "改图")
            logger.info(f"[改图] 完成: {display_name}..., 耗时={t_end - t_start:.2f}s")

        except Exception as e:
            logger.error(f"[改图] 失败: {e}", exc_info=True)
            await mark_failed(event)
        finally:
            await self._end_user_job(user_id, kind="image")

    async def _do_edit(
        self,
        event: AstrMessageEvent,
        prompt: str,
        backend: str | None = None,
        preset: str | None = None,
    ):
        """统一改图执行入口

        预设触发逻辑:
        1. 如果 preset 参数已指定，直接使用
        2. 否则检查 prompt 是否匹配预设名，若匹配则自动转为预设
        3. 都不匹配则作为普通提示词处理
        """
        user_id = str(event.get_sender_id() or "")
        request_id = self._debounce_key(event, "edit", user_id)

        # 防抖
        if self.debouncer.hit(request_id):
            await mark_failed(event)
            return

        p = (prompt or "").strip()
        history_prompt = p or str(preset or "").strip()
        override, rest = self._parse_provider_override_prefix(p)
        if override:
            backend = override
            prompt = rest
            history_prompt = str(prompt or "").strip()

        # 预设自动检测: prompt 完全匹配预设名时，自动转为预设
        if not preset and prompt:
            prompt_stripped = prompt.strip()
            preset_names = self.edit.get_preset_names()
            if prompt_stripped in preset_names:
                preset = prompt_stripped
                prompt = ""
                logger.debug(f"[改图] 自动匹配预设: {preset}")

        # 获取图片
        image_segs = await get_images_from_event(
            event,
            include_avatar=True,
            include_sender_avatar_fallback=False,
        )
        if not image_segs:
            await mark_failed(event)
            return

        bytes_images: list[bytes] = []
        for seg in image_segs:
            try:
                b64 = await seg.convert_to_base64()
                bytes_images.append(decode_base64_image_payload(b64))
            except Exception as e:
                logger.warning(f"[改图] 图片转换失败，跳过: {e}")

        if not bytes_images:
            await mark_failed(event)
            return

        if not await self._begin_user_job(user_id, kind="image"):
            await mark_failed(event)
            return

        try:
            # 标记处理中
            await mark_processing(event)
            t_start = time.perf_counter()
            image_path = await self.edit.edit(
                prompt=prompt,
                images=bytes_images,
                backend=backend,
                preset=preset,
            )
            t_end = time.perf_counter()

            self._remember_last_image(event, image_path)
            sent = await self._send_image_with_fallback(event, image_path)
            if not sent:
                await mark_failed(event)
                logger.warning(
                    "[改图] 结果发送失败，已仅使用表情标注: reason=%s",
                    sent.reason,
                )
                return

            # 标记成功
            await mark_success(event)
            await self._append_image_history_note(
                event,
                prompt=history_prompt,
                mode="edit",
            )
            display_name = preset or (prompt[:20] if prompt else "改图")
            logger.info(f"[改图] 完成: {display_name}..., 耗时={t_end - t_start:.2f}s")

        except Exception as e:
            logger.error(f"[改图] 失败: {e}")
            await mark_failed(event)
        finally:
            await self._end_user_job(user_id, kind="image")

    # ==================== 自拍参考照：内部实现 ====================

    def _get_selfie_conf(self) -> dict:
        return self._get_feature("selfie")

    async def _ensure_tool_image_cache_dir(self) -> None:
        tool_image_dir = Path(get_astrbot_temp_path()) / "tool_images"
        await asyncio.to_thread(tool_image_dir.mkdir, parents=True, exist_ok=True)

    async def _build_llm_tool_image_result(
        self, image_path: Path
    ) -> mcp.types.CallToolResult | None:
        try:
            image_bytes = await asyncio.to_thread(Path(image_path).read_bytes)
        except Exception as exc:
            logger.warning(
                "[aiimg_generate] failed to read image for LLM context: path=%s err=%s",
                image_path,
                exc,
            )
            return None

        if not image_bytes:
            logger.warning(
                "[aiimg_generate] skip empty image for LLM context: path=%s",
                image_path,
            )
            return None

        mime_type, _ = guess_image_mime_and_ext(image_bytes)
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        return mcp.types.CallToolResult(
            content=[
                mcp.types.ImageContent(
                    type="image",
                    data=image_b64,
                    mimeType=mime_type,
                )
            ]
        )

    async def _finalize_llm_tool_image(
        self,
        event: AstrMessageEvent,
        image_path: Path,
        *,
        task_meta: dict[str, Any],
    ) -> mcp.types.CallToolResult | None:
        self._remember_last_image(event, image_path)

        sent = await self._send_image_with_fallback(event, image_path)
        if not sent:
            await self._signal_llm_tool_failure(event)
            logger.warning(
                "[aiimg_generate] image send failed, emoji fallback only: reason=%s",
                sent.reason,
            )
            return self._llm_tool_text_result(
                "Image generation finished, but sending the image to the user failed. This request has ended. Do not retry automatically unless the user explicitly asks."
            )

        await mark_success(event)
        await self._save_last_image_task_meta(event, task_meta)
        await self._append_image_history_note(
            event,
            prompt=task_meta.get("user_prompt"),
            mode=task_meta.get("mode"),
            dedupe_key=self._image_history_dedupe_key(
                event,
                prompt=task_meta.get("user_prompt"),
                mode=task_meta.get("mode"),
            ),
        )
        return None

    def _get_selfie_ref_store_key(self, event: AstrMessageEvent) -> str:
        """用于 ReferenceStore 的固定 key（按 bot self_id 隔离）。"""
        self_id = ""
        try:
            if hasattr(event, "get_self_id"):
                self_id = str(event.get_self_id() or "").strip()
        except Exception:
            self_id = ""
        return f"bot_selfie_{self_id}" if self_id else "bot_selfie"

    def _resolve_data_rel_path(self, rel_path: str) -> Path | None:
        """将 data_dir 下的相对路径解析为绝对路径，并阻止路径穿越。"""
        if not isinstance(rel_path, str) or not rel_path.strip():
            return None
        rel = rel_path.replace("\\", "/").lstrip("/")
        parts = [p for p in rel.split("/") if p]
        if any(p in {".", ".."} for p in parts):
            return None
        base = Path(self.data_dir).resolve(strict=False)
        target = (base / "/".join(parts)).resolve(strict=False)
        try:
            target.relative_to(base)
        except ValueError:
            return None
        return target

    def _get_config_selfie_reference_paths(self) -> list[Path]:
        """从 WebUI file 配置项读取参考图路径。"""
        conf = self._get_selfie_conf()
        ref_list = conf.get("reference_images", [])
        if not isinstance(ref_list, list):
            return []

        paths: list[Path] = []
        for rel_path in ref_list:
            p = self._resolve_data_rel_path(str(rel_path))
            if not p:
                continue
            if p.is_file():
                paths.append(p)
        return paths

    async def _get_selfie_reference_paths(
        self, event: AstrMessageEvent
    ) -> tuple[list[Path], str]:
        """返回(路径列表, 来源)；来源=webui/store/none"""
        webui_paths = self._get_config_selfie_reference_paths()
        if webui_paths:
            return webui_paths, "webui"

        store_key = self._get_selfie_ref_store_key(event)
        store_paths = await self.refs.get_paths(store_key)
        if store_paths:
            return store_paths, "store"

        return [], "none"

    async def _read_paths_bytes(self, paths: list[Path]) -> list[bytes]:
        out: list[bytes] = []
        for p in paths:
            try:
                data = await asyncio.to_thread(p.read_bytes)
            except Exception:
                continue
            if data:
                out.append(data)
        return out

    async def _image_segs_to_bytes(self, image_segs: list) -> list[bytes]:
        """将 Image 组件列表转换为 bytes。"""
        out: list[bytes] = []
        for seg in image_segs:
            try:
                b64 = await seg.convert_to_base64()
                out.append(decode_base64_image_payload(b64))
            except Exception as e:
                logger.warning(f"[图片] 转换失败，跳过: {e}")
        return out

    async def _has_message_images(self, event: AstrMessageEvent) -> bool:
        """仅检测用户消息/引用里的图片（不含头像兜底）。"""
        image_segs = await get_images_from_event(event, include_avatar=False)
        return bool(image_segs)

    async def _has_message_images_or_avatar_mentions(
        self, event: AstrMessageEvent
    ) -> bool:
        if await self._has_message_images(event):
            return True
        return any(str(uid).isdigit() for uid in collect_at_user_ids(event))

    @staticmethod
    def _expand_time_placeholders(text: str) -> str:
        """Replace {now}/{date}/{time}/{weekday} with current values."""
        from datetime import datetime as _dt

        _weekday_names = [
            "星期一",
            "星期二",
            "星期三",
            "星期四",
            "星期五",
            "星期六",
            "星期日",
        ]
        now = _dt.now()
        return (
            text.replace("{now}", now.strftime("%Y-%m-%d %H:%M"))
            .replace("{date}", now.strftime("%Y-%m-%d"))
            .replace("{time}", now.strftime("%H:%M"))
            .replace("{weekday}", _weekday_names[now.weekday()])
        )

    @staticmethod
    def _split_config_keywords(value: Any, defaults: tuple[str, ...]) -> tuple[str, ...]:
        raw = value if isinstance(value, (list, tuple)) else []
        result = tuple(str(item).strip().casefold() for item in raw if str(item).strip())
        return result or defaults

    @staticmethod
    def _contains_keyword(text: str, keywords: tuple[str, ...]) -> bool:
        folded = str(text or "").casefold()
        return any(keyword and keyword in folded for keyword in keywords)

    @staticmethod
    def _strip_outfit_underwear(outfit: str) -> str:
        hidden_labels = ("内衣", "内裤", "文胸", "胸罩", "bra", "panties", "underwear", "lingerie")
        kept = []
        for line in str(outfit or "").splitlines():
            if any(label in line.casefold() for label in hidden_labels):
                continue
            if line.strip():
                kept.append(line.strip())
        return "\n".join(kept)

    def _get_busy_schedule_outfit(self) -> str:
        getter = getattr(self.context, "_busy_schedule_get_facts", None)
        if callable(getter):
            try:
                facts = getter()
                if isinstance(facts, dict):
                    outfit = str(facts.get("outfit") or "").strip()
                    if outfit:
                        return self._strip_outfit_underwear(outfit)
            except Exception:
                pass
        return self._strip_outfit_underwear(
            str(getattr(self.context, "_busy_schedule_outfit", "") or "").strip()
        )

    def _resolve_lighting_placeholder(self) -> str:
        from datetime import datetime
        conf = self._get_selfie_conf()
        now = datetime.now()
        now_minutes = now.hour * 60 + now.minute
        rules = conf.get("lighting_rules", [])
        if isinstance(rules, list):
            for item in rules:
                text = str(item or "").strip()
                if "=" not in text or "-" not in text:
                    continue
                span, value = text.split("=", 1)
                try:
                    start_text, end_text = span.strip().split("-", 1)
                    sh, sm = (int(part) for part in start_text.split(":", 1))
                    eh, em = (int(part) for part in end_text.split(":", 1))
                    start, end = sh * 60 + sm, eh * 60 + em
                except (ValueError, TypeError):
                    continue
                if not (0 <= start < 1440 and 0 <= end <= 1440 and value.strip()):
                    continue
                if end == 1440:
                    matched = now_minutes >= start
                else:
                    matched = start <= now_minutes < end if start < end else now_minutes >= start or now_minutes < end
                if matched:
                    return value.strip()
        if 5 * 60 <= now_minutes < 7 * 60:
            return "黎明、日出前后柔和的金色自然光"
        if 7 * 60 <= now_minutes < 17 * 60:
            return "日间自然光"
        if 17 * 60 <= now_minutes < 19 * 60:
            return "黄昏、日落前后柔和的金色自然光"
        return "夜间环境、室内人造灯光"

    def _prepare_selfie_prompt_context(
        self, prompt: str, control_text: str = ""
    ) -> tuple[str, str]:
        conf = self._get_selfie_conf()
        current = str(prompt or "").strip()
        force = self._split_config_keywords(conf.get("outfit_force_keywords"), ("沿用今日穿搭", "沿用今天的穿搭", "强制使用今日穿搭", "强制使用今天的穿搭"))
        skip = self._split_config_keywords(conf.get("outfit_skip_keywords"), ("不要使用今日穿搭", "不要使用今天的穿搭", "不使用今日穿搭", "不使用今天的穿搭"))
        auto = self._split_config_keywords(conf.get("outfit_auto_keywords"), ("裙子", "短裙", "长裙", "半身裙", "连衣裙", "裤子", "短裤", "长裤", "牛仔裤", "上衣", "外套", "衬衫", "针织衫", "卫衣", "夹克", "袜子", "鞋子", "靴子", "帽子", "围巾", "配饰", "服装", "衣服", "换装", "穿搭", "outfit", "dress", "skirt", "pants", "shirt", "jacket"))
        use_outfit = bool(conf.get("today_outfit_enabled", True))
        explicit_source = f"{control_text}\n{current}"
        if self._contains_keyword(explicit_source, force):
            use_outfit = True
        elif self._contains_keyword(explicit_source, skip) or self._contains_keyword(current, auto):
            use_outfit = False
        cleaned = current
        for control in tuple(dict.fromkeys((*force, *skip))):
            cleaned = re.sub(re.escape(control), "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip(" ，,。.!！\n\t"), "" if not use_outfit else self._get_busy_schedule_outfit()

    def _build_selfie_prompt(
        self,
        prompt: str,
        *,
        reference_count: int,
        extra_reference_count: int,
        control_text: str = "",
    ) -> str:
        conf = self._get_selfie_conf()
        prefix = str(conf.get("prompt_prefix", "") or "").strip()
        if not prefix:
            prefix = (
                "请根据参考图生成一张新的自拍照：\n"
                "1) 以固定人物参考图的人脸身份为准，保持五官和气质一致。\n"
                "2) 本次用户参考图仅用于用户指定的服装、姿势、构图或场景。\n"
                "3) 输出一张高质量照片风格自拍，不要拼图，不要水印。\n"
                "今日外显穿搭：{today_outfit}\n"
                "当前时间光线：{lighting}"
            )

        user_prompt, outfit = self._prepare_selfie_prompt_context(prompt, control_text)
        has_outfit_placeholder = "{today_outfit}" in prefix
        has_lighting_placeholder = "{lighting}" in prefix
        prefix = prefix.replace("{today_outfit}", outfit)
        lighting = self._resolve_lighting_placeholder()
        prefix = prefix.replace("{lighting}", lighting)
        if outfit and not has_outfit_placeholder:
            prefix = f"{prefix}\n今日外显穿搭：{outfit}"
        if not has_lighting_placeholder:
            prefix = f"{prefix}\n当前时间光线：{lighting}"
        prefix = self._expand_time_placeholders(prefix)
        user_prompt = self._expand_time_placeholders(user_prompt or "日常自拍照")
        reference_count = max(0, int(reference_count or 0))
        extra_reference_count = max(0, int(extra_reference_count or 0))
        if extra_reference_count > 0:
            extra_start = reference_count + 1
            extra_end = reference_count + extra_reference_count
            if extra_start == extra_end:
                extra_range = f"第 {extra_start} 张"
            else:
                extra_range = f"第 {extra_start}-{extra_end} 张"
            reference_note = (
                f"图片顺序：第 1-{reference_count} 张是固定人物参考图；"
                f"{extra_range}是本次用户附带或引用的参考图。"
                "用户参考图不是待修改原图，只参考用户要求中明确指定的服装、姿势、构图或场景。"
            )
        else:
            reference_note = f"图片顺序：第 1-{reference_count} 张均为固定人物参考图。"
        prefix = "\n".join(line.rstrip() for line in prefix.splitlines()).strip()
        sections = [part for part in (prefix, reference_note, f"用户要求：{user_prompt}") if part]
        return "\n\n".join(sections)

    def _merge_selfie_chain_with_edit_chain(
        self, selfie_chain: list[object]
    ) -> list[dict]:
        """将自拍链路与改图链路合并（自拍优先，去重 provider_id）。"""
        merged: list[dict] = []
        seen: set[str] = set()

        def append_unique(items: list) -> None:
            for item in items:
                normalized = self._normalize_chain_item(item)
                if not normalized:
                    continue
                pid = str(normalized.get("provider_id") or "").strip()
                if not pid or pid in seen:
                    continue
                merged.append(normalized)
                seen.add(pid)

        append_unique(selfie_chain)

        edit_chain_raw = self._get_feature("edit").get("chain", [])
        if isinstance(edit_chain_raw, list):
            append_unique(edit_chain_raw)

        return merged

    async def _generate_selfie_image_with_meta(
        self,
        event: AstrMessageEvent,
        prompt: str,
        backend: str | None,
        *,
        size: str | None = None,
        resolution: str | None = None,
        follow_up_meta: dict[str, Any] | None = None,
    ) -> tuple[Path, dict[str, Any]]:
        conf = self._get_selfie_conf()
        if not self._is_selfie_enabled():
            raise RuntimeError(self._selfie_disabled_message())

        # 1) 读取参考照（WebUI 优先，其次命令设置的 store）
        ref_paths, ref_source = await self._get_selfie_reference_paths(event)
        ref_images = await self._read_paths_bytes(ref_paths)
        if not ref_images:
            raise RuntimeError(
                "未设置自拍参考照。请先：发送图片 + /自拍参考 设置，或在 WebUI 配置 features.selfie.reference_images 上传。"
            )

        # 2) 读取额外参考图（衣服/姿势/场景）
        extra_segs = await get_images_from_event(event, include_avatar=False)
        extra_bytes = await self._image_segs_to_bytes(extra_segs)

        # 3) 拼接输入图：参考照在前
        images = [*ref_images, *extra_bytes]

        effective_user_prompt = self._build_selfie_follow_up_prompt(
            prompt, follow_up_meta
        )
        final_prompt = self._build_selfie_prompt(
            effective_user_prompt,
            reference_count=len(ref_images),
            extra_reference_count=len(extra_bytes),
            control_text=str(getattr(event, "message_str", "") or ""),
        )

        chain_override: list[dict] | None = None
        use_edit_chain = bool(conf.get("use_edit_chain_when_empty", True))
        raw_chain = conf.get("chain", [])
        if isinstance(raw_chain, list):
            chain_items = [
                normalized
                for normalized in (self._normalize_chain_item(x) for x in raw_chain)
                if normalized is not None
            ]
            if chain_items:
                chain_override = chain_items

        if backend is None:
            if chain_override is None:
                if not use_edit_chain:
                    raise RuntimeError(
                        "No selfie provider chain configured. Please set features.selfie.chain or enable features.selfie.use_edit_chain_when_empty."
                    )
            elif use_edit_chain:
                # 自拍链路可作为主链，改图链路作为补充兜底，避免“自拍链仅一项导致无兜底”。
                chain_override = self._merge_selfie_chain_with_edit_chain(
                    chain_override
                )

        if chain_override:
            logger.debug(
                "[selfie] effective providers=%s",
                [
                    str(x.get("provider_id") or "").strip()
                    for x in chain_override
                    if isinstance(x, dict)
                ],
            )

        # 4) 千问后端可选 task_types（仅对 gitee 生效）
        task_types = conf.get("gitee_task_types")
        if isinstance(task_types, list) and task_types:
            gitee_task_types = [str(x).strip() for x in task_types if str(x).strip()]
        else:
            gitee_task_types = ["id", "background", "style"]

        default_output = str(conf.get("default_output") or "").strip() or None

        image_path = await self.edit.edit(
            prompt=final_prompt,
            images=images,
            backend=backend,
            task_types=gitee_task_types,
            size=size,
            resolution=resolution,
            default_output=default_output,
            chain_override=chain_override,
        )
        task_meta = self._build_image_task_meta(
            mode="selfie_ref",
            user_prompt=prompt,
            effective_user_prompt=effective_user_prompt,
            effective_prompt=final_prompt,
            reference_source=ref_source,
            reference_count=len(ref_images),
            extra_reference_count=len(extra_bytes),
            continue_with="selfie_ref",
            follow_up=follow_up_meta is not None,
            backend=backend,
        )
        return image_path, task_meta

    async def _generate_selfie_image(
        self,
        event: AstrMessageEvent,
        prompt: str,
        backend: str | None,
        *,
        size: str | None = None,
        resolution: str | None = None,
    ) -> Path:
        image_path, _ = await self._generate_selfie_image_with_meta(
            event,
            prompt,
            backend,
            size=size,
            resolution=resolution,
        )
        return image_path

    async def _do_selfie(
        self,
        event: AstrMessageEvent,
        prompt: str,
        backend: str | None = None,
    ):
        """指令 /自拍 执行入口。"""
        if not self._is_selfie_enabled():
            await mark_failed(event)
            return

        user_id = str(event.get_sender_id() or "")
        request_id = self._debounce_key(event, "selfie", user_id)

        if self.debouncer.hit(request_id):
            await mark_failed(event)
            return

        if not await self._begin_user_job(user_id, kind="image"):
            await mark_failed(event)
            return

        p = (prompt or "").strip()
        override, rest = self._parse_provider_override_prefix(p)
        if override:
            backend = override
            prompt = rest
        try:
            await mark_processing(event)
            image_path, task_meta = await self._generate_selfie_image_with_meta(
                event, prompt, backend
            )
            self._remember_last_image(event, image_path)
            sent = await self._send_image_with_fallback(event, image_path)
            if not sent:
                await mark_failed(event)
                logger.warning(
                    "[自拍] 结果发送失败，已仅使用表情标注: reason=%s",
                    sent.reason,
                )
                return
            await mark_success(event)
            await self._save_last_image_task_meta(event, task_meta)
            await self._append_image_history_note(
                event,
                prompt=task_meta.get("user_prompt"),
                mode=task_meta.get("mode"),
            )
        except Exception as e:
            logger.error(f"[自拍] 失败: {e}", exc_info=True)
            await mark_failed(event)
        finally:
            await self._end_user_job(user_id, kind="image")

    async def _set_selfie_reference(self, event: AstrMessageEvent):
        if not self._is_selfie_enabled():
            await mark_failed(event)
            return

        image_segs = await get_images_from_event(event, include_avatar=False)
        if not image_segs:
            await mark_failed(event)
            return

        bytes_images = await self._image_segs_to_bytes(image_segs)
        if not bytes_images:
            await mark_failed(event)
            return

        # 限制数量，避免一次塞太多
        max_images = 8
        bytes_images = bytes_images[:max_images]

        store_key = self._get_selfie_ref_store_key(event)
        try:
            await self.refs.set(store_key, bytes_images)
        except Exception:
            await mark_failed(event)
            return

        await mark_success(event)

    async def _show_selfie_reference(self, event: AstrMessageEvent):
        if not self._is_selfie_enabled():
            await mark_failed(event)
            return

        paths, source = await self._get_selfie_reference_paths(event)
        if not paths:
            await mark_failed(event)
            return

        # 最多回显 5 张，避免刷屏
        max_show = 5
        show_paths = paths[:max_show]
        yield event.chain_result([Image.fromFileSystem(str(p)) for p in show_paths])
        yield event.plain_result(
            f"📌 当前自拍参考照来源：{source}，共 {len(paths)} 张（已展示 {len(show_paths)} 张）"
        )

    async def _delete_selfie_reference(self, event: AstrMessageEvent):
        if not self._is_selfie_enabled():
            await mark_failed(event)
            return

        store_key = self._get_selfie_ref_store_key(event)
        deleted = await self.refs.delete(store_key)

        webui_paths = self._get_config_selfie_reference_paths()
        if webui_paths:
            logger.info(
                "[自拍参考] 命令保存的参考照已删除，但 WebUI reference_images 仍生效（优先级更高）"
            )

        if deleted:
            await mark_success(event)
        else:
            await mark_failed(event)
