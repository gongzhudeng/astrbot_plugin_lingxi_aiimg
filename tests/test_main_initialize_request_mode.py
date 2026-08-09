import ast
import asyncio
import importlib.util
import json
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "main_init_request_mode_testpkg"
CORE_PACKAGE_NAME = f"{PACKAGE_NAME}.core"
PROVIDER_REGISTRY_MODULE_NAME = f"{CORE_PACKAGE_NAME}.provider_registry"
MAIN_MODULE_NAME = f"{PACKAGE_NAME}.main"


class _Logger:
    def __init__(self):
        self.warning_messages: list[str] = []

    def debug(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, msg, *args, **kwargs):
        if args:
            try:
                msg = msg % args
            except Exception:
                msg = f"{msg} {' '.join(str(x) for x in args)}"
        self.warning_messages.append(str(msg))
        return None

    def error(self, *args, **kwargs):
        return None


class _StubBackend:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _StubVertexSettings:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _StubService:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _StubRouter(_StubService):
    def get_available_backends(self):
        return []

    def get_preset_names(self):
        return []


class _StubStore(_StubService):
    pass


class _StubVideoManager(_StubService):
    pass


@dataclass
class _StubImageTaskSpec:
    mode: str = ""


@dataclass
class _StubParsedImageRequest:
    spec: object | None = None


@dataclass
class _StubPlannedPromptItem:
    title: str = ""
    prompt: str = ""
    variation_focus: str = ""


class _DummyTextPart:
    def __init__(self, text: str = "", **kwargs):
        self.text = text
        self.kwargs = kwargs

    def mark_as_temp(self):
        return self


class _DummyMessageComponent:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @staticmethod
    def fromFileSystem(path: str):
        return _DummyMessageComponent(path=path)


class _DummyStar:
    def __init__(self, context):
        self.context = context


class _DummyStarTools:
    @staticmethod
    def get_data_dir(name: str):
        return Path("/tmp") / name

    @staticmethod
    async def create_message(**kwargs):
        return types.SimpleNamespace(**kwargs)


class _DummyFilter:
    def __getattr__(self, name):
        def decorator_factory(*args, **kwargs):
            def decorator(func):
                return func

            return decorator

        return decorator_factory


class _SubscriptableType:
    @classmethod
    def __class_getitem__(cls, item):
        return cls


class _McpValue:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _clear_modules():
    for name in list(sys.modules):
        if name.startswith(PACKAGE_NAME) or name in {
            "astrbot",
            "astrbot.api",
            "astrbot.api.event",
            "astrbot.api.message_components",
            "astrbot.api.star",
            "astrbot.core",
            "astrbot.core.utils",
            "astrbot.core.utils.astrbot_path",
            "mcp",
        }:
            sys.modules.pop(name, None)


def _install_stub_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


def _load_module():
    _clear_modules()
    logger = _Logger()

    pkg = types.ModuleType(PACKAGE_NAME)
    pkg.__path__ = [str(ROOT)]
    sys.modules[PACKAGE_NAME] = pkg

    core_pkg = types.ModuleType(CORE_PACKAGE_NAME)
    core_pkg.__path__ = [str(ROOT / "core")]
    sys.modules[CORE_PACKAGE_NAME] = core_pkg

    mcp_mod = types.ModuleType("mcp")
    mcp_mod.types = types.SimpleNamespace(
        CallToolResult=_McpValue,
        TextContent=_McpValue,
        ImageContent=_McpValue,
    )
    sys.modules["mcp"] = mcp_mod

    astrbot_mod = types.ModuleType("astrbot")
    sys.modules["astrbot"] = astrbot_mod

    api_mod = types.ModuleType("astrbot.api")
    api_mod.logger = logger
    sys.modules["astrbot.api"] = api_mod

    _install_stub_module(
        "astrbot.api.event",
        AstrMessageEvent=type("AstrMessageEvent", (), {}),
        filter=_DummyFilter(),
    )
    _install_stub_module(
        "astrbot.api.message_components",
        At=_DummyMessageComponent,
        AtAll=_DummyMessageComponent,
        File=_DummyMessageComponent,
        Image=_DummyMessageComponent,
        Node=_DummyMessageComponent,
        Nodes=_DummyMessageComponent,
        Plain=_DummyMessageComponent,
        Reply=_DummyMessageComponent,
        Video=_DummyMessageComponent,
    )
    _install_stub_module(
        "astrbot.api.star",
        Context=type("Context", (), {}),
        Star=_DummyStar,
        StarTools=_DummyStarTools,
    )
    _install_stub_module(
        "astrbot.core.agent.message",
        TextPart=_DummyTextPart,
    )
    _install_stub_module(
        "astrbot.core.utils.astrbot_path",
        get_astrbot_temp_path=lambda: Path("/tmp"),
    )

    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.gemini_edit",
        GeminiEditBackend=_StubBackend,
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.gemini_flow2api",
        Flow2ApiVideoBackend=_StubBackend,
        GeminiFlow2ApiBackend=_StubBackend,
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.gitee_edit",
        GiteeEditBackend=_StubBackend,
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.gitee_sizes",
        GITEE_SUPPORTED_SIZES=["1024x1024"],
        GITEE_SUPPORTED_RATIOS={"1:1": ["1024x1024"]},
        normalize_size_text=lambda value: str(value or "").strip(),
        resolve_ratio_size=lambda *args, **kwargs: "1024x1024",
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.grok2api_images_backend",
        Grok2ApiImagesBackend=_StubBackend,
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.grok_images_backend",
        GrokImagesBackend=_StubBackend,
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.grok_video_service",
        GrokVideoService=_StubBackend,
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.jimeng_api_backend",
        JimengApiBackend=_StubBackend,
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.openai_chat_image_backend",
        OpenAIChatImageBackend=_StubBackend,
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.openai_compat_backend",
        OpenAICompatBackend=_StubBackend,
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.openai_full_url_backend",
        OpenAIFullURLBackend=_StubBackend,
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.sora2_video_service",
        Sora2VideoService=_StubBackend,
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.vertex_ai_anonymous_backend",
        VertexAIAnonymousBackend=_StubBackend,
        VertexAIAnonymousSettings=_StubVertexSettings,
    )

    provider_registry_spec = importlib.util.spec_from_file_location(
        PROVIDER_REGISTRY_MODULE_NAME,
        ROOT / "core" / "provider_registry.py",
    )
    provider_registry_module = importlib.util.module_from_spec(provider_registry_spec)
    sys.modules[PROVIDER_REGISTRY_MODULE_NAME] = provider_registry_module
    assert provider_registry_spec and provider_registry_spec.loader
    provider_registry_spec.loader.exec_module(provider_registry_module)

    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.batch_executor",
        BatchRunResult=type("BatchRunResult", (_SubscriptableType,), {}),
        run_batch=lambda *args, **kwargs: None,
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.debouncer",
        Debouncer=_StubService,
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.draw_service",
        ImageDrawService=_StubService,
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.edit_router",
        EditRouter=_StubRouter,
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.emoji_feedback",
        mark_failed=lambda *args, **kwargs: None,
        mark_processing=lambda *args, **kwargs: None,
        mark_success=lambda *args, **kwargs: None,
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.image_task_parser",
        ImageTaskSpec=_StubImageTaskSpec,
        ParsedImageRequest=_StubParsedImageRequest,
        parse_image_request=lambda *args, **kwargs: _StubParsedImageRequest(),
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.llm_batch_planner",
        PlannedPromptItem=_StubPlannedPromptItem,
        build_batch_planning_prompt=lambda *args, **kwargs: "",
        parse_planned_prompt_items=lambda *args, **kwargs: [],
        validate_planned_prompt_items=lambda *args, **kwargs: [],
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.image_format",
        decode_base64_image_payload=lambda *args, **kwargs: b"",
        guess_image_mime_and_ext=lambda *args, **kwargs: ("image/png", ".png"),
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.image_manager",
        ImageManager=_StubService,
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.nanobanana",
        NanoBananaService=_StubService,
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.ref_store",
        ReferenceStore=_StubStore,
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.utils",
        close_session=lambda *args, **kwargs: None,
        collect_at_user_ids=lambda *args, **kwargs: [],
        get_images_from_event=lambda *args, **kwargs: [],
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.video_manager",
        VideoManager=_StubVideoManager,
    )

    spec = importlib.util.spec_from_file_location(
        MAIN_MODULE_NAME,
        ROOT / "main.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[MAIN_MODULE_NAME] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module, logger


class MainInitializeRequestModeTests(unittest.IsolatedAsyncioTestCase):
    async def test_initialize_logs_fallback_warning_and_builds_consistent_backend(self):
        mod, logger = _load_module()
        plugin = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(),
            config={
                "providers": [
                    {
                        "id": "chat-provider",
                        "__template_key": "openai_chat",
                        "base_url": "https://api.example.com/v1",
                        "api_keys": ["test-key"],
                        "model": "gpt-image",
                        "generate_request_mode": "bogus",
                        "enable_stream_generate": False,
                    }
                ]
            },
        )
        plugin._patch_tool_image_cache_runtime = lambda: None
        plugin._register_preset_commands = lambda: None

        await plugin.initialize()

        backend = plugin.registry.get_backend("chat-provider")

        self.assertEqual(backend.kwargs["generate_request_mode"], "non_stream")
        self.assertFalse(backend.kwargs["enable_stream_generate"])
        self.assertTrue(
            any(
                "invalid generate_request_mode: bogus; runtime will fallback to non_stream via enable_stream_generate=false"
                in msg
                for msg in logger.warning_messages
            )
        )

    async def test_background_owner_conflict_retries_until_takeover(self):
        mod, logger = _load_module()

        class RetryManager:
            def __init__(self, *args, **kwargs):
                self.start_calls = 0
                self.started = False
                self.closed = False

            async def start(self):
                self.start_calls += 1
                if self.start_calls == 1:
                    raise mod.BackgroundTaskOwnerError("owner busy")
                self.started = True
                return []

            async def close(self):
                self.closed = True
                self.started = False

            @staticmethod
            def sanitize_error(exc):
                return str(exc)

        mod.BackgroundImageTaskManager = RetryManager
        plugin = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(),
            config={"features": {"background_llm_image": {"enabled": True}}},
        )
        plugin.BACKGROUND_OWNER_RETRY_SECONDS = 0
        plugin._patch_tool_image_cache_runtime = lambda: None
        plugin._register_preset_commands = lambda: None

        await plugin.initialize()
        retry_task = plugin._background_start_task

        self.assertIsNone(plugin.background_tasks)
        self.assertIsNotNone(retry_task)
        await asyncio.wait_for(retry_task, timeout=1)
        self.assertIsInstance(plugin.background_tasks, RetryManager)
        self.assertEqual(plugin.background_tasks.start_calls, 2)
        self.assertTrue(
            any("takeover will retry" in msg for msg in logger.warning_messages)
        )
        await plugin.background_tasks.close()
        plugin.background_tasks = None

    async def test_selfie_regex_fallback_handles_direct_slash_command(self):
        mod, _ = _load_module()
        plugin = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(),
            config={},
        )
        plugin._is_selfie_enabled = lambda: True

        calls = []

        async def fake_do_selfie(event, prompt, backend=None):
            calls.append((event, prompt, backend))

        plugin._do_selfie = fake_do_selfie

        plain = mod.Plain()
        plain.text = "/自拍 窗边自然光"

        class DummyEvent:
            message_str = "/自拍 窗边自然光"

            def __init__(self):
                self.call_llm = False
                self.stopped = False

            def get_messages(self):
                return [plain]

            def should_call_llm(self, value):
                self.call_llm = value

            def stop_event(self):
                self.stopped = True

        event = DummyEvent()

        await plugin.selfie_regex_fallback(event)

        self.assertEqual(calls, [(event, "窗边自然光", None)])
        self.assertTrue(event.call_llm)
        self.assertTrue(event.stopped)

    async def test_selfie_regex_fallback_handles_wake_stripped_command(self):
        mod, _ = _load_module()
        plugin = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(),
            config={},
        )
        plugin._is_selfie_enabled = lambda: True

        calls = []

        async def fake_do_selfie(event, prompt, backend=None):
            calls.append((event, prompt, backend))

        plugin._do_selfie = fake_do_selfie

        class DummyEvent:
            message_str = "自拍 窗边自然光"
            is_at_or_wake_command = True

            def __init__(self):
                self.call_llm = False
                self.stopped = False

            def get_extra(self, key, default=None):
                return default

            def should_call_llm(self, value):
                self.call_llm = value

            def stop_event(self):
                self.stopped = True

        event = DummyEvent()

        await plugin.selfie_regex_fallback(event)

        self.assertEqual(calls, [(event, "窗边自然光", None)])
        self.assertTrue(event.call_llm)
        self.assertTrue(event.stopped)

    async def test_selfie_regex_fallback_skips_when_command_handler_active(self):
        mod, _ = _load_module()
        plugin = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(),
            config={},
        )
        plugin._is_selfie_enabled = lambda: True

        calls = []

        async def fake_do_selfie(event, prompt, backend=None):
            calls.append((event, prompt, backend))

        plugin._do_selfie = fake_do_selfie

        class DummyHandler:
            handler_name = "selfie_command"

        class DummyEvent:
            message_str = "自拍 窗边自然光"
            is_at_or_wake_command = True

            def get_extra(self, key, default=None):
                if key == "activated_handlers":
                    return [DummyHandler()]
                return default

        await plugin.selfie_regex_fallback(DummyEvent())

        self.assertEqual(calls, [])

    async def test_selfie_regex_fallback_ignores_unwoken_bare_text(self):
        mod, _ = _load_module()
        plugin = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(),
            config={},
        )
        plugin._is_selfie_enabled = lambda: True

        calls = []

        async def fake_do_selfie(event, prompt, backend=None):
            calls.append((event, prompt, backend))

        plugin._do_selfie = fake_do_selfie

        class DummyEvent:
            message_str = "自拍 窗边自然光"
            is_at_or_wake_command = False

            def get_extra(self, key, default=None):
                return default

        await plugin.selfie_regex_fallback(DummyEvent())

        self.assertEqual(calls, [])

    async def test_selfie_reference_regex_fallback_handles_image_prefixed_command(self):
        mod, _ = _load_module()
        plugin = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(),
            config={},
        )
        plugin._is_selfie_enabled = lambda: True

        calls = []

        async def fake_set_selfie_reference(event):
            calls.append(event)

        plugin._set_selfie_reference = fake_set_selfie_reference

        class DummyEvent:
            message_str = "图片 /自拍参考 设置"
            is_at_or_wake_command = False

            def __init__(self):
                self.call_llm = False
                self.stopped = False

            def get_extra(self, key, default=None):
                return default

            def should_call_llm(self, value):
                self.call_llm = value

            def stop_event(self):
                self.stopped = True

        event = DummyEvent()
        yielded = []

        async for result in plugin.selfie_reference_regex_fallback(event):
            yielded.append(result)

        self.assertEqual(calls, [event])
        self.assertEqual(yielded, [])
        self.assertTrue(event.call_llm)
        self.assertTrue(event.stopped)

    async def test_aiimg_generate_defaults_to_exactly_one_single_task(self):
        mod, _ = _load_module()
        plugin = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(),
            config={"features": {"batch": {"max_count": 8}}},
        )
        calls = []

        async def fake_single(event, **kwargs):
            calls.append(("single", event, kwargs))
            return "single-result"

        async def fake_batch(event, **kwargs):
            calls.append(("batch", event, kwargs))
            return "batch-result"

        plugin._aiimg_generate_single = fake_single
        plugin._aiimg_batch_generate = fake_batch
        event = object()

        result = await plugin.aiimg_generate(event, "画一个苹果", mode="text")

        self.assertEqual(result, "single-result")
        self.assertEqual(
            calls,
            [
                (
                    "single",
                    event,
                    {
                        "prompt": "画一个苹果",
                        "mode": "text",
                        "backend": "auto",
                        "output": "",
                    },
                )
            ],
        )

    async def test_aiimg_generate_routes_explicit_count_to_one_batch(self):
        mod, _ = _load_module()
        plugin = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(),
            config={"features": {"batch": {"max_count": 8}}},
        )
        calls = []

        async def fake_single(event, **kwargs):
            calls.append(("single", event, kwargs))
            return "single-result"

        async def fake_batch(event, prompt, **kwargs):
            calls.append(("batch", event, prompt, kwargs))
            return "batch-result"

        plugin._aiimg_generate_single = fake_single
        plugin._aiimg_batch_generate = fake_batch
        event = object()

        result = await plugin.aiimg_generate(
            event,
            "同一主题，不同构图",
            mode="selfie_ref",
            backend="provider-a",
            output="4K",
            count=3,
        )

        self.assertEqual(result, "batch-result")
        self.assertEqual(
            calls,
            [
                (
                    "batch",
                    event,
                    "同一主题，不同构图",
                    {
                        "count": 3,
                        "mode": "selfie_ref",
                        "backend": "provider-a",
                        "output": "4K",
                    },
                )
            ],
        )

    def test_aiimg_count_validation_rejects_invalid_or_excess_values(self):
        mod, _ = _load_module()
        plugin = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(),
            config={"features": {"batch": {"max_count": 4}}},
        )

        self.assertEqual(plugin._validate_llm_image_count(1), 1)
        self.assertEqual(plugin._validate_llm_image_count("4"), 4)
        for value in (0, 5, True, 1.5, "2.0", "invalid"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                plugin._validate_llm_image_count(value)

    def test_llm_registers_only_one_image_tool(self):
        tree = ast.parse((ROOT / "main.py").read_text(encoding="utf-8"))
        registered_tools: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                func = decorator.func
                if not (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "filter"
                    and func.attr == "llm_tool"
                ):
                    continue
                registered_tools.append(node.name)

        image_tools = [
            name
            for name in registered_tools
            if name
            in {
                "gitee_draw_image",
                "gitee_edit_image",
                "aiimg_generate",
                "aiimg_batch_generate",
            }
        ]
        self.assertEqual(image_tools, ["aiimg_generate"])

    def test_image_history_note_contains_only_original_prompt_and_normalized_mode(self):
        mod, _ = _load_module()
        plugin = object.__new__(mod.GiteeAIImagePlugin)

        for raw_mode, mode in (
            ("text", "text"),
            ("edit", "edit"),
            ("selfie", "selfie_ref"),
        ):
            note = plugin._build_image_history_note(
                prompt="user prompt",
                mode=raw_mode,
            )
            self.assertIn(f"Mode: {mode}", note)
            self.assertIn("Prompt: user prompt", note)
            self.assertNotIn("effective_prompt", note)
            self.assertNotIn("internal rule", note)
            self.assertNotIn("prompt prefix", note)

    async def test_image_history_persists_one_assistant_note_and_is_idempotent(self):
        mod, _ = _load_module()
        plugin = object.__new__(mod.GiteeAIImagePlugin)
        conversation = types.SimpleNamespace(cid="conversation", history="[]")

        class ConversationManager:
            async def get_curr_conversation_id(self, origin):
                return "conversation"

            async def get_conversation(self, origin, conversation_id):
                return conversation

            async def update_conversation(self, origin, conversation_id, **changes):
                conversation.history = json.dumps(
                    changes["history"], ensure_ascii=False
                )

        plugin.context = types.SimpleNamespace(
            conversation_manager=ConversationManager()
        )
        event = types.SimpleNamespace(
            unified_msg_origin="platform:user",
            get_extra=lambda key, default=None: default,
        )

        await plugin._append_image_history_note(
            event,
            prompt="original prompt",
            mode="selfie_ref",
            dedupe_key="task-1",
        )
        await plugin._append_image_history_note(
            event,
            prompt="original prompt",
            mode="selfie_ref",
            dedupe_key="task-1",
        )

        history = json.loads(conversation.history)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["role"], "assistant")
        self.assertIn("Mode: selfie_ref", history[0]["content"])
        self.assertIn("Prompt: original prompt", history[0]["content"])
        self.assertNotIn("effective_prompt", history[0]["content"])
        self.assertNotIn("reference_source", history[0]["content"])
        self.assertNotIn("media_id", history[0]["content"])

    def test_metadata_version_is_current(self):
        metadata = (ROOT / "metadata.yaml").read_text(encoding="utf-8")
        self.assertIn("version: 1.2.3", metadata)

    def test_selfie_prompt_describes_fixed_and_user_reference_ranges(self):
        mod, _ = _load_module()
        plugin = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(),
            config={"features": {"selfie": {"prompt_prefix": "固定图逐张规则"}}},
        )

        prompt = plugin._build_selfie_prompt(
            "参考衣服和姿势",
            reference_count=4,
            extra_reference_count=2,
        )

        self.assertIn("固定图逐张规则", prompt)
        self.assertIn("第 1-4 张是固定人物参考图", prompt)
        self.assertIn("第 5-6 张是本次用户附带或引用的参考图", prompt)
        self.assertIn("用户参考图不是待修改原图", prompt)
        self.assertIn("用户要求：参考衣服和姿势", prompt)

    def test_selfie_prompt_without_user_reference_keeps_fixed_range(self):
        mod, _ = _load_module()
        plugin = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(),
            config={"features": {"selfie": {"prompt_prefix": "固定图规则"}}},
        )

        prompt = plugin._build_selfie_prompt(
            "夜间自拍",
            reference_count=4,
            extra_reference_count=0,
        )

        self.assertIn("第 1-4 张均为固定人物参考图", prompt)
        self.assertNotIn("本次用户附带或引用", prompt)


if __name__ == "__main__":
    unittest.main()
