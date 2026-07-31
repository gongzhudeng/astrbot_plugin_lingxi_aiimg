import asyncio
import json
import types
from pathlib import Path

import pytest
from test_main_initialize_request_mode import _load_module


class _Event:
    def __init__(self):
        self.unified_msg_origin = "platform:GroupMessage:group"
        self.message_obj = types.SimpleNamespace(
            message_id="source-message",
            message=[],
        )
        self.message_str = ""
        self._extras = {}
        self._has_send_oper = False
        self.stopped = False
        self.call_llm = False

    def set_extra(self, key, value):
        self._extras[key] = value

    def get_extra(self, key, default=None):
        return self._extras.get(key, default)

    def stop_event(self):
        self.stopped = True

    def should_call_llm(self, value):
        self.call_llm = value

    def get_sender_id(self):
        return "user"

    def get_platform_name(self):
        return "aiocqhttp"

    def get_platform_id(self):
        return "platform"

    def get_self_id(self):
        return "bot"

    def request_llm(self, **kwargs):
        tool_set = kwargs.pop("tool_set", None)
        return types.SimpleNamespace(
            func_tool=tool_set,
            extra_user_content_parts=[],
            system_prompt=kwargs.pop("system_prompt", ""),
            **kwargs,
        )


def _target(mod):
    return mod.TaskDeliveryTarget(
        platform_id="platform",
        platform_name="aiocqhttp",
        message_type="GroupMessage",
        umo="platform:GroupMessage:group",
        session_id="group",
        group_id="group",
        self_id="bot",
        sender_id="user",
        sender_name="Alice",
        source_message_id="source-message",
        conversation_id="conversation",
    )


def _plugin(mod, manager):
    plugin = object.__new__(mod.GiteeAIImagePlugin)
    plugin.config = {
        "features": {
            "draw": {"enabled": True, "llm_tool_enabled": True},
            "batch": {"max_count": 8},
        }
    }
    plugin.background_tasks = manager
    plugin._background_send_gates = {}
    plugin._last_image_by_user = {}
    plugin._last_image_task_meta_cache = {}
    return plugin


def _prepared_single(mod, prompt="take a portrait"):
    return mod.PreparedImageJob(
        mode="draw",
        user_prompt=prompt,
        effective_prompt=prompt,
        backend=None,
        output={"size": None, "resolution": None},
        task_meta={"mode": "text", "effective_prompt": prompt},
    )


async def _create_record(manager, task_id, *, state="queued"):
    scope = manager.scope_hash(
        "platform:GroupMessage:group", "bot", "user", "conversation"
    )
    record, created = await manager.create_task_record(
        {
            "task_id": task_id,
            "task_kind": "single",
            "state": state,
            "scope_hash": scope,
            "request_fingerprint": manager.request_fingerprint(
                scope, task_id, {"prompt": task_id}
            ),
            "platform_id": "platform",
            "platform_name": "aiocqhttp",
            "message_type": "GroupMessage",
            "umo": "platform:GroupMessage:group",
            "session_id": "group",
            "group_id": "group",
            "self_id": "bot",
            "sender_id": "user",
            "sender_name": "Alice",
            "source_message_id": "source-message",
            "conversation_id": "conversation",
            "user_prompt": task_id,
            "effective_prompt": task_id,
        },
        reservation=1,
    )
    assert created
    return record


async def _terminal_record(manager, task_id):
    record = await _create_record(manager, task_id)
    return await manager.transition(
        record["task_id"],
        "failed",
        {"error_code": "test_failure"},
        queue_notification=True,
    )


def test_aiocqhttp_background_entry_does_not_require_adapter_create_event():
    mod, _ = _load_module()
    manager = types.SimpleNamespace(accepting=True)
    plugin = _plugin(mod, manager)
    plugin.context = types.SimpleNamespace(
        get_platform_inst=lambda platform_id: types.SimpleNamespace(
            metadata=types.SimpleNamespace(id=platform_id, name="aiocqhttp"),
            bot=object(),
        ),
        get_config=lambda umo=None: {
            "provider_settings": {"streaming_response": False}
        },
    )

    assert plugin._background_manager_for_event(_Event()) is manager


def test_batch_notification_text_matches_terminal_facts():
    mod, _ = _load_module()
    notify = mod.GiteeAIImagePlugin._background_notification_text

    assert "已发出 2 张，失败 1 张" in notify(
        {
            "task_kind": "batch",
            "state": "partial",
            "requested_count": 3,
            "sent_count": 2,
            "failed_count": 1,
        }
    )
    assert "取消 2 张" in notify(
        {
            "task_kind": "batch",
            "state": "cancelled",
            "requested_count": 3,
            "sent_count": 1,
            "cancelled_count": 2,
        }
    )
    unknown = notify(
        {
            "task_kind": "batch",
            "state": "interrupted",
            "requested_count": 3,
            "sent_count": 1,
            "unknown_count": 2,
        }
    )
    assert "发送状态无法确认" in unknown
    assert "没有自动重发" in unknown


@pytest.mark.asyncio
async def test_single_tool_returns_before_provider_finishes(tmp_path):
    mod, _ = _load_module()
    manager = mod.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    plugin = _plugin(mod, manager)
    event = _Event()
    provider_started = asyncio.Event()
    provider_release = asyncio.Event()
    plugin.context = types.SimpleNamespace(
        get_platform_inst=lambda platform_id: types.SimpleNamespace(
            metadata=types.SimpleNamespace(id=platform_id, name="aiocqhttp"),
            bot=object(),
        ),
        get_config=lambda umo=None: {
            "provider_settings": {"streaming_response": False}
        },
    )

    async def build_target(self, current_event):
        return _target(mod)

    async def prepare(self, current_event, **kwargs):
        return _prepared_single(mod, kwargs["prompt"]), []

    async def blocked_provider(self, current_manager, job):
        provider_started.set()
        await provider_release.wait()
        return Path(tmp_path / "never-sent.png"), dict(job.task_meta)

    async def processing(current_event):
        return None

    plugin._build_background_delivery_target = types.MethodType(build_target, plugin)
    plugin._prepare_background_image_job = types.MethodType(prepare, plugin)
    plugin._execute_prepared_image_job = types.MethodType(blocked_provider, plugin)
    mod.mark_processing = processing

    result = await plugin.aiimg_generate(
        event,
        prompt="take a portrait",
        mode="text",
        backend="auto",
        output="",
    )
    payload = json.loads(result.content[0].text)

    assert payload["status"] == "accepted"
    assert payload["effective_prompt"] == "take a portrait"
    await asyncio.wait_for(provider_started.wait(), timeout=1)
    record = await manager.get_task(payload["task_id"])
    assert record["state"] == "running"
    assert not provider_release.is_set()

    await manager.cancel_task(payload["task_id"], "test cleanup")
    await manager.close()


def test_terminal_tool_filter_does_not_mutate_shared_tool_set():
    mod, _ = _load_module()
    image_tool = types.SimpleNamespace(name="aiimg_generate")
    other_tool = types.SimpleNamespace(name="search")
    shared = types.SimpleNamespace(tools=[image_tool, other_tool])

    filtered = mod.GiteeAIImagePlugin._without_image_generation_tools(shared)

    assert filtered is not shared
    assert [tool.name for tool in filtered.tools] == ["search"]
    assert [tool.name for tool in shared.tools] == ["aiimg_generate", "search"]


@pytest.mark.asyncio
async def test_internal_completion_handler_runs_agent_once_then_stops_pipeline():
    mod, _ = _load_module()
    plugin = _plugin(mod, manager=None)
    event = _Event()
    request = types.SimpleNamespace(
        func_tool=types.SimpleNamespace(
            tools=[
                types.SimpleNamespace(name="aiimg_generate"),
                types.SimpleNamespace(name="search"),
            ]
        )
    )
    event.set_extra(mod._BACKGROUND_COMPLETION_EVENT_EXTRA, True)
    event.set_extra(mod._BACKGROUND_COMPLETION_REQUEST_EXTRA, request)

    yielded = [item async for item in plugin.handle_background_completion_event(event)]

    assert yielded == [request]
    assert [tool.name for tool in request.func_tool.tools] == ["search"]
    assert event.call_llm is True
    assert event.stopped is True


@pytest.mark.asyncio
async def test_terminal_llm_hook_removes_image_tool_without_affecting_normal_request():
    mod, _ = _load_module()
    plugin = _plugin(mod, manager=None)
    image_tool = types.SimpleNamespace(name="aiimg_generate")
    other_tool = types.SimpleNamespace(name="search")

    terminal_event = _Event()
    terminal_event.set_extra(mod._BACKGROUND_COMPLETION_EVENT_EXTRA, True)
    terminal_request = types.SimpleNamespace(
        func_tool=types.SimpleNamespace(tools=[image_tool, other_tool])
    )
    await plugin.enforce_background_completion_tool_contract(
        terminal_event, terminal_request
    )

    normal_event = _Event()
    normal_tools = types.SimpleNamespace(tools=[image_tool, other_tool])
    normal_request = types.SimpleNamespace(func_tool=normal_tools)
    await plugin.enforce_background_completion_tool_contract(
        normal_event, normal_request
    )

    assert [tool.name for tool in terminal_request.func_tool.tools] == ["search"]
    assert normal_request.func_tool is normal_tools
    assert [tool.name for tool in normal_request.func_tool.tools] == [
        "aiimg_generate",
        "search",
    ]


@pytest.mark.asyncio
async def test_same_umo_terminal_notifications_wait_for_send_confirmation(tmp_path):
    mod, _ = _load_module()
    manager = mod.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    plugin = _plugin(mod, manager)
    committed = []
    first_committed = asyncio.Event()
    second_committed = asyncio.Event()

    class _ConversationManager:
        async def get_curr_conversation_id(self, umo):
            return "conversation"

        async def get_conversation(self, umo, conversation_id):
            return types.SimpleNamespace(cid=conversation_id)

    class _Adapter:
        def commit_event(self, event):
            committed.append(event)
            (first_committed if len(committed) == 1 else second_committed).set()

    adapter = _Adapter()
    shared_tools = types.SimpleNamespace(
        tools=[
            types.SimpleNamespace(name="aiimg_generate"),
            types.SimpleNamespace(name="search"),
        ]
    )
    plugin.context = types.SimpleNamespace(
        conversation_manager=_ConversationManager(),
        get_platform_inst=lambda platform_id: adapter,
        get_llm_tool_manager=lambda: types.SimpleNamespace(
            get_full_tool_set=lambda: shared_tools
        ),
    )

    async def rebuild(self, target, message=None):
        event = _Event()
        event.unified_msg_origin = target.umo
        event.message = message or []
        return event

    plugin._rebuild_background_event = types.MethodType(rebuild, plugin)
    first = await _terminal_record(manager, "notify-first")
    second = await _terminal_record(manager, "notify-second")
    target = _target(mod)

    first_delivery = asyncio.create_task(
        plugin._deliver_background_completion(manager, first, target)
    )
    await asyncio.wait_for(first_committed.wait(), timeout=1)
    second_delivery = asyncio.create_task(
        plugin._deliver_background_completion(manager, second, target)
    )
    await asyncio.sleep(0.05)

    assert len(committed) == 1
    assert not second_committed.is_set()
    first_event = committed[0]
    assert first_event.message_obj.message == []
    assert first_event.get_extra(mod._BACKGROUND_COMPLETION_EVENT_EXTRA) is True
    assert first_event.get_extra("provider_request") is None
    first_request = first_event.get_extra(mod._BACKGROUND_COMPLETION_REQUEST_EXTRA)
    assert [tool.name for tool in first_request.func_tool.tools] == ["search"]
    assert [tool.name for tool in shared_tools.tools] == [
        "aiimg_generate",
        "search",
    ]
    committed[0]._has_send_oper = True
    await plugin.confirm_background_task_result(committed[0])
    await asyncio.wait_for(second_committed.wait(), timeout=1)

    assert len(committed) == 2
    committed[1]._has_send_oper = True
    await plugin.confirm_background_task_result(committed[1])
    await asyncio.wait_for(
        asyncio.gather(first_delivery, second_delivery),
        timeout=1,
    )
    assert (await manager.get_task("notify-first"))["notification_state"] == "sent"
    assert (await manager.get_task("notify-second"))["notification_state"] == "sent"
    await manager.close()


@pytest.mark.asyncio
async def test_notification_timeout_releases_same_umo_turn(tmp_path):
    mod, _ = _load_module()
    manager = mod.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    plugin = _plugin(mod, manager)
    plugin.BACKGROUND_NOTIFICATION_WAIT_SECONDS = 0.05
    plugin.BACKGROUND_NOTIFICATION_WATCHDOG_SECONDS = 60
    committed = []
    second_committed = asyncio.Event()

    class _ConversationManager:
        async def get_curr_conversation_id(self, umo):
            return "conversation"

        async def get_conversation(self, umo, conversation_id):
            return types.SimpleNamespace(cid=conversation_id)

    class _Adapter:
        def commit_event(self, event):
            committed.append(event)
            if len(committed) == 2:
                second_committed.set()

    plugin.context = types.SimpleNamespace(
        conversation_manager=_ConversationManager(),
        get_platform_inst=lambda platform_id: _Adapter(),
    )

    async def rebuild(self, target, message=None):
        event = _Event()
        event.unified_msg_origin = target.umo
        return event

    plugin._rebuild_background_event = types.MethodType(rebuild, plugin)
    first = await _terminal_record(manager, "timeout-first")
    second = await _terminal_record(manager, "timeout-second")
    target = _target(mod)

    await asyncio.wait_for(
        asyncio.gather(
            plugin._deliver_background_completion(manager, first, target),
            plugin._deliver_background_completion(manager, second, target),
        ),
        timeout=1,
    )

    assert len(committed) == 2
    assert second_committed.is_set()
    await manager.close()


@pytest.mark.asyncio
async def test_background_image_delivery_rebuilds_umo_and_uses_existing_sender(
    tmp_path,
):
    mod, _ = _load_module()
    manager = mod.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    plugin = _plugin(mod, manager)
    rebuilt = _Event()
    seen = {}

    async def rebuild(self, target, message=None):
        rebuilt.unified_msg_origin = target.umo
        return rebuilt

    async def send_with_fallback(self, event, image_path):
        seen["event"] = event
        seen["path"] = image_path
        return mod.SendImageResult(ok=True, cached_path=image_path)

    plugin._rebuild_background_event = types.MethodType(rebuild, plugin)
    plugin._send_image_with_fallback = types.MethodType(send_with_fallback, plugin)
    image_path = tmp_path / "photo.png"
    image_path.write_bytes(b"image")

    event = await plugin._send_background_image_once(_target(mod), image_path)

    assert event is rebuilt
    assert seen == {"event": rebuilt, "path": image_path}
    assert rebuilt.unified_msg_origin == _target(mod).umo
    await manager.close()


@pytest.mark.asyncio
async def test_switched_conversation_uses_direct_notification_not_agent(tmp_path):
    mod, _ = _load_module()
    manager = mod.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    plugin = _plugin(mod, manager)
    record = await _terminal_record(manager, "switched-conversation")
    direct = []

    class _ConversationManager:
        async def get_curr_conversation_id(self, umo):
            return "new-conversation"

        async def get_conversation(self, umo, conversation_id):
            raise AssertionError("stale conversation must not be loaded")

    plugin.context = types.SimpleNamespace(conversation_manager=_ConversationManager())

    async def direct_notification(
        self, current_manager, claimed, target, *, attempt_id
    ):
        direct.append((claimed["task_id"], target.conversation_id, attempt_id))
        await current_manager.mark_notification(
            claimed["notification_token"], "sent", attempt_id=attempt_id
        )

    plugin._send_deterministic_background_notification = types.MethodType(
        direct_notification, plugin
    )

    await plugin._deliver_background_completion(manager, record, _target(mod))

    assert len(direct) == 1
    assert direct[0][0] == "switched-conversation"
    assert (await manager.get_task("switched-conversation"))[
        "notification_state"
    ] == "sent"
    await manager.close()


@pytest.mark.asyncio
async def test_stop_and_successful_reset_cancel_current_background_scope(tmp_path):
    mod, _ = _load_module()
    manager = mod.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    plugin = _plugin(mod, manager)
    plugin.context = types.SimpleNamespace()
    dispatched = []

    async def dispatch(self, current_manager, record, target):
        dispatched.append((record["task_id"], record["suppress_future_injection"]))

    plugin._dispatch_background_completion = types.MethodType(dispatch, plugin)
    await _create_record(manager, "stop-source")
    stop_event = _Event()
    stop_event.message_str = "/stop"

    await plugin.handle_background_session_commands(stop_event)

    assert stop_event.get_extra("_gitee_bg_stop_cancelled") == 1
    assert (await manager.get_task("stop-source"))["state"] == "cancelled"
    await _create_record(manager, "reset-source")
    reset_event = _Event()
    reset_event.set_extra("_gitee_bg_reset_candidate", "reset")
    reset_event.set_extra("_clean_group_context_session", True)

    await plugin.confirm_background_task_result(reset_event)

    cancelled = await manager.get_task("reset-source")
    assert cancelled["state"] == "cancelled"
    assert cancelled["suppress_future_injection"] is True
    assert dispatched == [("stop-source", False), ("reset-source", True)]
    await manager.close()


@pytest.mark.asyncio
async def test_batch_tool_returns_while_planner_is_still_running(tmp_path):
    mod, _ = _load_module()
    manager = mod.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    plugin = _plugin(mod, manager)
    event = _Event()
    planner_started = asyncio.Event()
    planner_release = asyncio.Event()

    plugin._background_manager_for_event = types.MethodType(
        lambda self, current_event: manager,
        plugin,
    )

    async def build_target(self, current_event):
        return _target(mod)

    async def prepare(self, current_event, **kwargs):
        return _prepared_single(mod, kwargs["prompt"]), []

    async def blocked_planner(self, **kwargs):
        planner_started.set()
        await planner_release.wait()
        return []

    async def processing(current_event):
        return None

    plugin._build_background_delivery_target = types.MethodType(build_target, plugin)
    plugin._prepare_background_image_job = types.MethodType(prepare, plugin)
    plugin._plan_batch_prompt_items = types.MethodType(blocked_planner, plugin)
    mod.mark_processing = processing

    result = await plugin._accept_background_batch(
        event,
        prompt="four different portraits",
        count=4,
        mode="text",
        backend="auto",
        output="",
    )
    payload = json.loads(result.content[0].text)

    assert payload["status"] == "accepted"
    assert payload["state"] == "planning"
    assert payload["requested_count"] == 4
    await asyncio.wait_for(planner_started.wait(), timeout=1)
    record = await manager.get_task(payload["task_id"])
    assert record["state"] == "planning"
    assert not planner_release.is_set()

    await manager.cancel_task(payload["task_id"], "test cleanup")
    await manager.close()
