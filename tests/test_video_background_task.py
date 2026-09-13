"""Background video task pipeline tests (task_kind="video")."""

import asyncio
import types
from pathlib import Path

import pytest

from test_background_pipeline_event import _Event, _plugin, _target
from test_main_initialize_request_mode import _load_module


async def _async_none(*args, **kwargs):
    return None


async def _no_images(event, **kwargs):
    return []


async def _bad_images(event, **kwargs):
    return [object()]


async def _async_true(*args, **kwargs):
    return True


def _video_plugin(mod, manager, tmp_path):
    plugin = _plugin(mod, manager)
    plugin.config = {
        "features": {
            "video": {
                "enabled": True,
                "llm_tool_enabled": True,
                "presets": {},
                "chain": ["comfyui_vid"],
            },
        }
    }
    plugin.context = types.SimpleNamespace(
        get_platform_inst=lambda platform_id: types.SimpleNamespace(
            metadata=types.SimpleNamespace(id=platform_id, name="aiocqhttp"),
            bot=object(),
        ),
        get_config=lambda umo=None: {
            "provider_settings": {"streaming_response": False}
        },
    )
    plugin._video_tasks = set()
    plugin.debouncer = types.SimpleNamespace(hit=lambda request_id: False)
    plugin._debounce_key = lambda event, kind, user_id: "debounce-key"
    plugin._video_begin = types.MethodType(_async_true, plugin)
    plugin._video_end = types.MethodType(_async_none, plugin)
    plugin._wait_for_background_ack = types.MethodType(_async_none, plugin)

    async def build_target(_self, current_event):
        return _target(mod)

    plugin._build_background_delivery_target = types.MethodType(
        build_target, plugin
    )
    return plugin


def _video_state_error(manager):
    """从打桩加载的 manager 模块里取 BackgroundTaskStateError（异常类须同源）。"""
    return type(manager).transition.__globals__["BackgroundTaskStateError"]


def _stub_video_generation(plugin, *, url=None, error=None):
    calls = []

    async def generate(_self, prompt, provider_id=None, image_bytes=None):
        calls.append({"prompt": prompt, "provider_id": provider_id})
        if error is not None:
            raise error
        return url, "comfyui_vid"

    plugin._generate_video_url_via_chain = types.MethodType(generate, plugin)
    return calls


def _stub_video_send(plugin):
    sent = []

    async def send(_self, target, video_url):
        sent.append((target.umo, video_url))

    plugin._send_background_video_once = types.MethodType(send, plugin)
    return sent


def _stub_dispatch(plugin):
    dispatched = []

    async def dispatch(_self, manager, record, target):
        dispatched.append((record["task_id"], record["state"]))

    plugin._dispatch_background_completion = types.MethodType(dispatch, plugin)
    return dispatched


async def _wait_terminal(manager, task_id, timeout=5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        record = await manager.get_task(task_id)
        if record is not None and record.get("state") in {
            "completed",
            "failed",
            "cancelled",
            "interrupted",
        }:
            return record
        await asyncio.sleep(0.05)
    raise AssertionError("video worker did not reach a terminal state in time")


def test_video_notification_text_matches_terminal_facts():
    mod, _ = _load_module()
    notify = mod.GiteeAIImagePlugin._background_notification_text

    assert "视频拍好了" in notify({"task_kind": "video", "state": "completed"})
    assert "停下来了" in notify({"task_kind": "video", "state": "cancelled"})
    interrupted = notify({"task_kind": "video", "state": "interrupted"})
    assert "没有自动重发" in interrupted
    assert "没能生成成功" in notify({"task_kind": "video", "state": "failed"})


def test_video_terminal_completed_requires_confirmed_delivery(tmp_path):
    mod, _ = _load_module()
    manager = mod.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    state_error = _video_state_error(manager)

    valid = {
        "video_generated": True,
        "video_sent": True,
        "delivery_state": "confirmed",
    }
    manager._validate_terminal_record(dict(valid, task_kind="video"), "completed")

    with pytest.raises(state_error):
        manager._validate_terminal_record(
            dict(valid, video_sent=False, task_kind="video"),
            "completed",
        )


@pytest.mark.asyncio
async def test_video_tool_accepts_background_record_and_returns_none(tmp_path):
    mod, _ = _load_module()
    manager = mod.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    plugin = _video_plugin(mod, manager, tmp_path)
    release = asyncio.Event()

    async def blocked_generate(_self, prompt, provider_id=None, image_bytes=None):
        await release.wait()
        return "http://127.0.0.1:8200/files/x.mp4", "comfyui_vid"

    plugin._generate_video_url_via_chain = types.MethodType(
        blocked_generate, plugin
    )
    mod.get_images_from_event = _no_images
    mod.mark_processing = _async_none

    event = _Event()
    result = await plugin.grok_generate_video(event, prompt="a cat video")

    assert result is None  # 与拍照工具一致：不返回可复述文本
    task_id = event.get_extra("_gitee_bg_ack_task_id")
    assert task_id
    record = await manager.get_task(task_id)
    assert record["task_kind"] == "video"
    assert record["state"] in {"queued", "running"}
    assert record["user_prompt"] == "a cat video"

    await manager.cancel_task(task_id, "test cleanup")
    release.set()
    await manager.close(grace_seconds=1)


@pytest.mark.asyncio
async def test_video_worker_completes_and_dispatches(tmp_path):
    mod, _ = _load_module()
    manager = mod.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    plugin = _video_plugin(mod, manager, tmp_path)
    calls = _stub_video_generation(
        plugin, url="http://127.0.0.1:8200/files/ok.mp4"
    )
    sent = _stub_video_send(plugin)
    dispatched = _stub_dispatch(plugin)
    mod.get_images_from_event = _no_images
    mod.mark_processing = _async_none

    event = _Event()
    await plugin.grok_generate_video(event, prompt="coffee shop reading")
    task_id = event.get_extra("_gitee_bg_ack_task_id")

    record = await _wait_terminal(manager, task_id)
    assert record["state"] == "completed"
    assert record["video_generated"] is True
    assert record["video_sent"] is True
    assert record["delivery_state"] == "confirmed"
    assert calls and calls[0]["prompt"] == "coffee shop reading"
    assert sent and sent[0][1] == "http://127.0.0.1:8200/files/ok.mp4"
    assert dispatched == [(task_id, "completed")]

    await manager.close(grace_seconds=1)


@pytest.mark.asyncio
async def test_video_worker_failure_dispatches_failed(tmp_path):
    mod, _ = _load_module()
    manager = mod.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    plugin = _video_plugin(mod, manager, tmp_path)
    _stub_video_generation(plugin, error=RuntimeError("Server disconnected"))
    sent = _stub_video_send(plugin)
    dispatched = _stub_dispatch(plugin)
    mod.get_images_from_event = _no_images
    mod.mark_processing = _async_none

    event = _Event()
    await plugin.grok_generate_video(event, prompt="a dancing cat")
    task_id = event.get_extra("_gitee_bg_ack_task_id")

    record = await _wait_terminal(manager, task_id)
    assert record["state"] == "failed"
    assert record["video_sent"] is False
    assert "Server disconnected" in str(record.get("error_message"))
    assert sent == []  # 失败时不发送视频
    assert dispatched == [(task_id, "failed")]

    await manager.close(grace_seconds=1)


@pytest.mark.asyncio
async def test_video_tool_with_unreadable_image_fails_fast(tmp_path):
    mod, _ = _load_module()
    manager = mod.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    plugin = _video_plugin(mod, manager, tmp_path)
    _stub_video_generation(plugin, url="http://unused")
    dispatched = _stub_dispatch(plugin)
    mod.get_images_from_event = _bad_images
    mod.mark_processing = _async_none
    mod.mark_failed = _async_none
    mod.decode_base64_image_payload = lambda payload: (_ for _ in ()).throw(
        ValueError("bad image")
    )

    event = _Event()
    event.is_private_chat = lambda: False  # 非私聊 → mark_failed（已打桩）
    result = await plugin.grok_generate_video(event, prompt="with image")

    # 图片读不出来：直接失败，不创建后台任务
    assert result is not None
    assert event.get_extra("_gitee_bg_ack_task_id") is None
    assert dispatched == []

    await manager.close(grace_seconds=1)
