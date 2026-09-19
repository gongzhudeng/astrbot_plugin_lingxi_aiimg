"""媒体记录保留上限（media_history_note_limit）行为测试。"""

import json
import unittest
from types import SimpleNamespace

from test_main_initialize_request_mode import _load_module


def _media_note(tag: str, marker: str) -> dict:
    return {
        "role": "assistant",
        "content": (
            f"<{tag}_history_record>\n"
            f"事实：测试记录 {marker}。\n"
            f"</{tag}_history_record>"
        ),
    }


class _Event:
    unified_msg_origin = "aiocqhttp:FriendMessage:1"


class MediaHistoryNoteLimitTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _plugin(config):
        mod, _ = _load_module()
        plugin = mod.GiteeAIImagePlugin(
            context=SimpleNamespace(),
            config=config,
        )
        return mod, plugin

    @staticmethod
    def _conversation_with(history, updates):
        conversation = SimpleNamespace(
            cid="c1",
            history=json.dumps(history, ensure_ascii=False),
        )

        async def fake_update(umo, cid, *, history=None):
            updates.append(list(history or []))

        return conversation, fake_update

    def test_limit_default_and_clamp(self):
        cases = [
            ({}, 2),
            ({"media_history_note_limit": 0}, 0),
            ({"media_history_note_limit": 99}, 10),
            ({"media_history_note_limit": -3}, 0),
            ({"media_history_note_limit": "bogus"}, 2),
            ({"media_history_note_limit": 5}, 5),
        ]
        for config, expected in cases:
            _, plugin = self._plugin(config)
            self.assertEqual(plugin._media_history_note_limit(), expected)

    def test_is_media_history_note(self):
        mod, plugin = self._plugin({})
        self.assertTrue(
            plugin._is_media_history_note(_media_note("image", "a")["content"])
        )
        self.assertTrue(
            plugin._is_media_history_note(_media_note("video", "b")["content"])
        )
        self.assertFalse(
            plugin._is_media_history_note(
                "The last video generation task has completed and the video was already sent."
            )
        )
        self.assertFalse(
            plugin._is_media_history_note(mod.GiteeAIImagePlugin._MEDIA_HISTORY_PRUNED_NOTE)
        )

    def test_prune_keeps_latest_and_replaces_oldest(self):
        mod, plugin = self._plugin({"media_history_note_limit": 2})
        history = [
            {"role": "user", "content": "hello"},
            _media_note("image", "old-1"),
            _media_note("video", "old-2"),
            _media_note("image", "new-1"),
            _media_note("video", "new-2"),
            {"role": "assistant", "content": "normal reply"},
        ]
        replaced = plugin._prune_media_history_notes(history)
        self.assertEqual(replaced, 2)
        self.assertEqual(history[1]["content"], mod.GiteeAIImagePlugin._MEDIA_HISTORY_PRUNED_NOTE)
        self.assertEqual(history[2]["content"], mod.GiteeAIImagePlugin._MEDIA_HISTORY_PRUNED_NOTE)
        self.assertIn("<image_history_record>", history[3]["content"])
        self.assertIn("<video_history_record>", history[4]["content"])
        self.assertEqual(history[0], {"role": "user", "content": "hello"})
        self.assertEqual(history[5]["content"], "normal reply")
        # 轻量标记不再计入媒体记录，重复修剪无变化
        self.assertEqual(plugin._prune_media_history_notes(history), 0)
        self.assertEqual(len(history), 6)

    def test_prune_zero_replaces_all(self):
        _, plugin = self._plugin({"media_history_note_limit": 0})
        history = [
            _media_note("image", "a"),
            {"role": "user", "content": "hi"},
            _media_note("video", "b"),
        ]
        replaced = plugin._prune_media_history_notes(history)
        self.assertEqual(replaced, 2)
        self.assertNotIn("_history_record", history[0]["content"])
        self.assertNotIn("_history_record", history[2]["content"])
        self.assertEqual(history[1], {"role": "user", "content": "hi"})

    def test_prune_noop_when_under_limit(self):
        _, plugin = self._plugin({"media_history_note_limit": 5})
        history = [_media_note("image", "a"), _media_note("video", "b")]
        self.assertEqual(plugin._prune_media_history_notes(history), 0)
        self.assertEqual(len(history), 2)

    async def test_append_media_note_positive_mode_prunes_in_same_write(self):
        _, plugin = self._plugin({"media_history_note_limit": 1})
        history = [
            {"role": "user", "content": "拍一个"},
            _media_note("image", "old"),
        ]
        updates = []
        conversation, fake_update = self._conversation_with(history, updates)
        plugin.context = SimpleNamespace(
            conversation_manager=SimpleNamespace(update_conversation=fake_update),
        )

        async def fake_resolve(event):
            return conversation

        plugin._resolve_plugin_conversation = fake_resolve

        note = plugin._build_video_history_note(prompt="一只猫")
        await plugin._append_media_history_note(_Event(), note)

        # append 与修剪合并在同一次写回
        self.assertEqual(len(updates), 1)
        final = updates[0]
        self.assertEqual(len(final), 3)
        self.assertEqual(
            final[1]["content"],
            plugin.__class__._MEDIA_HISTORY_PRUNED_NOTE,
        )
        self.assertIn("<video_history_record>", final[2]["content"])
        self.assertIn(plugin.__class__._MEDIA_HISTORY_PRUNED_NOTE, conversation.history)

    async def test_append_media_note_zero_mode_skips_new_and_prunes(self):
        _, plugin = self._plugin({"media_history_note_limit": 0})
        history = [
            {"role": "user", "content": "拍一个"},
            _media_note("image", "old"),
        ]
        updates = []
        conversation, fake_update = self._conversation_with(history, updates)
        plugin.context = SimpleNamespace(
            conversation_manager=SimpleNamespace(update_conversation=fake_update),
        )

        async def fake_resolve(event):
            return conversation

        plugin._resolve_plugin_conversation = fake_resolve

        note = plugin._build_video_history_note(prompt="一只猫")
        await plugin._append_media_history_note(_Event(), note)

        # 只有一次清理写回，新记录没有 append
        self.assertEqual(len(updates), 1)
        pruned = updates[0]
        self.assertEqual(len(pruned), 2)
        self.assertEqual(
            pruned[1]["content"],
            plugin.__class__._MEDIA_HISTORY_PRUNED_NOTE,
        )
        self.assertNotIn("<video_history_record>", conversation.history)
        self.assertNotIn("<image_history_record>", conversation.history)

    async def test_english_tool_note_not_affected_by_media_prune(self):
        _, plugin = self._plugin({"media_history_note_limit": 1})
        english_note = (
            "The last video generation task has completed and the video was "
            "already sent to the user. Do not continue or resubmit this task."
        )
        history = [
            {"role": "user", "content": "拍一个"},
            {"role": "assistant", "content": english_note},
        ]
        updates = []
        conversation, fake_update = self._conversation_with(history, updates)
        plugin.context = SimpleNamespace(
            conversation_manager=SimpleNamespace(update_conversation=fake_update),
        )

        async def fake_resolve(event):
            return conversation

        plugin._resolve_plugin_conversation = fake_resolve

        # 英文工具提示走公共入口（非媒体路径），不应触发修剪、不应被替换
        await plugin._append_plugin_conversation_note(_Event(), english_note + " dedupe-test")

        self.assertEqual(len(updates), 1)
        final = updates[0]
        self.assertEqual(final[1]["content"], english_note)
        self.assertEqual(final[2]["content"], english_note + " dedupe-test")


if __name__ == "__main__":
    unittest.main()
