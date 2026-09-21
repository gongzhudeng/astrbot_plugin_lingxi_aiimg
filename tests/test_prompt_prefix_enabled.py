"""提示词前缀开关（prompt_prefix_enabled）行为测试。"""

import types
import unittest

from test_main_initialize_request_mode import _load_module


def _plugin(config):
    mod, _ = _load_module()
    plugin = mod.GiteeAIImagePlugin(
        context=types.SimpleNamespace(),
        config=config,
    )
    return mod, plugin


class PromptPrefixEnabledTests(unittest.IsolatedAsyncioTestCase):
    def test_feature_prompt_prefix_switch(self):
        _, plugin = _plugin({})
        conf_on = {"prompt_prefix": "风格模板 {today_outfit}", "prompt_prefix_enabled": True}
        self.assertEqual(
            plugin._feature_prompt_prefix(conf_on), "风格模板 {today_outfit}"
        )
        conf_off = {"prompt_prefix": "风格模板", "prompt_prefix_enabled": False}
        self.assertEqual(plugin._feature_prompt_prefix(conf_off), "")
        conf_default = {"prompt_prefix": "风格模板"}
        self.assertEqual(plugin._feature_prompt_prefix(conf_default), "风格模板")
        self.assertEqual(plugin._feature_prompt_prefix({}), "")

    def test_video_prompt_prefix_disabled(self):
        _, plugin = _plugin(
            {
                "features": {
                    "video": {
                        "prompt_prefix": "自定义视频前缀",
                        "prompt_prefix_enabled": False,
                    }
                }
            }
        )
        out = plugin._build_video_prompt("一只猫在窗台")
        self.assertEqual(out, "一只猫在窗台")
        self.assertNotIn("自定义视频前缀", out)
        self.assertNotIn("今日外显穿搭", out)
        self.assertNotIn("当前时间光线", out)

    def test_video_prompt_prefix_enabled_uses_default(self):
        _, plugin = _plugin({"features": {"video": {}}})
        out = plugin._build_video_prompt("一只猫在窗台")
        self.assertIn("真实生活感随手拍视频", out)
        self.assertIn("一只猫在窗台", out)

    def test_selfie_prompt_prefix_disabled(self):
        _, plugin = _plugin(
            {
                "features": {
                    "selfie": {
                        "prompt_prefix": "自定义自拍前缀",
                        "prompt_prefix_enabled": False,
                    }
                }
            }
        )
        out = plugin._build_selfie_prompt(
            "在海边", reference_count=1, extra_reference_count=0
        )
        self.assertNotIn("自定义自拍前缀", out)
        self.assertNotIn("今日外显穿搭", out)
        self.assertNotIn("当前时间光线", out)
        # 参考图顺序说明是功能性内容，不受开关影响
        self.assertIn("均为固定人物参考图", out)
        self.assertIn("在海边", out)

    def test_selfie_prompt_prefix_enabled_uses_default(self):
        _, plugin = _plugin({"features": {"selfie": {}}})
        out = plugin._build_selfie_prompt(
            "在海边", reference_count=1, extra_reference_count=0
        )
        self.assertIn("请根据参考图拍摄一张新的照片", out)
        self.assertIn("在海边", out)


if __name__ == "__main__":
    unittest.main()
