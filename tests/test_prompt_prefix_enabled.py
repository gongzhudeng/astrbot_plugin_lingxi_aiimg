"""提示词前缀开关（prompt_prefix_enabled）与拼接规则行为测试。"""

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


LIGHTING_ALL_DAY = ["00:00-24:00=测试光线"]


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

    def test_feature_prompt_prefix_translates_literal_newline(self):
        _, plugin = _plugin({})
        conf = {"prompt_prefix": "A\\nB", "prompt_prefix_enabled": True}
        self.assertEqual(plugin._feature_prompt_prefix(conf), "A\nB")
        # 结尾的真实换行仍会被 strip 掉，字面 \n 不受影响
        conf_mixed = {"prompt_prefix": "A\\nB\n", "prompt_prefix_enabled": True}
        self.assertEqual(plugin._feature_prompt_prefix(conf_mixed), "A\nB")

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
        # 兜底模板结尾自带换行：正文独立成行
        self.assertIn("\n一只猫在窗台", out)

    def test_video_prompt_direct_concat_without_literal_newline(self):
        _, plugin = _plugin(
            {
                "features": {
                    "selfie": {"lighting_rules": LIGHTING_ALL_DAY},
                    "video": {
                        "prompt_prefix": "前缀{lighting}",
                        "prompt_prefix_enabled": True,
                    },
                }
            }
        )
        out = plugin._build_video_prompt("一只猫在窗台")
        self.assertEqual(out, "前缀测试光线一只猫在窗台")

    def test_video_prompt_literal_newline_controls_seam(self):
        _, plugin = _plugin(
            {
                "features": {
                    "selfie": {"lighting_rules": LIGHTING_ALL_DAY},
                    "video": {
                        "prompt_prefix": "前缀{lighting}\\n",
                        "prompt_prefix_enabled": True,
                    },
                }
            }
        )
        out = plugin._build_video_prompt("一只猫在窗台")
        self.assertEqual(out, "前缀测试光线\n一只猫在窗台")

    def test_video_prompt_trailing_real_newline_still_stripped(self):
        _, plugin = _plugin(
            {
                "features": {
                    "selfie": {"lighting_rules": LIGHTING_ALL_DAY},
                    "video": {
                        "prompt_prefix": "前缀{lighting}\n",
                        "prompt_prefix_enabled": True,
                    },
                }
            }
        )
        out = plugin._build_video_prompt("一只猫在窗台")
        self.assertEqual(out, "前缀测试光线一只猫在窗台")

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
        out = plugin._build_selfie_prompt("在海边")
        self.assertEqual(out, "在海边")
        self.assertNotIn("自定义自拍前缀", out)
        self.assertNotIn("今日外显穿搭", out)
        self.assertNotIn("当前时间光线", out)

    def test_selfie_prompt_prefix_enabled_uses_default(self):
        _, plugin = _plugin({"features": {"selfie": {}}})
        out = plugin._build_selfie_prompt("在海边")
        self.assertIn("请根据参考图拍摄一张新的照片", out)
        self.assertIn("在海边", out)

    def test_selfie_prompt_reference_note_removed_and_direct_concat(self):
        _, plugin = _plugin(
            {
                "features": {
                    "selfie": {
                        "prompt_prefix": "前缀{lighting}\\n",
                        "lighting_rules": LIGHTING_ALL_DAY,
                        "prompt_prefix_enabled": True,
                    }
                }
            }
        )
        out = plugin._build_selfie_prompt("在海边")
        self.assertEqual(out, "前缀测试光线\n在海边")
        # 代码固定的"图片顺序"说明段已整体移除
        self.assertNotIn("图片顺序", out)
        self.assertNotIn("固定人物参考图", out)
        self.assertNotIn("本次用户附带或引用", out)


if __name__ == "__main__":
    unittest.main()
