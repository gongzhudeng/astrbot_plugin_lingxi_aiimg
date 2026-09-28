"""分时段穿搭兜底规则（outfit_time_rules）行为测试。

优先级链：控制词 force/skip > 临时穿搭检测 > 时间规则 > 今日穿搭。
时间通过覆写 _current_now_minutes 控制，不依赖真实时钟。
"""

import types
import unittest

from test_main_initialize_request_mode import _load_module


SELFIE_PREFIX = {
    "prompt_prefix": "穿搭：{today_outfit}",
    "lighting_rules": ["00:00-24:00=光线"],
}

NIGHT_RULE = ["22:30-06:30=睡衣"]


def _plugin(config, now_minutes, outfit="上装：白衬衫"):
    mod, _ = _load_module()
    plugin = mod.GiteeAIImagePlugin(
        context=types.SimpleNamespace(_busy_schedule_outfit=outfit),
        config=config,
    )
    plugin._current_now_minutes = lambda: now_minutes
    return plugin


def _selfie_config(extra=None):
    conf = dict(SELFIE_PREFIX)
    conf.update(extra or {})
    return {"features": {"selfie": conf}}


class OutfitTimeRulesTests(unittest.IsolatedAsyncioTestCase):
    def test_time_rule_hit_replaces_today_outfit(self):
        plugin = _plugin(_selfie_config({"outfit_time_rules": NIGHT_RULE}), 23 * 60)
        prompt = plugin._build_selfie_prompt("窗边自拍")
        self.assertIn("穿搭：睡衣", prompt)
        self.assertNotIn("白衬衫", prompt)

    def test_time_rule_crosses_midnight_both_sides(self):
        config = _selfie_config({"outfit_time_rules": NIGHT_RULE})
        for minutes in (22 * 60 + 30, 23 * 60 + 59, 1 * 60, 6 * 60 + 29):
            plugin = _plugin(config, minutes)
            self.assertEqual(
                plugin._prepare_selfie_prompt_context("拍一张")[1],
                "睡衣",
                f"now={minutes} 应命中跨午夜规则",
            )
        for minutes in (6 * 60 + 30, 12 * 60, 22 * 60 + 29):
            plugin = _plugin(config, minutes)
            self.assertEqual(
                plugin._prepare_selfie_prompt_context("拍一张")[1],
                "上装：白衬衫",
                f"now={minutes} 不应命中规则",
            )

    def test_time_rule_miss_falls_back_to_today_outfit(self):
        plugin = _plugin(_selfie_config({"outfit_time_rules": NIGHT_RULE}), 12 * 60)
        _, outfit = plugin._prepare_selfie_prompt_context("拍一张")
        self.assertEqual(outfit, "上装：白衬衫")

    def test_time_rule_works_when_today_outfit_empty(self):
        plugin = _plugin(_selfie_config({"outfit_time_rules": NIGHT_RULE}), 23 * 60, outfit="")
        _, outfit = plugin._prepare_selfie_prompt_context("拍一张")
        self.assertEqual(outfit, "睡衣")

    def test_time_rule_value_not_underwear_stripped(self):
        # 规则值是用户显式配置的最终值，直接注入不做内衣剥离
        plugin = _plugin(_selfie_config({"outfit_time_rules": ["23:00-24:00=吊带睡裙"]}), 23 * 60 + 30)
        _, outfit = plugin._prepare_selfie_prompt_context("拍一张")
        self.assertEqual(outfit, "吊带睡裙")

    def test_force_control_word_skips_time_rule(self):
        plugin = _plugin(_selfie_config({"outfit_time_rules": NIGHT_RULE}), 23 * 60)
        cleaned, outfit = plugin._prepare_selfie_prompt_context("沿用今日穿搭，拍特写")
        self.assertEqual(outfit, "上装：白衬衫")
        self.assertNotIn("沿用今日穿搭", cleaned)
        self.assertIn("拍特写", cleaned)

    def test_skip_control_word_still_empty(self):
        plugin = _plugin(_selfie_config({"outfit_time_rules": NIGHT_RULE}), 23 * 60)
        cleaned, outfit = plugin._prepare_selfie_prompt_context("不要使用今日穿搭，拍特写")
        self.assertEqual(outfit, "")
        self.assertNotIn("不要使用今日穿搭", cleaned)

    def test_temporary_clothing_still_empty_at_night(self):
        plugin = _plugin(_selfie_config({"outfit_time_rules": NIGHT_RULE}), 23 * 60)
        _, outfit = plugin._prepare_selfie_prompt_context("穿黑色短裙自拍")
        self.assertEqual(outfit, "")

    def test_today_outfit_disabled_disables_rules(self):
        plugin = _plugin(
            _selfie_config({"today_outfit_enabled": False, "outfit_time_rules": NIGHT_RULE}),
            23 * 60,
        )
        _, outfit = plugin._prepare_selfie_prompt_context("拍一张")
        self.assertEqual(outfit, "")

    def test_invalid_rules_are_skipped(self):
        plugin = _plugin(
            _selfie_config(
                {
                    "outfit_time_rules": [
                        "bad line",
                        "25:00-26:00=睡衣",
                        "22:30-06:30=",
                        "=-睡衣",
                    ]
                }
            ),
            23 * 60,
        )
        _, outfit = plugin._prepare_selfie_prompt_context("拍一张")
        self.assertEqual(outfit, "上装：白衬衫")

    def test_end_midnight_normalized(self):
        # "22:30-00:00" 的结束端按午夜 24:00 处理
        config = _selfie_config({"outfit_time_rules": ["22:30-00:00=睡衣"]})
        plugin = _plugin(config, 23 * 60)
        self.assertEqual(plugin._prepare_selfie_prompt_context("拍一张")[1], "睡衣")
        plugin = _plugin(config, 12 * 60)
        self.assertEqual(plugin._prepare_selfie_prompt_context("拍一张")[1], "上装：白衬衫")

    def test_video_prompt_uses_time_rule(self):
        config = {
            "features": {
                "selfie": {"outfit_time_rules": NIGHT_RULE, "lighting_rules": ["00:00-24:00=光线"]},
                "video": {"prompt_prefix": "视频穿搭：{today_outfit}"},
            }
        }
        plugin = _plugin(config, 23 * 60)
        out = plugin._build_video_prompt("在客厅走动")
        self.assertIn("视频穿搭：睡衣", out)

    def test_no_rules_keeps_original_behavior(self):
        plugin = _plugin(_selfie_config(), 23 * 60)
        _, outfit = plugin._prepare_selfie_prompt_context("拍一张")
        self.assertEqual(outfit, "上装：白衬衫")


if __name__ == "__main__":
    unittest.main()
