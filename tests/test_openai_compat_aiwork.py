import importlib.util
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "openai_compat_aiwork_testpkg"
CORE_PACKAGE_NAME = f"{PACKAGE_NAME}.core"
MODULE_NAME = f"{CORE_PACKAGE_NAME}.openai_compat_backend"


class _Logger:
    def debug(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


def _clear_modules():
    for name in [
        MODULE_NAME,
        f"{CORE_PACKAGE_NAME}.gitee_sizes",
        f"{CORE_PACKAGE_NAME}.image_format",
        CORE_PACKAGE_NAME,
        PACKAGE_NAME,
        "astrbot",
        "astrbot.api",
    ]:
        sys.modules.pop(name, None)


def _load_module():
    _clear_modules()

    pkg = types.ModuleType(PACKAGE_NAME)
    pkg.__path__ = [str(ROOT)]
    sys.modules[PACKAGE_NAME] = pkg

    core_pkg = types.ModuleType(CORE_PACKAGE_NAME)
    core_pkg.__path__ = [str(ROOT / "core")]
    sys.modules[CORE_PACKAGE_NAME] = core_pkg

    astrbot_mod = types.ModuleType("astrbot")
    sys.modules["astrbot"] = astrbot_mod

    api_mod = types.ModuleType("astrbot.api")
    api_mod.logger = _Logger()
    sys.modules["astrbot.api"] = api_mod

    spec = importlib.util.spec_from_file_location(
        MODULE_NAME,
        ROOT / "core" / "openai_compat_backend.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class OpenAICompatAIWorkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module()

    def test_is_aiwork_base_url(self):
        self.assertTrue(self.mod.is_aiwork_base_url("https://aiwork.fans/v1"))
        self.assertFalse(self.mod.is_aiwork_base_url("https://api.example.com/v1"))

    def test_build_aiwork_edit_form_fields(self):
        fields = self.mod.build_aiwork_edit_form_fields(
            model="gpt-image-2",
            prompt="额外加个苹果",
            size="1024x1024",
        )
        self.assertEqual(fields["model"], "gpt-image-2")
        self.assertEqual(fields["prompt"], "额外加个苹果")
        self.assertEqual(fields["size"], "1024x1024")
        self.assertEqual(fields["response_format"], "b64_json")
        self.assertEqual(fields["quality"], "auto")
        self.assertEqual(fields["n"], "1")

    def test_build_aiwork_edit_file_fields_uses_image_array(self):
        png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
            b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
            b"\r\n-\xdb\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        files = self.mod.build_aiwork_edit_file_fields([png, png])
        self.assertEqual(len(files), 2)
        self.assertEqual(files[0][0], "image[]")
        self.assertEqual(files[1][0], "image[]")

    def test_build_aiwork_edit_file_fields_keeps_input_order(self):
        first = b"first-image"
        second = b"second-image"
        third = b"third-image"
        files = self.mod.build_aiwork_edit_file_fields([first, second, third])
        self.assertEqual([field for field, _ in files], ["image[]", "image[]", "image[]"])
        self.assertEqual([payload[1] for _, payload in files], [first, second, third])
        self.assertEqual([payload[0] for _, payload in files], ["input_1.jpg", "input_2.jpg", "input_3.jpg"])

    def test_aiwork_edit_posts_multiple_image_array_fields_in_order(self):
        captured = {}

        class FakeImageManager:
            async def save_base64_image(self, value):
                return Path("out.png")

        class FakeResponse:
            status_code = 200
            text = '{"data":[{"b64_json":"ok"}]}'
            headers = {"content-type": "application/json"}
            content = b""

            def json(self):
                return {"data": [{"b64_json": "ok"}]}

        backend = self.mod.OpenAICompatBackend(
            imgr=FakeImageManager(),
            base_url="https://aiwork.fans",
            api_keys=["sk-test"],
            default_model="gpt-image-2",
        )

        async def fake_post(url, api_key, data, files):
            captured["url"] = url
            captured["api_key"] = api_key
            captured["data"] = data
            captured["files"] = files
            return FakeResponse()

        backend._raw_post_multipart = fake_post
        first = b"first-image"
        second = b"second-image"

        import asyncio

        asyncio.run(
            backend.edit(
                "自拍",
                [first, second],
                size="1024x1024",
            )
        )

        self.assertEqual(captured["url"], "https://aiwork.fans/v1/images/edits")
        self.assertEqual(captured["api_key"], "sk-test")
        self.assertEqual(captured["data"]["response_format"], "b64_json")
        self.assertEqual([field for field, _ in captured["files"]], ["image[]", "image[]"])
        self.assertEqual([payload[1] for _, payload in captured["files"]], [first, second])

    def test_backend_normalizes_aiwork_base_url(self):
        backend = self.mod.OpenAICompatBackend(
            imgr=object(),
            base_url="https://aiwork.fans",
            api_keys=["sk-test"],
            default_model="gpt-image-2",
        )
        self.assertEqual(backend.base_url, "https://aiwork.fans/v1")

    def test_aiwork_generate_payload_uses_b64_response_format(self):
        captured = {}

        class FakeImageManager:
            async def save_base64_image(self, value):
                return Path("out.png")

        class FakeResponse:
            status_code = 200
            text = '{"data":[{"b64_json":"ok"}]}'
            headers = {"content-type": "application/json"}
            content = b""

            def json(self):
                return {"data": [{"b64_json": "ok"}]}

        backend = self.mod.OpenAICompatBackend(
            imgr=FakeImageManager(),
            base_url="https://aiwork.fans/v1",
            api_keys=["sk-test"],
            default_model="gpt-image-2",
        )

        async def fake_post(url, api_key, payload):
            captured["url"] = url
            captured["api_key"] = api_key
            captured["payload"] = payload
            return FakeResponse()

        backend._raw_post_json = fake_post

        import asyncio

        asyncio.run(backend.generate("画个苹果", size="1024x1024"))

        self.assertEqual(captured["url"], "https://aiwork.fans/v1/images/generations")
        self.assertEqual(captured["api_key"], "sk-test")
        self.assertEqual(captured["payload"]["response_format"], "b64_json")

    def test_generic_provider_posts_multiple_image_array_fields_first(self):
        captured = {}

        class FakeImageManager:
            async def save_base64_image(self, value):
                return Path("out.png")

        class FakeResponse:
            status_code = 200
            text = '{"data":[{"b64_json":"ok"}]}'
            headers = {"content-type": "application/json"}
            content = b""

            def json(self):
                return {"data": [{"b64_json": "ok"}]}

        backend = self.mod.OpenAICompatBackend(
            imgr=FakeImageManager(),
            base_url="https://lucen.cc/v1",
            api_keys=["sk-test"],
            default_model="gpt-image-2",
        )

        async def fake_post(url, api_key, data, files):
            captured["url"] = url
            captured["files"] = files
            return FakeResponse()

        backend._raw_post_multipart = fake_post

        import asyncio

        asyncio.run(backend.edit("自拍", [b"first-image", b"second-image"], size="1024x1024"))

        self.assertEqual(captured["url"], "https://lucen.cc/v1/images/edits")
        self.assertEqual([field for field, _ in captured["files"]], ["image[]", "image[]"])
        self.assertEqual([payload[1] for _, payload in captured["files"]], [b"first-image", b"second-image"])

    def test_generic_provider_falls_back_to_collage_on_file_format_error(self):
        calls = []

        class FakeImageManager:
            async def save_base64_image(self, value):
                return Path("out.png")

        class ErrorResponse:
            status_code = 422
            text = '{"error":"too many image files"}'
            headers = {"content-type": "application/json"}
            content = b""

        class OkResponse:
            status_code = 200
            text = '{"data":[{"b64_json":"ok"}]}'
            headers = {"content-type": "application/json"}
            content = b""

            def json(self):
                return {"data": [{"b64_json": "ok"}]}

        backend = self.mod.OpenAICompatBackend(
            imgr=FakeImageManager(),
            base_url="https://lucen.cc/v1",
            api_keys=["sk-test"],
            default_model="gpt-image-2",
        )

        async def fake_post(url, api_key, data, files):
            calls.append(files)
            return ErrorResponse() if len(calls) == 1 else OkResponse()

        backend._raw_post_multipart = fake_post

        import asyncio

        asyncio.run(backend.edit("自拍", [b"first-image", b"second-image"], size="1024x1024"))

        self.assertEqual(len(calls), 2)
        self.assertIsInstance(calls[0], list)
        self.assertIsInstance(calls[1], dict)
        self.assertIn("image[]", calls[1])

    def test_aiwork_insufficient_balance_does_not_retry_collage(self):
        calls = []

        class FakeImageManager:
            async def save_base64_image(self, value):
                return Path("out.png")

        class FakeResponse:
            status_code = 403
            text = '{"code":"INSUFFICIENT_BALANCE","message":"Insufficient account balance"}'
            headers = {"content-type": "application/json"}
            content = b""

            def json(self):
                return {"code": "INSUFFICIENT_BALANCE", "message": "Insufficient account balance"}

        backend = self.mod.OpenAICompatBackend(
            imgr=FakeImageManager(),
            base_url="https://aiwork.fans",
            api_keys=["sk-test"],
            default_model="gpt-image-2",
        )

        async def fake_post(url, api_key, data, files):
            calls.append(files)
            return FakeResponse()

        backend._raw_post_multipart = fake_post

        import asyncio

        with self.assertRaises(RuntimeError):
            asyncio.run(backend.edit("自拍", [b"first-image", b"second-image"], size="1024x1024"))

        self.assertEqual(len(calls), 1)
        self.assertIsInstance(calls[0], list)


if __name__ == "__main__":
    unittest.main()
