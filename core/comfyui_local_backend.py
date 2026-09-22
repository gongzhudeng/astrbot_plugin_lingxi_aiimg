from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import httpx

from astrbot.api import logger


class ComfyUILocalBackend:
    """本地 ComfyUI 桥接（comfyui-bridge /v1/images）后端。

    generate() -> 无图请求（文生图/拍照轴，路由由桥接按提示词规则完成）
    edit()     -> drop_reference_images=true 时忽略参考图输入（本地自拍模式：
                  人物身份由工作流内置参考图节点提供，走无图轴自拍集）；
                  否则把第一张用户图作为引用图发送（路由到引用编辑集）。
    """

    def __init__(self, *, imgr, settings: dict[str, Any] | None = None):
        self.imgr = imgr
        s = settings or {}
        self.base_url = str(s.get("base_url") or "http://127.0.0.1:8200").strip().rstrip("/")
        self.timeout = int(s.get("timeout") or 900)
        self.poll_interval = max(0.5, float(s.get("poll_interval") or 2))
        self.default_size = str(s.get("default_size") or "768x1024").strip()
        self.drop_reference_images = bool(s.get("drop_reference_images", False))

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=5.0))

    def _size(self, size: str | None, resolution: str | None) -> str:
        s = str(size or "").strip()
        if s:
            return s
        r = str(resolution or "").strip()
        if r:
            return r
        return self.default_size

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        size: str | None = None,
        resolution: str | None = None,
        extra_body: dict | None = None,
    ) -> Path:
        return await self._run_text_only(prompt, size=size, resolution=resolution)

    async def edit(
        self,
        prompt: str,
        images: list[bytes],
        *,
        model: str | None = None,
        size: str | None = None,
        resolution: str | None = None,
        extra_body: dict | None = None,
    ) -> Path:
        if self.drop_reference_images:
            if images:
                logger.info(
                    "[ComfyUILocal] 本地模式：忽略 %d 张参考图（人物身份由工作流参考图节点提供）",
                    len(images),
                )
            prompt = self._strip_stale_reference_note(prompt)
            return await self._run_text_only(prompt, size=size, resolution=resolution)
        if not images:
            raise ValueError("至少需要一张图片")
        return await self._run_with_image(prompt, images[0], size=size, resolution=resolution)

    @staticmethod
    def _strip_stale_reference_note(prompt: str) -> str:
        """本地模式丢图后，插件为云端多图上传写的「图片顺序：第 X-N 张…」说明就成了
        无效残留（请求里根本没有那些图），整段剔除避免误导模型。"""
        paras = [p for p in (prompt or "").split("\n\n") if not p.lstrip().startswith("图片顺序：")]
        cleaned = "\n\n".join(paras).strip()
        if cleaned != (prompt or "").strip():
            logger.info("[ComfyUILocal] 已剔除本地模式下无意义的参考图顺序说明段")
        return cleaned

    async def _run_text_only(self, prompt: str, *, size: str | None, resolution: str | None) -> Path:
        payload = {"prompt": prompt, "size": self._size(size, resolution)}
        task_id = await self._submit_json(payload)
        return await self._wait_and_save(task_id)

    async def _run_with_image(
        self, prompt: str, image: bytes, *, size: str | None, resolution: str | None
    ) -> Path:
        async with self._client() as client:
            files = {"image": ("reference.png", image, "image/png")}
            form = {"prompt": prompt, "size": self._size(size, resolution)}
            resp = await client.post(f"{self.base_url}/v1/images", data=form, files=files)
            resp.raise_for_status()
            task_id = str(resp.json().get("task_id") or "").strip()
        if not task_id:
            raise RuntimeError("桥接未返回 task_id")
        return await self._wait_and_save(task_id)

    async def _submit_json(self, payload: dict) -> str:
        async with self._client() as client:
            resp = await client.post(f"{self.base_url}/v1/images", json=payload)
            resp.raise_for_status()
            task_id = str(resp.json().get("task_id") or "").strip()
        if not task_id:
            raise RuntimeError("桥接未返回 task_id")
        return task_id

    async def _wait_and_save(self, task_id: str) -> Path:
        deadline = time.time() + self.timeout
        async with self._client() as client:
            while time.time() < deadline:
                await asyncio.sleep(self.poll_interval)
                resp = await client.get(f"{self.base_url}/v1/images/{task_id}")
                resp.raise_for_status()
                body = resp.json()
                status = str(body.get("status") or "").lower()
                if status == "failed":
                    raise RuntimeError(f"本地生图失败: {str(body.get('error'))[:300]}")
                if status == "completed":
                    url = str(body.get("image_url") or "").strip()
                    if not url:
                        raise RuntimeError("本地生图完成但未返回 image_url")
                    img_resp = await client.get(url)
                    img_resp.raise_for_status()
                    logger.info("[ComfyUILocal] 本地生图完成: %s", task_id)
                    return await self.imgr.save_image(img_resp.content)
        raise RuntimeError(f"本地生图超时（{self.timeout}s）")
