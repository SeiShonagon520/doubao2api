"""
Unified API server for Doubao (Playwright browser-based).

Exposes OpenAI-compatible endpoints:
  POST /v1/chat/completions     (chat, streaming & non-streaming)
  GET  /v1/models               (list available models)
  GET  /health                  (health check)
  GET  /auth                    (QR login page)

Start with:
    python -m doubao2api
"""
from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import ipaddress
import socket
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from .account_manager import AccountManager
from .browser_client import BrowserClient, VideoQuotaExceeded
from .browser_operation import BrowserOperationGate, BrowserOperationMiddleware
from .upstream_errors import UpstreamAPIError, UpstreamError, parse_upstream_error


def is_safe_url(url: str) -> bool:
    """
    Validate that the URL is safe to fetch (prevents SSRF).
    Disallows localhost, private IP ranges (RFC 1918), link-local, cloud metadata, and non-http/https.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        hostname = parsed.hostname
        if not hostname:
            return False
        if hostname.lower() in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
            return False

        blocked_networks = [
            ipaddress.ip_network("127.0.0.0/8"),
            ipaddress.ip_network("10.0.0.0/8"),
            ipaddress.ip_network("172.16.0.0/12"),
            ipaddress.ip_network("192.168.0.0/16"),
            ipaddress.ip_network("169.254.0.0/16"),
            ipaddress.ip_network("0.0.0.0/8"),
            ipaddress.ip_network("::1/128"),
            ipaddress.ip_network("fc00::/7"),
            ipaddress.ip_network("fe80::/10"),
        ]

        addr_info = socket.getaddrinfo(hostname, None)
        for entry in addr_info:
            ip_str = entry[4][0]
            ip = ipaddress.ip_address(ip_str)
            if any(ip in net for net in blocked_networks):
                return False
            if ip_str.startswith("169.254."):
                return False
        return True
    except Exception:
        return False

from .tool_calling import (
    build_tool_system_prompt,
    convert_messages_with_tools,
    parse_tool_calls_xml,
    is_tool_call_start,
    has_complete_tool_calls,
    StreamingGuard,
    detect_truncated_tool_call,
    build_continuation_prompt,
    filter_history_by_topic,
    ToolNameObfuscator,
    coerce_tool_arguments,
    deduplicate_continuation,
)
from .token_counter import count_tokens, count_messages_tokens, SAFETY_FACTOR

log = logging.getLogger("doubao_unified")

# ── Auto-delete ephemeral conversations (即用即焚 / Scheme B) ──
_auto_delete_conv = os.environ.get("DOUBAO_AUTO_DELETE_CONV", "true").lower() in ("true", "1", "yes")

# ── Request Dispatch Smoothing (avoids burst rate-limiting from translation plugins) ──
_dispatch_lock = asyncio.Lock()
_last_dispatch_time: float = 0.0
_MIN_DISPATCH_INTERVAL = float(os.environ.get("DOUBAO_MIN_INTERVAL", "0.2"))  # 200ms

# ── Model definitions & Aliases ──────────────────────────────

CHAT_MODELS: Dict[str, int] = {
    "doubao-2.1-turbo": 0,
    "doubao-2.1-pro": 0,
    "doubao-2.1": 0,
    "doubao-think": 1,
    "doubao-2.1-think": 1,
    "doubao-pro": 0,
    "doubao": 0,
    "doubao-expert": 3,
}

MODEL_ALIASES: Dict[str, str] = {
    # OpenAI Chat models
    "gpt-4o": "doubao-2.1-turbo",
    "gpt-4o-mini": "doubao",
    "gpt-4": "doubao-2.1-pro",
    "gpt-4-turbo": "doubao-2.1-pro",
    "gpt-3.5-turbo": "doubao",
    "text-davinci-003": "doubao",
    # Claude models
    "claude-3-5-sonnet": "doubao-2.1-turbo",
    "claude-3-5-sonnet-20241022": "doubao-2.1-turbo",
    "claude-3-opus": "doubao-2.1-pro",
    "claude-3-haiku": "doubao",
    # DeepSeek models
    "deepseek-chat": "doubao-2.1",
    "deepseek-v3": "doubao-2.1",
    "deepseek-reasoner": "doubao-think",
    "deepseek-r1": "doubao-think",
    # OpenAI Reasoning models
    "o1": "doubao-think",
    "o1-mini": "doubao-think",
    "o1-preview": "doubao-think",
    "o3-mini": "doubao-think",
    # Image models
    "dall-e-3": "doubao-image",
    "dall-e-2": "doubao-image",
}

DEFAULT_FALLBACK_MODEL = os.environ.get("DOUBAO_DEFAULT_MODEL", "doubao-2.1-turbo")

ALL_MODELS = [
    {"id": m, "object": "model", "owned_by": "doubao", "created": 0}
    for m in CHAT_MODELS
] + [
    {"id": m, "object": "model", "owned_by": "doubao-alias", "created": 0}
    for m in MODEL_ALIASES
] + [
    {"id": "doubao-image", "object": "model", "owned_by": "doubao", "created": 0},
    {"id": "doubao-music", "object": "model", "owned_by": "doubao", "created": 0},
    {"id": "doubao-video", "object": "model", "owned_by": "doubao", "created": 0},
]

# ── Recent Media Buffer (P2: Gallery) ────────────────────────
from collections import deque
_recent_media: deque = deque(maxlen=30)

def _record_recent_media(media_type: str, url: str, prompt: str, cover_url: str = "", model: str = ""):
    """Record generated image or video for admin gallery display."""
    _recent_media.appendleft({
        "id": f"med_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}",
        "type": media_type,
        "url": url,
        "cover_url": cover_url or url,
        "prompt": prompt,
        "model": model,
        "created_at": time.time(),
    })


# ── Expert Mode Quota Tracker ──
class ExpertQuotaTracker:
    """Detects when expert mode is silently downgraded and falls back to think."""

    def __init__(self, consecutive_threshold: int = 2, retry_interval: int = 1800):
        self._no_reasoning_count = 0  # consecutive expert requests without reasoning
        self._threshold = consecutive_threshold  # how many before marking degraded
        self._degraded = False
        self._last_retry_time = 0.0
        self._retry_interval = retry_interval  # seconds before retrying expert (30 min)

    @property
    def is_degraded(self) -> bool:
        """True if expert mode appears to be quota-limited."""
        if not self._degraded:
            return False
        # Periodically retry
        import time
        if time.time() - self._last_retry_time > self._retry_interval:
            return False  # Allow a retry
        return True

    def report_response(self, had_reasoning: bool):
        """Call after each expert-mode request with whether reasoning was present."""
        import time
        if had_reasoning:
            self._no_reasoning_count = 0
            if self._degraded:
                log.info("Expert mode recovered (reasoning detected)")
            self._degraded = False
        else:
            self._no_reasoning_count += 1
            if self._no_reasoning_count >= self._threshold and not self._degraded:
                self._degraded = True
                self._last_retry_time = time.time()
                log.warning("Expert mode appears degraded (no reasoning for %d requests), falling back to think", self._threshold)

    def mark_retry(self):
        """Mark that we're doing a retry probe."""
        import time
        self._last_retry_time = time.time()

    def get_effective_mode(self, requested_deep_think: int) -> tuple[int, str]:
        """Return (deep_think_value, model_name) considering degradation.

        If expert (3) is degraded, falls back to think (1).
        """
        if requested_deep_think == 3 and self.is_degraded:
            return 1, "doubao-think"
        model_map = {0: "doubao", 1: "doubao-think", 3: "doubao-expert"}
        return requested_deep_think, model_map.get(requested_deep_think, "doubao")


_expert_tracker = ExpertQuotaTracker()


def _size_to_ratio(size):
    """Convert OpenAI size format to Doubao ratio."""
    if not size:
        return "1:1"
    size_map = {
        "1024x1024": "1:1",
        "1792x1024": "16:9",
        "1024x1792": "9:16",
        "1024x768": "4:3",
        "768x1024": "3:4",
    }
    if size in size_map:
        return size_map[size]
    if ":" in size:
        return size
    return "1:1"

# ── Request log ring buffer ───────────────────────────────────

_REQUEST_LOG: collections.deque = collections.deque(maxlen=100)
_SERVER_START_TIME: float = time.time()


# ── Rate limiter ─────────────────────────────────────────────


class _TokenBucket:
    """Simple async token-bucket rate limiter."""

    def __init__(self, rpm: float):
        self._interval = 60.0 / rpm if rpm > 0 else 0.0
        self._next_allowed = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        if self._interval <= 0:
            return
        while True:
            async with self._lock:
                now = time.monotonic()
                if now >= self._next_allowed:
                    self._next_allowed = now + self._interval
                    return
                wait_time = self._next_allowed - now
            await asyncio.sleep(wait_time)


# ── Pydantic request models ──────────────────────────────────


class _Message(BaseModel):
    role: str
    content: Any  # str | list[dict]
    tool_calls: Optional[list] = None  # for assistant messages with tool calls
    tool_call_id: Optional[str] = None  # for role:tool messages
    name: Optional[str] = None  # tool name for role:tool messages


class ChatCompletionRequest(BaseModel):
    model: str = "doubao"
    messages: List[_Message]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    conversation_id: Optional[str] = None
    bot_id: Optional[str] = None
    tools: Optional[List[dict]] = None
    tool_choice: Optional[Any] = None  # "auto" | "none" | {"type":"function","function":{"name":"..."}}
    enable_thinking: Optional[bool] = None  # triggers deep_search="1" for thinking mode
    reasoning_effort: Optional[str] = None  # "low"|"medium"|"high" — also triggers thinking



class ImageGenerationRequest(BaseModel):
    prompt: str
    model: str = "doubao-image"
    n: int = 1
    size: Optional[str] = "1024x1024"
    ratio: Optional[str] = None
    ref_image_key: Optional[str] = None
    ref_image_keys: Optional[List[str]] = None
    image: Optional[Any] = None
    images: Optional[List[Any]] = None
    image_url: Optional[Any] = None
    image_urls: Optional[List[str]] = None
    response_format: Optional[str] = "url"
    save_history: Optional[bool] = None
    memory_type: Optional[int] = None

def _extract_prompt_and_file_refs(messages: List[_Message]) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract text prompt, OpenAI-style file_url references, and image_url references."""
    parts: list[str] = []
    file_refs: list[dict[str, Any]] = []
    image_refs: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg.content, str):
            if len(messages) == 1:
                parts.append(msg.content)
            else:
                parts.append(f"[{msg.role}]: {msg.content}")
            continue
        if not isinstance(msg.content, list):
            continue
        for part in msg.content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "text":
                text = part.get("text", "")
                if text:
                    if len(messages) == 1:
                        parts.append(text)
                    else:
                        parts.append(f"[{msg.role}]: {text}")
            elif part_type == "image_url":
                img_data = part.get("image_url", {})
                if isinstance(img_data, str):
                    image_refs.append({"url": img_data})
                elif isinstance(img_data, dict):
                    image_refs.append(img_data)
            elif part_type == "file_url":
                file_url = part.get("file_url", {})
                ref = {"url": file_url} if isinstance(file_url, str) else dict(file_url)
                url_str = str(ref.get("url", "")).lower()
                name_str = str(ref.get("name", "")).lower()
                is_img = any(ext in url_str or ext in name_str for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif", "image/"))
                if is_img:
                    image_refs.append(ref)
                else:
                    file_refs.append(ref)
    return "\n".join(parts), file_refs, image_refs


# ── Application factory ──────────────────────────────────────


def create_app(
    *,
    api_key: Optional[str] = None,
    rpm_limit: float = 20.0,
    browser_client: Optional[Any] = None,
) -> FastAPI:
    """Build and return a configured FastAPI application."""

    _browser: Dict[str, Any] = {"client": browser_client} if browser_client is not None else {}
    _conversation_account_ids: Dict[str, str] = {}
    _resource_account_ids: Dict[str, str] = {}

    async def _browser_watchdog():
        """Background task: check browser health, auto-detect login, auto-restart on crash, auto-heal captcha."""
        consecutive_dead = 0
        while True:
            client = _browser.get("client")
            sleep_time = 10 if (client and client.is_ready and not client.needs_captcha) else 2
            await asyncio.sleep(sleep_time)

            if client is None:
                continue
            try:
                async with client.operation_gate.lease():
                    alive = await client.is_alive()
                    if not alive:
                        consecutive_dead += 1
                        log.warning("Browser watchdog: health check failed (%d/3)", consecutive_dead)
                        if consecutive_dead >= 3:
                            log.error("Browser watchdog: process dead 3 consecutive checks, auto-relaunching window...")
                            await client.restart()
                            consecutive_dead = 0
                    else:
                        consecutive_dead = 0
                        if not client.is_ready:
                            await client._check_login_state()
                            if client.is_ready:
                                log.info("Browser client logged in successfully! sessionid confirmed.")
            except Exception as e:
                log.error("Browser watchdog error: %s", e)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if browser_client is not None:
            yield
            return

        # Ensure browser_client logs are visible
        logging.getLogger("doubao2api.browser_client").setLevel(logging.INFO)
        logging.getLogger("doubao2api.browser_client").addHandler(logging.StreamHandler())

        # Start browser client (auto-resolves headless mode from DOUBAO_HEADLESS)
        user_data_dir = os.environ.get(
            "DOUBAO_BROWSER_DATA",
            os.path.join(os.path.expanduser("~"), ".doubao_browser"),
        )
        client = BrowserClient(headless=None, user_data_dir=user_data_dir)
        _browser["client"] = client
        _browser["startup_error"] = None

        try:
            await client.start()
            if client.is_ready:
                log.info("Browser client ready (already logged in)")
            else:
                log.warning(
                    "Browser not logged in. Visit /admin to scan QR code or import cookies."
                )
        except Exception as e:
            log.error("Browser client failed to start: %s", e)
            _browser["startup_error"] = str(e)

        # Start browser watchdog
        watchdog_task = asyncio.create_task(_browser_watchdog())

        # Auto open admin dashboard in default browser
        if os.environ.get("DOUBAO_AUTO_OPEN", "true").lower() == "true":
            server_port = int(os.environ.get("DOUBAO_PORT", "9090"))
            async def _auto_open():
                await asyncio.sleep(1.5)
                try:
                    import webbrowser
                    webbrowser.open(f"http://127.0.0.1:{server_port}/admin")
                except Exception:
                    pass
            asyncio.create_task(_auto_open())

        yield

        # Shutdown
        watchdog_task.cancel()
        client = _browser.pop("client", None)
        if client:
            try:
                await client.stop()
            except Exception:
                pass

    app = FastAPI(title="Doubao API", version="1.0.0", lifespan=lifespan)

    @app.exception_handler(UpstreamAPIError)
    async def _upstream_exc(request: Request, exc: UpstreamAPIError):
        error = exc.upstream_error
        return JSONResponse(
            status_code=error.status_code,
            content=error.to_openai_error(_admin_url(request)),
        )

    @app.exception_handler(HTTPException)
    async def _http_exc(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": exc.detail, "type": "api_error", "code": exc.status_code}},
        )

    @app.exception_handler(Exception)
    async def _unhandled_exc(request: Request, exc: Exception):
        log.exception("Unhandled exception")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(exc), "type": "internal_error", "code": 500}},
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(
        BrowserOperationMiddleware,
        get_client=lambda: _browser.get("client"),
        paths={
            "/v1/chat/completions", "/v1/images/generations", "/v1/images/edits",
            "/v1/audio/generations", "/v1/video/generations", "/v1/files",
            "/v1/files/download", "/v1/images/upload", "/v1/chat/completions/with-file",
            "/v1/session/qr-login", "/admin/api/cookies", "/admin/api/cookies/import",
            "/admin/api/accounts", "/admin/api/accounts/add", "/admin/api/accounts/save_current",
            "/admin/api/accounts/switch", "/admin/api/accounts/delete", "/admin/api/accounts/strategy",
            "/admin/api/accounts/reset_quota", "/admin/api/accounts/rename", "/admin/api/accounts/probe_all",
            "/admin/api/accounts/clear_captcha", "/admin/api/probe", "/auth/login", "/auth/reset_captcha",
            "/auth/eval", "/auth/screenshot", "/auth/status", "/admin/api/status",
            "/admin/api/browser/status", "/admin/api/browser/mode", "/admin/api/browser/screenshot",
            "/admin/api/browser/force-window", "/admin/api/browser/install-chromium",
        },
    )

    bucket = _TokenBucket(rpm_limit)

    # ── Auth helper ──

    def _check_auth(request: Request) -> None:
        if not api_key:
            return
        auth = request.headers.get("Authorization", "")
        token = auth[7:].strip() if auth.startswith("Bearer ") else ""
        if not token:
            token = request.headers.get("X-API-Key", "").strip()
        # Query parameter ?key= is explicitly disallowed for security reasons
        if api_key == "any":
            if not token:
                raise HTTPException(
                    status_code=401,
                    detail="API key required in Authorization or X-API-Key header",
                )
            return
        if token != api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")

    def _get_client() -> BrowserClient:
        client = _browser.get("client")
        startup_err = _browser.get("startup_error")
        if client is None or startup_err:
            err_msg = f": {startup_err}" if startup_err else ""
            raise HTTPException(
                status_code=503,
                detail=f"Browser client failed to start{err_msg}. Please check /admin for diagnosis.",
            )
        if not client.is_ready and not client.needs_captcha:
            raise HTTPException(
                status_code=503,
                detail="Not logged in. Visit /admin to scan QR code or import cookies.",
            )
        return client

    # ── Prompt extraction ──

    def _extract_prompt(messages: List[_Message]) -> str:
        """Extract text prompt from OpenAI-format messages."""
        parts: list[str] = []
        for msg in messages:
            if isinstance(msg.content, str):
                if len(messages) == 1:
                    parts.append(msg.content)
                else:
                    parts.append(f"[{msg.role}]: {msg.content}")
            elif isinstance(msg.content, list):
                for p in msg.content:
                    if isinstance(p, dict) and p.get("type") == "text":
                        text = p.get("text", "")
                        if text:
                            if len(messages) == 1:
                                parts.append(text)
                            else:
                                parts.append(f"[{msg.role}]: {text}")
        return "\n".join(parts)

    async def _materialize_image_refs(client: BrowserClient, image_refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Resolve image references (data URI, remote URL, TOS URI) to uploaded image metadata."""
        import base64
        import mimetypes
        from urllib.parse import urlparse
        images: list[dict[str, Any]] = []
        for img_ref in image_refs:
            url = str(img_ref.get("url", "")).strip()
            if not url:
                continue
            name = img_ref.get("name") or "image.png"
            if url.startswith("tos-"):
                images.append({
                    "uri": url,
                    "cdn_url": "",
                    "name": name,
                    "width": int(img_ref.get("width") or 1024),
                    "height": int(img_ref.get("height") or 1024),
                })
                continue
            img_bytes = None
            if url.startswith("data:"):
                try:
                    header, encoded = url.split(",", 1)
                    img_bytes = base64.b64decode(encoded)
                except (ValueError, TypeError) as exc:
                    raise HTTPException(status_code=400, detail=f"Invalid base64 in image_url: {exc}")
                if name == "image.png" and ";" in header:
                    mime_type = header[5:].split(";", 1)[0]
                    ext = mimetypes.guess_extension(mime_type) or ".png"
                    name = f"image{ext}"
            elif url.startswith("http://") or url.startswith("https://"):
                if not is_safe_url(url):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Prohibited or unsafe URL for image_url (SSRF protection): {url[:80]}",
                    )
                parsed = urlparse(url)
                inferred_name = parsed.path.rsplit("/", 1)[-1] or "image.png"
                if "." not in inferred_name:
                    inferred_name += ".png"
                name = inferred_name
                try:
                    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as http_client:
                        resp = await http_client.get(url)
                        if resp.status_code != 200:
                            raise HTTPException(
                                status_code=502,
                                detail=f"Failed to fetch remote image from {url[:80]} (status {resp.status_code})",
                            )
                        img_bytes = resp.content
                except HTTPException:
                    raise
                except Exception as exc:
                    raise HTTPException(status_code=502, detail=f"Failed to fetch remote image: {exc}")
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported image_url format: {url[:80]}")

            if img_bytes is not None:
                uploaded = await client.upload_image(image_bytes=img_bytes, filename=name)
                _register_account_resource(client, uploaded.get("uri", ""))
                uploaded["_source_bytes"] = img_bytes
                uploaded["_source_name"] = name
                images.append(uploaded)
        return images

    async def _materialize_file_refs(client: BrowserClient, file_refs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
        """Resolve TOS/data/http file_url references to uploaded file metadata and extract text context if readable."""
        import base64
        import mimetypes
        from urllib.parse import urlparse
        files: list[dict[str, Any]] = []
        text_contexts: list[str] = []
        for file_ref in file_refs:
            url = str(file_ref.get("url", "")).strip()
            if not url:
                raise HTTPException(status_code=400, detail="file_url.url is required")
            name = file_ref.get("name") or "file"
            size = int(file_ref.get("size") or 0)
            if url.startswith("tos-"):
                files.append({"uri": url, "name": name, "size": size, "_source_bytes": None})
                continue
            file_data = None
            if url.startswith("data:"):
                try:
                    header, encoded = url.split(",", 1)
                    file_data = base64.b64decode(encoded)
                except (ValueError, TypeError) as exc:
                    raise HTTPException(status_code=400, detail="Invalid data URI") from exc
                if name == "file":
                    mime_type = header[5:].split(";", 1)[0]
                    ext = mimetypes.guess_extension(mime_type) or ".txt"
                    name = f"upload{ext}"
            elif url.startswith("http://") or url.startswith("https://"):
                if not is_safe_url(url):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Prohibited or unsafe URL for file_url (SSRF protection): {url[:80]}",
                    )
                parsed = urlparse(url)
                inferred_name = parsed.path.rsplit("/", 1)[-1] or "downloaded_file"
                if "." not in inferred_name:
                    inferred_name += ".txt"
                if name == "file":
                    name = inferred_name
                max_download_bytes = int(os.environ.get("MAX_DOWNLOAD_SIZE_MB", "50")) * 1024 * 1024
                try:
                    async with httpx.AsyncClient(timeout=15.0) as http_client:
                        async with http_client.stream("GET", url) as response:
                            response.raise_for_status()
                            chunks = []
                            downloaded_size = 0
                            async for chunk in response.aiter_bytes():
                                downloaded_size += len(chunk)
                                if downloaded_size > max_download_bytes:
                                    raise HTTPException(
                                        status_code=413,
                                        detail=f"Remote file exceeds maximum allowed size ({max_download_bytes // (1024*1024)}MB)",
                                    )
                                chunks.append(chunk)
                            file_data = b"".join(chunks)
                except HTTPException:
                    raise
                except Exception as exc:
                    raise HTTPException(status_code=502, detail=f"Failed to fetch remote file: {exc}")
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file_url: {url[:80]}")

            if file_data is not None:
                try:
                    text = file_data.decode("utf-8")
                    if text.strip() and len(text) < 200000:
                        text_contexts.append(f"[附带文件: {name}]\n{text}\n[附带文件结束]")
                except UnicodeDecodeError:
                    try:
                        text = file_data.decode("gbk")
                        if text.strip() and len(text) < 200000:
                            text_contexts.append(f"[附带文件: {name}]\n{text}\n[附带文件结束]")
                    except UnicodeDecodeError:
                        pass
                uploaded = await client.upload_file(file_data=file_data, filename=name)
                _register_account_resource(client, uploaded["uri"])
                files.append({
                    "uri": uploaded["uri"], "name": uploaded["name"], "size": uploaded["size"],
                    "_source_bytes": file_data,
                })

        context_str = "\n\n".join(text_contexts)
        return files, context_str

    def _attachments_reuploadable(
        image_attachments: Optional[list[dict[str, Any]]],
        file_attachments: Optional[list[dict[str, Any]]],
    ) -> bool:
        return all(
            isinstance(item.get("_source_bytes"), bytes)
            for item in (image_attachments or []) + (file_attachments or [])
        )

    async def _reupload_chat_attachments(
        client: BrowserClient,
        image_attachments: Optional[list[dict[str, Any]]],
        file_attachments: Optional[list[dict[str, Any]]],
    ) -> None:
        if not _attachments_reuploadable(image_attachments, file_attachments):
            raise HTTPException(
                status_code=409,
                detail="Account-bound TOS references cannot be reused after account switching; provide source bytes.",
            )
        for image in image_attachments or []:
            uploaded = await client.upload_image(
                image_bytes=image["_source_bytes"],
                filename=image.get("_source_name") or image.get("name") or "image.png",
            )
            _register_account_resource(client, uploaded.get("uri", ""))
            image.update(uploaded)
        for file in file_attachments or []:
            uploaded = await client.upload_file(
                file_data=file["_source_bytes"],
                filename=file.get("name") or "file.txt",
            )
            _register_account_resource(client, uploaded["uri"])
            file.update({"uri": uploaded["uri"], "name": uploaded["name"], "size": uploaded["size"]})

    async def _reupload_image_sources(
        client: BrowserClient,
        image_keys: list[str],
        sources: list[dict[str, Any]],
    ) -> list[str]:
        source_by_key = {item["uri"]: item for item in sources}
        replacements: dict[str, str] = {}
        for key in image_keys:
            source = source_by_key.get(key)
            if source is None or not isinstance(source.get("source_bytes"), bytes):
                raise HTTPException(
                    status_code=409,
                    detail="Account-bound TOS references cannot be reused after account switching; provide source bytes.",
                )
            uploaded = await client.upload_image(
                image_bytes=source["source_bytes"],
                filename=source.get("filename") or "ref_image.png",
            )
            new_key = uploaded.get("uri", "")
            _register_account_resource(client, new_key)
            replacements[key] = new_key
            source["uri"] = new_key
        return [replacements.get(key, key) for key in image_keys]

    async def _resolve_image_to_key(
        client: BrowserClient,
        image_input: Any,
        *,
        sources: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        """Resolve TOS key, data URI, base64 string, or remote HTTP URL to a ByteDance TOS key."""
        import base64
        import mimetypes
        from urllib.parse import urlparse

        if not image_input:
            return ""

        # Handle dict wrapping, e.g. {"url": "..."} or {"image_url": "..."}
        if isinstance(image_input, dict):
            image_input = (
                image_input.get("url")
                or image_input.get("image_url")
                or image_input.get("key")
                or image_input.get("uri")
                or image_input.get("b64_json")
                or ""
            )

        if not isinstance(image_input, str):
            raise HTTPException(status_code=400, detail="Invalid image input: expected string or dict")

        image_str = image_input.strip()
        if not image_str:
            return ""

        # 1. Already a TOS key
        if image_str.startswith("tos-"):
            if sources is not None:
                sources.append({"uri": image_str, "source_bytes": None, "filename": "ref_image.png"})
            return image_str

        # 2. Data URI
        if image_str.startswith("data:"):
            try:
                header, encoded = image_str.split(",", 1)
                image_bytes = base64.b64decode(encoded)
                mime_type = header[5:].split(";", 1)[0]
                ext = mimetypes.guess_extension(mime_type) or ".png"
                if ext == ".jpe":
                    ext = ".jpg"
                filename = f"ref_image{ext}"
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid image data URI: {exc}") from exc
            uploaded = await client.upload_image(image_bytes=image_bytes, filename=filename)
            key = uploaded.get("uri", "")
            if sources is not None and key:
                sources.append({"uri": key, "source_bytes": image_bytes, "filename": filename})
            _register_account_resource(client, key)
            return key

        # 3. HTTP or HTTPS URL
        if image_str.startswith("http://") or image_str.startswith("https://"):
            if not is_safe_url(image_str):
                raise HTTPException(
                    status_code=400,
                    detail=f"Prohibited or unsafe URL for image (SSRF protection): {image_str[:80]}",
                )
            parsed = urlparse(image_str)
            filename = parsed.path.rsplit("/", 1)[-1] or "ref_image.png"
            if not any(filename.lower().endswith(e) for e in (".png", ".jpg", ".jpeg", ".webp")):
                filename = "ref_image.png"

            max_download_bytes = int(os.environ.get("MAX_DOWNLOAD_SIZE_MB", "50")) * 1024 * 1024
            try:
                async with httpx.AsyncClient(timeout=20.0) as http_client:
                    async with http_client.stream("GET", image_str) as response:
                        response.raise_for_status()
                        chunks = []
                        downloaded_size = 0
                        async for chunk in response.aiter_bytes():
                            downloaded_size += len(chunk)
                            if downloaded_size > max_download_bytes:
                                raise HTTPException(
                                    status_code=413,
                                    detail=f"Remote image exceeds maximum allowed size ({max_download_bytes // (1024*1024)}MB)",
                                )
                            chunks.append(chunk)
                        image_bytes = b"".join(chunks)
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(status_code=502, detail=f"Failed to fetch remote image: {exc}")

            uploaded = await client.upload_image(image_bytes=image_bytes, filename=filename)
            key = uploaded.get("uri", "")
            if sources is not None and key:
                sources.append({"uri": key, "source_bytes": image_bytes, "filename": filename})
            _register_account_resource(client, key)
            return key

        # 4. Raw base64 string (without data: prefix)
        try:
            image_bytes = base64.b64decode(image_str)
            if len(image_bytes) > 0:
                uploaded = await client.upload_image(image_bytes=image_bytes, filename="ref_image.png")
                key = uploaded.get("uri", "")
                if sources is not None and key:
                    sources.append({"uri": key, "source_bytes": image_bytes, "filename": "ref_image.png"})
                _register_account_resource(client, key)
                return key
        except Exception:
            pass

        raise HTTPException(status_code=400, detail=f"Unsupported image format: {image_str[:80]}")

    async def _resolve_multi_images_to_keys(
        client: Any,
        image_inputs: List[Any],
        *,
        sources: Optional[list[dict[str, Any]]] = None,
    ) -> List[str]:
        """Resolve a list of image inputs (files, TOS keys, Data URIs, URLs, or dicts) to TOS keys."""
        keys = []
        for item in image_inputs:
            if not item:
                continue
            if hasattr(item, "read"):
                file_bytes = await item.read()
                filename = getattr(item, "filename", "ref_image.png") or "ref_image.png"
                uploaded = await client.upload_image(image_bytes=file_bytes, filename=filename)
                key = uploaded.get("uri", "")
                if sources is not None and key:
                    sources.append({"uri": key, "source_bytes": file_bytes, "filename": filename})
                _register_account_resource(client, key)
                if key and key not in keys:
                    keys.append(key)
            elif isinstance(item, (str, dict)):
                key = await _resolve_image_to_key(client, item, sources=sources)
                if key and key not in keys:
                    keys.append(key)
        return keys

    # ── Request logging middleware ──

    @app.middleware("http")
    async def _log_requests(request: Request, call_next):
        path = request.url.path
        if path.startswith("/auth") or path.startswith("/admin"):
            return await call_next(request)
        start = time.time()
        response = await call_next(request)
        elapsed = round((time.time() - start) * 1000)
        _REQUEST_LOG.append({
            "ts": time.time(),
            "method": request.method,
            "path": path,
            "status": response.status_code,
            "ms": elapsed,
        })
        return response

    # ── Endpoints ──

    @app.get("/health/live")
    async def health_live():
        """Liveness probe: returns 200 as long as the API server process is alive."""
        return {"status": "alive"}

    @app.get("/health/ready")
    async def health_ready():
        """Readiness probe: returns 200 if browser is logged in and ready, else 503."""
        client = _browser.get("client")
        startup_error = _browser.get("startup_error")
        if startup_error:
            raise HTTPException(
                status_code=503,
                detail={"status": "not_ready", "reason": "startup_error", "error": startup_error},
            )
        if not client:
            raise HTTPException(
                status_code=503,
                detail={"status": "not_ready", "reason": "browser_not_initialized"},
            )
        if not client.is_ready:
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "not_ready",
                    "reason": "not_logged_in",
                    "needs_captcha": client.needs_captcha,
                    "last_error_code": client.last_error_code,
                },
            )
        return {"status": "ready", "logged_in": True}

    @app.get("/health")
    async def health():
        """Combined health status (backward compatible)."""
        client = _browser.get("client")
        startup_error = _browser.get("startup_error")
        ready = client.is_ready if client else False
        result = {
            "status": "ok" if ready else "not_ready",
            "logged_in": ready,
            "live": True,
            "ready": ready,
        }
        if startup_error:
            result["startup_error"] = startup_error
        if client:
            result["consecutive_failures"] = client.consecutive_failures
            result["needs_captcha"] = client.needs_captcha
            result["last_error_code"] = client.last_error_code
            result["headless"] = client.headless
            result["browser_name"] = getattr(client, "browser_name", "chromium")
        result["expert_degraded"] = _expert_tracker.is_degraded
        return result

    def _admin_url(request: Request) -> str:
        return f"{str(request.base_url).rstrip('/')}/admin"

    def _event_has_output(event: dict) -> bool:
        def _has_blocks(blocks: Any) -> bool:
            if not isinstance(blocks, list):
                return False
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                block_content = block.get("content", {})
                if not isinstance(block_content, dict):
                    continue
                if block.get("block_type") == 10000:
                    text_block = block_content.get("text_block", {})
                    if isinstance(text_block, dict) and text_block.get("text"):
                        return True
                elif block.get("block_type") == 10025:
                    if block_content.get("search_query_result_block"):
                        return True
                elif block.get("block_type") not in (10040, 10052) and block_content:
                    return True
            return False

        if isinstance(event.get("text"), str) and event["text"]:
            return True
        for patch in event.get("patch_op", []):
            if not isinstance(patch, dict):
                continue
            patch_value = patch.get("patch_value", {})
            if not isinstance(patch_value, dict):
                continue
            if _has_blocks(patch_value.get("content_block", [])):
                return True
            content_str = patch_value.get("content")
            if isinstance(content_str, str):
                try:
                    if json.loads(content_str).get("text"):
                        return True
                except (json.JSONDecodeError, AttributeError, TypeError):
                    pass
        content = event.get("content", {})
        if isinstance(content, dict):
            return _has_blocks(content.get("content_block", []))
        return False

    async def _recover_upstream_error(
        client: BrowserClient,
        error: UpstreamError,
        admin_url: str,
        *,
        allow_account_switch: bool,
        record_failure: bool = True,
        feature: str = "chat",
    ) -> Optional[str]:
        """Resolve a verification error without clearing it before a successful request."""
        code = int(error.code) if error.code and error.code.isdigit() else 0
        if record_failure:
            client.record_failure(
                code or error.status_code,
                verification_required=error.category == "verification",
                error_category=error.category,
            )
        if error.category != "verification":
            return None

        visible = await client.is_captcha_visible()
        if visible:
            try:
                if await client.try_auto_solve_captcha():
                    return "same_account"
            except Exception as exc:
                log.warning("Captcha auto-solver failed (%s)", type(exc).__name__)

        manager = client.account_manager if isinstance(client.account_manager, AccountManager) else None
        current_acc = manager.get_active_account() if manager else None
        if manager and current_acc:
            manager.mark_captcha_required(current_acc.id)
        next_acc = (
            manager.get_next_available_account(feature)
            if manager and allow_account_switch else None
        )
        if next_acc and (not current_acc or next_acc.id != current_acc.id):
            try:
                switched = await client.switch_to_account(next_acc.id)
            except Exception as exc:
                log.warning("Account switch failed (%s)", type(exc).__name__)
                switched = False
            if switched:
                return "account_switch"
            if current_acc:
                manager.set_active_account(current_acc.id)

        if visible:
            await client.auto_popup_for_captcha()
        return None

    async def _resolve_captcha_or_failover(
        client: BrowserClient,
        request: Request,
        *,
        allow_account_switch: bool,
        feature: str = "chat",
    ) -> Optional[str]:
        """Try explicit verification recovery; let a real request prove hidden recovery."""
        if not client.needs_captcha:
            return None

        safe_code = str(client.last_error_code) if str(client.last_error_code).isdigit() else None
        error = UpstreamError(
            "verification",
            503,
            code=safe_code,
        )
        recovered = await _recover_upstream_error(
            client, error, _admin_url(request), allow_account_switch=allow_account_switch,
            record_failure=False, feature=feature,
        )
        if recovered:
            manager = getattr(client, "account_manager", None)
            active = manager.get_active_account() if isinstance(manager, AccountManager) else None
            return active.id if active else None

        # A missing DOM challenge cannot prove recovery. A hidden challenge also
        # gets one actual upstream request, whose success clears this state.
        if not await client.is_captcha_visible():
            manager = getattr(client, "account_manager", None)
            active = manager.get_active_account() if isinstance(manager, AccountManager) else None
            return active.id if active else None

        await client.auto_popup_for_captcha()
        raise UpstreamAPIError(error)

    async def _select_account_for_request(
        client: BrowserClient,
        feature: str,
        *,
        conversation_id: Optional[str] = None,
        account_bound_inputs: bool = False,
        preferred_account_id: Optional[str] = None,
    ) -> Optional[str]:
        """Choose one account for a new operation, committing only after a real switch."""
        manager = getattr(client, "account_manager", None)
        if not isinstance(manager, AccountManager) or not manager.accounts:
            return None

        active = manager.get_active_account()
        if conversation_id:
            owner_id = _conversation_account_ids.get(conversation_id)
            if owner_id:
                if not active or active.id != owner_id:
                    raise HTTPException(
                        status_code=409,
                        detail="This conversation belongs to a different account. Switch to its owner account to continue.",
                    )
                owner = manager.get_account(owner_id)
                if owner is None or owner.status in ("disabled", "invalid"):
                    raise HTTPException(status_code=409, detail="The account bound to this conversation is unavailable.")
                if owner.status == "captcha" and not client.needs_captcha:
                    raise HTTPException(status_code=409, detail="The account bound to this conversation requires verification.")
                return owner_id
            if active:
                if active.status in ("disabled", "invalid") or (
                    active.status == "captcha" and not client.needs_captcha
                ):
                    raise HTTPException(status_code=409, detail="The current account is unavailable for this conversation.")
                return active.id
            raise HTTPException(status_code=409, detail="No active account is available to pin this conversation.")

        if preferred_account_id:
            preferred = manager.get_account(preferred_account_id)
            probe_current = (
                preferred is not None
                and preferred.status == "captcha"
                and client.needs_captcha
                and feature != "video"
                and active is not None
                and active.id == preferred_account_id
            )
            if preferred is None or (preferred not in manager.get_available_accounts(feature) and not probe_current):
                raise HTTPException(status_code=503, detail=f"Recovered account is unavailable for {feature} requests.")
            if not active or active.id != preferred_account_id:
                raise HTTPException(status_code=503, detail="Recovered account changed before request dispatch.")
            return preferred_account_id

        candidate = manager.get_next_available_account(feature)
        if candidate is None:
            status_code = 429 if feature == "video" else 503
            raise HTTPException(status_code=status_code, detail=f"No available account for {feature} requests.")

        if account_bound_inputs:
            if not active or candidate.id != active.id:
                raise HTTPException(
                    status_code=409,
                    detail="Account-bound reference data cannot be reused after account rotation. Provide source image or file bytes.",
                )

        if active and candidate.id == active.id:
            manager.set_active_account(candidate.id)
            return candidate.id

        if not await client.switch_to_account(candidate.id):
            raise HTTPException(status_code=503, detail="Failed to switch to the selected account.")
        return candidate.id

    def _bind_conversation_account(conversation_id: Optional[str], account_id: Optional[str]) -> None:
        if conversation_id and account_id:
            _conversation_account_ids[conversation_id] = account_id

    def _register_account_resource(client: Any, uri: str) -> None:
        manager = getattr(client, "account_manager", None)
        active = manager.get_active_account() if isinstance(manager, AccountManager) else None
        if uri and active:
            _resource_account_ids[uri] = active.id

    def _check_account_bound_resources(client: Any, resource_ids: list[str]) -> None:
        manager = getattr(client, "account_manager", None)
        active = manager.get_active_account() if isinstance(manager, AccountManager) else None
        if active is None:
            return
        for resource_id in resource_ids:
            owner_id = _resource_account_ids.get(resource_id)
            if owner_id and owner_id != active.id:
                raise HTTPException(
                    status_code=409,
                    detail="This TOS reference belongs to another account. Switch to its owner before using it.",
                )

    def _tos_resource_id(value: Any) -> Optional[str]:
        if isinstance(value, dict):
            value = (
                value.get("url") or value.get("image_url") or value.get("file_url")
                or value.get("key") or value.get("uri") or ""
            )
        if isinstance(value, str) and value.strip().startswith("tos-"):
            return value.strip()
        return None

    @app.get("/v1/models")
    async def list_models(request: Request):
        _check_auth(request)
        return {"object": "list", "data": ALL_MODELS}

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest, request: Request):
        _check_auth(request)

        # ── Tool calling mode ──
        has_tools = bool(body.tools)
        file_refs = []
        image_refs = []
        if has_tools:
            # Use expert model for tool calling, with auto-fallback to think if degraded
            requested_deep_think = CHAT_MODELS["doubao-expert"]
            use_deep_think, model_name = _expert_tracker.get_effective_mode(requested_deep_think)
            # Convert messages with tool definitions injected
            messages_raw = [m.model_dump(exclude_none=True) for m in body.messages]
            prompt = convert_messages_with_tools(messages_raw, body.tools)
        else:
            requested_model = (body.model or "").strip()
            target_model = requested_model
            if target_model not in CHAT_MODELS:
                if target_model in MODEL_ALIASES:
                    target_model = MODEL_ALIASES[target_model]
                    log.info("Model alias mapped: '%s' -> '%s'", requested_model, target_model)
                else:
                    log.warning("Unknown model '%s', automatically falling back to default '%s'", requested_model, DEFAULT_FALLBACK_MODEL)
                    target_model = DEFAULT_FALLBACK_MODEL

            use_deep_think = CHAT_MODELS.get(target_model, 0)
            model_name = requested_model or target_model
            # Allow enable_thinking or reasoning_effort to dynamically control thinking mode
            if body.enable_thinking is True or (body.reasoning_effort and body.reasoning_effort in ("medium", "high")):
                use_deep_think = 1 if use_deep_think != 3 else 3
            elif body.enable_thinking is False or (body.reasoning_effort and body.reasoning_effort == "none"):
                use_deep_think = 0

            prompt, file_refs, image_refs = _extract_prompt_and_file_refs(body.messages)
            if not prompt and not file_refs and not image_refs:
                raise HTTPException(status_code=400, detail="No text content")
            if not prompt:
                prompt = "请描述这张图片" if image_refs else "请总结或分析所附带的文件内容"

        await bucket.acquire()
        client = _get_client()
        bound_resource_ids = [
            key for ref in image_refs + file_refs
            if (key := _tos_resource_id(ref)) is not None
        ]
        _check_account_bound_resources(client, bound_resource_ids)
        account_bound_inputs = bool(bound_resource_ids)

        # ── 3-Tier Captcha Defense & Account Failover ──
        recovered_account_id = await _resolve_captcha_or_failover(
            client,
            request,
            allow_account_switch=not bool(body.conversation_id) and not account_bound_inputs,
            feature="chat",
        )
        account_id = await _select_account_for_request(
            client,
            "chat",
            conversation_id=body.conversation_id,
            account_bound_inputs=account_bound_inputs,
            preferred_account_id=recovered_account_id,
        )

        # ── Request Dispatch Smoothing (anti-burst rate limit) ──
        global _last_dispatch_time
        async with _dispatch_lock:
            now = time.time()
            gap = now - _last_dispatch_time
            if gap < _MIN_DISPATCH_INTERVAL:
                await asyncio.sleep(_MIN_DISPATCH_INTERVAL - gap)
            _last_dispatch_time = time.time()

        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        # Determine whether to auto-delete ephemeral conversation after completion (Scheme B: 即用即焚)
        header_del = request.headers.get("x-auto-delete")
        if header_del is not None:
            auto_delete = header_del.lower() in ("true", "1", "yes")
        else:
            auto_delete = _auto_delete_conv and (not body.conversation_id)

        # Materialize image and file references if provided
        image_attachments = None
        file_attachments = None
        if image_refs:
            image_attachments = await _materialize_image_refs(client, image_refs)
        if file_refs:
            file_attachments, text_ctx = await _materialize_file_refs(client, file_refs)
            if text_ctx:
                prompt = f"{prompt}\n\n{text_ctx}" if prompt else text_ctx

        if body.stream:
            return StreamingResponse(
                _stream_chat(client, prompt, use_deep_think, request_id, model_name,
                             conversation_id=body.conversation_id, bot_id=body.bot_id,
                             has_tools=has_tools,
                             messages_for_counting=body.messages,
                             auto_delete=auto_delete,
                             admin_url=_admin_url(request),
                             allow_account_switch=(
                                 not bool(body.conversation_id)
                                 and _attachments_reuploadable(image_attachments, file_attachments)
                             ),
                             image_attachments=image_attachments,
                             file_attachments=file_attachments),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # Non-streaming: collect all chunks with thinking state machine
        try:
            if has_tools:
                # Tool calling non-streaming path
                message = await _collect_chat_response(
                    client, prompt, use_deep_think,
                    conversation_id=body.conversation_id, bot_id=body.bot_id,
                    admin_url=_admin_url(request),
                    allow_account_switch=(
                        not bool(body.conversation_id)
                        and _attachments_reuploadable(image_attachments, file_attachments)
                    ),
                    image_attachments=image_attachments,
                    file_attachments=file_attachments,
                )
                # Report to expert tracker (detect silent downgrade)
                had_reasoning = bool(message.get("reasoning_content"))
                if use_deep_think >= 1:
                    _expert_tracker.report_response(had_reasoning)
                # Check if response contains tool calls
                content = message.get("content", "")
                parsed_tools = parse_tool_calls_xml(content) if content else None
                if parsed_tools:
                    message = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": parsed_tools,
                    }
                    finish_reason = "tool_calls"
                else:
                    finish_reason = "stop"
            else:
                message = await _collect_chat_response(
                    client, prompt, use_deep_think,
                    conversation_id=body.conversation_id, bot_id=body.bot_id,
                    admin_url=_admin_url(request),
                    allow_account_switch=(
                        not bool(body.conversation_id)
                        and _attachments_reuploadable(image_attachments, file_attachments)
                    ),
                    image_attachments=image_attachments,
                    file_attachments=file_attachments,
                )
                finish_reason = "stop"
        except UpstreamAPIError as exc:
            return JSONResponse(
                exc.upstream_error.to_openai_error(_admin_url(request)),
                status_code=exc.upstream_error.status_code,
            )
        except RuntimeError:
            error = parse_upstream_error(None)
            return JSONResponse(error.to_openai_error(_admin_url(request)), status_code=error.status_code)

        account_manager = getattr(client, "account_manager", None)
        active_account = account_manager.get_active_account() if isinstance(account_manager, AccountManager) else None
        if active_account:
            account_id = active_account.id

        # max_tokens truncation (non-streaming only)
        content = message.get("content") or ""
        if body.max_tokens and content and not message.get("tool_calls"):
            max_chars = int(body.max_tokens * 2.5)  # rough tokens->chars
            if len(content) > max_chars:
                message["content"] = content[:max_chars]
                finish_reason = "length"

        # Token counting
        prompt_tokens = count_messages_tokens(
            [m.model_dump(exclude_none=True) for m in body.messages]
        )
        completion_content = message.get("content") or ""
        reasoning_content = message.get("reasoning_content") or ""
        completion_tokens = count_tokens(completion_content + reasoning_content)
        _bind_conversation_account(message.get("conversation_id"), account_id)

        resp_data = {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        if message.get("conversation_id"):
            resp_data["conversation_id"] = message["conversation_id"]

        # Auto-delete ephemeral conversation in background (即用即焚 / Scheme B)
        if auto_delete and message.get("conversation_id"):
            asyncio.create_task(
                _delayed_delete(client, message["conversation_id"], owner_account_id=account_id)
            )

        return JSONResponse(resp_data)

    @app.post("/v1/images/generations")
    async def image_generations(body: ImageGenerationRequest, request: Request):
        _check_auth(request)
        await bucket.acquire()
        client = _get_client()

        ratio = body.ratio or _size_to_ratio(body.size)
        ref_image_keys = []
        if body.ref_image_keys:
            ref_image_keys.extend([str(k).strip() for k in body.ref_image_keys if str(k).strip()])
        if body.ref_image_key and body.ref_image_key not in ref_image_keys:
            ref_image_keys.insert(0, body.ref_image_key)

        img_inputs = []
        for val in (body.images, body.image_urls, body.image, body.image_url):
            if isinstance(val, list):
                img_inputs.extend(val)
            elif val:
                img_inputs.append(val)

        bound_resource_ids = [
            key for item in ref_image_keys + img_inputs
            if (key := _tos_resource_id(item)) is not None
        ]
        _check_account_bound_resources(client, bound_resource_ids)
        account_bound_inputs = bool(bound_resource_ids)
        recovered_account_id = await _resolve_captcha_or_failover(
            client, request,
            allow_account_switch=not account_bound_inputs,
            feature="image",
        )
        await _select_account_for_request(
            client, "image", account_bound_inputs=account_bound_inputs,
            preferred_account_id=recovered_account_id,
        )

        if img_inputs:
            uploaded_keys = await _resolve_multi_images_to_keys(client, img_inputs)
            for uk in uploaded_keys:
                if uk not in ref_image_keys:
                    ref_image_keys.append(uk)

        ref_image_key = ref_image_keys[0] if ref_image_keys else None

        gen_kwargs = {}
        if len(ref_image_keys) > 1:
            gen_kwargs["ref_image_keys"] = ref_image_keys

        header_save = request.headers.get("x-save-history", "").lower() in ("true", "1", "yes")
        header_no_del = request.headers.get("x-auto-delete", "").lower() in ("false", "0", "no")
        if body.memory_type is not None:
            gen_kwargs["memory_type"] = body.memory_type
        elif body.save_history is not None:
            gen_kwargs["save_history"] = body.save_history
        elif header_save or header_no_del:
            gen_kwargs["save_history"] = True
        elif not _auto_delete_conv:
            gen_kwargs["memory_type"] = 0

        try:
            result = await client.generate_image(
                prompt=body.prompt,
                ratio=ratio,
                ref_image_key=ref_image_key,
                **gen_kwargs,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

        images = result.get("images", [])
        if not images:
            raise HTTPException(
                status_code=502, detail="No images generated"
            )

        data = []
        for img in images:
            data.append({
                "url": img["url"],
                "revised_prompt": body.prompt,
            })
            _record_recent_media("image", img["url"], body.prompt, model=body.model)

        return JSONResponse({
            "created": int(time.time()),
            "data": data,
        })

    @app.post("/v1/images/edits")
    async def image_edits(request: Request):
        """OpenAI-compatible image edits (img2img). Supports multipart/form-data and JSON."""
        _check_auth(request)
        await bucket.acquire()
        client = _get_client()

        content_type = request.headers.get("content-type", "")
        prompt = ""
        size = "1024x1024"
        ratio = None
        ref_image_key = None

        save_history = None
        memory_type = None

        ref_image_keys = []
        image_items = []
        image_sources: list[dict[str, Any]] = []
        if "multipart/form-data" in content_type:
            form = await request.form()
            prompt = str(form.get("prompt", "")).strip()
            size = form.get("size") or "1024x1024"
            ratio = form.get("ratio")
            raw_keys = form.getlist("ref_image_keys") or form.getlist("ref_image_key")
            for rk in raw_keys:
                if isinstance(rk, str):
                    for k in rk.split(","):
                        k = k.strip()
                        if k and k not in ref_image_keys:
                            ref_image_keys.append(k)
            if form.get("save_history") is not None:
                save_history = str(form.get("save_history")).lower() in ("true", "1", "yes")
            if form.get("memory_type") is not None:
                try:
                    memory_type = int(form.get("memory_type"))
                except ValueError:
                    pass
            for field_name in ("images", "image", "files", "file"):
                for it in form.getlist(field_name):
                    if it and it not in image_items:
                        image_items.append(it)
        else:
            body = await request.json()
            prompt = str(body.get("prompt", "")).strip()
            size = body.get("size") or "1024x1024"
            ratio = body.get("ratio")
            raw_keys = body.get("ref_image_keys") or body.get("ref_image_key")
            if isinstance(raw_keys, str):
                for k in raw_keys.split(","):
                    k = k.strip()
                    if k and k not in ref_image_keys:
                        ref_image_keys.append(k)
            elif isinstance(raw_keys, list):
                for k in raw_keys:
                    k = str(k).strip()
                    if k and k not in ref_image_keys:
                        ref_image_keys.append(k)
            if body.get("save_history") is not None:
                save_history = bool(body.get("save_history"))
            if body.get("memory_type") is not None:
                try:
                    memory_type = int(body.get("memory_type"))
                except ValueError:
                    pass
            for field_name in ("images", "image_urls", "ref_images", "image", "image_url", "ref_image"):
                val = body.get(field_name)
                if isinstance(val, list):
                    image_items.extend(val)
                elif val:
                    image_items.append(val)

        if not prompt:
            raise HTTPException(status_code=400, detail="Missing prompt")
        bound_resource_ids = [
            key for item in ref_image_keys + image_items
            if (key := _tos_resource_id(item)) is not None
        ]
        _check_account_bound_resources(client, bound_resource_ids)
        account_bound_inputs = bool(bound_resource_ids)
        recovered_account_id = await _resolve_captcha_or_failover(
            client, request,
            allow_account_switch=not account_bound_inputs,
            feature="image",
        )
        await _select_account_for_request(
            client, "image", account_bound_inputs=account_bound_inputs,
            preferred_account_id=recovered_account_id,
        )
        if image_items:
            uploaded_keys = await _resolve_multi_images_to_keys(client, image_items, sources=image_sources)
            for uk in uploaded_keys:
                if uk not in ref_image_keys:
                    ref_image_keys.append(uk)
        if not ref_image_keys:
            raise HTTPException(status_code=400, detail="Missing reference image (image or ref_image_key)")

        actual_ratio = ratio or _size_to_ratio(size)
        ref_image_key = ref_image_keys[0]

        gen_kwargs = {}
        if len(ref_image_keys) > 1:
            gen_kwargs["ref_image_keys"] = ref_image_keys

        header_save = request.headers.get("x-save-history", "").lower() in ("true", "1", "yes")
        header_no_del = request.headers.get("x-auto-delete", "").lower() in ("false", "0", "no")
        if memory_type is not None:
            gen_kwargs["memory_type"] = memory_type
        elif save_history is not None:
            gen_kwargs["save_history"] = save_history
        elif header_save or header_no_del:
            gen_kwargs["save_history"] = True
        elif not _auto_delete_conv:
            gen_kwargs["memory_type"] = 0

        try:
            result = await client.generate_image(
                prompt=prompt,
                ratio=actual_ratio,
                ref_image_key=ref_image_key,
                **gen_kwargs,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

        images = result.get("images", [])
        if not images:
            raise HTTPException(status_code=502, detail="No images generated")

        data = []
        for img in images:
            data.append({
                "url": img["url"],
                "revised_prompt": prompt,
            })
            _record_recent_media("image", img["url"], prompt, model="doubao-image-edit")

        return JSONResponse({
            "created": int(time.time()),
            "data": data,
        })


    @app.post("/v1/audio/generations")
    async def audio_generations(request: Request):
        _check_auth(request)
        await bucket.acquire()
        client = _get_client()

        body = await request.json()
        prompt = body.get("prompt", "")
        if not prompt:
            raise HTTPException(status_code=400, detail="Missing prompt")

        save_history = body.get("save_history")
        memory_type = body.get("memory_type")
        gen_kwargs = {}
        header_save = request.headers.get("x-save-history", "").lower() in ("true", "1", "yes")
        header_no_del = request.headers.get("x-auto-delete", "").lower() in ("false", "0", "no")
        if memory_type is not None:
            gen_kwargs["memory_type"] = memory_type
        elif save_history is not None:
            gen_kwargs["save_history"] = save_history
        elif header_save or header_no_del:
            gen_kwargs["save_history"] = True
        elif not _auto_delete_conv:
            gen_kwargs["memory_type"] = 0

        try:
            result = await client.generate_music(
                prompt=prompt,
                lyric=body.get("lyric"),
                genre=body.get("genre"),
                **gen_kwargs,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

        tracks = result.get("tracks", [])
        if not tracks:
            raise HTTPException(
                status_code=502, detail="No music tracks generated"
            )

        return JSONResponse({
            "created": int(time.time()),
            "data": tracks,
        })

    @app.post("/v1/video/generations")
    async def video_generations(request: Request):
        _check_auth(request)
        await bucket.acquire()
        client = _get_client()

        content_type = request.headers.get("content-type", "")
        prompt = ""
        ratio = None
        ref_image_key = None

        save_history = None
        memory_type = None

        ref_image_keys = []
        image_items = []
        image_sources: list[dict[str, Any]] = []
        if "multipart/form-data" in content_type:
            form = await request.form()
            prompt = str(form.get("prompt", "")).strip()
            ratio = form.get("ratio") or form.get("size")
            raw_keys = form.getlist("ref_image_keys") or form.getlist("ref_image_key")
            for rk in raw_keys:
                if isinstance(rk, str):
                    for k in rk.split(","):
                        k = k.strip()
                        if k and k not in ref_image_keys:
                            ref_image_keys.append(k)
            if form.get("save_history") is not None:
                save_history = str(form.get("save_history")).lower() in ("true", "1", "yes")
            if form.get("memory_type") is not None:
                try:
                    memory_type = int(form.get("memory_type"))
                except ValueError:
                    pass
            for field_name in ("images", "image", "files", "file"):
                for it in form.getlist(field_name):
                    if it and it not in image_items:
                        image_items.append(it)
        else:
            body = await request.json()
            prompt = str(body.get("prompt", "")).strip()
            ratio = body.get("ratio") or body.get("size")
            raw_keys = body.get("ref_image_keys") or body.get("ref_image_key")
            if isinstance(raw_keys, str):
                for k in raw_keys.split(","):
                    k = k.strip()
                    if k and k not in ref_image_keys:
                        ref_image_keys.append(k)
            elif isinstance(raw_keys, list):
                for k in raw_keys:
                    k = str(k).strip()
                    if k and k not in ref_image_keys:
                        ref_image_keys.append(k)
            if body.get("save_history") is not None:
                save_history = bool(body.get("save_history"))
            if body.get("memory_type") is not None:
                try:
                    memory_type = int(body.get("memory_type"))
                except ValueError:
                    pass
            for field_name in ("images", "image_urls", "ref_images", "image", "image_url", "ref_image"):
                val = body.get(field_name)
                if isinstance(val, list):
                    image_items.extend(val)
                elif val:
                    image_items.append(val)

        if not prompt:
            raise HTTPException(status_code=400, detail="Missing prompt")

        bound_resource_ids = [
            key for item in ref_image_keys + image_items
            if (key := _tos_resource_id(item)) is not None
        ]
        _check_account_bound_resources(client, bound_resource_ids)
        account_bound_inputs = bool(bound_resource_ids)
        recovered_account_id = await _resolve_captcha_or_failover(
            client, request,
            allow_account_switch=not account_bound_inputs,
            feature="video",
        )
        await _select_account_for_request(
            client, "video", account_bound_inputs=account_bound_inputs,
            preferred_account_id=recovered_account_id,
        )
        if image_items:
            uploaded_keys = await _resolve_multi_images_to_keys(client, image_items, sources=image_sources)
            for uk in uploaded_keys:
                if uk not in ref_image_keys:
                    ref_image_keys.append(uk)

        if ratio and "x" in str(ratio):
            ratio = _size_to_ratio(ratio)

        ref_image_key = ref_image_keys[0] if ref_image_keys else None

        history_kwargs = {}
        header_save = request.headers.get("x-save-history", "").lower() in ("true", "1", "yes")
        header_no_del = request.headers.get("x-auto-delete", "").lower() in ("false", "0", "no")
        if memory_type is not None:
            history_kwargs["memory_type"] = memory_type
        elif save_history is not None:
            history_kwargs["save_history"] = save_history
        elif header_save or header_no_del:
            history_kwargs["save_history"] = True
        elif not _auto_delete_conv:
            history_kwargs["memory_type"] = 0

        manager = getattr(client, "account_manager", None)
        attempted_account_ids: set[str] = set()
        while True:
            active = manager.get_active_account() if isinstance(manager, AccountManager) else None
            if active:
                attempted_account_ids.add(active.id)
            call_kwargs = dict(history_kwargs)
            if len(ref_image_keys) > 1:
                call_kwargs["ref_image_keys"] = list(ref_image_keys)
            try:
                result = await client.generate_video(
                    prompt=prompt,
                    ratio=ratio,
                    ref_image_key=ref_image_key,
                    **call_kwargs,
                )
                break
            except VideoQuotaExceeded:
                if not isinstance(manager, AccountManager):
                    raise HTTPException(status_code=429, detail="Video generation quota exceeded.")
                candidate = manager.get_next_available_account(
                    "video", exclude_ids=attempted_account_ids
                )
                if candidate is None:
                    raise HTTPException(status_code=429, detail="Video quota is exhausted for all available accounts.")
                source_by_key = {source["uri"]: source for source in image_sources}
                if any(
                    key not in source_by_key
                    or not isinstance(source_by_key[key].get("source_bytes"), bytes)
                    for key in ref_image_keys
                ):
                    raise HTTPException(
                        status_code=409,
                        detail="Video quota failover cannot reuse account-bound TOS references; provide source image bytes.",
                    )
                while candidate is not None:
                    attempted_account_ids.add(candidate.id)
                    try:
                        switched = await client.switch_to_account(candidate.id)
                    except Exception as exc:
                        log.warning("Video quota failover switch failed (%s)", type(exc).__name__)
                        switched = False
                    if switched:
                        break
                    candidate = manager.get_next_available_account(
                        "video", exclude_ids=attempted_account_ids
                    )
                if candidate is None:
                    raise HTTPException(status_code=429, detail="Video quota is exhausted for all available accounts.")
                ref_image_keys = await _reupload_image_sources(client, ref_image_keys, image_sources)
                ref_image_key = ref_image_keys[0] if ref_image_keys else None
            except RuntimeError as exc:
                raise HTTPException(status_code=502, detail=str(exc))

        videos = result.get("videos", [])
        msg = result.get("message", "")
        if not videos and msg:
            return JSONResponse({"created": int(time.time()), "data": [], "message": msg})
        if not videos:
            raise HTTPException(status_code=502, detail="No videos generated")

        for vid in videos:
            _record_recent_media("video", vid.get("video_url", ""), prompt, cover_url=vid.get("cover_url", ""), model="doubao-video")

        return JSONResponse({
            "created": int(time.time()),
            "data": videos,
        })

    @app.post("/v1/files")
    async def upload_file(request: Request):
        """Upload a file. Returns file metadata for use in chat."""
        _check_auth(request)
        await bucket.acquire()
        client = _get_client()

        form = await request.form()
        file_field = form.get("file")
        if not file_field:
            raise HTTPException(status_code=400, detail="Missing file field")

        filename = file_field.filename or "file.txt"
        ext = os.path.splitext(filename)[1].lower()
        allowed_exts = {
            ".txt", ".pdf", ".docx", ".doc", ".csv", ".xlsx", ".xls", ".pptx", ".ppt",
            ".md", ".json", ".xml", ".yaml", ".yml", ".html", ".htm",
            ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp",
            ".mp3", ".wav", ".m4a", ".ogg", ".aac",
            ".mp4", ".mov", ".avi", ".webm", ".mkv",
        }
        if ext and ext not in allowed_exts:
            raise HTTPException(status_code=400, detail=f"Unsupported file extension: {ext}")

        max_upload_bytes = int(os.environ.get("MAX_UPLOAD_SIZE_MB", "20")) * 1024 * 1024
        file_data = await file_field.read()
        if len(file_data) > max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum allowed size ({max_upload_bytes // (1024*1024)}MB)",
            )

        try:
            result = await client.upload_file(file_data, filename)
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        _register_account_resource(client, result["uri"])

        return JSONResponse({
            "id": result["uri"],
            "object": "file",
            "filename": result["name"],
            "bytes": result["size"],
            "uri": result["uri"],
            "file_type": result.get("file_type", ""),
            "purpose": "assistants",
        })


    @app.get("/v1/files/download")
    async def file_download(request: Request, uri: str, expire: int = 3600):
        _check_auth(request)
        await bucket.acquire()
        client = _get_client()
        _check_account_bound_resources(client, [uri] if _tos_resource_id(uri) else [])
        try:
            url = await client.get_file_download_url(uri=uri, expire_seconds=expire)
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        return JSONResponse({"url": url, "uri": uri, "expires_in": expire})

    @app.post("/v1/images/upload")
    async def upload_image(request: Request):
        _check_auth(request)
        await bucket.acquire()
        client = _get_client()
        form = await request.form()
        upload = form.get("file") or form.get("image")
        if not upload:
            raise HTTPException(status_code=400, detail="Missing file field")
        filename = upload.filename or "image.png"
        ext = os.path.splitext(filename)[1].lower()
        if ext and ext not in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"):
            raise HTTPException(status_code=400, detail=f"Unsupported image extension: {ext}")

        max_upload_bytes = int(os.environ.get("MAX_UPLOAD_SIZE_MB", "20")) * 1024 * 1024
        image_data = await upload.read()
        if len(image_data) > max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Image exceeds maximum allowed size ({max_upload_bytes // (1024*1024)}MB)",
            )

        try:
            result = await client.upload_image(image_bytes=image_data, filename=filename)
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        _register_account_resource(client, result.get("uri", ""))
        return JSONResponse({
            "uri": result["uri"],
            "cdn_url": result["cdn_url"],
            "url": result["cdn_url"],
            "name": result["name"],
            "format": result["format"],
            "width": result["width"],
            "height": result["height"],
        })

    @app.post("/v1/chat/completions/with-file")
    async def chat_with_file(request: Request):
        """Chat with file attachment. Body: {file_id, prompt, model}."""
        _check_auth(request)
        await bucket.acquire()
        client = _get_client()

        body = await request.json()
        file_id = body.get("file_id", "")
        prompt = body.get("prompt", "")
        file_name = body.get("file_name", "file.txt")
        file_size = body.get("file_size", 0)
        model = body.get("model", "doubao")

        if not file_id or not prompt:
            raise HTTPException(status_code=400, detail="Missing file_id or prompt")

        use_deep_think = CHAT_MODELS.get(model, 0)
        _check_account_bound_resources(client, [file_id] if _tos_resource_id(file_id) else [])
        recovered_account_id = await _resolve_captcha_or_failover(
            client, request, allow_account_switch=False, feature="chat"
        )
        await _select_account_for_request(
            client, "chat", account_bound_inputs=True,
            preferred_account_id=recovered_account_id,
        )

        try:
            result = await client.chat_with_file(
                text=prompt,
                file_uri=file_id,
                file_name=file_name,
                file_size=file_size,
                use_deep_think=use_deep_think,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        return JSONResponse({
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result["text"]},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    async def _delayed_delete(
        client: BrowserClient,
        conversation_id: str,
        delay: float = 0.8,
        *,
        owner_account_id: Optional[str] = None,
    ):
        """Asynchronously delete ephemeral conversation in background (即用即焚)."""
        try:
            if delay > 0:
                await asyncio.sleep(delay)
            manager = getattr(client, "account_manager", None)
            gate = getattr(client, "operation_gate", None)
            if isinstance(gate, BrowserOperationGate):
                async with gate.fresh_lease():
                    active = manager.get_active_account() if isinstance(manager, AccountManager) else None
                    if owner_account_id and (not active or active.id != owner_account_id):
                        return
                    deleted = await client.delete_conversation(conversation_id)
            else:
                deleted = await client.delete_conversation(conversation_id)
            if deleted:
                _conversation_account_ids.pop(conversation_id, None)
        except Exception as exc:
            log.warning("Delayed delete failed (%s)", type(exc).__name__)

    async def _collect_chat_response(
        client: BrowserClient,
        prompt: str,
        use_deep_think: int,
        *,
        conversation_id: Optional[str] = None,
        bot_id: Optional[str] = None,
        image_attachments: Optional[list[dict[str, Any]]] = None,
        file_attachments: Optional[list[dict[str, Any]]] = None,
        admin_url: Optional[str] = None,
        allow_account_switch: bool = True,
        _retries_remaining: int = 1,
    ) -> dict:
        """Collect full chat response with thinking separation.

        Returns an OpenAI message dict:
        {"role": "assistant", "content": "...", "reasoning_content": "..."}
        reasoning_content is only present when thinking was detected.
        """
        thinking_count = 0
        in_thinking = False
        thinking_parts: list = []
        content_parts: list = []
        result_conversation_id: Optional[str] = None
        saw_upstream_output = False
        manager = getattr(client, "account_manager", None)
        initial_account = manager.get_active_account() if isinstance(manager, AccountManager) else None
        initial_account_id = initial_account.id if initial_account else None

        def _iter_blocks(data: dict):
            for patch in data.get("patch_op", []):
                pv = patch.get("patch_value", {})
                yield from pv.get("content_block", [])
            dc = data.get("content", {})
            if isinstance(dc, dict):
                yield from dc.get("content_block", [])

        event_iter = client.chat_completion(
            prompt, use_deep_think=use_deep_think,
            conversation_id=conversation_id or None,
            bot_id=bot_id or None,
            image_attachments=image_attachments,
            file_attachments=file_attachments,
        ).__aiter__()
        async for event in event_iter:
            if _event_has_output(event):
                saw_upstream_output = True
            event_type = event.get("_event", "")
            if event.get("error") or event.get("error_code") or event_type == "STREAM_ERROR":
                error = parse_upstream_error(event, status_code=event.get("status"))
                close = getattr(event_iter, "aclose", None)
                if close:
                    await close()
                can_attempt_retry = (
                    _retries_remaining > 0
                    and not saw_upstream_output
                    and not content_parts
                    and not thinking_parts
                    and not _event_has_output(event)
                )
                recovered = await _recover_upstream_error(
                    client,
                    error,
                    admin_url or "",
                    allow_account_switch=allow_account_switch and can_attempt_retry,
                    feature="chat",
                )
                can_retry = (
                    can_attempt_retry
                    and (not conversation_id or recovered != "account_switch")
                )
                if recovered and can_retry:
                    active = manager.get_active_account() if isinstance(manager, AccountManager) else None
                    if active and active.id != initial_account_id:
                        await _reupload_chat_attachments(client, image_attachments, file_attachments)
                    return await _collect_chat_response(
                        client,
                        prompt,
                        use_deep_think,
                        conversation_id=conversation_id,
                        bot_id=bot_id,
                        image_attachments=image_attachments,
                        file_attachments=file_attachments,
                        admin_url=admin_url,
                        allow_account_switch=allow_account_switch,
                        _retries_remaining=_retries_remaining - 1,
                    )
                raise UpstreamAPIError(error)

            # Extract conversation_id for multi-turn
            if not result_conversation_id:
                cid = client.extract_conversation_id(event)
                if cid and cid != "0":
                    result_conversation_id = cid

            event_type = event.get("_event", "")

            # CHUNK_DELTA
            if (
                event_type == "CHUNK_DELTA"
                and "text" in event
                and isinstance(event.get("text"), str)
                and event["text"]
            ):
                if in_thinking:
                    thinking_parts.append(event["text"])
                else:
                    content_parts.append(event["text"])
                continue

            # content_block
            has_content_block = False
            for cb in _iter_blocks(event):
                has_content_block = True
                bt = cb.get("block_type", 0)
                block_content = cb.get("content", {})

                if bt == 10040:
                    thinking_count += 1
                    in_thinking = (thinking_count == 1)
                elif bt == 10000:
                    tb = block_content.get("text_block", {})
                    if isinstance(tb, dict) and tb.get("text"):
                        if in_thinking:
                            thinking_parts.append(tb["text"])
                        else:
                            content_parts.append(tb["text"])

            # patch_op content string fallback
            if not has_content_block:
                for patch in event.get("patch_op", []):
                    pv = patch.get("patch_value", {})
                    if isinstance(pv, dict) and "content" in pv:
                        content_str = pv.get("content", "")
                        if isinstance(content_str, str) and content_str:
                            try:
                                obj = json.loads(content_str)
                                t = obj.get("text", "")
                                if t:
                                    if in_thinking:
                                        thinking_parts.append(t)
                                    else:
                                        content_parts.append(t)
                            except (json.JSONDecodeError, TypeError):
                                pass

        message: dict = {"role": "assistant", "content": "".join(content_parts)}
        if thinking_parts:
            message["reasoning_content"] = "".join(thinking_parts)
        if result_conversation_id:
            message["conversation_id"] = result_conversation_id
        client.record_success()
        return message

    async def _stream_chat(
        client: BrowserClient,
        prompt: str,
        use_deep_think: int,
        request_id: str,
        model: str,
        *,
        conversation_id: Optional[str] = None,
        bot_id: Optional[str] = None,
        has_tools: bool = False,
        messages_for_counting: Optional[list] = None,
        auto_delete: bool = False,
        image_attachments: Optional[list[dict[str, Any]]] = None,
        file_attachments: Optional[list[dict[str, Any]]] = None,
        admin_url: Optional[str] = None,
        allow_account_switch: bool = True,
        _retry_count: int = 0,
    ):
        """Generate real-time SSE stream in OpenAI format via httpx streaming.

        Thinking state machine (mirrors old client.py logic):
        - block_type=10040 toggles thinking mode (1st=enter, 2nd=exit)
        - Text between markers -> delta.reasoning_content
        - Text after exit -> delta.content
        - block_type=10025 -> delta.search_results (incremental)
        - error_code in event -> emit error and stop
        """
        thinking_count = 0
        in_thinking = False
        had_reasoning_content = False  # Track if any reasoning was emitted
        stream_content_chars = 0  # Track total output chars for token estimation
        # Track last emitted result count per block_id for incremental updates
        search_last_count: dict = {}
        result_conversation_id: Optional[str] = None
        # Tool calling state
        tool_buffer = ""  # accumulates text when tool call detected
        tool_mode = False  # True when we're buffering potential tool call XML
        emitted_tool_calls = False  # True once we've emitted tool_calls chunks
        stream_output_started = False
        pending_event_task = None
        event_iter = None

        def _make_chunk(delta: dict, finish_reason=None):
            nonlocal stream_output_started
            if any(key in delta for key in ("content", "reasoning_content", "tool_calls", "search_results")):
                stream_output_started = True
            return {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }],
            }

        def _iter_blocks(data: dict):
            """Yield content_block dicts from patch_op or top-level content."""
            for patch in data.get("patch_op", []):
                pv = patch.get("patch_value", {})
                yield from pv.get("content_block", [])
            dc = data.get("content", {})
            if isinstance(dc, dict):
                yield from dc.get("content_block", [])

        try:
            event_iter = client.chat_completion(
                prompt, use_deep_think=use_deep_think,
                conversation_id=conversation_id or None,
                bot_id=bot_id or None,
                image_attachments=image_attachments,
                file_attachments=file_attachments,
            ).__aiter__()

            while True:
                if pending_event_task is None:
                    pending_event_task = asyncio.ensure_future(event_iter.__anext__())
                done, _ = await asyncio.wait({pending_event_task}, timeout=5.0)
                if not done:
                    # Emit SSE comment frame to keep connection alive through intermediate proxies
                    yield ": keep-alive\n\n"
                    continue
                try:
                    event = pending_event_task.result()
                except StopAsyncIteration:
                    pending_event_task = None
                    break
                pending_event_task = None

                if _event_has_output(event):
                    stream_output_started = True

                event_type = event.get("_event", "")
                if event.get("error") or event.get("error_code") or event_type == "STREAM_ERROR":
                    error = parse_upstream_error(event, status_code=event.get("status"))
                    close = getattr(event_iter, "aclose", None)
                    if close:
                        await close()
                    event_iter = None
                    can_attempt_retry = _retry_count < 1 and not stream_output_started
                    recovered = await _recover_upstream_error(
                        client,
                        error,
                        admin_url or "",
                        allow_account_switch=allow_account_switch and can_attempt_retry,
                        feature="chat",
                    )
                    can_retry = (
                        recovered is not None
                        and can_attempt_retry
                        and (not conversation_id or recovered != "account_switch")
                    )
                    if can_retry:
                        manager = getattr(client, "account_manager", None)
                        active = manager.get_active_account() if isinstance(manager, AccountManager) else None
                        initial = _conversation_account_ids.get(conversation_id or "")
                        if active and initial and active.id != initial:
                            await _reupload_chat_attachments(client, image_attachments, file_attachments)
                        elif active and not conversation_id and recovered == "account_switch":
                            await _reupload_chat_attachments(client, image_attachments, file_attachments)
                        async for frame in _stream_chat(
                            client,
                            prompt,
                            use_deep_think,
                            request_id,
                            model,
                            conversation_id=conversation_id,
                            bot_id=bot_id,
                            has_tools=has_tools,
                            messages_for_counting=messages_for_counting,
                            auto_delete=auto_delete,
                            image_attachments=image_attachments,
                            file_attachments=file_attachments,
                            admin_url=admin_url,
                            allow_account_switch=allow_account_switch,
                            _retry_count=_retry_count + 1,
                        ):
                            yield frame
                        return
                    yield f"data: {json.dumps(error.to_openai_error(admin_url), ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                # --- Extract conversation_id for multi-turn ---
                if not result_conversation_id:
                    cid = client.extract_conversation_id(event)
                    if cid and cid != "0":
                        result_conversation_id = cid

                # --- CHUNK_DELTA: compact {"text": "..."} (highest priority) ---
                if (
                    event_type == "CHUNK_DELTA"
                    and "text" in event
                    and isinstance(event.get("text"), str)
                    and event["text"]
                ):
                    t = event["text"]
                    # Tool calling: buffer text to detect XML tool_calls
                    if has_tools and not in_thinking:
                        tool_buffer += t
                        if not tool_mode and is_tool_call_start(tool_buffer):
                            tool_mode = True
                        if tool_mode:
                            # Check if we have complete tool calls
                            if has_complete_tool_calls(tool_buffer):
                                # Parse and emit as tool_calls
                                parsed = parse_tool_calls_xml(tool_buffer)
                                if parsed:
                                    # Emit tool_calls in OpenAI streaming format
                                    for idx, tc in enumerate(parsed):
                                        # First chunk: role + tool_call with function name
                                        delta_tc = {
                                            "role": "assistant",
                                            "content": None,
                                            "tool_calls": [{
                                                "index": idx,
                                                "id": tc["id"],
                                                "type": "function",
                                                "function": {
                                                    "name": tc["function"]["name"],
                                                    "arguments": "",
                                                },
                                            }],
                                        }
                                        chunk = _make_chunk(delta_tc)
                                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                                        # Second chunk: arguments content
                                        delta_args = {
                                            "tool_calls": [{
                                                "index": idx,
                                                "function": {
                                                    "arguments": tc["function"]["arguments"],
                                                },
                                            }],
                                        }
                                        chunk = _make_chunk(delta_args)
                                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                                    tool_buffer = ""
                                    tool_mode = False
                                    emitted_tool_calls = True
                                else:
                                    # XML complete but parse failed — flush as content
                                    log.warning("Tool call XML parse failed, flushing as content")
                                    delta = {"role": "assistant", "content": tool_buffer}
                                    chunk = _make_chunk(delta)
                                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                                    tool_buffer = ""
                                    tool_mode = False
                            continue  # don't emit raw text while in tool mode
                        else:
                            # Not a tool call start — flush buffer as normal content
                            if len(tool_buffer) > 20 and not is_tool_call_start(tool_buffer):
                                delta = {"role": "assistant", "content": tool_buffer}
                                chunk = _make_chunk(delta)
                                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                                tool_buffer = ""
                            elif not tool_buffer.strip().startswith("<"):
                                # Definitely not XML, flush immediately
                                delta = {"role": "assistant", "content": tool_buffer}
                                chunk = _make_chunk(delta)
                                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                                tool_buffer = ""
                            continue
                    # Normal (non-tool) path
                    if in_thinking:
                        delta = {"reasoning_content": t}
                        had_reasoning_content = True
                    else:
                        delta = {"role": "assistant", "content": t}
                    stream_content_chars += len(t)
                    chunk = _make_chunk(delta)
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    continue

                # --- Process content_block arrays for markers & search ---
                has_content_block = False
                for cb in _iter_blocks(event):
                    has_content_block = True
                    bt = cb.get("block_type", 0)
                    block_content = cb.get("content", {})

                    if bt == 10040:
                        thinking_count += 1
                        in_thinking = (thinking_count == 1)
                        continue

                    if bt == 10025:
                        sqrb = block_content.get(
                            "search_query_result_block", {}
                        )
                        if sqrb:
                            block_id = cb.get("block_id", "")
                            queries = sqrb.get("queries", [])
                            results = sqrb.get("results", [])
                            parsed = [
                                {
                                    "title": r.get("text_card", {}).get("title", ""),
                                    "url": r.get("text_card", {}).get("url", ""),
                                    "summary": r.get("text_card", {}).get("summary", ""),
                                    "source": r.get("text_card", {}).get("source_name", ""),
                                }
                                for r in results if r.get("text_card")
                            ]
                            prev = search_last_count.get(block_id, 0)
                            if (parsed and len(parsed) > prev) or (queries and prev == 0):
                                search_last_count[block_id] = len(parsed)
                                chunk = _make_chunk({
                                    "search_results": {
                                        "queries": queries,
                                        "results": parsed,
                                        "summary": f"搜索 {len(queries)} 个关键词，参考 {len(parsed)} 篇资料",
                                    },
                                })
                                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        continue

                    if bt == 10000:
                        tb = block_content.get("text_block", {})
                        if isinstance(tb, dict) and tb.get("text"):
                            t = tb["text"]
                            # Tool calling: buffer text for XML detection
                            if has_tools and not in_thinking:
                                tool_buffer += t
                                if not tool_mode and is_tool_call_start(tool_buffer):
                                    tool_mode = True
                                if tool_mode:
                                    if has_complete_tool_calls(tool_buffer):
                                        parsed = parse_tool_calls_xml(tool_buffer)
                                        if parsed:
                                            for idx, tc in enumerate(parsed):
                                                delta_tc = {
                                                    "role": "assistant", "content": None,
                                                    "tool_calls": [{"index": idx, "id": tc["id"], "type": "function",
                                                        "function": {"name": tc["function"]["name"], "arguments": ""}}],
                                                }
                                                chunk = _make_chunk(delta_tc)
                                                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                                                delta_args = {"tool_calls": [{"index": idx,
                                                    "function": {"arguments": tc["function"]["arguments"]}}]}
                                                chunk = _make_chunk(delta_args)
                                                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                                            tool_buffer = ""
                                            tool_mode = False
                                            emitted_tool_calls = True
                                        else:
                                            # XML complete but parse failed
                                            delta = {"role": "assistant", "content": tool_buffer}
                                            chunk = _make_chunk(delta)
                                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                                            tool_buffer = ""
                                            tool_mode = False
                                elif len(tool_buffer) > 20 and not is_tool_call_start(tool_buffer):
                                    delta = {"role": "assistant", "content": tool_buffer}
                                    chunk = _make_chunk(delta)
                                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                                    tool_buffer = ""
                                continue
                            if in_thinking:
                                delta = {"reasoning_content": t}
                                had_reasoning_content = True
                            else:
                                delta = {"role": "assistant", "content": t}
                            stream_content_chars += len(t)
                            chunk = _make_chunk(delta)
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        continue

                # --- patch_op content string (only if no content_block found) ---
                if not has_content_block:
                    for patch in event.get("patch_op", []):
                        pv = patch.get("patch_value", {})
                        if isinstance(pv, dict) and "content" in pv:
                            content_str = pv.get("content", "")
                            if isinstance(content_str, str) and content_str:
                                try:
                                    content_obj = json.loads(content_str)
                                    t = content_obj.get("text", "")
                                    if t:
                                        if in_thinking:
                                            delta = {"reasoning_content": t}
                                            had_reasoning_content = True
                                        else:
                                            delta = {"role": "assistant", "content": t}
                                        chunk = _make_chunk(delta)
                                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                                except (json.JSONDecodeError, TypeError):
                                    pass

        except Exception as exc:
            log.error("Stream error (%s)", type(exc).__name__)
            error = parse_upstream_error(None)
            yield f"data: {json.dumps(error.to_openai_error(admin_url), ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return
        finally:
            if pending_event_task is not None:
                if not pending_event_task.done():
                    pending_event_task.cancel()
                await asyncio.gather(pending_event_task, return_exceptions=True)
            if event_iter is not None:
                close = getattr(event_iter, "aclose", None)
                if close:
                    try:
                        await close()
                    except Exception:
                        pass

        # Flush any remaining tool buffer
        if tool_buffer:
            if tool_mode and has_complete_tool_calls(tool_buffer):
                parsed = parse_tool_calls_xml(tool_buffer)
                if parsed:
                    for idx, tc in enumerate(parsed):
                        delta_tc = {
                            "role": "assistant", "content": None,
                            "tool_calls": [{"index": idx, "id": tc["id"], "type": "function",
                                "function": {"name": tc["function"]["name"], "arguments": ""}}],
                        }
                        chunk = _make_chunk(delta_tc)
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        delta_args = {"tool_calls": [{"index": idx,
                            "function": {"arguments": tc["function"]["arguments"]}}]}
                        chunk = _make_chunk(delta_args)
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    emitted_tool_calls = True
                else:
                    # Parse failed — flush as content
                    delta = {"role": "assistant", "content": tool_buffer}
                    chunk = _make_chunk(delta)
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            elif tool_buffer.strip():
                # Emit as regular content
                delta = {"role": "assistant", "content": tool_buffer}
                chunk = _make_chunk(delta)
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        # Final chunk with usage
        client.record_success()
        manager = getattr(client, "account_manager", None)
        active = manager.get_active_account() if isinstance(manager, AccountManager) else None
        _bind_conversation_account(result_conversation_id, active.id if active else None)
        # Report to expert tracker for degradation detection
        if has_tools and use_deep_think >= 1:
            _expert_tracker.report_response(had_reasoning_content)

        # Estimate token usage
        prompt_tokens = 0
        if messages_for_counting:
            prompt_tokens = count_messages_tokens(
                [m.model_dump(exclude_none=True) for m in messages_for_counting]
            )
        completion_tokens = int(stream_content_chars / 2.5 * SAFETY_FACTOR) if stream_content_chars else 0

        final_delta: dict = {}
        if result_conversation_id:
            final_delta["conversation_id"] = result_conversation_id
        final_chunk = _make_chunk(final_delta, 'tool_calls' if emitted_tool_calls else 'stop')
        final_chunk["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

        # Ephemeral conversation auto-delete (Scheme B: 即用即焚)
        if auto_delete and result_conversation_id:
            asyncio.create_task(
                _delayed_delete(
                    client,
                    result_conversation_id,
                    owner_account_id=active.id if active else None,
                )
            )

    # ── Admin Dashboard & Auth ──

    @app.get("/admin", response_class=HTMLResponse)
    async def admin_dashboard(request: Request):
        """Serve the admin dashboard (QR login + system + API test + logs)."""
        from pathlib import Path
        html_path = Path(__file__).parent / "static" / "admin.html"
        html = html_path.read_text(encoding="utf-8")
        auth_required = "true" if bool(api_key) else "false"
        content = html.replace("{{AUTH_REQUIRED}}", auth_required)
        return HTMLResponse(content=content, status_code=200)

    @app.get("/")
    @app.get("/auth")
    async def auth_redirect(request: Request):
        """Redirect / and /auth to /admin for backwards compatibility and easy browser entry."""
        from fastapi.responses import RedirectResponse
        key = request.query_params.get("key", "")
        url = "/admin" + (f"?key={key}" if key else "")
        return RedirectResponse(url=url)

    @app.get("/admin/api/system")
    async def admin_system(request: Request):
        """Return system information."""
        _check_auth(request)
        import platform
        import sys
        uptime = int(time.time() - _SERVER_START_TIME)
        return JSONResponse({
            "python_version": sys.version,
            "platform": platform.platform(),
            "uptime_seconds": uptime,
            "rpm_limit": rpm_limit,
            "host": os.environ.get("DOUBAO_HOST", "0.0.0.0"),
            "port": int(os.environ.get("DOUBAO_PORT", "9090")),
            "models": {
                "chat": list(CHAT_MODELS.keys()),
                "image": ["doubao-image"],
                "video": ["doubao-video"],
                "audio": ["doubao-music"],
            },
            "auto_delete_conv": _auto_delete_conv,
        })

    @app.post("/admin/api/config/auto-delete")
    async def admin_toggle_auto_delete(request: Request):
        """Toggle or set auto_delete_conv mode (即用即焚 / 阅后即焚) at runtime."""
        _check_auth(request)
        global _auto_delete_conv
        try:
            body = await request.json()
        except Exception:
            body = {}
        if "enabled" in body:
            _auto_delete_conv = bool(body["enabled"])
        else:
            _auto_delete_conv = not _auto_delete_conv
        os.environ["DOUBAO_AUTO_DELETE_CONV"] = "true" if _auto_delete_conv else "false"
        log.info("Runtime auto_delete_conv toggled to: %s", _auto_delete_conv)
        return JSONResponse({"status": "ok", "auto_delete_conv": _auto_delete_conv})

    @app.get("/admin/api/logs")
    async def admin_logs(request: Request):
        """Return recent request logs from ring buffer."""
        _check_auth(request)
        return JSONResponse(list(_REQUEST_LOG))

    @app.get("/admin/api/cookies")
    async def admin_cookies(request: Request):
        """Return current browser cookies (masked for security)."""
        _check_auth(request)
        client = _browser.get("client")
        if client is None or client._context is None:
            return JSONResponse({"cookies": [], "total": 0})
        try:
            cookies = await client._context.cookies("https://www.doubao.com")
            cookies_list = [
                {
                    "name": c["name"],
                    "domain": c.get("domain", ""),
                    "path": c.get("path", "/"),
                    "expires": c.get("expires", -1),
                    "length": len(c.get("value", "")),
                    "masked": (c["value"][:3] + "..." + c["value"][-3:]) if len(c.get("value", "")) > 6 else "***",
                }
                for c in cookies
            ]
            return JSONResponse({"cookies": cookies_list, "total": len(cookies_list)})
        except Exception:
            return JSONResponse({"cookies": [], "total": 0})

    def _parse_cookie_payload(raw: Any) -> Dict[str, str]:
        """Parse various cookie input formats into a clean {name: value} dict.

        Supports:
        - dict: {"sessionid": "...", "other": "..."}
        - list of dicts (e.g. from EditThisCookie / Cookie-Editor):
            [{"name": "sessionid", "value": "..."}, ...]
        - JSON string of dict or list: '{"sessionid": "..."}' or '[{"name": "..."}]'
        - Semicolon-separated string: 'sessionid=xxx; other=yyy'
        - Raw sessionid string (length > 10, no '=' or ';')
        """
        cookie_dict: Dict[str, str] = {}
        if isinstance(raw, dict):
            cookie_dict = {str(k).strip(): str(v).strip().strip('"\'') for k, v in raw.items()}
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict) and "name" in item and "value" in item:
                    cookie_dict[str(item["name"]).strip()] = str(item["value"]).strip().strip('"\'')
        elif isinstance(raw, str):
            raw_str = raw.strip()
            if (raw_str.startswith("{") and raw_str.endswith("}")) or (raw_str.startswith("[") and raw_str.endswith("]")):
                try:
                    import json
                    parsed = json.loads(raw_str)
                    return _parse_cookie_payload(parsed)
                except Exception:
                    pass

            if not cookie_dict:
                if "=" not in raw_str and len(raw_str) > 10:
                    cookie_dict["sessionid"] = raw_str.strip('"\'')
                else:
                    for part in raw_str.split(";"):
                        part = part.strip()
                        if not part or "=" not in part:
                            continue
                        k, v = part.split("=", 1)
                        cookie_dict[k.strip()] = v.strip().strip('"\'')

        # Normalize sessionid key case
        for k in list(cookie_dict.keys()):
            if k.lower() == "sessionid" and k != "sessionid":
                cookie_dict["sessionid"] = cookie_dict.pop(k)

        return cookie_dict

    @app.post("/admin/api/cookies/import")
    async def admin_cookies_import(request: Request):
        """Import cookie string or dict into browser context."""
        _check_auth(request)
        client = _browser.get("client")
        if client is None:
            raise HTTPException(status_code=503, detail="Browser not initialized")
        body = await request.json()
        raw = body.get("cookies", "")
        cookie_dict = _parse_cookie_payload(raw)

        ok = await client.inject_cookies_and_reload(cookie_dict)
        return {
            "status": "ok" if ok else "fail",
            "success": ok,
            "logged_in": ok and client.is_ready,
            "cookies_count": len(cookie_dict),
            "has_sessionid": "sessionid" in cookie_dict,
            "ready": client.is_ready,
        }

    # ── Multi-Account Pool Endpoints ──

    @app.get("/admin/api/accounts")
    async def admin_accounts_list(request: Request):
        """List all accounts in the account pool."""
        _check_auth(request)
        client = _browser.get("client")
        am = getattr(client, "account_manager", None)
        if am is None:
            from .account_manager import AccountManager
            am = AccountManager()
        return JSONResponse({
            "strategy": am.strategy,
            "active_account_id": am.active_account_id,
            "accounts": am.list_accounts(mask_cookies=True),
        })

    @app.post("/admin/api/accounts/add")
    async def admin_accounts_add(request: Request):
        """Add or update an account with cookies in the pool."""
        _check_auth(request)
        client = _browser.get("client")
        am = getattr(client, "account_manager", None)
        if am is None:
            from .account_manager import AccountManager
            am = AccountManager()

        body = await request.json()
        name = str(body.get("name", "")).strip() or "新账号"
        raw = body.get("cookies", "")
        cookie_dict = _parse_cookie_payload(raw)

        if not cookie_dict:
            raise HTTPException(status_code=400, detail="未提供有效 Cookie 或 sessionid")

        account_id = body.get("account_id")
        acc = am.add_or_update_account(name=name, cookies=cookie_dict, account_id=account_id)
        return JSONResponse({
            "status": "ok",
            "account": acc.to_dict(mask_cookies=True),
        })

    @app.post("/admin/api/accounts/save_current")
    async def admin_accounts_save_current(request: Request):
        """Save current active browser session as a named account in the pool."""
        _check_auth(request)
        client = _browser.get("client")
        if client is None or not client._context:
            raise HTTPException(status_code=503, detail="浏览器尚未运行或未就绪")

        body = await request.json()
        name = str(body.get("name", "")).strip() or f"已保存会话-{int(time.time()) % 10000}"

        try:
            cookies = await client._context.cookies("https://www.doubao.com")
            cookie_dict = {c["name"]: c["value"] for c in cookies}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"获取浏览器 Cookie 失败: {e}")

        if not cookie_dict or "sessionid" not in cookie_dict:
            raise HTTPException(status_code=400, detail="当前浏览器未检测到已登录的 sessionid")

        am = getattr(client, "account_manager", None)
        if am is None:
            from .account_manager import AccountManager
            am = AccountManager(storage_dir=client.user_data_dir)
            client.account_manager = am

        acc = am.add_or_update_account(name=name, cookies=cookie_dict)
        am.set_active_account(acc.id)
        return JSONResponse({
            "status": "ok",
            "account": acc.to_dict(mask_cookies=True),
        })

    @app.post("/admin/api/accounts/switch")
    async def admin_accounts_switch(request: Request):
        """Manually switch active account in the browser."""
        _check_auth(request)
        client = _browser.get("client")
        body = await request.json()
        account_id = body.get("account_id")
        if not account_id:
            raise HTTPException(status_code=400, detail="Missing account_id")

        if client is None:
            raise HTTPException(status_code=503, detail="Browser not initialized")

        if hasattr(client, "switch_to_account"):
            ok = await client.switch_to_account(account_id)
        else:
            ok = False

        am = getattr(client, "account_manager", None)
        return JSONResponse({
            "status": "ok" if ok else "fail",
            "success": ok,
            "active_account_id": am.active_account_id if am else None,
            "logged_in": client.is_ready,
        })

    @app.post("/admin/api/accounts/delete")
    async def admin_accounts_delete(request: Request):
        """Delete an account from the pool."""
        _check_auth(request)
        client = _browser.get("client")
        am = getattr(client, "account_manager", None)
        if am is None:
            from .account_manager import AccountManager
            am = AccountManager()

        body = await request.json()
        account_id = body.get("account_id")
        if not account_id:
            raise HTTPException(status_code=400, detail="Missing account_id")

        ok = am.delete_account(account_id)
        return JSONResponse({"status": "ok" if ok else "not_found", "deleted": ok})

    @app.post("/admin/api/accounts/strategy")
    async def admin_accounts_strategy(request: Request):
        """Set rotation strategy (failover / round_robin)."""
        _check_auth(request)
        client = _browser.get("client")
        am = getattr(client, "account_manager", None)
        if am is None:
            from .account_manager import AccountManager
            am = AccountManager()

        body = await request.json()
        strategy = str(body.get("strategy", "")).lower()
        if strategy not in ("failover", "round_robin"):
            raise HTTPException(status_code=400, detail="Strategy must be 'failover' or 'round_robin'")

        am.strategy = strategy
        am.save()
        return JSONResponse({"status": "ok", "strategy": am.strategy})

    @app.post("/admin/api/accounts/reset_quota")
    async def admin_accounts_reset_quota(request: Request):
        """Manually reset quota for an account."""
        _check_auth(request)
        client = _browser.get("client")
        am = getattr(client, "account_manager", None)
        if am is None:
            from .account_manager import AccountManager
            am = AccountManager()

        body = await request.json()
        account_id = body.get("account_id")
        if not account_id:
            raise HTTPException(status_code=400, detail="Missing account_id")

        ok = am.reset_quota(account_id)
        return JSONResponse({"status": "ok" if ok else "not_found", "reset": ok})

    @app.post("/admin/api/accounts/rename")
    async def admin_accounts_rename(request: Request):
        """Rename an account in the pool."""
        _check_auth(request)
        client = _browser.get("client")
        am = getattr(client, "account_manager", None)
        if am is None:
            from .account_manager import AccountManager
            am = AccountManager()

        body = await request.json()
        account_id = body.get("account_id")
        name = str(body.get("name", "")).strip()
        if not account_id or not name:
            raise HTTPException(status_code=400, detail="Missing account_id or name")

        ok = am.rename_account(account_id, name)
        return JSONResponse({"status": "ok" if ok else "not_found", "renamed": ok})

    @app.post("/admin/api/accounts/probe_all")
    async def admin_accounts_probe_all(request: Request):
        """Probe and check validity of all accounts in the pool."""
        _check_auth(request)
        client = _browser.get("client")
        am = getattr(client, "account_manager", None)
        if am is None:
            from .account_manager import AccountManager
            am = AccountManager()

        results = []
        async with httpx.AsyncClient(timeout=5.0) as http:
            for acc in list(am.accounts.values()):
                cookie_str = "; ".join(f"{k}={v}" for k, v in acc.cookies.items())
                is_valid = False
                err = ""
                if client and client.is_ready and acc.id == am.active_account_id:
                    is_valid = True
                else:
                    try:
                        headers = {
                            "Cookie": cookie_str,
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
                            "Referer": "https://www.doubao.com/chat/",
                        }
                        resp = await http.get("https://www.doubao.com/api/user/info", headers=headers)
                        if resp.status_code == 200:
                            data = resp.json()
                            if data.get("code") == 0 or data.get("data", {}).get("user_id"):
                                is_valid = True
                            else:
                                err = data.get("msg") or "Session expired"
                        elif "sessionid" in acc.cookies and len(acc.cookies["sessionid"]) >= 16:
                            is_valid = True
                        else:
                            err = f"HTTP {resp.status_code}"
                    except Exception as e:
                        if "sessionid" in acc.cookies and len(acc.cookies["sessionid"]) >= 16:
                            is_valid = True
                        else:
                            err = str(e)

                if acc.status in ("disabled", "captcha", "quota_exceeded"):
                    status = acc.status
                else:
                    status = "active" if is_valid else "invalid"
                am.update_account_status(acc.id, status)
                results.append({
                    "id": acc.id,
                    "name": acc.name,
                    "status": status,
                    "valid": is_valid,
                    "error": err,
                })

        return JSONResponse({
            "status": "ok",
            "results": results,
            "accounts": am.list_accounts(mask_cookies=True),
        })

    @app.post("/admin/api/accounts/clear_captcha")
    async def admin_clear_account_captcha(request: Request):
        """Clear captcha status for an account, restoring it to active."""
        _check_auth(request)
        client = _browser.get("client")
        am = getattr(client, "account_manager", None)
        if am is None:
            from .account_manager import AccountManager
            am = AccountManager()

        body = await request.json()
        account_id = body.get("account_id")
        if not account_id:
            raise HTTPException(status_code=400, detail="account_id is required")

        success = am.clear_captcha_status(account_id)
        if not success:
            raise HTTPException(status_code=404, detail="Account not found or not in captcha state")

        if client and am.active_account_id == account_id:
            client.clear_captcha()
            client._ready = True

        return JSONResponse({
            "status": "ok",
            "message": f"已成功解除账号 {account_id} 的人机验证标记",
            "accounts": am.list_accounts(mask_cookies=True),
        })

    @app.get("/admin/api/media/recent")
    async def admin_recent_media(request: Request):
        """Return recent generated images and videos for admin gallery."""
        _check_auth(request)
        return JSONResponse({
            "media": list(_recent_media),
            "total": len(_recent_media),
        })

    @app.post("/admin/api/probe")
    async def admin_probe(request: Request):
        """Probe session by making a real chat request."""
        _check_auth(request)
        client = _browser.get("client")
        if client is None or not client.is_ready:
            return JSONResponse({"status": "error", "message": "未登录"})
        try:
            t0 = time.time()
            result = await client.chat("1+1=?只回答数字", use_deep_think=0)
            ms = int((time.time() - t0) * 1000)
            content = result.get("text", "")
            client.record_success()
            return JSONResponse({"status": "healthy", "ms": ms, "response": content[:100]})
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)[:200]})

    @app.post("/auth/login")
    async def auth_login(request: Request):
        """Trigger QR login flow. Returns status."""
        _check_auth(request)
        client = _browser.get("client")
        if client is None:
            raise HTTPException(status_code=503, detail="Browser not initialized")
        if client.is_ready:
            return {"status": "already_logged_in"}

        # Start login (non-blocking, returns immediately)
        gate = getattr(client, "operation_gate", None)
        if isinstance(gate, BrowserOperationGate):
            gate.spawn_detached(_do_login(client))
        else:
            asyncio.create_task(_do_login(client))
        return {"status": "login_started", "message": "QR code displayed in browser. Scan to login."}
    @app.post("/auth/reset_captcha")
    async def reset_captcha(request: Request):
        """Reset captcha flag after manual verification via VNC."""
        _check_auth(request)
        client = _browser.get("client")
        if client is None:
            raise HTTPException(status_code=503, detail="Browser not initialized")
        client.record_success()
        if not client.is_ready:
            client._ready = True
        return {"status": "ok", "message": "Captcha flag cleared, service resumed."}


    async def _do_login(client: BrowserClient):
        """Background login task."""
        try:
            ok = await client.wait_for_login(timeout=120)
            if ok:
                log.info("QR login successful via /auth")
            else:
                log.warning("QR login timed out")
        except Exception as exc:
            log.error("QR login error: %s", exc)


    @app.get("/auth/status")
    async def auth_status(request: Request):
        return await _get_login_status(request)

    @app.get("/admin/api/status")
    async def admin_api_status(request: Request):
        return await _get_login_status(request)

    async def _get_login_status(request: Request):
        _check_auth(request)
        client = _browser.get("client")
        if client is None:
            return {"logged_in": False, "browser": "not_started"}

        page_url = client.page.url if client.page else ""
        login_btn_count = 0
        has_session = False
        if client._context:
            try:
                cookies = await client._context.cookies("https://www.doubao.com")
                has_session = any(c["name"] == "sessionid" and c.get("value") for c in cookies)
            except Exception:
                pass
        if client.page:
            try:
                login_btn = client.page.locator('button:has-text("登录")')
                login_btn_count = await login_btn.count()
            except Exception:
                pass

        actual_logged_in = has_session or (client.is_ready and login_btn_count == 0)

        return {
            "logged_in": actual_logged_in,
            "is_ready_flag": client.is_ready,
            "login_button_visible": login_btn_count > 0,
            "page_url": page_url,
            "device_id": client._device_id or "",
            "web_id": client._web_id or "",
            "headless": client.headless,
            "mode": "headless" if client.headless else "window",
            "needs_captcha": client.needs_captcha,
            "browser_name": getattr(client, "browser_name", "chromium"),
        }

    @app.get("/admin/api/browser/status")
    async def admin_browser_status(request: Request):
        """Return browser mode and session details."""
        return await _get_login_status(request)

    @app.post("/admin/api/browser/mode")
    async def admin_browser_set_mode(request: Request):
        """Dynamically switch browser mode between headless and windowed GUI."""
        _check_auth(request)
        client = _browser.get("client")
        if client is None:
            raise HTTPException(status_code=503, detail="Browser client not initialized")
        body = await request.json()
        target_headless = bool(body.get("headless", False))
        log.info("API request: switch browser mode to headless=%s", target_headless)
        ok = await client.switch_mode(target_headless)
        return {
            "status": "ok" if ok else "failed",
            "headless": client.headless,
            "mode": "headless" if client.headless else "window",
            "logged_in": client.is_ready,
            "browser_name": getattr(client, "browser_name", "chromium"),
        }

    @app.post("/auth/eval")
    async def auth_eval(request: Request):
        """Evaluate JS on the browser page (debug/dev only)."""
        _check_auth(request)
        is_dev = os.environ.get("DEV_MODE", "false").lower() in ("true", "1", "yes") or \
                 os.environ.get("DEBUG", "false").lower() in ("true", "1", "yes")
        if not is_dev:
            raise HTTPException(
                status_code=403,
                detail="The /auth/eval endpoint is disabled in production. Set DEV_MODE=true or DEBUG=true to enable.",
            )
        client = _browser.get("client")
        if client is None or client.page is None:
            raise HTTPException(status_code=503, detail="Browser not available")
        body = await request.json()
        js = body.get("js", "")
        if not js:
            raise HTTPException(status_code=400, detail="Missing 'js' field")
        try:
            result = await client.page.evaluate(js)
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}

    @app.get("/auth/screenshot")
    async def auth_screenshot(request: Request):
        """Return a screenshot of the browser page (for remote QR viewing)."""
        _check_auth(request)
        client = _browser.get("client")
        if client is None or client.page is None:
            raise HTTPException(status_code=503, detail="Browser not available")
        png_bytes = await client.page.screenshot()
        from fastapi.responses import Response
        return Response(content=png_bytes, media_type="image/png")

    # ── QR Login (pure HTTP, no VNC needed) ──

    _qr_login_state: Dict[str, Any] = {}

    @app.post("/v1/session/qr-login")
    async def session_qr_login_start(request: Request):
        """Start QR login flow. Returns base64 QR code PNG."""
        _check_auth(request)
        from .qr_login import QRLogin, QRStatus

        # Cancel any existing login
        if _qr_login_state.get("instance"):
            _qr_login_state["instance"].cancel()

        qr = QRLogin()
        _qr_login_state.clear()
        _qr_login_state["instance"] = qr
        _qr_login_state["status"] = "starting"
        _qr_login_state["error"] = ""

        loop = asyncio.get_event_loop()

        def on_status(status: QRStatus, msg: str):
            _qr_login_state["status"] = status.value
            if msg == "qr_ready":
                _qr_login_state["qr_ready"] = True

        def on_done(result):
            if result.status == QRStatus.CONFIRMED:
                _qr_login_state["status"] = "success"
                _qr_login_state["cookies"] = result.cookies
                # Inject cookies into Playwright browser
                client = _browser.get("client")
                if client:
                    loop.call_soon_threadsafe(
                        lambda: asyncio.ensure_future(
                            _inject_qr_cookies(client, result.cookies)
                        )
                    )
                log.info("QR login success: %d cookies", len(result.cookies))
            else:
                _qr_login_state["status"] = "failed"
                _qr_login_state["error"] = result.error

        qr.start(on_status=on_status, on_done=on_done)

        # Wait briefly for QR code to be generated
        for _ in range(20):
            await asyncio.sleep(0.1)
            if qr.qrcode_data:
                break

        if qr.qrcode_data:
            import base64 as b64
            qr_b64 = b64.b64encode(qr.qrcode_data).decode()
            return JSONResponse({
                "status": "qr_ready",
                "qr_image_base64": qr_b64,
                "message": "请用豆包 App 扫码。轮询 GET /v1/session/qr-login 获取状态。",
            })
        else:
            return JSONResponse({
                "status": _qr_login_state.get("status", "error"),
                "error": _qr_login_state.get("error", "生成二维码失败"),
            }, status_code=502)

    @app.get("/v1/session/qr-login")
    async def session_qr_login_poll(request: Request):
        """Poll QR login status."""
        _check_auth(request)
        status = _qr_login_state.get("status", "idle")
        resp: Dict[str, Any] = {"status": status}

        if status == "success":
            resp["message"] = "登录成功，session 已更新"
            resp["cookies_count"] = len(_qr_login_state.get("cookies", {}))
        elif status == "failed":
            resp["error"] = _qr_login_state.get("error", "未知错误")
        elif status == "idle":
            resp["message"] = "无进行中的登录。POST /v1/session/qr-login 开始。"

        return JSONResponse(resp)

    async def _inject_qr_cookies(client: BrowserClient, cookies: Dict[str, str]):
        """Inject QR login cookies into Playwright and verify."""
        try:
            ok = await client.inject_cookies_and_reload(cookies)
            if ok:
                log.info("QR cookies injected successfully, browser is ready")
                _qr_login_state["browser_ready"] = True
            else:
                log.warning("QR cookies injected but login check failed")
                _qr_login_state["browser_ready"] = False
        except Exception as e:
            log.error("Failed to inject QR cookies: %s", e)
            _qr_login_state["browser_ready"] = False

    @app.get("/admin/api/browser/screenshot")
    async def admin_browser_screenshot(request: Request):
        """Return screenshot of browser page as PNG."""
        _check_auth(request)
        from fastapi.responses import Response
        client = _browser.get("client")
        if client is None or client.page is None:
            raise HTTPException(status_code=503, detail="Browser not initialized")
        try:
            buf = await client.page.screenshot(type="png")
            return Response(content=buf, media_type="image/png")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/admin/api/open-doubao")
    async def admin_open_doubao(request: Request):
        """Open doubao.com in system default browser for user login."""
        _check_auth(request)
        import webbrowser
        try:
            webbrowser.open("https://www.doubao.com/chat/")
            return {"status": "ok", "url": "https://www.doubao.com/chat/"}
        except Exception as e:
            return {"status": "error", "detail": str(e)}

    @app.post("/admin/api/browser/force-window")
    async def admin_browser_force_window(request: Request):
        """Force launch or bring up the visible browser window."""
        _check_auth(request)
        client = _browser.get("client")
        if client is None:
            raise HTTPException(status_code=503, detail="Browser not initialized")
        try:
            # Always ensure an active, visible window
            await client.switch_mode(headless=False)
            if client.page and not client.page.is_closed():
                try:
                    await client.page.bring_to_front()
                except Exception:
                    pass
                # Click login button if not logged in and modal not open
                if not client.is_ready:
                    try:
                        modal = client.page.locator('[role="dialog"], .semi-modal')
                        if await modal.count() == 0:
                            btn = client.page.locator('button:has-text("登录")')
                            if await btn.count() > 0 and await btn.first.is_visible():
                                 await btn.first.click()
                    except Exception:
                        pass
            return {"status": "ok", "mode": "window", "headless": client.headless, "browser_name": getattr(client, "browser_name", "chromium")}
        except Exception as e:
            log.error("Failed to force window: %s", e)
            return {"status": "error", "detail": str(e)}

    @app.post("/admin/api/browser/install-chromium")
    async def admin_install_chromium(request: Request):
        """Download and install official Playwright universal Chromium browser."""
        _check_auth(request)
        import subprocess, sys
        try:
            proc = await asyncio.to_thread(
                subprocess.run,
                [sys.executable, "-m", "playwright", "install", "chromium"],
                capture_output=True,
                text=True,
                timeout=300
            )
            if proc.returncode == 0:
                return {"status": "ok", "message": "Playwright 通用独立 Chromium 安装成功！"}
            else:
                return {"status": "error", "message": proc.stderr or "安装失败"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    return app




# ── Server runner ──


def _is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def run_server():
    """Start the uvicorn server with env-based configuration."""
    import uvicorn

    host = os.environ.get("DOUBAO_HOST", "0.0.0.0")
    port = int(os.environ.get("DOUBAO_PORT", "9090"))
    api_key = os.environ.get("DOUBAO_API_KEY", "")
    rpm = float(os.environ.get("DOUBAO_RPM_LIMIT", "20"))

    # Port conflict detection and friendly self-healing
    if _is_port_in_use(port, "127.0.0.1"):
        import urllib.request
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/health", headers={"User-Agent": "doubao2api-probe"})
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    print(f"\n" + "=" * 60)
                    print(f"  [提示] doubao2api 服务已经在端口 {port} 上正常运行中！")
                    print(f"  管理控制台: http://127.0.0.1:{port}/admin")
                    print(f"  无需重复启动。如需彻底重启，请先运行 repair.bat 停止旧服务。")
                    print(f"=" * 60 + "\n")
                    import webbrowser
                    webbrowser.open(f"http://127.0.0.1:{port}/admin")
                    return
        except Exception:
            pass
        print(f"\n[警告] 本地端口 {port} 已被其他程序占用，服务无法启动！")
        print(f"请使用 repair.bat 清理残留进程，或通过环境变量 DOUBAO_PORT=xxxx 指定其他端口。\n")
        return

    # Security check: downgrade 0.0.0.0 to 127.0.0.1 if no API key is set
    allow_unprotected = os.environ.get("ALLOW_UNPROTECTED_BIND", "false").lower() in ("true", "1", "yes")
    if host == "0.0.0.0" and not api_key and not allow_unprotected:
        log.warning(
            "SECURITY WARNING: DOUBAO_HOST is '0.0.0.0' but no DOUBAO_API_KEY is configured! "
            "To prevent unauthorized external access, automatically downgrading host to '127.0.0.1'. "
            "Set DOUBAO_API_KEY or set ALLOW_UNPROTECTED_BIND=true if you intentionally want public open access."
        )
        host = "127.0.0.1"

    app = create_app(api_key=api_key or None, rpm_limit=rpm)

    print(f"\n  Doubao API Server (Playwright Native)")
    print(f"  Listening on http://{host}:{port}")
    print(f"  Admin page: http://{host}:{port}/admin")
    if api_key:
        print(f"  API Key: {api_key[:4]}{'*' * (len(api_key) - 4)}")
    print()

    auto_open = os.environ.get("DOUBAO_AUTO_OPEN_BROWSER", "true").lower() in ("true", "1", "yes")
    if auto_open:
        def _open_browser_when_ready():
            import time
            import urllib.request
            target_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
            admin_url = f"http://{target_host}:{port}/admin"
            for _ in range(20):
                time.sleep(0.5)
                try:
                    req = urllib.request.Request(
                        f"http://{target_host}:{port}/health",
                        headers={"User-Agent": "doubao2api-probe"},
                    )
                    with urllib.request.urlopen(req, timeout=1) as resp:
                        if resp.status == 200:
                            import webbrowser
                            webbrowser.open(admin_url)
                            break
                except Exception:
                    pass

        import threading
        threading.Thread(target=_open_browser_when_ready, daemon=True).start()

    uvicorn.run(app, host=host, port=port, log_level="info")
