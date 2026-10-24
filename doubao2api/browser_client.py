"""Playwright-based Doubao client with in-browser fetch.

Architecture:
- Playwright: Login (QR scan via noVNC) + page session
- In-browser fetch(): API requests go through ByteDance's fetch hook which
  automatically injects a_bogus/msToken signatures with real browser fingerprint
- httpx: Only used for file upload (TOS/ImageX flow, no fetch hook needed)
- expose_function bridge: Streams SSE chunks from browser JS back to Python

ByteDance's frontend exposes window.bdms.frontierSign() which generates
X-Bogus signatures. We use Playwright only to maintain a logged-in page
and call this signing function. All actual API traffic goes through httpx.
"""

import asyncio
import inspect
import json
import logging
import os
import time
import uuid
from functools import wraps
from typing import AsyncGenerator, Optional, Dict, Any, List, Union
from urllib.parse import urlencode

import httpx
from playwright.async_api import async_playwright, BrowserContext, Page
from playwright_stealth import Stealth
from .browser_operation import BrowserOperationGate

log = logging.getLogger(__name__)

DOUBAO_URL = "https://www.doubao.com"
CHAT_URL = f"{DOUBAO_URL}/chat/"
COMPLETION_URL = f"{DOUBAO_URL}/chat/completion"
SAMANTHA_COMPLETION_URL = f"{DOUBAO_URL}/samantha/chat/completion"
DEFAULT_BOT_ID = "7338286299411103781"


class VideoQuotaExceeded(RuntimeError):
    """The active account has reached its daily video generation quota."""


def _serialized_browser_operation(method):
    if inspect.isasyncgenfunction(method):
        @wraps(method)
        async def _stream(self, *args, **kwargs):
            async with self.operation_gate.lease():
                iterator = method(self, *args, **kwargs).__aiter__()
                try:
                    async for item in iterator:
                        yield item
                finally:
                    close = getattr(iterator, "aclose", None)
                    if close:
                        await close()
        return _stream

    @wraps(method)
    async def _call(self, *args, **kwargs):
        async with self.operation_gate.lease():
            return await method(self, *args, **kwargs)
    return _call


def resolve_headless_mode(val: Optional[Union[bool, str]] = None) -> bool:
    """
    Resolve headless setting based on parameter, env DOUBAO_HEADLESS, and system display capability.
    - 'auto' / None:
        - Windows (os.name == 'nt') or macOS: False (headed for ease of login)
        - Linux/other: False if DISPLAY or WAYLAND_DISPLAY is set, else True (headless)
    - 'true' / True: True (always headless)
    - 'false' / False: False. If on Linux/headless env without DISPLAY, raise RuntimeError.
    """
    if val is None:
        env_val = os.environ.get("DOUBAO_HEADLESS", "auto").strip().lower()
    elif isinstance(val, bool):
        env_val = "true" if val else "false"
    else:
        env_val = str(val).strip().lower()

    has_display = True
    if os.name != "nt":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    if env_val in ("true", "1", "yes"):
        return True
    elif env_val in ("false", "0", "no"):
        if not has_display:
            raise RuntimeError(
                "DOUBAO_HEADLESS is set to 'false', but no display server (DISPLAY / WAYLAND_DISPLAY) "
                "is available in this environment. Please run with DOUBAO_HEADLESS=true, or provide an X11/wayland display."
            )
        return False
    elif env_val == "auto":
        if os.name == "nt":
            return False
        return not has_display
    else:
        log.warning("Unknown DOUBAO_HEADLESS='%s', defaulting to auto", env_val)
        if os.name == "nt":
            return False
        return not has_display


def resolve_memory_type(
    memory_type: Optional[int] = None,
    save_history: Optional[bool] = None,
) -> int:
    """
    Resolve Doubao memory_type:
    0 = normal conversation (saved to web history, not 阅后即焚)
    2 = ephemeral / burn-after-reading (does NOT show in web chat history)

    Priority:
    1. Explicit memory_type argument (0 or 2)
    2. save_history flag (True -> 0, False -> 2)
    3. SAVE_CHAT_HISTORY / DOUBAO_AUTO_DELETE_CONV env vars:
       - If SAVE_CHAT_HISTORY is true -> 0
       - If DOUBAO_AUTO_DELETE_CONV is false/0/no -> 0
       - Otherwise default to 2 (ephemeral)
    """
    if memory_type is not None:
        return int(memory_type)
    if save_history is not None:
        return 0 if save_history else 2
    save_env = os.environ.get("SAVE_CHAT_HISTORY", "").strip().lower()
    if save_env in ("true", "1", "yes"):
        return 0
    auto_del_env = os.environ.get("DOUBAO_AUTO_DELETE_CONV", "true").strip().lower()
    if auto_del_env in ("false", "0", "no"):
        return 0
    return 2


def bring_window_to_foreground():
    """Attempt to bring the browser window to the foreground on Windows."""
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        targets = []

        WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def _enum_proc(hwnd, lparam):
            if user32.IsWindowVisible(hwnd):
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buf = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buf, length + 1)
                    title = buf.value
                    if any(k in title for k in ("豆包", "Doubao", "doubao.com", "Chrome", "Edge")):
                        targets.append(hwnd)
            return True

        user32.EnumWindows(WNDENUMPROC(_enum_proc), 0)
        for hwnd in targets:
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            user32.SetForegroundWindow(hwnd)
            user32.BringWindowToTop(hwnd)
    except Exception as e:
        log.debug("bring_window_to_foreground error: %s", e)


class BrowserClient:
    """Manages Playwright for login and in-browser fetch for API calls."""

    def __init__(self, headless: Optional[Union[bool, str]] = None, user_data_dir: Optional[str] = None):
        self._raw_headless = headless
        self.headless = resolve_headless_mode(headless)
        self.operation_gate = BrowserOperationGate()
        self.user_data_dir = user_data_dir
        self._playwright = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._http: Optional[httpx.AsyncClient] = None
        self._ready = False
        self._device_id: Optional[str] = None
        self._web_id: Optional[str] = None
        self._fp: Optional[str] = None
        # msToken rotation: updated from x-ms-token response header
        self._ms_token: str = ""
        # Robustness: failure tracking
        self._consecutive_failures: int = 0
        self._last_error_code: int = 0
        self._last_error_category: str = ""
        self._needs_captcha: bool = False
        self._mode_lock: asyncio.Lock = asyncio.Lock()
        # Stream bridge: request_id -> asyncio.Queue for SSE chunks
        self._stream_queues: Dict[str, asyncio.Queue] = {}
        self._bridge_ready: bool = False
        self._user_agent: str = ""
        self._browser_name: str = "Chromium"

        from .account_manager import AccountManager
        self.account_manager = AccountManager(storage_dir=self.user_data_dir)

    @property
    def browser_name(self) -> str:
        return self._browser_name

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def page(self) -> Optional[Page]:
        return self._page

    @property
    def needs_captcha(self) -> bool:
        return self._needs_captcha

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def last_error_code(self) -> int:
        return self._last_error_code

    @property
    def last_error_category(self) -> str:
        return self._last_error_category

    def record_success(self):
        """Reset failure counters on successful request."""
        self._consecutive_failures = 0
        self._last_error_code = 0
        self._last_error_category = ""
        self._needs_captcha = False
        if self.account_manager:
            active = self.account_manager.get_active_account()
            if active:
                self.account_manager.clear_captcha_status(active.id)

    def clear_captcha(self):
        """Manually or automatically clear the needs_captcha flag."""
        self._needs_captcha = False
        self._consecutive_failures = 0
        self._last_error_code = 0
        self._last_error_category = ""
        log.info("Captcha flag cleared")

    @_serialized_browser_operation
    async def is_captcha_visible(self) -> bool:
        """Check whether a real captcha modal/slider/iframe is actually visible in the browser DOM."""
        if not self._page:
            return False
        try:
            # Fast check across all iframes for ByteDance captcha/verify endpoints
            has_iframe = await self._page.evaluate("""() => {
                const frames = Array.from(document.querySelectorAll('iframe'));
                return frames.some(f => {
                    const s = (f.src || '').toLowerCase();
                    return s.includes('verify') || s.includes('captcha') || s.includes('zijieapi') || s.includes('bytedance.com');
                });
            }""")
            if has_iframe:
                return True
        except Exception:
            pass

        try:
            selectors = [
                '#captcha_container',
                '.captcha_verify_container',
                '#captcha-verify-image',
                '.secsdk-captcha-wrapper',
                '.verify-bar-close',
                '.secsdk-captcha-drag-icon',
                '[class*="captcha-modal"]',
                '[class*="captcha_verify"]',
                '.semi-modal:has-text("验证")',
                'iframe[src*="verify"]',
                'iframe[src*="captcha"]',
                'iframe[src*="bytedance.com"]',
                'iframe[src*="zijieapi.com"]',
            ]
            for sel in selectors:
                loc = self._page.locator(sel)
                cnt = await loc.count()
                if cnt > 0:
                    for i in range(cnt):
                        if await loc.nth(i).is_visible():
                            return True
            return False
        except Exception:
            return False

    @_serialized_browser_operation
    async def auto_popup_for_captcha(self) -> bool:
        """If captcha is detected, ensure browser window is visible and brought to front."""
        can_display = (os.name == "nt") or bool(os.environ.get("DISPLAY"))
        if not can_display:
            log.info("Cannot auto-popup browser window: no desktop GUI display available")
            return False

        try:
            await self.switch_mode(headless=False)
        except Exception as e:
            log.error("Failed to bring browser window to front: %s", e)
            return False

        if self._page:
            try:
                await self._page.bring_to_front()
            except Exception:
                pass
        return True

    @_serialized_browser_operation
    async def try_auto_solve_captcha(self) -> bool:
        """Attempt to automatically detect and solve slider captcha.

        Returns True if solved, False otherwise.
        """
        if not self._page:
            return False
        try:
            from .captcha_solver import SliderCaptchaSolver
            solver = SliderCaptchaSolver()
            solved = await solver.detect_and_solve(self._page)
            if solved:
                self._ready = True
                log.info("Auto-solved slider captcha successfully!")
                return True
            return False
        except Exception as e:
            log.warning("Automated captcha solver error: %s", e)
            return False

    def record_failure(
        self,
        error_code: int = 0,
        *,
        verification_required: Optional[bool] = None,
        error_category: Optional[str] = None,
    ):
        """Track failures and mark verification only when the failure was classified as such."""
        self._consecutive_failures += 1
        self._last_error_code = error_code
        self._last_error_category = error_category or (
            "verification"
            if verification_required is True or (verification_required is None and error_code == 710022002)
            else ""
        )
        if verification_required is True or (verification_required is None and error_code == 710022002):
            self._needs_captcha = True
            log.warning("Verification flag marked on error code %s", error_code)
        if self._consecutive_failures >= 10 and not self._needs_captcha:
            log.error("10 consecutive failures (non-captcha) - marking not ready")
            self._ready = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def get_candidate_browsers(cls) -> List[Dict[str, Any]]:
        """
        Build a prioritized list of browser launch candidate configurations.
        Supports:
          1. Custom env override (DOUBAO_BROWSER_CHANNEL or DOUBAO_BROWSER_BIN)
          2. Google Chrome (channel="chrome" or system paths)
          3. Microsoft Edge (channel="msedge" - pre-installed on 100% of modern Windows)
          4. Brave Browser (via executable path)
          5. Other Chromium-based browsers (Vivaldi, Opera, 360, etc.)
          6. Bundled Playwright Chromium (channel=None)
        """
        import shutil
        import sys
        candidates = []

        # 1. Custom env overrides
        custom_ch = os.environ.get("DOUBAO_BROWSER_CHANNEL", "").strip().lower()
        if custom_ch:
            candidates.append({"channel": custom_ch, "desc": f"自定义通道 ({custom_ch})"})

        custom_bin = os.environ.get("DOUBAO_BROWSER_BIN", "").strip()
        if custom_bin and os.path.exists(custom_bin):
            candidates.append({"executable_path": custom_bin, "desc": f"自定义浏览器路径 ({custom_bin})"})

        # 2. OS-specific auto-discovery
        if os.name == "nt":
            # Playwright Universal Standalone Chromium (完全独立、免安装、不与系统Chrome冲突)
            base_dir = os.path.expandvars(r"%LOCALAPPDATA%\ms-playwright")
            if os.path.exists(base_dir):
                for item in sorted(os.listdir(base_dir), reverse=True):
                    if item.startswith("chromium-") and "headless" not in item:
                        exe = os.path.join(base_dir, item, "chrome-win", "chrome.exe")
                        if os.path.exists(exe):
                            candidates.append({"executable_path": exe, "desc": f"Playwright 通用独立 Chromium ({item})"})
                            break

            # Chrome
            chrome_paths = [
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
            ]
            if any(os.path.exists(p) for p in chrome_paths):
                candidates.append({"channel": "chrome", "desc": "Google Chrome (系统安装)"})

            # Edge (Windows 10/11 保证预装)
            edge_paths = [
                r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe"),
            ]
            if any(os.path.exists(p) for p in edge_paths):
                candidates.append({"channel": "msedge", "desc": "Microsoft Edge (系统自带)"})

            # Brave
            brave_paths = [
                r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
                os.path.expandvars(r"%LOCALAPPDATA%\BraveSoftware\Brave-Browser\Application\brave.exe"),
            ]
            for bp in brave_paths:
                if os.path.exists(bp):
                    candidates.append({"executable_path": bp, "desc": f"Brave Browser ({bp})"})
                    break

            # Other Chromium-based browsers (Vivaldi, Opera)
            other_paths = [
                r"C:\Program Files\Vivaldi\Application\vivaldi.exe",
                os.path.expandvars(r"%LOCALAPPDATA%\Vivaldi\Application\vivaldi.exe"),
                os.path.expandvars(r"%LOCALAPPDATA%\Programs\Opera\launcher.exe"),
            ]
            for op in other_paths:
                if os.path.exists(op):
                    candidates.append({"executable_path": op, "desc": f"Chromium 内核浏览器 ({op})"})
                    break

        elif sys.platform == "darwin":
            # macOS
            mac_chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
            mac_edge = "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"
            mac_brave = "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"
            mac_chromium = "/Applications/Chromium.app/Contents/MacOS/Chromium"
            if os.path.exists(mac_chrome):
                candidates.append({"channel": "chrome", "desc": "Google Chrome (macOS)"})
            if os.path.exists(mac_edge):
                candidates.append({"channel": "msedge", "desc": "Microsoft Edge (macOS)"})
            if os.path.exists(mac_brave):
                candidates.append({"executable_path": mac_brave, "desc": "Brave Browser (macOS)"})
            if os.path.exists(mac_chromium):
                candidates.append({"executable_path": mac_chromium, "desc": "Chromium (macOS)"})

        else:
            # Linux
            for bin_name in ("google-chrome", "google-chrome-stable"):
                if shutil.which(bin_name):
                    candidates.append({"channel": "chrome", "desc": f"Google Chrome ({bin_name})"})
                    break
            for bin_name in ("microsoft-edge", "microsoft-edge-stable"):
                if shutil.which(bin_name):
                    candidates.append({"channel": "msedge", "desc": f"Microsoft Edge ({bin_name})"})
                    break
            for bin_name in ("chromium", "chromium-browser", "brave-browser"):
                p = shutil.which(bin_name)
                if p:
                    candidates.append({"executable_path": p, "desc": f"Chromium/Brave ({p})"})
                    break

        # 3. Bundled Chromium (Playwright native)
        candidates.append({"desc": "Playwright 内置 Chromium 内核"})

        # 4. Standard channels fallback
        for ch in ("chrome", "msedge", "chromium"):
            if not any(c.get("channel") == ch for c in candidates):
                candidates.append({"channel": ch, "desc": f"系统 '{ch}' 通道"})

        return candidates

    @_serialized_browser_operation
    async def start(self):
        """Launch browser window or headless instance, navigate to Doubao, init httpx client."""
        self.headless = resolve_headless_mode(self._raw_headless)
        mode_str = "headless" if self.headless else "headed native desktop window"
        log.info("Starting BrowserClient (%s)", mode_str)
        # Clean stale browser lock files if any were left behind from previous runs
        if self.user_data_dir and os.path.exists(self.user_data_dir):
            for lock_name in ("lockfile", "SingletonLock", "SingletonSocket", "SingletonCookie"):
                lock_file = os.path.join(self.user_data_dir, lock_name)
                if os.path.exists(lock_file):
                    try:
                        os.remove(lock_file)
                        log.info("Cleaned stale profile lock: %s", lock_file)
                    except Exception as e:
                        log.debug("Could not remove lock %s: %s", lock_file, e)

        self._playwright = await async_playwright().start()

        launch_args = [
            "--disable-blink-features=AutomationControlled",
            "--no-first-run",
            "--no-default-browser-check",
            "--no-sandbox",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
        ]

        launch_kwargs = {
            "headless": self.headless,
            "args": launch_args,
            "viewport": {"width": 1280, "height": 720},
            "locale": "zh-CN",
        }

        async def _launch_context(extra_kwargs):
            kw = dict(launch_kwargs)
            kw.update(extra_kwargs)
            if self.user_data_dir:
                try:
                    ctx = await self._playwright.chromium.launch_persistent_context(
                        self.user_data_dir,
                        **kw
                    )
                except Exception as e:
                    err_str = str(e)
                    if "ProcessSingleton" in err_str or "Lock file" in err_str or "lockfile" in err_str:
                        log.warning("Detected lock collision during launch, retrying after clearing lockfile...")
                        await asyncio.sleep(1.0)
                        for lock_name in ("lockfile", "SingletonLock", "SingletonSocket", "SingletonCookie"):
                            lp = os.path.join(self.user_data_dir, lock_name)
                            if os.path.exists(lp):
                                try:
                                    os.remove(lp)
                                except Exception:
                                    pass
                        ctx = await self._playwright.chromium.launch_persistent_context(
                            self.user_data_dir,
                            **kw
                        )
                    else:
                        raise
                page = ctx.pages[0] if ctx.pages else await ctx.new_page()
                return ctx, page
            else:
                b_kwargs = {"headless": self.headless, "args": launch_args}
                if kw.get("channel"):
                    b_kwargs["channel"] = kw["channel"]
                if kw.get("executable_path"):
                    b_kwargs["executable_path"] = kw["executable_path"]
                browser = await self._playwright.chromium.launch(**b_kwargs)
                ctx = await browser.new_context(
                    viewport={"width": 1280, "height": 720}, locale="zh-CN",
                )
                page = await ctx.new_page()
                return ctx, page

        candidates = self.get_candidate_browsers()
        launched = False
        last_error = None

        for cand in candidates:
            cand_desc = cand.get("desc", "unknown")
            extra = {}
            if "channel" in cand:
                extra["channel"] = cand["channel"]
            if "executable_path" in cand:
                extra["executable_path"] = cand["executable_path"]

            try:
                log.info("尝试启动浏览器: %s (headless=%s) ...", cand_desc, self.headless)
                self._context, self._page = await _launch_context(extra)
                self._browser_name = cand_desc
                log.info("成功启动浏览器: %s (headless=%s)", cand_desc, self.headless)
                launched = True
                break
            except Exception as e:
                last_error = e
                log.debug("启动浏览器 %s 失败: %s", cand_desc, e)

        # If all candidates failed, attempt auto-install of Playwright Chromium
        if not launched:
            log.warning("未检测到可用的系统浏览器，正在自动下载并安装 Playwright Chromium 内核...")
            try:
                import subprocess, sys
                proc = await asyncio.to_thread(
                    subprocess.run,
                    [sys.executable, "-m", "playwright", "install", "chromium"],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                if proc.returncode == 0:
                    log.info("Playwright Chromium 内核安装成功，正在重新启动...")
                    self._context, self._page = await _launch_context({})
                    self._browser_name = "Playwright 内置 Chromium (自动安装)"
                    launched = True
                else:
                    log.error("自动安装 Playwright Chromium 失败: %s", proc.stderr)
            except Exception as ie:
                log.error("自动安装 Playwright Chromium 异常: %s", ie)

        if not launched:
            log.error("未找到任何可用的 Chromium 内核浏览器 (已尝试 Chrome, Edge, Brave, Chromium)。")
            raise RuntimeError(
                "未检测到可用的浏览器。本项目支持 Google Chrome、Microsoft Edge、Brave 或系统 Chromium。"
                "请安装 Chrome 或 Edge，或在终端执行: playwright install chromium"
            ) from last_error

        # Stealth patches
        stealth = Stealth(navigator_languages_override=("zh-CN", "zh"))
        await stealth.apply_stealth_async(self._page)

        # Pre-register fetch bridge
        await self._setup_fetch_bridge()

        # Navigate
        log.info("Navigating to %s", CHAT_URL)
        try:
            await self._page.goto(CHAT_URL, wait_until="domcontentloaded", timeout=45000)
            await asyncio.sleep(2)
        except Exception as e:
            log.warning("Navigation to %s warning: %s", CHAT_URL, e)
            if self._page and not self._page.is_closed():
                try:
                    await self._page.goto(CHAT_URL, wait_until="commit", timeout=15000)
                except Exception:
                    pass
        try:
            self._user_agent = await self._page.evaluate("() => navigator.userAgent")
            log.info("Detected browser User-Agent: %s", self._user_agent)
        except Exception:
            pass

        # Init httpx
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(180, connect=10))

        await self._check_login_state()
        if not self._ready and not self.headless:
            self.operation_gate.spawn_detached(self._trigger_login_dialog())

    @_serialized_browser_operation
    async def stop(self):
        """Close browser and httpx client."""
        if self._http:
            await self._http.aclose()
            self._http = None
        if self._context:
            try:
                await self._context.close()
            except Exception:
                pass
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception:
                pass
        self._context = None
        self._playwright = None
        self._page = None
        self._ready = False
        self._bridge_ready = False
        # Clean stale browser lock files if any were left behind
        if self.user_data_dir and os.path.exists(self.user_data_dir):
            for lock_name in ("lockfile", "SingletonLock", "SingletonSocket", "SingletonCookie"):
                lock_file = os.path.join(self.user_data_dir, lock_name)
                if os.path.exists(lock_file):
                    try:
                        os.remove(lock_file)
                    except Exception:
                        pass
        log.info("BrowserClient stopped")

    @_serialized_browser_operation
    async def is_alive(self) -> bool:
        """Check if browser process is still responsive."""
        if not self._page or not self._context:
            return False
        if self._page.is_closed():
            return False
        try:
            result = await asyncio.wait_for(
                self._page.evaluate("1+1"), timeout=15
            )
            return result == 2
        except Exception as e:
            log.warning("Browser health check failed: %s", e)
            return False

    @_serialized_browser_operation
    async def restart(self):
        """Stop and restart the browser client."""
        log.info("Restarting BrowserClient...")
        await self.stop()
        await asyncio.sleep(2)
        await self.start()
        log.info("BrowserClient restarted. ready=%s", self._ready)

    @_serialized_browser_operation
    async def switch_mode(self, headless: Optional[Union[bool, str]] = None) -> bool:
        """Ensure browser window is active, healthy, and optionally adjust headless mode."""
        if headless is not None:
            self._raw_headless = headless
            self.headless = resolve_headless_mode(headless)
        async with self._mode_lock:
            if self._page and not self._page.is_closed():
                try:
                    if await self.is_alive():
                        if not self.headless:
                            await self._page.bring_to_front()
                            bring_window_to_foreground()
                            log.info("Browser window brought to front")
                        return True
                except Exception:
                    pass
            log.info("Browser window not active or mode switched, relaunching window...")
            await self.restart()
            if not self.headless:
                bring_window_to_foreground()
            return self._ready

    @classmethod
    def is_profile_logged_in(cls, user_data_dir: Optional[str]) -> bool:
        """Check if user_data_dir exists and has logged-in marks."""
        if not user_data_dir or not os.path.exists(user_data_dir):
            return False
        # Check explicit flag file
        flag_file = os.path.join(user_data_dir, ".login_success")
        if os.path.exists(flag_file) and os.path.getsize(flag_file) > 0:
            return True
        cookies_db = os.path.join(user_data_dir, "Default", "Network", "Cookies")
        if os.path.exists(cookies_db) and os.path.getsize(cookies_db) > 10240:
            return True
        return False

    # ------------------------------------------------------------------
    # Login
    # ------------------------------------------------------------------

    @_serialized_browser_operation
    async def _check_login_state(self):
        """Check if logged in by inspecting session cookies and page DOM."""
        if self._page and "from_logout=1" in self._page.url:
            log.info("Page is on from_logout=1, navigating to clean chat URL...")
            try:
                await self._page.goto(CHAT_URL, wait_until="load", timeout=30000)
                await asyncio.sleep(2)
            except Exception as e:
                log.warning("Navigation back to CHAT_URL failed: %s", e)

        try:
            cookies = await self._context.cookies("https://www.doubao.com")
        except Exception as e:
            log.warning("Failed to inspect browser cookies (window may have been closed): %s", e)
            self._ready = False
            return
        has_session = any(c["name"] == "sessionid" and c.get("value") for c in cookies)

        # Check DOM indicators for true authenticated state
        has_login_btn = False
        try:
            btn = self._page.locator('button:has-text("登录"), a:has-text("登录")')
            if await btn.count() > 0 and await btn.first.is_visible():
                has_login_btn = True
        except Exception:
            pass

        has_avatar = False
        try:
            avatar = self._page.locator('img[class*="avatar"], div[class*="avatar"]')
            if await avatar.count() > 0:
                has_avatar = True
        except Exception:
            pass

        # True authenticated state: avatar present, or session cookie without visible login button
        is_logged_in = has_avatar or (has_session and not has_login_btn)
        log.info("Login check: has_sessionid=%s, has_avatar=%s, has_login_btn=%s -> is_logged_in=%s",
                 has_session, has_avatar, has_login_btn, is_logged_in)

        if not is_logged_in:
            log.info("Not logged in - valid sessionid or avatar not confirmed")
            self._ready = False
            if self.user_data_dir:
                flag_file = os.path.join(self.user_data_dir, ".login_success")
                if os.path.exists(flag_file):
                    try:
                        os.remove(flag_file)
                    except Exception:
                        pass
            return

        self._ready = True
        if self.user_data_dir:
            try:
                os.makedirs(self.user_data_dir, exist_ok=True)
                with open(os.path.join(self.user_data_dir, ".login_success"), "w", encoding="utf-8") as f:
                    f.write(str(int(time.time())))
            except Exception:
                pass

        await self._extract_params()
        await self._seed_ms_token()
        await self._setup_fetch_bridge()
        await self._verify_fetch_hook()
        await self._wait_for_signing()  # still needed for upload endpoints
        log.info("Ready! device_id=%s, fetch_hook=%s", self._device_id, self._bridge_ready)

        # Auto-register default account into account_manager if pool is empty
        if hasattr(self, "account_manager") and len(self.account_manager.accounts) == 0:
            try:
                cookies = await self._context.cookies("https://www.doubao.com")
                cookie_dict = {c["name"]: c["value"] for c in cookies}
                if "sessionid" in cookie_dict:
                    self.account_manager.add_or_update_account(
                        name="默认账号",
                        cookies=cookie_dict,
                        account_id="acc_default",
                    )
                    log.info("Auto-registered current session as '默认账号' (acc_default) in account pool")
            except Exception as e:
                log.debug("Auto-register default account failed: %s", e)

    async def _extract_params(self):
        """Extract device_id, web_id, fp from localStorage/cookies."""
        for _ in range(5):
            async with self.operation_gate.lease():
                params = await self._page.evaluate("""() => {
                    const result = {};
                    try {
                        const samWeb = JSON.parse(localStorage.getItem('samantha_web_web_id') || '{}');
                        result.device_id = samWeb.web_id || '';
                    } catch(e) {}
                    try {
                        const tea = JSON.parse(localStorage.getItem('__tea_cache_tokens_497858') || '{}');
                        result.web_id = tea.web_id || '';
                    } catch(e) {}
                    const fpCookie = document.cookie.split(';')
                        .map(c => c.trim())
                        .find(c => c.startsWith('s_v_web_id='));
                    result.fp = fpCookie ? fpCookie.split('=')[1] : '';
                    return result;
                }""")
            self._device_id = params.get("device_id", "")
            self._web_id = params.get("web_id", "")
            self._fp = params.get("fp", "")
            if self._device_id and self._web_id:
                break
            await asyncio.sleep(1)
        log.info("Params: device_id=%s, web_id=%s, fp=%s",
                 self._device_id, self._web_id, self._fp[:20] if self._fp else "")

    async def _wait_for_signing(self):
        """Wait for bdms.frontierSign to become available (legacy, kept for upload signing)."""
        for i in range(12):  # up to 60s
            async with self.operation_gate.lease():
                has_sign = await self._page.evaluate(
                    "() => typeof window.bdms?.frontierSign === 'function'"
                )
            if has_sign:
                log.info("bdms.frontierSign available after %ds", (i + 1) * 5)
                return
            await asyncio.sleep(5)
        log.warning("bdms.frontierSign not available after 60s - signing may fail")

    @_serialized_browser_operation
    async def _setup_fetch_bridge(self):
        """Register expose_function callback for streaming data from browser to Python."""
        if self._bridge_ready:
            return

        async def _on_stream_chunk(request_id: str, chunk_json: str):
            """Called from browser JS for each SSE chunk or completion signal."""
            queue = self._stream_queues.get(request_id)
            if queue:
                await queue.put(chunk_json)

        try:
            await self._page.expose_function("__doubaoStreamChunk", _on_stream_chunk)
            self._bridge_ready = True
            log.info("Fetch bridge registered (expose_function ready)")
        except Exception as e:
            # May already be registered if page didn't navigate
            if "already been registered" in str(e).lower():
                self._bridge_ready = True
                log.info("Fetch bridge already registered")
            else:
                log.error("Failed to register fetch bridge: %s", e)
                raise

    async def _verify_fetch_hook(self):
        """Verify ByteDance's fetch interceptor is active (adds a_bogus)."""
        for i in range(15):  # up to 30s
            async with self.operation_gate.lease():
                hooked = await self._page.evaluate("""() => {
                    try {
                        const s = window.fetch.toString();
                        return !s.includes('native code');
                    } catch(e) { return false; }
                }""")
            if hooked:
                log.info("Fetch hook verified active after %ds", (i + 1) * 2)
                return True
            await asyncio.sleep(2)
        log.warning("Fetch hook NOT detected after 30s - requests may fail")
        return False

    async def wait_for_login(self, timeout: int = 180) -> bool:
        """Wait for user to scan QR code."""
        await self._trigger_login_dialog()
        log.info("Waiting for QR scan login (timeout=%ds)...", timeout)
        try:
            for _ in range(timeout):
                await asyncio.sleep(1)
                async with self.operation_gate.lease():
                    avatar = self._page.locator('img[class*="avatar"], div[class*="avatar"], span:has-text("套餐"), div:has-text("套餐")')
                    has_avatar = await avatar.count() > 0
                    login_btn = self._page.locator('button:has-text("登录")')
                    has_login_button = await login_btn.count() > 0
                if has_avatar:
                    log.info("Detected user profile/avatar in DOM!")
                    break
                if not has_login_button:
                    log.info("Login button no longer visible!")
                    break

            await asyncio.sleep(2)
            await self._extract_params()
            await self._seed_ms_token()
            await self._setup_fetch_bridge()
            await self._verify_fetch_hook()
            await self._wait_for_signing()
            self._ready = True
            log.info("Login successful and client ready!")
            return True
        except Exception as e:
            log.error("Login check error: %s", e)
            return False

    async def _trigger_login_dialog(self):
        """Click login button to show QR code dialog."""
        for _ in range(10):
            try:
                async with self.operation_gate.lease():
                    btn = self._page.locator('button:has-text("登录")')
                    if await btn.count() > 0:
                        await btn.first.click()
                        log.info("Clicked '登录' button to display QR code dialog")
                        return
            except Exception:
                pass
            await asyncio.sleep(1)


    @_serialized_browser_operation
    async def inject_cookies_and_reload(self, cookies: Dict[str, str]) -> bool:
        """Inject cookies from QR login or user import into browser context and reload.

        Returns True if login state is confirmed after reload.
        """
        if not self._context or not self._page or not await self.is_alive():
            log.info("inject_cookies: browser not active, restarting before inject...")
            await self.restart()

        if not self._context or not self._page:
            log.error("inject_cookies: browser could not be started")
            return False

        # Build cookie list for Playwright
        pw_cookies = []
        for name, value in cookies.items():
            pw_cookies.append({
                "name": name,
                "value": value,
                "domain": ".doubao.com",
                "path": "/",
            })

        await self._context.add_cookies(pw_cookies)
        log.info("Injected %d cookies into browser context", len(pw_cookies))

        # Reload page to pick up new session
        await self._page.reload(wait_until="load", timeout=30000)
        await asyncio.sleep(3)

        # Re-check login state
        await self._check_login_state()
        return self._ready

    @_serialized_browser_operation
    async def switch_to_account(self, account_id: str) -> bool:
        """Switch browser session to specified account from the account pool."""
        acc = self.account_manager.get_account(account_id)
        if not acc:
            log.error("switch_to_account: account %s not found in pool", account_id)
            return False
        if not acc.cookies:
            log.error("switch_to_account: account %s has no cookies", account_id)
            return False

        previous = self.account_manager.get_active_account()
        previous_cookies = dict(previous.cookies) if previous else {}

        try:
            if not self._context or not self._page or not await self.is_alive():
                log.info("switch_to_account: browser not active, restarting...")
                await self.restart()
        except asyncio.CancelledError:
            self._ready = False
            raise
        except Exception as exc:
            log.warning("Browser restart failed during account switch (%s)", type(exc).__name__)
            self._ready = False
            return False

        if not self._context or not self._page:
            return False

        async def _restore_previous() -> bool:
            if not previous_cookies or not self._context or not self._page:
                return False
            try:
                await self._context.clear_cookies()
                return await self.inject_cookies_and_reload(previous_cookies)
            except Exception as exc:
                log.warning("Failed to restore prior account after switch failure (%s)", type(exc).__name__)
                return False

        # Clear existing cookies to avoid session cross-contamination
        self._ready = False
        try:
            await self._context.clear_cookies()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("clear_cookies failed before account switch (%s)", type(exc).__name__)
            return False

        try:
            ok = await self.inject_cookies_and_reload(acc.cookies)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("Account switch failed during cookie injection (%s)", type(exc).__name__)
            ok = False
        if ok:
            self.account_manager.set_active_account(acc.id)
            self._consecutive_failures = 0
            self._last_error_code = 0
            self._last_error_category = ""
            self._needs_captcha = False
            log.info("Successfully switched to account %s (%s)", acc.id, acc.name)
            return True

        restored = await _restore_previous()
        self._ready = restored
        return False
    # ------------------------------------------------------------------
    # Signing & Cookies
    # ------------------------------------------------------------------

    @_serialized_browser_operation
    async def _get_cookies_string(self) -> str:
        """Get full cookie string including httpOnly cookies."""
        cookies = await self._context.cookies("https://www.doubao.com")
        return "; ".join(f"{c['name']}={c['value']}" for c in cookies)

    @_serialized_browser_operation
    async def _get_csrf_token(self) -> str:
        """Get passport_csrf_token from browser cookies."""
        cookies = await self._context.cookies("https://www.doubao.com")
        for c in cookies:
            if c["name"] == "passport_csrf_token":
                return c["value"]
            if c["name"] == "passport_csrf_token_default":
                return c["value"]
        return ""

    @_serialized_browser_operation
    async def _seed_ms_token(self):
        """Seed initial msToken from browser cookies."""
        cookies = await self._context.cookies("https://www.doubao.com")
        for c in cookies:
            if c["name"] == "msToken":
                self._ms_token = c["value"]
                log.info("Seeded msToken from cookies (%d chars)", len(c["value"]))
                return
        log.warning("No msToken cookie found - first request may trigger rate limit")

    @_serialized_browser_operation
    async def _sign_url(self, base_url: str, params: Dict[str, str]) -> str:
        """Sign a URL using bdms.frontierSign with retry on failure."""
        sorted_params = dict(sorted(params.items()))
        query_string = urlencode(sorted_params)

        last_error = None
        for attempt in range(3):
            try:
                sig = await self._page.evaluate(
                    f'window.bdms.frontierSign("{query_string}")'
                )

                if isinstance(sig, dict):
                    if "a_bogus" in sig:
                        return f"{base_url}?{query_string}&a_bogus={sig['a_bogus']}"
                    elif "X-Bogus" in sig:
                        return f"{base_url}?{query_string}&X-Bogus={sig['X-Bogus']}"
                    elif sig:
                        k, v = next(iter(sig.items()))
                        return f"{base_url}?{query_string}&{k}={v}"
                elif isinstance(sig, str) and sig:
                    return f"{base_url}?{query_string}&X-Bogus={sig}"

                last_error = "empty signature"
            except Exception as e:
                last_error = f"signing failed ({type(e).__name__})"
                log.warning("frontierSign attempt %d failed (%s)", attempt + 1, type(e).__name__)

            if attempt < 2:
                await asyncio.sleep(1)

        log.error("frontierSign failed after 3 attempts: %s", last_error)
        raise RuntimeError(f"Failed to generate X-Bogus signature: {last_error}")

    def _build_query_params(self) -> Dict[str, str]:
        """Build the standard query parameters for API calls."""
        params = {
            "aid": "497858",
            "device_id": self._device_id or "",
            "device_platform": "web",
            "doubao_device_platform": "web",
            "web_platform": "browser",
            "fp": self._fp or "",
            "language": "zh",
            "pc_version": "3.36.0",
            "doubao_pc_version": "3.36.0",
            "pkg_type": "release_version",
            "real_aid": "497858",
            "region": "CN",
            "sys_region": "CN",
            "samantha_web": "1",
            "tea_uuid": self._web_id or "",
            "tz_name": "Asia/Shanghai",
            "use-olympus-account": "1",
            "version_code": "20800",
            "web_id": self._web_id or "",
            "web_tab_id": str(uuid.uuid4()),
        }
        if self._ms_token:
            params["msToken"] = self._ms_token
        return params

    def _build_headers(self, cookie_str: str, csrf_token: str = "") -> Dict[str, str]:
        """Build request headers."""
        # Extract CSRF token from cookie string if not provided
        if not csrf_token:
            for part in cookie_str.split("; "):
                if part.startswith("passport_csrf_token="):
                    csrf_token = part.split("=", 1)[1]
                    break
        ua = self._user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        )
        headers = {
            "Accept": "*/*",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Content-Type": "application/json",
            "Cookie": cookie_str,
            "Origin": DOUBAO_URL,
            "Referer": CHAT_URL,
            "User-Agent": ua,
            "agw-js-conv": "str, str",
        }
        if csrf_token:
            headers["x-tt-passport-csrf-token"] = csrf_token
        return headers

    # ------------------------------------------------------------------
    # Chat Completion (streaming via in-browser fetch)
    # ------------------------------------------------------------------

    @_serialized_browser_operation
    async def chat_completion(
        self,
        text: str,
        conversation_id: Optional[str] = None,
        bot_id: Optional[str] = None,
        use_deep_think: int = 0,
        image_attachments: Optional[List[Dict[str, Any]]] = None,
        file_attachments: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Send a chat message and yield SSE events via in-browser fetch."""
        if not self._page or self._page.is_closed() or not await self.is_alive():
            log.warning("Browser page is closed or not alive. Auto-recovering browser now...")
            await self.restart()

        if not self._ready:
            raise RuntimeError("Browser not ready - need login first")

        need_create = conversation_id is None or conversation_id == ""
        effective_bot_id = bot_id or DEFAULT_BOT_ID
        msg_uuid = str(uuid.uuid4())
        local_conv_id = f"local_{uuid.uuid4().int % 10**16}"
        now_ms = int(time.time() * 1000)
        now_sec = int(time.time())

        content_blocks: List[Dict[str, Any]] = []

        # 1. File attachments (type=3)
        if file_attachments:
            att_list = []
            for f in file_attachments:
                att_list.append({
                    "type": 3,
                    "identifier": str(uuid.uuid4()),
                    "file": {
                        "uri": f.get("uri", ""),
                        "url": "",
                        "file_type": 0,
                        "name": f.get("name", "file.txt"),
                        "size": int(f.get("size") or 0),
                    },
                    "parse_state": 1,
                    "review_state": 1,
                    "upload_status": 1,
                    "progress": 100,
                    "src": "",
                })
            content_blocks.append({
                "block_type": 10052,
                "content": {
                    "attachment_block": {"attachments": att_list},
                    "pc_event_block": "",
                },
                "block_id": str(uuid.uuid4()),
                "parent_id": "",
                "meta_info": [],
                "append_fields": [],
            })

        # 2. Image attachments (type=1) with genuine dimensions
        if image_attachments:
            for img in image_attachments:
                content_blocks.append({
                    "block_type": 10052,
                    "content": {
                        "attachment_block": {
                            "attachments": [{
                                "type": 1,
                                "identifier": str(uuid.uuid4()),
                                "image": {
                                    "name": img.get("name", "image.png"),
                                    "uri": img.get("uri", ""),
                                    "image_ori": {
                                        "url": img.get("cdn_url", ""),
                                        "width": int(img.get("width") or 1024),
                                        "height": int(img.get("height") or 1024),
                                        "format": img.get("format", "png"),
                                        "url_formats": {},
                                    },
                                },
                                "parse_state": 0,
                                "review_state": 0,
                            }],
                        },
                        "pc_event_block": "",
                    },
                    "block_id": str(uuid.uuid4()),
                    "parent_id": "",
                    "meta_info": [],
                    "append_fields": [],
                    "is_finish": True,
                    "patch_type": 2,
                })

        # 3. Text block (block_type 10000)
        content_blocks.append({
            "block_type": 10000,
            "content": {
                "text_block": {"text": text, "icon_url": "", "icon_url_dark": "", "summary": ""},
                "pc_event_block": "",
            },
            "block_id": str(uuid.uuid4()),
            "parent_id": "",
            "meta_info": [],
            "append_fields": [],
        })

        payload = {
            "client_meta": {
                "local_conversation_id": local_conv_id if need_create else "",
                "conversation_id": conversation_id or "",
                "bot_id": effective_bot_id,
                "last_section_id": "",
                "last_message_index": None,
                "local_permissions": [
                    {"permission_name": "ACCESS_COARSE_LOCATION", "status": 3},
                    {"permission_name": "ACCESS_FINE_LOCATION", "status": 3},
                    {"permission_name": "ACCESS_BACKGROUND_LOCATION", "status": 3},
                ],
            },
            "messages": [{
                "local_message_id": msg_uuid,
                "content_block": content_blocks,
                "message_status": 0,
            }],
            "option": {
                "send_message_scene": "",
                "create_time_ms": now_ms,
                "collect_id": "",
                "is_audio": False,
                "answer_with_suggest": False,
                "agent_mode": 2,
                "tts_switch": False,
                "need_deep_think": use_deep_think,
                "click_clear_context": False,
                "from_suggest": False,
                "is_regen": False,
                "is_replace": False,
                "is_from_click_option": False,
                "is_from_click_softlink": False,
                "disable_sse_cache": False,
                "select_text_action": "",
                "is_select_text": False,
                "resend_for_regen": False,
                "scene_type": 0,
                "unique_key": str(uuid.uuid4()),
                "start_seq": 0,
                "need_create_conversation": need_create,
                "conversation_init_option": {"need_ack_conversation": True},
                "conversation_init_ext": {
                    "model_item_key": "0",
                    "reasoning_effort": str(use_deep_think) if use_deep_think else "0",
                    "mode_id": "1",
                },
                "regen_query_id": [],
                "edit_query_id": [],
                "regen_instruction": "",
                "no_replace_for_regen": False,
                "message_from": 0,
                "shared_app_name": "",
                "shared_app_id": "",
                "sse_recv_event_options": {"support_chunk_delta": True},
                "is_ai_playground": False,
                "is_old_user": True,
                "recovery_option": {
                    "is_recovery": False,
                    "req_create_time_sec": now_sec,
                    "append_sse_event_scene": 0,
                },
                "message_storage_type": 0,
                "related_deleted_message_ids": {},
                "connector_info_list": [],
                "model_config": {
                    "model_item_key": "0",
                    "model_extra_params": {},
                    "reasoning_effort": use_deep_think,
                },
                "aggregate_params": {
                    "conversation_mode": "1",
                    "mode_id": "1",
                    "model_item_key": "0",
                    "agent_mode": "2",
                    "reasoning_effort": str(use_deep_think) if use_deep_think else "0",
                    "provider_id": "",
                },
                "conversation_mode": 1,
            },
            "user_context": [],
            "ext": {
                "agent_mode": "2",
                "use_deep_think": str(use_deep_think),
                "sub_conv_firstmet_type": "1" if need_create else "0",
                "collection_id": "",
                "is_finish": "1",
                "conversation_init_option": json.dumps({"need_ack_conversation": True}),
                "commerce_credit_config_enable": "0",
            },
        }

        # Build URL with query params and X-Bogus signature via bdms.frontierSign
        query_params = self._build_query_params()
        url = await self._sign_url("/chat/completion", query_params)

        request_id = f"req_{uuid.uuid4().hex[:16]}"
        queue: asyncio.Queue = asyncio.Queue()
        self._stream_queues[request_id] = queue

        log.info("POST %s (conv=%s, deep_think=%s) [browser fetch]",
                 url.split("?")[0], conversation_id or "new", use_deep_think)

        # Launch browser fetch in background with completion watcher
        eval_task = asyncio.create_task(
            self._browser_fetch_stream(url, payload, request_id)
        )

        def _on_eval_done(task: asyncio.Task):
            try:
                exc = task.exception()
                if exc:
                    log.error("eval_task crashed for %s (%s)", request_id, type(exc).__name__)
                    queue.put_nowait(f"__ERROR__:{exc}")
            except (asyncio.CancelledError, Exception):
                pass

        eval_task.add_done_callback(_on_eval_done)

        # Yield parsed SSE events from queue
        stream_complete = False
        try:
            while True:
                chunk_json = await asyncio.wait_for(queue.get(), timeout=180)
                if chunk_json is None:
                    # Stream complete
                    stream_complete = True
                    break
                if chunk_json.startswith("__ERROR__:"):
                    error_msg = chunk_json[10:]
                    log.error("Browser fetch failed for request %s", request_id)
                    yield {"error": True, "status": 0, "body": error_msg}
                    break
                if chunk_json.startswith("__HTTP_ERROR__:"):
                    status = int(chunk_json[15:].split(":", 1)[0])
                    body = chunk_json[15:].split(":", 1)[1] if ":" in chunk_json[15:] else ""
                    log.error("Upstream HTTP request failed with status %d", status)
                    yield {"error": True, "status": status, "body": body}
                    break
                # Parse SSE line
                try:
                    data = json.loads(chunk_json)
                    yield data
                except json.JSONDecodeError:
                    continue
        except asyncio.TimeoutError:
            log.error("Stream timeout (180s) for request %s", request_id)
            yield {"error": True, "status": 0, "body": "Stream timeout"}
        finally:
            self._stream_queues.pop(request_id, None)
            if stream_complete and not eval_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(eval_task), timeout=2.0)
                except (asyncio.TimeoutError, Exception):
                    pass
            if not eval_task.done():
                if self._page and not self._page.is_closed():
                    try:
                        await asyncio.wait_for(
                            self._page.evaluate(
                                """requestId => {
                                    const controllers = window.__doubaoStreamAbortControllers;
                                    const controller = controllers && controllers.get(requestId);
                                    if (controller) controller.abort();
                                }""",
                                request_id,
                            ),
                            timeout=2.0,
                        )
                    except Exception as exc:
                        log.debug("Browser stream abort failed (%s)", type(exc).__name__)
                eval_task.cancel()
                await asyncio.gather(eval_task, return_exceptions=True)
            else:
                # Check for exceptions
                try:
                    eval_task.result()
                except Exception:
                    pass

    @_serialized_browser_operation
    async def _browser_fetch_stream(
        self, url: str, payload: Dict[str, Any], request_id: str
    ):
        """Execute fetch() inside browser page and stream SSE chunks via callback."""
        js_code = """
        async ([url, payloadJson, requestId]) => {
            const controllers = window.__doubaoStreamAbortControllers ||
                (window.__doubaoStreamAbortControllers = new Map());
            const controller = new AbortController();
            controllers.set(requestId, controller);
            let reader = null;
            const sendChunk = async (chunk) => {
                if (typeof window.__doubaoStreamChunk === 'function') {
                    try {
                        await window.__doubaoStreamChunk(requestId, chunk);
                    } catch(e) {}
                }
            };
            try {
                if (typeof window.__doubaoStreamChunk !== 'function') {
                    throw new Error('window.__doubaoStreamChunk bridge not registered on page');
                }
                const csrf = document.cookie.match(/passport_csrf_token=([^;]+)/);
                const csrfToken = csrf ? csrf[1] : '';
                const headers = {
                    'Content-Type': 'application/json',
                    'agw-js-conv': 'str',
                };
                if (csrfToken) {
                    headers['x-tt-passport-csrf-token'] = csrfToken;
                }
                const res = await fetch(url, {
                    method: 'POST',
                    headers: headers,
                    body: payloadJson,
                    credentials: 'include',
                    signal: controller.signal,
                });
                if (!res.ok) {
                    const errBody = await res.text();
                    await sendChunk('__HTTP_ERROR__:' + res.status + ':' + errBody.slice(0, 500));
                    return;
                }
                reader = res.body.getReader();
                const decoder = new TextDecoder();
                let currentEvent = '';
                let buffer = '';
                while (true) {
                    const {done, value} = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, {stream: true});
                    const lines = buffer.split('\\n');
                    buffer = lines.pop();
                    for (const line of lines) {
                        const trimmed = line.trim();
                        if (!trimmed) continue;
                        if (trimmed.startsWith('event: ')) {
                            currentEvent = trimmed.slice(7);
                            continue;
                        }
                        if (trimmed.startsWith('id: ')) continue;
                        if (!trimmed.startsWith('data: ')) continue;
                        const dataStr = trimmed.slice(6);
                        if (!dataStr || dataStr === '{}') continue;
                        try {
                            const obj = JSON.parse(dataStr);
                            obj._event = currentEvent;
                            await sendChunk(JSON.stringify(obj));
                        } catch(e) {}
                    }
                }
                // Process remaining buffer
                if (buffer.trim()) {
                    const trimmed = buffer.trim();
                    if (trimmed.startsWith('data: ')) {
                        const dataStr = trimmed.slice(6);
                        if (dataStr && dataStr !== '{}') {
                            try {
                                const obj = JSON.parse(dataStr);
                                obj._event = currentEvent;
                                await sendChunk(JSON.stringify(obj));
                            } catch(e) {}
                        }
                    }
                }
                // Signal completion
                await sendChunk(null);
            } catch(e) {
                if (e.name !== 'AbortError') {
                    await sendChunk('__ERROR__:' + e.message);
                    throw e;
                }
            } finally {
                if (reader) {
                    try { await reader.cancel(); } catch(e) {}
                }
                controllers.delete(requestId);
            }
        }
        """
        payload_json = json.dumps(payload, ensure_ascii=False)
        await self._page.evaluate(js_code, [url, payload_json, request_id])

    # ------------------------------------------------------------------
    # High-level chat helper
    # ------------------------------------------------------------------

    @_serialized_browser_operation
    async def chat(
        self,
        text: str,
        conversation_id: Optional[str] = None,
        bot_id: Optional[str] = None,
        use_deep_think: int = 0,
    ) -> Dict[str, Any]:
        """Send message, collect full response. Returns {text, conversation_id}."""
        full_text = ""
        result_conv_id = conversation_id
        events = []

        async for event in self.chat_completion(
            text, conversation_id=conversation_id,
            bot_id=bot_id, use_deep_think=use_deep_think
        ):
            events.append(event)
            if event.get("error"):
                raise RuntimeError(
                    f"API error {event.get('status')}: {event.get('body', '')[:200]}"
                )
            if not result_conv_id:
                cid = self.extract_conversation_id(event)
                if cid and cid != "0":
                    result_conv_id = cid
            full_text += self._extract_text(event)

        return {"text": full_text, "conversation_id": result_conv_id}

    # ------------------------------------------------------------------
    # SSE parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(event: Dict[str, Any]) -> str:
        """Extract text content from a SSE event."""
        event_type = event.get("_event", "")

        if event_type == "CHUNK_DELTA" and "text" in event:
            return event["text"]

        if "patch_op" in event:
            for op in event["patch_op"]:
                pv = op.get("patch_value", {})
                for block in pv.get("content_block", []):
                    content = block.get("content", {})
                    tb = content.get("text_block", {})
                    if tb.get("text"):
                        return tb["text"]
                if op.get("patch_object") == 102:
                    raw = pv.get("content", "")
                    if raw:
                        try:
                            parsed = json.loads(raw)
                            if parsed.get("text"):
                                return parsed["text"]
                        except (json.JSONDecodeError, TypeError):
                            pass

        if event_type == "STREAM_MSG_NOTIFY":
            content = event.get("content", {})
            if isinstance(content, dict):
                for block in content.get("content_block", []):
                    tb = block.get("content", {}).get("text_block", {})
                    if tb.get("text"):
                        return tb["text"]

        return ""

    @staticmethod
    def extract_conversation_id(event: Dict[str, Any]) -> Optional[str]:
        """Extract conversation_id from SSE events."""
        ack = event.get("ack_client_meta", {})
        if ack.get("conversation_id"):
            return ack["conversation_id"]
        meta = event.get("meta", {})
        if meta.get("conversation_id"):
            return meta["conversation_id"]
        return None

    @_serialized_browser_operation
    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation from Doubao via /im/conversation/batch_del_user_conv.

        Keeps the web UI sidebar completely clean (即用即焚 / Ephemeral Conversations).
        Returns True if deletion succeeded, False otherwise.
        """
        if not conversation_id or str(conversation_id).strip() in ("0", ""):
            return False

        if not self._ready or self._page is None:
            log.warning("Cannot delete conversation %s: browser not ready", conversation_id)
            return False

        query_params = self._build_query_params()
        try:
            signed_url = await self._sign_url("/im/conversation/batch_del_user_conv", query_params)
        except Exception as exc:
            log.warning("Failed to sign delete url for %s: %s", conversation_id, exc)
            return False

        payload = {
            "cmd": 4171,
            "uplink_body": {
                "batch_delete_user_conversation_uplink_body": {
                    "conversation_id": [str(conversation_id)],
                    "delete_all": False,
                    "conversation_type": 3,
                }
            },
            "sequence_id": str(uuid.uuid4()),
            "channel": 2,
            "version": "1",
        }

        try:
            res = await self._page.evaluate('''async (args) => {
                const [url, payload] = args;
                const csrfMatch = document.cookie.match(/passport_csrf_token=([^;]+)/);
                const csrf = csrfMatch ? csrfMatch[1] : "";
                try {
                    const resp = await fetch(url, {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json; encoding=utf-8",
                            "x-tt-passport-csrf-token": csrf,
                            "agw-js-conv": "str",
                        },
                        body: JSON.stringify(payload),
                    });
                    const text = await resp.text();
                    return { status: resp.status, body: text };
                } catch(e) {
                    return { error: e.message };
                }
            }''', [signed_url, payload])

            if res.get("error"):
                log.warning("delete_conversation %s failed: %s", conversation_id, res["error"])
                return False

            data = json.loads(res.get("body", "{}"))
            if data.get("status_code") == 0:
                log.info("Successfully deleted ephemeral conversation %s (即用即焚)", conversation_id)
                return True
            else:
                log.warning("delete_conversation %s: status_code=%s desc=%s",
                            conversation_id, data.get("status_code"), data.get("status_desc"))
                return False
        except Exception as exc:
            log.warning("delete_conversation %s exception: %s", conversation_id, exc)
            return False

    # ------------------------------------------------------------------
    # Samantha endpoint (image/video/music generation)
    # ------------------------------------------------------------------

    @_serialized_browser_operation
    async def _samantha_request(
        self,
        payload: Dict[str, Any],
        timeout: float = 120,
    ) -> str:
        """Send a request to /samantha/chat/completion via in-browser fetch."""
        if not self._ready:
            raise RuntimeError("Browser not ready - need login first")

        query_params = self._build_query_params()
        query_string = "&".join(f"{k}={v}" for k, v in sorted(query_params.items()))
        url = f"/samantha/chat/completion?{query_string}"

        js_code = """
        async ([url, payloadJson, timeoutMs]) => {
            const csrf = document.cookie.match(/passport_csrf_token=([^;]+)/);
            const csrfToken = csrf ? csrf[1] : '';
            const headers = {
                'Content-Type': 'application/json',
                'agw-js-conv': 'str',
            };
            if (csrfToken) {
                headers['x-tt-passport-csrf-token'] = csrfToken;
            }
            const controller = new AbortController();
            const timer = setTimeout(() => controller.abort(), timeoutMs);
            try {
                const res = await fetch(url, {
                    method: 'POST',
                    headers: headers,
                    body: payloadJson,
                    credentials: 'include',
                    signal: controller.signal,
                });
                clearTimeout(timer);
                if (!res.ok) {
                    const errBody = await res.text();
                    return {error: true, status: res.status, body: errBody.slice(0, 500)};
                }
                const body = await res.text();
                return {error: false, body: body};
            } catch(e) {
                clearTimeout(timer);
                return {error: true, status: 0, body: e.message};
            }
        }
        """
        payload_json = json.dumps(payload, ensure_ascii=False)
        timeout_ms = int(timeout * 1000)

        log.info("POST %s [browser fetch, timeout=%ds]", url.split("?")[0], timeout)
        result = await self._page.evaluate(
            js_code, [url, payload_json, timeout_ms]
        )

        if result.get("error"):
            status = result.get("status", 0)
            body = result.get("body", "")
            raise RuntimeError(
                f"samantha/chat/completion failed ({status}): {body[:500]}"
            )

        body = result.get("body", "")
        if body.lstrip().startswith("{"):
            try:
                err = json.loads(body)
                if isinstance(err, dict) and "code" in err:
                    raise RuntimeError(
                        f"samantha auth error: code={err.get('code')} "
                        f"msg={err.get('msg') or err.get('message', '')}"
                    )
            except json.JSONDecodeError:
                pass
        return body

    @staticmethod
    def _parse_samantha_sse(raw: str) -> List[Dict[str, Any]]:
        """Parse samantha SSE body into list of event dicts."""
        events = []
        for block in raw.split("\n\n"):
            if not block.strip():
                continue
            data_str = ""
            for line in block.strip().split("\n"):
                if line.startswith("data:"):
                    data_str = line[5:].strip()
            if not data_str:
                continue
            try:
                events.append(json.loads(data_str))
            except json.JSONDecodeError:
                continue
        return events

    @_serialized_browser_operation
    async def generate_image(
        self,
        prompt: str,
        ratio: Optional[str] = None,
        ref_image_key: Optional[str] = None,
        ref_image_keys: Optional[List[str]] = None,
        memory_type: Optional[int] = None,
        save_history: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Generate images using /samantha/chat/completion.

        Args:
            prompt: Text description of the image to generate.
            ratio: Aspect ratio ("1:1", "16:9", "9:16", "4:3", "3:4").
            ref_image_key: Optional uploaded image key for reference (single image).
            ref_image_keys: Optional list of uploaded image keys for reference (multi-image).
            memory_type: Optional 0 (save to web history) or 2 (ephemeral).
            save_history: If True, sets memory_type=0.

        Returns:
            Dict with 'images' list, each having url/width/height/key.
        """
        content_data: Dict[str, Any] = {"text": prompt}
        if ratio:
            content_data["ratio"] = ratio

        message: Dict[str, Any] = {
            "content": json.dumps(content_data, ensure_ascii=False),
            "content_type": 2009,
            "attachments": [],
            "references": [],
            "skill": {
                "skill_type": 3,
                "skill_type_no_default": 3,
                "skill_id": "3",
                "skill_id_no_default": "3",
            },
        }

        keys = []
        if ref_image_keys:
            if isinstance(ref_image_keys, str):
                keys.append(ref_image_keys)
            elif isinstance(ref_image_keys, (list, tuple)):
                keys.extend([k for k in ref_image_keys if k])
        if ref_image_key and ref_image_key not in keys:
            keys.insert(0, ref_image_key)

        if keys:
            message["attachments"] = [
                {"type": "image", "key": k, "extra": {"refer_types": "overall"}}
                for k in keys
            ]

        eff_mem_type = resolve_memory_type(memory_type, save_history)
        payload = {
            "messages": [message],
            "completion_option": {
                "is_regen": False,
                "with_suggest": True,
                "need_create_conversation": True,
                "launch_stage": 1,
                "is_replace": False,
                "is_delete": False,
                "is_ai_playground": False,
                "memory_type": eff_mem_type,
                "message_from": 0,
                "use_deep_think": False,
                "use_auto_cot": False,
                "resend_for_regen": False,
                "enable_commerce_credit": False,
                "action_bar_skill_id": 3,
            },
            "evaluate_option": {"web_ab_params": ""},
            "local_conversation_id": str(uuid.uuid4()),
            "local_message_id": str(uuid.uuid4()),
        }

        log.info("generate_image: prompt=%s, ratio=%s", prompt[:50], ratio)
        raw = await self._samantha_request(payload, timeout=120)

        # Parse response - look for content_type=2010 (image output)
        images = []
        for data in self._parse_samantha_sse(raw):
            et = data.get("event_type")
            if et == 2005:
                detail = data.get("event_data", "")
                raise RuntimeError(f"generate_image error: {str(detail)[:500]}")
            if et != 2001:
                continue

            ed = data.get("event_data", {})
            if isinstance(ed, str):
                try:
                    ed = json.loads(ed)
                except json.JSONDecodeError:
                    continue

            msg = ed.get("message", {})
            if isinstance(msg, str):
                try:
                    msg = json.loads(msg)
                except json.JSONDecodeError:
                    continue

            if msg.get("content_type") != 2010:
                continue

            content_raw = msg.get("content", "")
            if isinstance(content_raw, str):
                try:
                    content = json.loads(content_raw)
                except json.JSONDecodeError:
                    continue
            else:
                content = content_raw

            for item in content.get("data", []):
                if not isinstance(item, dict):
                    continue
                ori = item.get("image_ori", {}) or {}
                raw_img = item.get("image_raw", {}) or {}
                thumb = item.get("image_thumb", {}) or {}
                images.append({
                    "key": item.get("key", ""),
                    "url": ori.get("url") or raw_img.get("url") or thumb.get("url", ""),
                    "width": ori.get("width") or thumb.get("width", 0),
                    "height": ori.get("height") or thumb.get("height", 0),
                    "format": ori.get("format") or thumb.get("format", ""),
                })

        log.info("generate_image: got %d images", len(images))
        return {"images": images, "prompt": prompt}

    @_serialized_browser_operation
    async def generate_music(
        self,
        prompt: str,
        lyric: Optional[str] = None,
        genre: Optional[str] = None,
        memory_type: Optional[int] = None,
        save_history: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Generate music using /samantha/chat/completion.

        Args:
            prompt: Text description of the music to generate.
            lyric: Explicit lyrics (optional).
            genre: Music genre (optional).
            memory_type: Optional 0 (save to web history) or 2 (ephemeral).
            save_history: If True, sets memory_type=0.

        Returns:
            Dict with 'tracks' list, each having audio_url/title/lyrics/duration.
        """
        import base64

        content_data: Dict[str, Any] = {"text": prompt}
        if lyric:
            content_data["lyric"] = lyric
        if genre:
            content_data["genre"] = genre

        message: Dict[str, Any] = {
            "content": json.dumps(content_data, ensure_ascii=False),
            "content_type": 2005,
            "attachments": [],
            "references": [],
            "skill": {
                "skill_type": 9,
                "skill_type_no_default": 9,
                "skill_id": "9",
                "skill_id_no_default": "9",
            },
        }

        eff_mem_type = resolve_memory_type(memory_type, save_history)
        payload = {
            "messages": [message],
            "completion_option": {
                "is_regen": False,
                "with_suggest": True,
                "need_create_conversation": True,
                "launch_stage": 1,
                "is_replace": False,
                "is_delete": False,
                "is_ai_playground": False,
                "memory_type": eff_mem_type,
                "message_from": 0,
                "use_deep_think": False,
                "use_auto_cot": False,
                "resend_for_regen": False,
                "enable_commerce_credit": False,
                "action_bar_skill_id": 9,
            },
            "evaluate_option": {"web_ab_params": ""},
            "local_conversation_id": str(uuid.uuid4()),
            "local_message_id": str(uuid.uuid4()),
        }

        log.info("generate_music: prompt=%s", prompt[:50])
        raw = await self._samantha_request(payload, timeout=300)

        # Parse: find last content_type=2006 with video_model
        tracks = []
        final_content = None
        for data in self._parse_samantha_sse(raw):
            et = data.get("event_type")
            if et == 2005:
                detail = data.get("event_data", "")
                raise RuntimeError(f"generate_music error: {str(detail)[:500]}")
            if et != 2001:
                continue

            ed = data.get("event_data", {})
            if isinstance(ed, str):
                try:
                    ed = json.loads(ed)
                except json.JSONDecodeError:
                    continue

            msg = ed.get("message", {})
            if isinstance(msg, str):
                try:
                    msg = json.loads(msg)
                except json.JSONDecodeError:
                    continue

            if msg.get("content_type") not in (2006, 2004):
                continue

            content_raw = msg.get("content", "")
            if isinstance(content_raw, str):
                try:
                    content = json.loads(content_raw)
                except json.JSONDecodeError:
                    continue
            else:
                content = content_raw

            # Keep updating - we want the final (most complete) version
            final_content = content

        if not final_content:
            log.warning("generate_music: no content_type=2006 found")
            return {"tracks": [], "prompt": prompt}

        # Parse tasks
        tasks = final_content.get("tasks", {})
        if isinstance(tasks, dict):
            tasks_list = list(tasks.values())
        elif isinstance(tasks, list):
            tasks_list = tasks
        else:
            tasks_list = []

        for task in tasks_list:
            if not isinstance(task, dict):
                continue

            audio_url = ""
            duration = 0.0
            vm_str = task.get("video_model", "")
            if vm_str:
                try:
                    vm = json.loads(vm_str) if isinstance(vm_str, str) else vm_str
                    duration = vm.get("video_duration", 0.0)
                    vlist = vm.get("video_list", {})
                    for _q, vinfo in vlist.items():
                        main_url_b64 = vinfo.get("main_url", "")
                        if main_url_b64:
                            audio_url = base64.b64decode(main_url_b64).decode(
                                "utf-8", errors="replace"
                            )
                            break
                except (json.JSONDecodeError, Exception):
                    pass

            cover_url = ""
            cover = task.get("cover", {})
            if isinstance(cover, dict):
                cover_ori = cover.get("image_ori", {}) or {}
                cover_url = cover_ori.get("url", "")

            if audio_url or task.get("title"):
                tracks.append({
                    "audio_url": audio_url,
                    "title": task.get("title", ""),
                    "lyrics": task.get("lyric", ""),
                    "duration": duration,
                    "cover_url": cover_url,
                })

        log.info("generate_music: got %d tracks", len(tracks))
        return {"tracks": tracks, "prompt": prompt}

    @_serialized_browser_operation
    async def generate_video(
        self,
        prompt: str,
        ratio: Optional[str] = None,
        ref_image_key: Optional[str] = None,
        ref_image_keys: Optional[List[str]] = None,
        memory_type: Optional[int] = None,
        save_history: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Generate video using /samantha/chat/completion (async 2-step).

        Args:
            prompt: Text description of the video to generate.
            ratio: Aspect ratio ("16:9", "9:16", "1:1").
            ref_image_key: Optional image key for image-to-video (single image).
            ref_image_keys: Optional list of image keys for multi-image reference.
            memory_type: Optional 0 (save to web history) or 2 (ephemeral).
            save_history: If True, sets memory_type=0.

        Returns:
            Dict with 'videos' list, each having video_url/cover_url/duration.
        """
        import base64

        content_data: Dict[str, Any] = {"text": prompt}
        if ratio:
            content_data["ratio"] = ratio

        # Support both single and multiple reference images
        keys = []
        if ref_image_keys:
            if isinstance(ref_image_keys, str):
                keys.append(ref_image_keys)
            elif isinstance(ref_image_keys, (list, tuple)):
                keys.extend([k for k in ref_image_keys if k])
        if ref_image_key and ref_image_key not in keys:
            keys.insert(0, ref_image_key)

        attachments = [{"type": "image", "key": k} for k in keys]

        message: Dict[str, Any] = {
            "content": json.dumps(content_data, ensure_ascii=False),
            "content_type": 2020,
            "attachments": attachments,
            "references": [],
            "skill": {
                "skill_type": 17,
                "skill_type_no_default": 17,
                "skill_id": "17",
                "skill_id_no_default": "17",
            },
        }

        eff_mem_type = resolve_memory_type(memory_type, save_history)
        payload = {
            "messages": [message],
            "completion_option": {
                "is_regen": False,
                "with_suggest": True,
                "need_create_conversation": True,
                "launch_stage": 1,
                "is_replace": False,
                "is_delete": False,
                "is_ai_playground": False,
                "memory_type": eff_mem_type,
                "message_from": 0,
                "use_deep_think": False,
                "use_auto_cot": False,
                "resend_for_regen": False,
                "enable_commerce_credit": False,
                "action_bar_skill_id": 17,
            },
            "evaluate_option": {"web_ab_params": ""},
            "local_conversation_id": str(uuid.uuid4()),
            "local_message_id": str(uuid.uuid4()),
        }

        log.info("generate_video: prompt=%s, ratio=%s, memory_type=%s, attachments_count=%d", prompt[:50], ratio, eff_mem_type, len(attachments))
        raw = await self._samantha_request(payload, timeout=60)

        # Phase 1: Extract async task_id from fin_reason
        task_id = None
        text_parts = []
        for data in self._parse_samantha_sse(raw):
            et = data.get("event_type")
            if et == 2005:
                detail = data.get("event_data", "")
                is_quota = any(k in str(detail) for k in ("今日生成次数已达上限", "次数已达上限", "额度已用尽", "达到今日上限", "次数超限"))
                if is_quota and hasattr(self, "account_manager"):
                    self.account_manager.mark_quota_exceeded(
                        account_id=self.account_manager.active_account_id,
                        feature="video",
                    )
                    raise VideoQuotaExceeded("Video generation quota exceeded for active account")
                raise RuntimeError(f"generate_video error: {str(detail)[:500]}")
            if et != 2001:
                continue

            ed = data.get("event_data", {})
            if isinstance(ed, str):
                try:
                    ed = json.loads(ed)
                except json.JSONDecodeError:
                    continue

            # Check for async task
            fin_reason = ed.get("fin_reason", {})
            if fin_reason and fin_reason.get("reason") == 1:
                async_task = fin_reason.get("async_task", {})
                task_id = async_task.get("id", "")

            # Collect text for error messages
            msg = ed.get("message", {})
            if isinstance(msg, str):
                try:
                    msg = json.loads(msg)
                except json.JSONDecodeError:
                    continue
            if msg.get("content_type") == 2001:
                content_raw = msg.get("content", "")
                if isinstance(content_raw, str):
                    try:
                        c = json.loads(content_raw)
                        text_parts.append(c.get("text", ""))
                    except json.JSONDecodeError:
                        pass

        full_text = "".join(text_parts)
        is_quota = any(k in full_text for k in ("今日生成次数已达上限", "次数已达上限", "额度已用尽", "达到今日上限", "次数超限"))
        if is_quota and hasattr(self, "account_manager"):
            self.account_manager.mark_quota_exceeded(
                account_id=self.account_manager.active_account_id,
                feature="video",
            )
            raise VideoQuotaExceeded("Video generation quota exceeded for active account")

        if "服务过载" in full_text or "重试" in full_text:
            raise RuntimeError("视频生成服务过载，请稍后重试")

        if not task_id:
            # Maybe sync result with content_type=2021, or just text response
            if full_text:
                return {"videos": [], "prompt": prompt, "message": full_text}
            raise RuntimeError("Video generation: no task_id returned")

        # Phase 2: Poll for result
        log.info("generate_video: polling task_id=%s", task_id)
        return await self._poll_video_result(task_id, prompt)

    @_serialized_browser_operation
    async def _poll_video_result(
        self, task_id: str, prompt: str, timeout: float = 300
    ) -> Dict[str, Any]:
        """Poll /samantha/chat/completion with task_id for video result."""
        import base64

        poll_payload = {"task_id": task_id, "event_id": 0}
        # Use _samantha_request which now uses browser fetch
        raw = await self._samantha_request(poll_payload, timeout=timeout)

        videos = []
        for data in self._parse_samantha_sse(raw):
            et = data.get("event_type")
            if et != 2001:
                continue

            ed = data.get("event_data", {})
            if isinstance(ed, str):
                try:
                    ed = json.loads(ed)
                except json.JSONDecodeError:
                    continue

            msg = ed.get("message", {})
            if isinstance(msg, str):
                try:
                    msg = json.loads(msg)
                except json.JSONDecodeError:
                    continue

            if msg.get("content_type") != 2021:
                continue

            content_raw = msg.get("content", "")
            if isinstance(content_raw, str):
                try:
                    content = json.loads(content_raw)
                except json.JSONDecodeError:
                    continue
            else:
                content = content_raw

            for item in content.get("data", [content]):
                if not isinstance(item, dict):
                    continue
                video_url = item.get("video_url", "") or item.get("url", "")
                if not video_url:
                    vm_str = item.get("video_model", "")
                    if vm_str:
                        try:
                            vm = json.loads(vm_str) if isinstance(vm_str, str) else vm_str
                            vlist = vm.get("video_list", {})
                            for _q, vinfo in vlist.items():
                                main_b64 = vinfo.get("main_url", "")
                                if main_b64:
                                    video_url = base64.b64decode(main_b64).decode(
                                        "utf-8", errors="replace"
                                    )
                                    break
                        except (json.JSONDecodeError, Exception):
                            pass

                cover_url = item.get("cover_url", "") or item.get("cover", {}).get("url", "")
                if video_url:
                    videos.append({
                        "video_url": video_url,
                        "cover_url": cover_url,
                        "width": item.get("width", 0),
                        "height": item.get("height", 0),
                        "duration": item.get("duration", 0.0),
                    })

        log.info("generate_video: got %d videos", len(videos))
        return {"videos": videos, "prompt": prompt}

    # ------------------------------------------------------------------
    # File upload (TOS / ImageX flow)
    # ------------------------------------------------------------------

    @_serialized_browser_operation
    async def upload_file(
        self,
        file_data: bytes,
        filename: str,
    ) -> Dict[str, Any]:
        """Upload a file to Doubao's storage (ByteDance TOS via ImageX proxy).

        4-step flow:
          1. POST /alice/resource/prepare_upload -> STS credentials
          2. GET  /top/v1?Action=ApplyImageUpload -> upload address
          3. POST https://{tos_host}/upload/v1/{store_uri} -> upload binary
          4. POST /top/v1?Action=CommitImageUpload -> confirm

        Returns:
            Dict with uri, name, size, file_type.
        """
        import zlib
        import hashlib
        import hmac as hmac_mod
        from datetime import datetime, timezone
        from urllib.parse import urlparse, parse_qs, quote as url_quote

        if not self._ready:
            raise RuntimeError("Browser not ready - need login first")

        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        file_size = len(file_data)
        crc32 = format(zlib.crc32(file_data) & 0xFFFFFFFF, "08x")

        query_params = self._build_query_params()
        signed_url = await self._sign_url(
            f"{DOUBAO_URL}/alice/resource/prepare_upload", query_params
        )
        cookie_str = await self._get_cookies_string()
        headers = self._build_headers(cookie_str)

        # Step 1: prepare_upload
        resp = await self._http.post(
            signed_url, headers=headers,
            json={"tenant_id": "5", "scene_id": "5", "resource_type": 1},
            timeout=30,
        )
        body = resp.json()
        if body.get("code") != 0:
            raise RuntimeError(f"prepare_upload failed: {body.get('msg', body)}")
        data = body["data"]
        service_id = data["service_id"]
        auth_token = data["upload_auth_token"]
        ak = auth_token["access_key"]
        sk = auth_token["secret_key"]
        st = auth_token["session_token"]

        # AWS V4 signing helper
        def _aws_sign_v4(method, url, req_body):
            parsed = urlparse(url)
            host = parsed.hostname or ""
            path = parsed.path or "/"
            now = datetime.now(timezone.utc)
            amz_date = now.strftime("%Y%m%dT%H%M%SZ")
            date_stamp = now.strftime("%Y%m%d")
            qparams = parse_qs(parsed.query, keep_blank_values=True)
            sorted_qp = sorted((k, v[0] if v else "") for k, v in qparams.items())
            canonical_qs = "&".join(
                f"{url_quote(k, safe='~')}={url_quote(v, safe='~')}" for k, v in sorted_qp
            )
            h2s = {"host": host, "x-amz-date": amz_date}
            if st:
                h2s["x-amz-security-token"] = st
            signed_h = ";".join(sorted(h2s.keys()))
            canonical_h = "".join(f"{k}:{v}\n" for k, v in sorted(h2s.items()))
            body_b = req_body if isinstance(req_body, bytes) else req_body.encode()
            payload_hash = hashlib.sha256(body_b).hexdigest()
            cr = f"{method}\n{path}\n{canonical_qs}\n{canonical_h}\n{signed_h}\n{payload_hash}"
            scope = f"{date_stamp}/cn-north-1/imagex/aws4_request"
            cr_hash = hashlib.sha256(cr.encode()).hexdigest()
            sts = f"AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{cr_hash}"
            def _s(key, msg):
                return hmac_mod.new(key, msg.encode("utf-8"), hashlib.sha256).digest()
            k_d = _s(f"AWS4{sk}".encode("utf-8"), date_stamp)
            k_r = _s(k_d, "cn-north-1")
            k_sv = _s(k_r, "imagex")
            k_sg = _s(k_sv, "aws4_request")
            sig = hmac_mod.new(k_sg, sts.encode("utf-8"), hashlib.sha256).hexdigest()
            auth_str = f"AWS4-HMAC-SHA256 Credential={ak}/{scope}, SignedHeaders={signed_h}, Signature={sig}"
            result = {"Authorization": auth_str, "x-amz-date": amz_date, "x-amz-content-sha256": payload_hash}
            if st:
                result["x-amz-security-token"] = st
            return result

        # Step 2: ApplyImageUpload
        file_ext = f".{ext}" if ext else ""
        apply_url = (
            f"{DOUBAO_URL}/top/v1?"
            f"Action=ApplyImageUpload&Version=2018-08-01"
            f"&ServiceId={service_id}&NeedFallback=true"
            f"&FileSize={file_size}&FileExtension={file_ext}"
            f"&s=jdnfglwfkl"
        )
        sign_h = _aws_sign_v4("GET", apply_url, "")
        sign_h["Cookie"] = cookie_str
        resp = await self._http.get(apply_url, headers=sign_h, timeout=30)
        result_data = resp.json().get("Result")
        if not result_data:
            raise RuntimeError(f"ApplyImageUpload failed: {resp.json()}")
        upload_addr = result_data["UploadAddress"]
        store_info = upload_addr["StoreInfos"][0]
        store_uri = store_info["StoreUri"]
        tos_auth = store_info["Auth"]
        session_key = upload_addr["SessionKey"]
        upload_hosts = upload_addr.get("UploadHosts", [])

        # Step 3: Upload binary to TOS
        tos_host = upload_hosts[0] if upload_hosts else "tos-mya2lf.vodupload.com"
        upload_url = f"https://{tos_host}/upload/v1/{store_uri}"
        resp = await self._http.post(
            upload_url, content=file_data,
            headers={"Authorization": tos_auth, "Content-CRC32": crc32},
            timeout=120,
        )
        tos_resp = resp.json()
        if tos_resp.get("code") != 2000:
            raise RuntimeError(f"TOS upload failed: {tos_resp}")

        # Step 4: CommitImageUpload
        commit_url = (
            f"{DOUBAO_URL}/top/v1?"
            f"Action=CommitImageUpload&Version=2018-08-01"
            f"&ServiceId={service_id}"
        )
        commit_body = json.dumps({"SessionKey": session_key})
        sign_h2 = _aws_sign_v4("POST", commit_url, commit_body)
        sign_h2["Content-Type"] = "application/json"
        sign_h2["Cookie"] = cookie_str
        resp = await self._http.post(commit_url, content=commit_body, headers=sign_h2, timeout=30)
        body = resp.json()
        results = body.get("Result", {}).get("Results", [])
        if not results or results[0].get("UriStatus") != 2000:
            raise RuntimeError(f"CommitImageUpload failed: {body}")

        log.info("File uploaded: %s -> %s", filename, store_uri)
        return {"uri": store_uri, "name": filename, "size": file_size, "file_type": ext}


    @_serialized_browser_operation
    async def get_file_download_url(
        self,
        uri: str,
        expire_seconds: int = 3600,
    ) -> Dict[str, Any]:
        """Get a temporary CDN URL for a previously uploaded file."""
        if not self._ready:
            raise RuntimeError("Browser not ready - need login first")
        query_params = self._build_query_params()
        signed_url = await self._sign_url(
            f"{DOUBAO_URL}/alice/message/get_file_url", query_params
        )
        cookie_str = await self._get_cookies_string()
        headers = self._build_headers(cookie_str)
        ext = uri.rsplit(".", 1)[-1] if "." in uri else ""
        resp = await self._http.post(
            signed_url,
            headers=headers,
            json={
                "uris": [uri],
                "type": "file",
                "format": ext,
                "expire_second": expire_seconds,
            },
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"get_file_url failed ({resp.status_code}): {resp.text[:500]}")
        body = resp.json()
        if body.get("code") != 0:
            raise RuntimeError(f"get_file_url error: {body.get('msg', body)}")
        file_urls = body.get("data", {}).get("file_urls", [])
        if not file_urls:
            raise RuntimeError("get_file_url returned no file_urls")
        return file_urls[0].get("main_url", "")

    @_serialized_browser_operation
    async def upload_image(
        self,
        image_bytes: bytes,
        filename: str = "image.png",
    ) -> Dict[str, Any]:
        """Upload an image and return metadata usable by chat/image generation."""
        if not self._ready:
            raise RuntimeError("Browser not ready - need login first")
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "png"
        query_params = self._build_query_params()
        signed_url = await self._sign_url(
            f"{DOUBAO_URL}/samantha/pages/upload_image", query_params
        )
        cookie_str = await self._get_cookies_string()
        headers = self._build_headers(cookie_str)
        headers.pop("Content-Type", None)
        files = {
            "data": (filename, image_bytes, f"image/{ext}"),
            "file_type": (None, ext),
        }
        resp = await self._http.post(signed_url, headers=headers, files=files, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"Image upload failed ({resp.status_code}): {resp.text[:500]}")
        body = resp.json()
        if body.get("code") != 0:
            raise RuntimeError(f"Image upload error: {body.get('msg', body)}")
        uri = body.get("data", {}).get("uri", "")
        if not uri:
            raise RuntimeError(f"Image upload returned no uri: {body}")
        query_params = self._build_query_params()
        file_url = await self._sign_url(
            f"{DOUBAO_URL}/alice/message/get_file_url", query_params
        )
        cookie_str = await self._get_cookies_string()
        headers = self._build_headers(cookie_str)
        resp = await self._http.post(
            file_url,
            headers=headers,
            json={
                "uris": [uri],
                "type": "image",
                "format": ext,
                "expire_second": 3600,
            },
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"get_file_url failed ({resp.status_code}): {resp.text[:500]}")
        body = resp.json()
        if body.get("code") != 0:
            raise RuntimeError(f"get_file_url error: {body.get('msg', body)}")
        file_urls = body.get("data", {}).get("file_urls", [])
        if not file_urls:
            raise RuntimeError("get_file_url returned no file_urls")
        info = file_urls[0]
        img_w = 1024
        img_h = 1024
        try:
            import io
            from PIL import Image
            with Image.open(io.BytesIO(image_bytes)) as pil_img:
                img_w, img_h = pil_img.size
                if pil_img.format:
                    ext = pil_img.format.lower()
        except Exception:
            pass

        return {
            "uri": info.get("uri", uri),
            "cdn_url": info.get("main_url") or info.get("url", ""),
            "name": filename,
            "format": ext,
            "width": int(info.get("width") or img_w),
            "height": int(info.get("height") or img_h),
        }

    @_serialized_browser_operation
    async def chat_with_file(
        self,
        text: str,
        file_uri: str,
        file_name: str,
        file_size: int,
        use_deep_think: int = 0,
    ) -> Dict[str, Any]:
        """Chat with a file attachment. The AI will read the file and answer.

        Args:
            text: Question about the file.
            file_uri: URI from upload_file() or upload_image().
            file_name: Original filename.
            file_size: File size in bytes.
            use_deep_think: 0=quick, 1=think, 3=expert.

        Returns:
            Dict with 'text' and 'conversation_id'.
        """
        if not self._ready:
            raise RuntimeError("Browser not ready - need login first")

        msg_uuid = str(uuid.uuid4())
        local_conv_id = f"local_{uuid.uuid4().int % 10**16}"
        now_ms = int(time.time() * 1000)
        now_sec = int(time.time())

        if isinstance(file_uri, list):
            file_refs = file_uri
        else:
            file_refs = [{"uri": file_uri, "name": file_name, "size": file_size}]

        content_blocks: List[Dict[str, Any]] = []
        file_attachments = []
        image_attachments = []
        for file_ref in file_refs:
            fname = file_ref.get("name", "").lower()
            fext = fname.rsplit(".", 1)[-1] if "." in fname else ""
            is_img = fext in ("png", "jpg", "jpeg", "webp", "gif", "bmp") or file_ref.get("is_image")
            if is_img:
                image_attachments.append(file_ref)
            else:
                file_attachments.append({
                    "type": 3,
                    "identifier": str(uuid.uuid4()),
                    "file": {
                        "uri": file_ref.get("uri", ""),
                        "url": "",
                        "file_type": 0,
                        "name": file_ref.get("name", "file.txt"),
                        "size": int(file_ref.get("size") or 0),
                    },
                    "parse_state": 1,
                    "review_state": 1,
                    "upload_status": 1,
                    "progress": 100,
                    "src": "",
                })

        if file_attachments:
            content_blocks.append({
                "block_type": 10052,
                "content": {
                    "attachment_block": {
                        "attachments": file_attachments
                    },
                    "pc_event_block": "",
                },
                "block_id": str(uuid.uuid4()),
                "parent_id": "",
                "meta_info": [],
                "append_fields": [],
            })

        if image_attachments:
            for img in image_attachments:
                content_blocks.append({
                    "block_type": 10052,
                    "content": {
                        "attachment_block": {
                            "attachments": [{
                                "type": 1,
                                "identifier": str(uuid.uuid4()),
                                "image": {
                                    "name": img.get("name", "image.png"),
                                    "uri": img.get("uri", ""),
                                    "image_ori": {
                                        "url": img.get("cdn_url", ""),
                                        "width": int(img.get("width") or 1024),
                                        "height": int(img.get("height") or 1024),
                                        "format": img.get("format", "png"),
                                        "url_formats": {},
                                    },
                                },
                                "parse_state": 0,
                                "review_state": 0,
                            }],
                        },
                        "pc_event_block": "",
                    },
                    "block_id": str(uuid.uuid4()),
                    "parent_id": "",
                    "meta_info": [],
                    "append_fields": [],
                    "is_finish": True,
                    "patch_type": 2,
                })

        content_blocks.append({
            "block_type": 10000,
            "content": {
                "text_block": {"text": text, "icon_url": "", "icon_url_dark": "", "summary": ""},
                "pc_event_block": "",
            },
            "block_id": str(uuid.uuid4()),
            "parent_id": "",
            "meta_info": [],
            "append_fields": [],
        })

        payload = {
            "client_meta": {
                "local_conversation_id": local_conv_id,
                "conversation_id": "",
                "bot_id": DEFAULT_BOT_ID,
                "last_section_id": "",
                "last_message_index": None,
            },
            "messages": [{
                "local_message_id": msg_uuid,
                "content_block": content_blocks,
                "message_status": 0,
            }],
            "option": {
                "send_message_scene": "", "create_time_ms": now_ms, "collect_id": "",
                "is_audio": False, "answer_with_suggest": False, "tts_switch": False,
                "need_deep_think": use_deep_think, "click_clear_context": False,
                "from_suggest": False, "is_regen": False, "is_replace": False,
                "disable_sse_cache": False, "select_text_action": "",
                "resend_for_regen": False, "scene_type": 0,
                "unique_key": str(uuid.uuid4()), "start_seq": 0,
                "need_create_conversation": True, "regen_query_id": [],
                "edit_query_id": [], "regen_instruction": "",
                "no_replace_for_regen": False, "message_from": 0,
                "shared_app_name": "", "shared_app_id": "",
                "sse_recv_event_options": {"support_chunk_delta": True},
                "is_ai_playground": False,
                "recovery_option": {"is_recovery": False, "req_create_time_sec": now_sec, "append_sse_event_scene": 0},
                "message_storage_type": 0,
            },
            "ext": {
                "use_deep_think": str(use_deep_think), "fp": self._fp or "",
                "collection_id": "", "commerce_credit_config_enable": "0",
                "sub_conv_firstmet_type": "1",
            },
        }

        query_params = self._build_query_params()
        query_string = "&".join(f"{k}={v}" for k, v in sorted(query_params.items()))
        url = f"/chat/completion?{query_string}"

        # Use browser fetch (non-streaming, collect full response)
        js_code = """
        async ([url, payloadJson]) => {
            const csrf = document.cookie.match(/passport_csrf_token=([^;]+)/);
            const csrfToken = csrf ? csrf[1] : '';
            const headers = {
                'Content-Type': 'application/json',
                'agw-js-conv': 'str',
            };
            if (csrfToken) headers['x-tt-passport-csrf-token'] = csrfToken;
            const res = await fetch(url, {
                method: 'POST',
                headers: headers,
                body: payloadJson,
                credentials: 'include',
            });
            if (!res.ok) {
                const errBody = await res.text();
                return {error: true, status: res.status, body: errBody.slice(0, 500)};
            }
            const body = await res.text();
            return {error: false, body: body};
        }
        """
        payload_json = json.dumps(payload, ensure_ascii=False)
        log.info("POST /chat/completion [chat_with_file, browser fetch]")
        result = await self._page.evaluate(js_code, [url, payload_json])

        if result.get("error"):
            raise RuntimeError(
                f"chat_with_file error {result.get('status')}: {result.get('body', '')[:200]}"
            )

        full_text = ""
        conv_id = None
        raw_body = result.get("body", "")
        for block in raw_body.split("\n"):
            line = block.strip()
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if not data_str or data_str == "{}":
                continue
            try:
                data = json.loads(data_str)
                full_text += self._extract_text(data)
                if not conv_id:
                    cid = self.extract_conversation_id(data)
                    if cid and cid != "0":
                        conv_id = cid
            except json.JSONDecodeError:
                continue

        return {"text": full_text, "conversation_id": conv_id}
