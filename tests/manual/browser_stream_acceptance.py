"""Independent acceptance: actual browser fetch must stop on iterator close."""
import asyncio
import json
import sys
import tempfile
import socket
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aiohttp import web
from playwright.async_api import async_playwright
from doubao2api.browser_client import BrowserClient
from doubao2api.unified_server import create_app
import httpx
import uvicorn


async def main():
    disconnected = asyncio.Event()

    async def home(request):
        return web.Response(text="<html>Local stream acceptance fixture</html>", content_type="text/html")

    async def slow_stream(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        try:
            await response.write(b'event: CHUNK_DELTA\ndata: {"text":"fixture"}\n\n')
            while True:
                await asyncio.sleep(0.05)
                await response.write(b": fixture heartbeat\n\n")
        except (ConnectionResetError, ConnectionError, asyncio.CancelledError):
            disconnected.set()
        return response

    app = web.Application()
    app.router.add_get("/", home)
    app.router.add_post("/stream", slow_stream)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(channel="chrome", headless=True)
            try:
                page = await browser.new_page()
                await page.goto(f"http://127.0.0.1:{port}/")
                with tempfile.TemporaryDirectory() as profile:
                    client = BrowserClient(headless=True, user_data_dir=profile)
                    client._page = page
                    client._ready = True

                    async def alive():
                        return True

                    async def sign(*args, **kwargs):
                        return "/stream"

                    client.is_alive = alive
                    client._sign_url = sign

                    def receive(request_id, value):
                        queue = client._stream_queues.get(request_id)
                        if queue:
                            queue.put_nowait(value)

                    await page.expose_function("__doubaoStreamChunk", receive)
                    stream = client.chat_completion("local fixture only")
                    first = await asyncio.wait_for(stream.__anext__(), 10)
                    assert first.get("text") == "fixture", first
                    await asyncio.wait_for(stream.aclose(), 5)
                    await asyncio.wait_for(disconnected.wait(), 5)
                    for _ in range(100):
                        remaining = await page.evaluate("window.__doubaoStreamAbortControllers.size")
                        if remaining == 0:
                            break
                        await asyncio.sleep(0.01)
                    assert remaining == 0
                    assert not client._stream_queues
                    result = {"fixture": "local_only", "server_observed_disconnect": True,
                              "browser_controllers_remaining": remaining, "python_queues_remaining": 0}

                    # Exercise the actual HTTP disconnect/cancellation path too.
                    disconnected.clear()
                    sock = socket.socket()
                    sock.bind(("127.0.0.1", 0))
                    api_port = sock.getsockname()[1]
                    control_arrived = asyncio.Event()
                    control_executed = asyncio.Event()

                    async def fake_switch(headless=None):
                        control_executed.set()
                        return True

                    client.switch_mode = fake_switch
                    api = create_app(browser_client=client)

                    async def observed_api(scope, receive, send):
                        if scope.get("path") == "/admin/api/browser/mode":
                            control_arrived.set()
                        await api(scope, receive, send)

                    server = uvicorn.Server(uvicorn.Config(
                        observed_api, access_log=False,
                        log_level="critical", log_config=None, timeout_graceful_shutdown=2))
                    server_task = asyncio.create_task(server.serve(sockets=[sock]))
                    for _ in range(100):
                        if server.started:
                            break
                        await asyncio.sleep(0.01)
                    try:
                        async with httpx.AsyncClient(timeout=10) as http:
                            async with http.stream("POST", f"http://127.0.0.1:{api_port}/v1/chat/completions",
                                json={"messages": [{"role": "user", "content": "local fixture"}], "stream": True}) as response:
                                assert response.status_code == 200
                                async for line in response.aiter_lines():
                                    if "fixture" in line:
                                        break
                                control = asyncio.create_task(http.post(
                                    f"http://127.0.0.1:{api_port}/admin/api/browser/mode",
                                    json={"headless": True}))
                                await asyncio.wait_for(control_arrived.wait(), 5)
                                assert not control_executed.is_set(), "window operation bypassed live stream gate"
                            control_result = await asyncio.wait_for(control, 5)
                            assert control_result.status_code == 200
                            assert control_executed.is_set()
                        await asyncio.wait_for(disconnected.wait(), 5)
                        async def acquire_after_disconnect():
                            async with client.operation_gate.lease():
                                return True
                        released = await asyncio.wait_for(acquire_after_disconnect(), 5)
                        remaining = await page.evaluate("window.__doubaoStreamAbortControllers.size")
                        assert remaining == 0
                        assert not client._stream_queues
                        result.update({"http_disconnect_aborted_browser": True,
                                       "http_disconnect_released_gate": released,
                                       "window_operation_waited_for_stream_disconnect": True})
                    finally:
                        server.should_exit = True
                        await asyncio.wait_for(server_task, 10)
                    evidence = Path(__file__).resolve().parents[2] / "scratch" / "issue-validation"
                    evidence.mkdir(parents=True, exist_ok=True)
                    (evidence / "browser-abort-result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                    print(json.dumps(result))
            finally:
                await browser.close()
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
