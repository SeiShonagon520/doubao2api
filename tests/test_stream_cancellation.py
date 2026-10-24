"""Browser fetch stream cleanup must abort before the iterator closes."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from doubao2api.browser_client import BrowserClient


def test_closing_stream_requests_abort_for_its_browser_request_id(tmp_path):
    async def run():
        client = BrowserClient(headless=True, user_data_dir=str(tmp_path))
        page = MagicMock()
        page.is_closed.return_value = False
        page.evaluate = AsyncMock(return_value=None)
        client._page = page
        client._ready = True
        client.is_alive = AsyncMock(return_value=True)
        client._setup_fetch_bridge = AsyncMock()
        client._sign_url = AsyncMock(return_value="/chat/completion")
        fetch_started = asyncio.Event()
        fetch_closed = asyncio.Event()

        async def blocked_fetch(url, payload, request_id):
            fetch_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                fetch_closed.set()

        client._browser_fetch_stream = blocked_fetch
        stream = client.chat_completion("test")
        next_event = asyncio.create_task(stream.__anext__())
        await fetch_started.wait()
        next_event.cancel()
        try:
            await next_event
        except asyncio.CancelledError:
            pass

        assert page.evaluate.await_count == 1
        abort_script, request_id = page.evaluate.await_args.args
        assert "controller.abort()" in abort_script
        assert request_id.startswith("req_")
        assert request_id not in client._stream_queues
        assert fetch_closed.is_set()

    asyncio.run(run())
