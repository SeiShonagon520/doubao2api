import asyncio
from unittest.mock import AsyncMock, MagicMock
from doubao2api.browser_client import BrowserClient

def test_record_failure_risk_codes():
    client = BrowserClient(headless=True)
    client._ready = True

    # 1. Error 710022002 should trigger needs_captcha
    client.record_failure(710022002)
    assert client.needs_captcha is True
    assert client.last_error_code == 710022002
    assert client.is_ready is True

    # 2. Even after 10 failures on captcha, ready flag is preserved for auto-heal
    for _ in range(10):
        client.record_failure(710022002)
    assert client.is_ready is True

    # 3. Clearing captcha
    client.clear_captcha()
    assert client.needs_captcha is False
    assert client.consecutive_failures == 0

    # 4. A risk code alone must not mark verification without classification
    client.record_failure(710022004)
    assert client.needs_captcha is False

    client.record_failure(710022004, verification_required=True)
    assert client.needs_captcha is True

def test_auto_popup_for_captcha():
    async def _run():
        client = BrowserClient(headless=True)
        client.switch_mode = AsyncMock(return_value=True)
        mock_page = MagicMock()
        mock_page.bring_to_front = AsyncMock()
        client._page = mock_page

        popped = await client.auto_popup_for_captcha()
        assert popped is True
        client.switch_mode.assert_called_once_with(headless=False)
        mock_page.bring_to_front.assert_called_once()
    asyncio.run(_run())


def test_switch_mode_reports_open_unlogged_page_as_available():
    async def _run():
        client = BrowserClient(headless=True)
        client._ready = False
        client._page = MagicMock()
        client._page.is_closed.return_value = True
        client._context = MagicMock()
        live_page = MagicMock()
        live_page.is_closed.return_value = False

        async def restart():
            client._page = live_page
            client._context = MagicMock()
            client._ready = False

        client.restart = AsyncMock(side_effect=restart)
        client.is_alive = AsyncMock(return_value=True)

        available = await client.switch_mode(headless=False)

        assert available is True
        assert client.is_ready is False
        assert client.headless is False
        client.restart.assert_awaited_once()
        client.is_alive.assert_awaited_once()

    asyncio.run(_run())
