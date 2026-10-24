"""Tests for 3-tier Captcha Defense:
1. Humanized mouse trajectory generation.
2. Gap offset calculation.
3. Account pool captcha failover and recovery.
4. Auto-solve integration and fallback to 503.
5. Admin clear_captcha API endpoint.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from doubao2api.captcha_solver import (
    generate_human_trajectory,
    calculate_gap_offset,
    SliderCaptchaSolver,
)
from doubao2api.account_manager import Account, AccountManager
from doubao2api.browser_client import BrowserClient
from doubao2api.unified_server import create_app


def test_generate_human_trajectory():
    """Test human trajectory characteristics."""
    start_x, start_y, distance = 100.0, 200.0, 150.0
    trajectory = generate_human_trajectory(start_x, start_y, distance, total_time=1.0)

    assert len(trajectory) >= 30
    first_x, first_y, first_delay = trajectory[0]
    last_x, last_y, last_delay = trajectory[-1]

    # Starts near start_x
    assert first_x > start_x
    # Ends close to target distance (within 5px tolerance)
    assert abs(last_x - (start_x + distance)) < 5.0
    # Y-axis has slight jitter
    y_values = [p[1] for p in trajectory]
    assert any(abs(y - start_y) > 0.1 for y in y_values)
    # Delays are positive
    assert all(p[2] > 0 for p in trajectory)


def test_account_manager_captcha_lifecycle(tmp_path):
    """Test mark_captcha_required, clear_captcha_status and failover exclusion."""
    filepath = str(tmp_path / "accounts.json")
    am = AccountManager(filepath=filepath, strategy="failover")

    acc1 = am.add_or_update_account("账号1", {"sessionid": "s1_valid"})
    acc2 = am.add_or_update_account("账号2", {"sessionid": "s2_valid"})

    # Initially acc1 is active
    assert am.get_active_account().id == acc1.id
    assert am.get_next_available_account().id == acc1.id

    # Mark acc1 as captcha
    am.mark_captcha_required(acc1.id)
    assert am.get_account(acc1.id).status == "captcha"

    # acc1 should now be excluded from available accounts; failover to acc2
    next_acc = am.get_next_available_account()
    assert next_acc is not None
    assert next_acc.id == acc2.id
    assert am.active_account_id == acc1.id
    am.set_active_account(acc2.id)

    # Clear captcha status on acc1
    cleared = am.clear_captcha_status(acc1.id)
    assert cleared is True
    assert am.get_account(acc1.id).status == "active"


def test_auto_solve_captcha_integration():
    """Test SliderCaptchaSolver detect_and_solve returns False when no container is present."""
    mock_page = MagicMock()
    mock_page.frames = []
    loc = MagicMock()
    loc.count = AsyncMock(return_value=0)
    mock_page.locator = MagicMock(return_value=loc)

    solver = SliderCaptchaSolver()
    result = asyncio.run(solver.detect_and_solve(mock_page))
    assert result is False


def test_admin_clear_captcha_endpoint(tmp_path):
    """Test POST /admin/api/accounts/clear_captcha endpoint."""
    filepath = str(tmp_path / "accounts.json")
    am = AccountManager(filepath=filepath)
    acc = am.add_or_update_account("测试账号", {"sessionid": "test_sess"})
    am.mark_captcha_required(acc.id)
    assert am.get_account(acc.id).status == "captcha"

    mock_client = BrowserClient(headless=True, user_data_dir=str(tmp_path))
    mock_client.account_manager = am
    mock_client._needs_captcha = True
    mock_client._last_error_code = 710022002
    mock_client._ready = True
    mock_client.is_captcha_visible = AsyncMock(return_value=False)
    mock_client.try_auto_solve_captcha = AsyncMock(return_value=False)

    app = create_app(browser_client=mock_client)
    tc = TestClient(app)

    # Call clear_captcha
    resp = tc.post("/admin/api/accounts/clear_captcha", json={"account_id": acc.id})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert am.get_account(acc.id).status == "active"


def test_captcha_failover_in_chat_completions(tmp_path):
    """Test that when account A hits captcha and account B is available,
    chat_completions auto-fails over to account B without raising 503."""
    filepath = str(tmp_path / "accounts.json")
    am = AccountManager(filepath=filepath, strategy="failover")
    acc1 = am.add_or_update_account("主号", {"sessionid": "s1_main"})
    acc2 = am.add_or_update_account("副号", {"sessionid": "s2_backup"})
    am.set_active_account(acc1.id)

    mock_client = BrowserClient(headless=True, user_data_dir=filepath)
    mock_client.account_manager = am
    mock_client._needs_captcha = True
    mock_client._last_error_code = 710022002
    mock_client._ready = True
    # Captcha is visible, auto-solver fails
    mock_client.is_captcha_visible = AsyncMock(return_value=True)
    mock_client.try_auto_solve_captcha = AsyncMock(return_value=False)
    async def switch_to_account(account_id):
        am.set_active_account(account_id)
        return True

    mock_client.switch_to_account = AsyncMock(side_effect=switch_to_account)
    mock_client.extract_conversation_id = MagicMock(return_value="c_test")

    # Mock chat completion stream
    async def mock_chat(*args, **kwargs):
        yield {"event_type": 2001, "event_data": '{"message": {"content": {"text": "hello"}}}'}

    mock_client.chat_completion = mock_chat

    app = create_app(browser_client=mock_client)
    tc = TestClient(app)

    resp = tc.post("/v1/chat/completions", json={
        "model": "doubao-2.1-turbo",
        "messages": [{"role": "user", "content": "hi"}]
    })

    # Should succeed with 200 OK because it failed over to acc2!
    assert resp.status_code == 200
    # acc1 was marked as captcha
    assert am.get_account(acc1.id).status == "captcha"
    # switch_to_account was called with acc2
    mock_client.switch_to_account.assert_awaited_once_with(acc2.id)


def test_missing_captcha_dom_does_not_clear_before_successful_request(tmp_path):
    """The actual request, not a missing DOM node, must prove recovery."""
    manager = AccountManager(filepath=str(tmp_path / "accounts.json"))
    account = manager.add_or_update_account("主号", {"sessionid": "s1_main"})
    client = BrowserClient(headless=True, user_data_dir=str(tmp_path))
    client.account_manager = manager
    client._ready = True
    client._needs_captcha = True
    client._last_error_code = 710022002
    client.is_captcha_visible = AsyncMock(return_value=False)
    client.try_auto_solve_captcha = AsyncMock(return_value=False)
    client.extract_conversation_id = MagicMock(return_value=None)

    async def mock_chat(*args, **kwargs):
        assert client.needs_captcha is True
        yield {"_event": "CHUNK_DELTA", "text": "verified"}

    client.chat_completion = mock_chat
    response = TestClient(create_app(browser_client=client)).post(
        "/v1/chat/completions",
        json={"model": "doubao-2.1-turbo", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "verified"
    assert client.needs_captcha is False
    assert manager.get_account(account.id).status == "active"


def test_failed_account_switch_does_not_report_recovery(tmp_path):
    manager = AccountManager(filepath=str(tmp_path / "accounts.json"), strategy="failover")
    acc1 = manager.add_or_update_account("主号", {"sessionid": "s1_main"})
    acc2 = manager.add_or_update_account("副号", {"sessionid": "s2_backup"})
    manager.set_active_account(acc1.id)
    client = MagicMock()
    client.account_manager = manager
    client.needs_captcha = True
    client.last_error_code = 710022002
    client.is_captcha_visible = AsyncMock(return_value=True)
    client.try_auto_solve_captcha = AsyncMock(return_value=False)
    client.switch_to_account = AsyncMock(return_value=False)
    client.auto_popup_for_captcha = AsyncMock(return_value=True)

    response = TestClient(create_app(browser_client=client)).post(
        "/v1/chat/completions",
        json={"model": "doubao-2.1-turbo", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 503
    assert "打开独立桌面浏览器窗口" in response.text
    assert "http://testserver/admin" in response.text
    error = response.json()["error"]
    assert isinstance(error["message"], str)
    assert error["status"] == 503
    assert "upstream_type" not in error
    assert "upstream_subtype" not in error
    assert "error" not in error
    client.switch_to_account.assert_awaited_once_with(acc2.id)
    assert manager.active_account_id == acc1.id


def test_collector_does_not_switch_or_retry_after_patch_content(tmp_path):
    manager = AccountManager(filepath=str(tmp_path / "accounts.json"), strategy="failover")
    acc1 = manager.add_or_update_account("主号", {"sessionid": "s1_main"})
    manager.add_or_update_account("副号", {"sessionid": "s2_backup"})
    manager.set_active_account(acc1.id)
    client = BrowserClient(headless=True, user_data_dir=str(tmp_path))
    client.account_manager = manager
    client._ready = True
    client.is_captcha_visible = AsyncMock(return_value=False)
    client.try_auto_solve_captcha = AsyncMock(return_value=False)
    client.switch_to_account = AsyncMock(return_value=True)
    client.extract_conversation_id = MagicMock(return_value=None)
    attempts = 0

    async def mock_chat(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        yield {"patch_op": [{"patch_value": {"content": '{"text":"already started"}'}}]}
        yield {
            "error_code": 710022004,
            "error_msg": "rate limited",
            "extra": {"decision": {"type": "verify", "subtype": "semantic_reasoning"}},
        }

    client.chat_completion = mock_chat
    response = TestClient(create_app(browser_client=client)).post(
        "/v1/chat/completions",
        json={"model": "doubao-2.1-turbo", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "verification"
    assert attempts == 1
    client.switch_to_account.assert_not_awaited()
    assert manager.active_account_id == acc1.id


def test_verified_event_is_preserved_for_preflight_on_next_request(tmp_path):
    manager = AccountManager(filepath=str(tmp_path / "accounts.json"))
    account = manager.add_or_update_account("主号", {"sessionid": "s1_main"})
    client = BrowserClient(headless=True, user_data_dir=str(tmp_path))
    client.account_manager = manager
    client._ready = True
    client.is_captcha_visible = AsyncMock(return_value=True)
    client.try_auto_solve_captcha = AsyncMock(return_value=False)
    client.auto_popup_for_captcha = AsyncMock(return_value=True)
    client.extract_conversation_id = MagicMock(return_value=None)

    async def verification_error(*args, **kwargs):
        yield {
            "error_code": 710022004,
            "error_msg": "rate limited",
            "extra": {"decision": json.dumps({"type": "verify", "subtype": "semantic_reasoning"})},
        }

    client.chat_completion = verification_error
    test_client = TestClient(create_app(browser_client=client))
    first = test_client.post(
        "/v1/chat/completions",
        json={
            "model": "doubao-2.1-turbo",
            "conversation_id": "existing-conversation",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    second = test_client.post(
        "/v1/chat/completions",
        json={
            "model": "doubao-2.1-turbo",
            "conversation_id": "existing-conversation",
            "messages": [{"role": "user", "content": "continue"}],
        },
    )

    assert first.status_code == 503
    assert client.needs_captcha is True
    assert client.last_error_category == "verification"
    assert second.status_code == 503
    assert second.json()["error"]["code"] == "verification"
    assert "http://testserver/admin" in second.json()["error"]["message"]
    assert manager.get_account(account.id).status == "captcha"


def test_new_stream_retries_once_after_verify_switch_before_output(tmp_path):
    manager = AccountManager(filepath=str(tmp_path / "accounts.json"), strategy="failover")
    acc1 = manager.add_or_update_account("主号", {"sessionid": "s1_main"})
    acc2 = manager.add_or_update_account("副号", {"sessionid": "s2_backup"})
    manager.set_active_account(acc1.id)
    client = BrowserClient(headless=True, user_data_dir=str(tmp_path))
    client.account_manager = manager
    client._ready = True
    client.is_captcha_visible = AsyncMock(return_value=False)

    async def switch_to_account(account_id):
        manager.set_active_account(account_id)
        client._needs_captcha = False
        return True

    client.switch_to_account = AsyncMock(side_effect=switch_to_account)
    client.extract_conversation_id = MagicMock(return_value=None)
    attempts = 0

    async def mock_chat(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            yield {
                "_event": "STREAM_ERROR",
                "error_code": 710022004,
                "error_msg": "rate limited",
                "extra": {"decision": json.dumps({"type": "verify", "subtype": "semantic_reasoning"})},
            }
        else:
            yield {"_event": "CHUNK_DELTA", "text": "recovered"}

    client.chat_completion = mock_chat
    response = TestClient(create_app(browser_client=client)).post(
        "/v1/chat/completions",
        json={
            "model": "doubao-2.1-turbo",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 200
    assert "recovered" in response.text
    assert '"error"' not in response.text
    assert attempts == 2
    client.switch_to_account.assert_awaited_once_with(acc2.id)
    assert client.needs_captcha is False
    assert manager.active_account_id == acc2.id
    assert manager.get_account(acc1.id).status == "captcha"


def test_existing_stream_error_is_structured_without_account_switch(tmp_path):
    manager = AccountManager(filepath=str(tmp_path / "accounts.json"), strategy="failover")
    acc1 = manager.add_or_update_account("主号", {"sessionid": "s1_main"})
    manager.add_or_update_account("副号", {"sessionid": "s2_backup"})
    manager.set_active_account(acc1.id)
    client = BrowserClient(headless=True, user_data_dir=str(tmp_path))
    client.account_manager = manager
    client._ready = True
    client.is_captcha_visible = AsyncMock(return_value=False)
    client.switch_to_account = AsyncMock(return_value=True)
    client.extract_conversation_id = MagicMock(return_value=None)
    attempts = 0

    async def mock_chat(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        yield {
            "_event": "STREAM_ERROR",
            "error_code": 710022004,
            "error_msg": "rate limited",
            "extra": {"decision": {"type": "verify", "subtype": "semantic_reasoning"}},
        }

    client.chat_completion = mock_chat
    response = TestClient(create_app(browser_client=client)).post(
        "/v1/chat/completions",
        json={
            "model": "doubao-2.1-turbo",
            "stream": True,
            "conversation_id": "existing-conversation",
            "messages": [{"role": "user", "content": "continue"}],
        },
    )

    assert response.status_code == 200
    assert "verification_required" in response.text
    assert "[Error" not in response.text
    assert attempts == 1
    client.switch_to_account.assert_not_awaited()


def test_output_started_stream_does_not_switch_or_retry(tmp_path):
    manager = AccountManager(filepath=str(tmp_path / "accounts.json"), strategy="failover")
    manager.add_or_update_account("主号", {"sessionid": "s1_main"})
    manager.add_or_update_account("副号", {"sessionid": "s2_backup"})
    client = BrowserClient(headless=True, user_data_dir=str(tmp_path))
    client.account_manager = manager
    client._ready = True
    client.is_captcha_visible = AsyncMock(return_value=False)
    client.switch_to_account = AsyncMock(return_value=True)
    client.extract_conversation_id = MagicMock(return_value=None)
    attempts = 0

    async def mock_chat(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        yield {"_event": "CHUNK_DELTA", "text": "partial"}
        yield {
            "_event": "STREAM_ERROR",
            "error_code": 710022004,
            "error_msg": "rate limited",
            "extra": {"decision": {"type": "verify", "subtype": "semantic_reasoning"}},
        }

    client.chat_completion = mock_chat
    response = TestClient(create_app(browser_client=client)).post(
        "/v1/chat/completions",
        json={
            "model": "doubao-2.1-turbo",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 200
    assert "partial" in response.text
    assert "verification_required" in response.text
    assert attempts == 1
    client.switch_to_account.assert_not_awaited()


def test_stream_keepalive_preserves_pending_upstream_event(tmp_path):
    client = BrowserClient(headless=True, user_data_dir=str(tmp_path))
    client._ready = True
    client.extract_conversation_id = MagicMock(return_value=None)

    async def slow_chat(*args, **kwargs):
        await asyncio.sleep(5.1)
        yield {"_event": "CHUNK_DELTA", "text": "after-heartbeat"}

    client.chat_completion = slow_chat
    response = TestClient(create_app(browser_client=client)).post(
        "/v1/chat/completions",
        json={
            "model": "doubao-2.1-turbo",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 200
    assert ": keep-alive" in response.text
    assert "after-heartbeat" in response.text
