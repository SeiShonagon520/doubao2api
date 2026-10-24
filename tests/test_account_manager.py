"""
Tests for AccountManager and Multi-Account Admin APIs.
"""
import os
import tempfile
from datetime import date, timedelta
import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient

from doubao2api.account_manager import Account, AccountManager
from doubao2api.unified_server import create_app


@pytest.fixture
def temp_accounts_file():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    if os.path.exists(path):
        os.remove(path)
    yield path
    if os.path.exists(path):
        os.remove(path)


def test_account_manager_crud(temp_accounts_file):
    mgr = AccountManager(filepath=temp_accounts_file)
    assert len(mgr.accounts) == 0

    # Add account 1
    acc1 = mgr.add_or_update_account(
        name="主账号",
        cookies={"sessionid": "sess_111", "other": "val1"},
    )
    assert acc1.id in mgr.accounts
    assert acc1.name == "主账号"
    assert mgr.active_account_id == acc1.id

    # Add account 2
    acc2 = mgr.add_or_update_account(
        name="备用账号",
        cookies={"sessionid": "sess_222"},
    )
    assert len(mgr.accounts) == 2
    assert mgr.active_account_id == acc1.id  # active stays acc1

    # Switch active
    assert mgr.set_active_account(acc2.id) is True
    assert mgr.active_account_id == acc2.id

    # List accounts (masked)
    acc_list = mgr.list_accounts(mask_cookies=True)
    assert len(acc_list) == 2
    assert acc_list[1]["is_active"] is True
    assert acc_list[0]["has_sessionid"] is True

    # Delete account 1
    assert mgr.delete_account(acc1.id) is True
    assert len(mgr.accounts) == 1
    assert mgr.active_account_id == acc2.id


def test_account_manager_persistence(temp_accounts_file):
    mgr1 = AccountManager(filepath=temp_accounts_file)
    mgr1.add_or_update_account("持久化账号", {"sessionid": "sess_persist"}, account_id="acc_p1")
    mgr1.strategy = "round_robin"
    mgr1.save()

    # Load in a second manager instance from the same file
    mgr2 = AccountManager(filepath=temp_accounts_file)
    assert len(mgr2.accounts) == 1
    assert mgr2.strategy == "round_robin"
    assert "acc_p1" in mgr2.accounts
    assert mgr2.accounts["acc_p1"].cookies["sessionid"] == "sess_persist"


def test_failover_and_round_robin_strategy(temp_accounts_file):
    mgr = AccountManager(filepath=temp_accounts_file, strategy="failover")
    acc1 = mgr.add_or_update_account("Acc1", {"sessionid": "s1"}, account_id="a1")
    acc2 = mgr.add_or_update_account("Acc2", {"sessionid": "s2"}, account_id="a2")
    mgr.set_active_account("a1")

    # Failover: sticks with active account while available
    next_acc = mgr.get_next_available_account("video")
    assert next_acc.id == "a1"

    # Mark a1 quota exceeded -> failover picks a2
    mgr.mark_quota_exceeded("a1", "video")
    next_acc = mgr.get_next_available_account("video")
    assert next_acc.id == "a2"
    assert mgr.active_account_id == "a1"
    assert mgr.set_active_account(next_acc.id) is True
    assert mgr.active_account_id == "a2"

    # Mark a2 quota exceeded -> no available account
    mgr.mark_quota_exceeded("a2", "video")
    assert mgr.get_next_available_account("video") is None

    # Reset quota
    assert mgr.reset_quota("a1") is True
    next_acc = mgr.get_next_available_account("video")
    assert next_acc.id == "a1"

    # Test Round-Robin
    mgr.strategy = "round_robin"
    mgr.reset_quota("a2")
    # Now both a1 and a2 are available
    r1 = mgr.get_next_available_account("video")
    r1_again = mgr.get_next_available_account("video")
    assert r1_again.id == r1.id
    assert mgr.active_account_id == "a2"
    mgr.set_active_account(r1.id)
    r2 = mgr.get_next_available_account("video")
    assert r2.id != r1.id


def test_candidate_selection_is_pure_until_active_account_commit(temp_accounts_file):
    mgr = AccountManager(filepath=temp_accounts_file, strategy="round_robin")
    acc1 = mgr.add_or_update_account("Acc1", {"sessionid": "s1"}, account_id="a1")
    acc2 = mgr.add_or_update_account("Acc2", {"sessionid": "s2"}, account_id="a2")
    mgr.set_active_account(acc1.id)
    before_last_used = {acc.id: acc.last_used for acc in (acc1, acc2)}
    before_cursor = mgr._round_robin_last_id

    first = mgr.get_next_available_account("chat")
    second = mgr.get_next_available_account("chat")

    assert first.id == acc2.id
    assert second.id == first.id
    assert mgr.active_account_id == acc1.id
    assert mgr._round_robin_last_id == before_cursor
    assert {acc.id: acc.last_used for acc in (acc1, acc2)} == before_last_used

    mgr.set_active_account(first.id)
    assert mgr.get_next_available_account("chat").id == acc1.id


def test_video_quota_does_not_disable_chat_and_legacy_quota_resets(temp_accounts_file):
    mgr = AccountManager(filepath=temp_accounts_file, strategy="failover")
    acc1 = mgr.add_or_update_account("Acc1", {"sessionid": "s1"}, account_id="a1")
    acc2 = mgr.add_or_update_account("Acc2", {"sessionid": "s2"}, account_id="a2")
    mgr.mark_quota_exceeded(acc1.id, "video")

    assert acc1.status == "active"
    assert mgr.get_next_available_account("chat").id == acc1.id
    assert mgr.get_next_available_account("image").id == acc1.id
    assert mgr.get_next_available_account("video").id == acc2.id

    acc1.video_quota_exceeded = False
    acc1.status = "quota_exceeded"  # Legacy accounts used a shared status.
    acc1.quota_reset_date = date.today().isoformat()
    assert mgr.get_next_available_account("chat").id == acc1.id
    assert mgr.get_next_available_account("video").id == acc2.id

    mgr.reset_daily_quotas_if_needed(date.today() + timedelta(days=1))
    assert acc1.status == "active"
    assert acc1.video_quota_exceeded is False

    acc2.status = "disabled"
    acc2.video_quota_exceeded = True
    acc2.quota_reset_date = (date.today() - timedelta(days=1)).isoformat()
    mgr.reset_daily_quotas_if_needed(date.today())
    assert acc2.status == "disabled"
    assert acc2.video_quota_exceeded is False


def test_admin_accounts_api(temp_accounts_file):
    mgr = AccountManager(filepath=temp_accounts_file)
    mock_client = MagicMock()
    mock_client.account_manager = mgr
    mock_client.is_ready = True
    mock_client.switch_to_account = AsyncMock(return_value=True)

    app = create_app(browser_client=mock_client)
    client = TestClient(app)

    # 1. List accounts (empty initially)
    resp = client.get("/admin/api/accounts")
    assert resp.status_code == 200
    assert resp.json()["accounts"] == []

    # 2. Add account via API
    resp = client.post(
        "/admin/api/accounts/add",
        json={"name": "测试账号1", "cookies": "sessionid=test_session_12345; other=abc"},
    )
    assert resp.status_code == 200
    acc_data = resp.json()["account"]
    acc_id = acc_data["id"]
    assert acc_data["name"] == "测试账号1"

    # 3. Add second account
    resp = client.post(
        "/admin/api/accounts/add",
        json={"name": "测试账号2", "cookies": {"sessionid": "test_session_67890"}},
    )
    assert resp.status_code == 200
    acc_id2 = resp.json()["account"]["id"]

    # 4. Switch account via API
    resp = client.post("/admin/api/accounts/switch", json={"account_id": acc_id2})
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    mock_client.switch_to_account.assert_called_with(acc_id2)

    # 5. Set strategy
    resp = client.post("/admin/api/accounts/strategy", json={"strategy": "round_robin"})
    assert resp.status_code == 200
    assert resp.json()["strategy"] == "round_robin"

    # 6. Reset quota
    resp = client.post("/admin/api/accounts/reset_quota", json={"account_id": acc_id})
    assert resp.status_code == 200

    # 7. Delete account
    resp = client.post("/admin/api/accounts/delete", json={"account_id": acc_id})
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True

    # 8. Add account via JSON list format (e.g. Cookie-Editor export)
    resp = client.post(
        "/admin/api/accounts/add",
        json={
            "name": "插件导出账号",
            "cookies": '[{"domain": ".doubao.com", "name": "sessionid", "value": "sess_from_extension"}]'
        },
    )
    assert resp.status_code == 200
    acc_ext = resp.json()["account"]
    assert acc_ext["has_sessionid"] is True

    # 9. Test cookie import with direct list of dicts
    mock_client.inject_cookies_and_reload = AsyncMock(return_value=True)
    resp = client.post(
        "/admin/api/cookies/import",
        json={
            "cookies": [{"name": "sessionid", "value": "sess_import_list"}]
        }
    )
    assert resp.status_code == 200
    assert resp.json()["has_sessionid"] is True
