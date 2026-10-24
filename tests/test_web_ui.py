from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from doubao2api.browser_client import BrowserClient
from doubao2api.unified_server import create_app

def test_web_ui_routes():
    app = create_app(api_key=None)
    client = TestClient(app, follow_redirects=False)

    # 1. Root route redirect
    resp = client.get("/")
    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == "/admin"

    # 2. Auth redirect
    resp = client.get("/auth?key=test1234")
    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == "/admin?key=test1234"

    # 3. Admin dashboard HTML
    resp = client.get("/admin")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Doubao 2API 控制台" in resp.text
    assert "apiBaseUrl" in resp.text
    assert "纯净免窗口扫码登录" in resp.text

    # 4. Health
    resp = client.get("/health")
    assert resp.status_code == 200

    # 5. Models list
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "data" in data
    model_ids = [m["id"] for m in data["data"]]
    assert "doubao-2.1-turbo" in model_ids
    assert "doubao-think" in model_ids

def test_web_ui_with_api_key():
    app = create_app(api_key="secret123")
    client = TestClient(app, follow_redirects=False)

    # Admin dashboard should render (allows entering key in UI)
    resp = client.get("/admin")
    assert resp.status_code == 200
    assert "Doubao 2API 控制台" in resp.text

    # System API without key should be 401
    resp = client.get("/admin/api/system")
    assert resp.status_code == 401

    # System API with query key should be rejected (P0 security requirement)
    resp = client.get("/admin/api/system?key=secret123")
    assert resp.status_code == 401

    # System API with X-API-Key header should succeed
    resp = client.get("/admin/api/system", headers={"X-API-Key": "secret123"})
    assert resp.status_code == 200
    sys_info = resp.json()
    assert "platform" in sys_info
    assert "models" in sys_info

    # System API with Bearer token should succeed
    resp = client.get("/admin/api/system", headers={"Authorization": "Bearer secret123"})
    assert resp.status_code == 200


@pytest.mark.parametrize(
    "switch_result,page_present,page_closed,expected_status",
    [
        (False, True, False, 503),
        (True, False, False, 503),
        (True, True, True, 503),
        (True, True, False, 200),
    ],
)
def test_force_window_reports_managed_page_result(
    tmp_path, switch_result, page_present, page_closed, expected_status
):
    browser = BrowserClient(headless=True, user_data_dir=str(tmp_path))
    browser._ready = True
    if switch_result:
        browser.headless = False
    browser.switch_mode = AsyncMock(return_value=switch_result)
    page = MagicMock()
    page.is_closed.return_value = page_closed
    page.bring_to_front = AsyncMock()
    browser._page = page if page_present else None

    with TestClient(create_app(browser_client=browser)) as client:
        response = client.post("/admin/api/browser/force-window")

    assert response.status_code == expected_status
    if expected_status == 200:
        assert response.json()["status"] == "ok"
        page.bring_to_front.assert_awaited_once()
    else:
        assert response.json()["status"] == "error"
        assert response.json()["detail"] == "Managed browser window is unavailable"
        if page_present:
            page.bring_to_front.assert_not_awaited()


def test_force_window_requires_admin_authentication(tmp_path):
    browser = BrowserClient(headless=True, user_data_dir=str(tmp_path))
    browser.switch_mode = AsyncMock(return_value=True)

    with TestClient(create_app(api_key="secret", browser_client=browser)) as client:
        response = client.post("/admin/api/browser/force-window")

    assert response.status_code == 401
    browser.switch_mode.assert_not_awaited()
