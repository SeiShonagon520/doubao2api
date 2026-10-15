import os
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from doubao2api.browser_client import resolve_memory_type
from doubao2api.unified_server import create_app


def test_resolve_memory_type():
    # 1. Default (clean env) -> 2
    with patch.dict(os.environ, {"SAVE_CHAT_HISTORY": "", "DOUBAO_AUTO_DELETE_CONV": "true"}, clear=False):
        assert resolve_memory_type() == 2

    # 2. Explicit memory_type
    assert resolve_memory_type(memory_type=0) == 0
    assert resolve_memory_type(memory_type=2) == 2

    # 3. Explicit save_history flag
    assert resolve_memory_type(save_history=True) == 0
    assert resolve_memory_type(save_history=False) == 2

    # 4. Env var SAVE_CHAT_HISTORY=true
    with patch.dict(os.environ, {"SAVE_CHAT_HISTORY": "true"}):
        assert resolve_memory_type() == 0

    # 5. Env var DOUBAO_AUTO_DELETE_CONV=false
    with patch.dict(os.environ, {"SAVE_CHAT_HISTORY": "", "DOUBAO_AUTO_DELETE_CONV": "false"}):
        assert resolve_memory_type() == 0


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.is_ready = True
    client.needs_captcha = False
    client.consecutive_failures = 0
    client.last_error_code = None
    client.headless = True
    client.browser_name = "chromium"
    client.is_alive = AsyncMock(return_value=True)

    client.upload_image = AsyncMock(return_value={
        "uri": "tos-mock/ref.png",
        "cdn_url": "https://tos.mock.com/ref.png",
    })
    client.generate_image = AsyncMock(return_value={
        "images": [{"url": "https://tos.mock.com/img.png"}],
    })
    client.generate_video = AsyncMock(return_value={
        "videos": [{"video_url": "https://tos.mock.com/vid.mp4", "duration": 5}],
    })
    return client


def test_video_generation_with_save_history(mock_client):
    app = create_app(browser_client=mock_client)
    client = TestClient(app)

    # 1. Without save_history (default: no extra kwargs)
    resp = client.post(
        "/v1/video/generations",
        json={"prompt": "森林日出", "ratio": "16:9"}
    )
    assert resp.status_code == 200
    mock_client.generate_video.assert_called_with(
        prompt="森林日出", ratio="16:9", ref_image_key=None
    )

    # 2. With save_history=True in body -> passes save_history=True
    resp2 = client.post(
        "/v1/video/generations",
        json={"prompt": "海边日落", "ratio": "16:9", "save_history": True}
    )
    assert resp2.status_code == 200
    mock_client.generate_video.assert_called_with(
        prompt="海边日落", ratio="16:9", ref_image_key=None, save_history=True
    )

    # 3. With explicit memory_type=0 in body -> passes memory_type=0
    resp3 = client.post(
        "/v1/video/generations",
        json={"prompt": "极光星空", "ratio": "16:9", "memory_type": 0}
    )
    assert resp3.status_code == 200
    mock_client.generate_video.assert_called_with(
        prompt="极光星空", ratio="16:9", ref_image_key=None, memory_type=0
    )


def test_images_generation_with_save_history(mock_client):
    app = create_app(browser_client=mock_client)
    client = TestClient(app)

    # With header X-Save-History: true -> passes save_history=True
    resp = client.post(
        "/v1/images/generations",
        headers={"x-save-history": "true"},
        json={"prompt": "水彩画城堡", "size": "1024x1024"}
    )
    assert resp.status_code == 200
    mock_client.generate_image.assert_called_with(
        prompt="水彩画城堡", ratio="1:1", ref_image_key=None, save_history=True
    )


def test_admin_toggle_auto_delete(mock_client):
    app = create_app(browser_client=mock_client)
    client = TestClient(app)

    # Toggle to false (留存网页)
    resp = client.post("/admin/api/config/auto-delete", json={"enabled": False})
    assert resp.status_code == 200
    assert resp.json()["auto_delete_conv"] is False

    # Check /admin/api/system reflects change
    resp_sys = client.get("/admin/api/system")
    assert resp_sys.status_code == 200
    assert resp_sys.json()["auto_delete_conv"] is False

    # Toggle back to true
    resp2 = client.post("/admin/api/config/auto-delete", json={"enabled": True})
    assert resp2.status_code == 200
    assert resp2.json()["auto_delete_conv"] is True
