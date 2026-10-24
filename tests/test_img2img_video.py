"""
Tests for Img2Img (image edits) and Img2Video (video generations with reference image).
"""
import base64
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from doubao2api.unified_server import create_app
from doubao2api.upstream_errors import UpstreamAPIError, parse_upstream_error


@pytest.fixture
def mock_browser_client():
    client = MagicMock()
    client.is_ready = True
    client.needs_captcha = False
    client.consecutive_failures = 0
    client.last_error_code = None
    client.headless = True
    client.browser_name = "chromium"

    client.is_alive = AsyncMock(return_value=True)
    client.upload_image = AsyncMock(return_value={
        "uri": "tos-cn-i-mock/ref_image_123.png",
        "cdn_url": "https://tos.mock.com/ref_image_123.png",
        "name": "ref_image.png",
        "format": "png",
        "width": 640,
        "height": 480,
    })
    client.generate_image = AsyncMock(return_value={
        "images": [{"url": "https://tos.mock.com/generated_img.png"}],
    })
    client.generate_video = AsyncMock(return_value={
        "videos": [{"video_url": "https://tos.mock.com/generated_vid.mp4", "duration": 5}],
    })
    return client


@pytest.fixture
def test_app(mock_browser_client):
    app = create_app(browser_client=mock_browser_client)
    yield app, mock_browser_client


def test_images_generations_with_tos_key(test_app):
    app, mock_client = test_app
    with patch.dict(app.extra if hasattr(app, "extra") else {}, {}):
        # Inject client into app closure
        from doubao2api import unified_server
        client = TestClient(app)
        
        # Test images/generations with direct ref_image_key
        resp = client.post(
            "/v1/images/generations",
            json={
                "prompt": "一只戴墨镜的猫",
                "ref_image_key": "tos-cn-i-mock/original.png",
                "size": "1024x1024",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["url"] == "https://tos.mock.com/generated_img.png"
        mock_client.generate_image.assert_called_with(
            prompt="一只戴墨镜的猫",
            ratio="1:1",
            ref_image_key="tos-cn-i-mock/original.png",
        )


def test_images_generations_with_base64_data_uri(test_app):
    app, mock_client = test_app
    client = TestClient(app)

    # 1x1 transparent png in base64
    b64_png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    resp = client.post(
        "/v1/images/generations",
        json={
            "prompt": "生成相似图片",
            "image": b64_png,
            "size": "1792x1024",
        },
    )
    assert resp.status_code == 200
    mock_client.upload_image.assert_called_once()
    mock_client.generate_image.assert_called_with(
        prompt="生成相似图片",
        ratio="16:9",
        ref_image_key="tos-cn-i-mock/ref_image_123.png",
    )


def test_images_edits_endpoint_json(test_app):
    app, mock_client = test_app
    client = TestClient(app)

    resp = client.post(
        "/v1/images/edits",
        json={
            "prompt": "变成赛博朋克风格",
            "image": "tos-cn-i-mock/city.png",
            "size": "1024x1024",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["data"][0]["revised_prompt"] == "变成赛博朋克风格"
    mock_client.generate_image.assert_called_with(
        prompt="变成赛博朋克风格",
        ratio="1:1",
        ref_image_key="tos-cn-i-mock/city.png",
    )


def test_images_edits_endpoint_multipart(test_app):
    app, mock_client = test_app
    client = TestClient(app)

    file_content = b"fake-png-data"
    resp = client.post(
        "/v1/images/edits",
        data={"prompt": "水彩画风格", "size": "1024x1024"},
        files={"image": ("test.png", file_content, "image/png")},
    )
    assert resp.status_code == 200
    mock_client.upload_image.assert_called_once()
    mock_client.generate_image.assert_called_with(
        prompt="水彩画风格",
        ratio="1:1",
        ref_image_key="tos-cn-i-mock/ref_image_123.png",
    )


def test_video_generations_rejects_tos_reference_without_trusted_metadata(test_app):
    app, mock_client = test_app
    client = TestClient(app)

    resp = client.post(
        "/v1/video/generations",
        json={
            "prompt": "让镜头向前推进，树叶随风飘落",
            "ratio": "16:9",
            "ref_image_key": "tos-cn-i-mock/autumn_forest.png",
        },
    )
    assert resp.status_code == 409
    assert "metadata is unavailable" in resp.json()["error"]["message"]
    mock_client.generate_video.assert_not_called()


def test_video_generations_uses_trusted_upload_metadata_for_tos_reference(test_app):
    app, mock_client = test_app
    uri = "tos-cn-i-mock/known-reference.png"
    mock_client._uploaded_image_metadata = {
        uri: {
            "uri": uri,
            "cdn_url": "https://tos.mock.com/known-reference.png",
            "name": "known-reference.png",
            "format": "png",
            "width": 900,
            "height": 600,
            "_account_id": None,
        }
    }

    response = TestClient(app).post(
        "/v1/video/generations",
        json={"prompt": "trusted metadata", "ref_image_key": uri},
    )

    assert response.status_code == 200
    mock_client.generate_video.assert_awaited_once_with(
        prompt="trusted metadata",
        ratio=None,
        ref_image_key=uri,
        image_attachments=[{
            "uri": uri,
            "cdn_url": "https://tos.mock.com/known-reference.png",
            "name": "known-reference.png",
            "format": "png",
            "width": 900,
            "height": 600,
        }],
    )


def test_ssrf_blocked_in_image_resolution(test_app):
    app, mock_client = test_app
    client = TestClient(app)

    # Private IP should be blocked
    resp = client.post(
        "/v1/images/generations",
        json={
            "prompt": "测试内网拦截",
            "image": "http://127.0.0.1:8080/secret.png",
        },
    )
    assert resp.status_code == 400
    assert "SSRF" in resp.json()["error"]["message"]


def test_video_generations_rejects_multiple_tos_keys_without_metadata(test_app):
    app, mock_client = test_app
    client = TestClient(app)

    resp = client.post(
        "/v1/video/generations",
        json={
            "prompt": "融合两张图片的风格与角色动作",
            "ratio": "16:9",
            "ref_image_keys": [
                "tos-cn-i-mock/character.png",
                "tos-cn-i-mock/background.png",
            ],
        },
    )
    assert resp.status_code == 409
    mock_client.generate_video.assert_not_called()


def test_video_generations_rejects_comma_separated_keys_without_metadata(test_app):
    app, mock_client = test_app
    client = TestClient(app)

    resp = client.post(
        "/v1/video/generations",
        json={
            "prompt": "视频生成测试逗号分隔",
            "ratio": "16:9",
            "ref_image_keys": "tos-cn-i-mock/k1.png, tos-cn-i-mock/k2.png",
        },
    )
    assert resp.status_code == 409
    mock_client.generate_video.assert_not_called()


def test_video_generations_with_multiple_base64_images(test_app):
    app, mock_client = test_app
    client = TestClient(app)

    b64_img = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    upload_responses = [
        {"uri": "tos-cn-i-mock/img_1.png", "cdn_url": "https://tos.mock.com/img_1.png", "name": "img_1.png", "format": "png", "width": 1, "height": 1},
        {"uri": "tos-cn-i-mock/img_2.png", "cdn_url": "https://tos.mock.com/img_2.png", "name": "img_2.png", "format": "png", "width": 1, "height": 1},
    ]
    mock_client.upload_image = AsyncMock(side_effect=upload_responses)

    resp = client.post(
        "/v1/video/generations",
        json={
            "prompt": "使用两个base64图片生成视频",
            "images": [b64_img, b64_img],
        },
    )
    assert resp.status_code == 200
    assert mock_client.upload_image.call_count == 2
    mock_client.generate_video.assert_called_with(
        prompt="使用两个base64图片生成视频",
        ratio=None,
        ref_image_key="tos-cn-i-mock/img_1.png",
        ref_image_keys=[
            "tos-cn-i-mock/img_1.png",
            "tos-cn-i-mock/img_2.png",
        ],
        image_attachments=[
            {"uri": "tos-cn-i-mock/img_1.png", "cdn_url": "https://tos.mock.com/img_1.png", "name": "img_1.png", "format": "png", "width": 1, "height": 1},
            {"uri": "tos-cn-i-mock/img_2.png", "cdn_url": "https://tos.mock.com/img_2.png", "name": "img_2.png", "format": "png", "width": 1, "height": 1},
        ],
    )


def test_video_generations_multipart_multiple_files(test_app):
    app, mock_client = test_app
    client = TestClient(app)

    upload_responses = [
        {"uri": "tos-cn-i-mock/file_1.png", "cdn_url": "https://tos.mock.com/file_1.png", "name": "file_1.png", "format": "png", "width": 1, "height": 1},
        {"uri": "tos-cn-i-mock/file_2.png", "cdn_url": "https://tos.mock.com/file_2.png", "name": "file_2.png", "format": "png", "width": 1, "height": 1},
    ]
    mock_client.upload_image = AsyncMock(side_effect=upload_responses)

    resp = client.post(
        "/v1/video/generations",
        data={"prompt": "多文件上传视频生成", "ratio": "9:16"},
        files=[
            ("images", ("frame1.png", b"image1-data", "image/png")),
            ("images", ("frame2.png", b"image2-data", "image/png")),
        ],
    )
    assert resp.status_code == 200
    assert mock_client.upload_image.call_count == 2
    mock_client.generate_video.assert_called_with(
        prompt="多文件上传视频生成",
        ratio="9:16",
        ref_image_key="tos-cn-i-mock/file_1.png",
        ref_image_keys=[
            "tos-cn-i-mock/file_1.png",
            "tos-cn-i-mock/file_2.png",
        ],
        image_attachments=[
            {"uri": "tos-cn-i-mock/file_1.png", "cdn_url": "https://tos.mock.com/file_1.png", "name": "file_1.png", "format": "png", "width": 1, "height": 1},
            {"uri": "tos-cn-i-mock/file_2.png", "cdn_url": "https://tos.mock.com/file_2.png", "name": "file_2.png", "format": "png", "width": 1, "height": 1},
        ],
    )


def test_images_generations_with_multiple_keys(test_app):
    app, mock_client = test_app
    client = TestClient(app)

    resp = client.post(
        "/v1/images/generations",
        json={
            "prompt": "多图参考生图",
            "ref_image_keys": ["tos-cn-i-mock/r1.png", "tos-cn-i-mock/r2.png"],
            "size": "1024x1024",
        },
    )
    assert resp.status_code == 200
    mock_client.generate_image.assert_called_with(
        prompt="多图参考生图",
        ratio="1:1",
        ref_image_key="tos-cn-i-mock/r1.png",
        ref_image_keys=["tos-cn-i-mock/r1.png", "tos-cn-i-mock/r2.png"],
    )


def test_browser_client_video_uses_native_layout_and_one_confirmed_continuation(caplog):
    import asyncio
    import logging
    from unittest.mock import AsyncMock
    from doubao2api.browser_client import BrowserClient

    async def _run():
        client = BrowserClient(headless=True)
        client.account_manager = None
        prompt = "private user video prompt"
        conversation_id = "current-video-conversation"
        image_keys = ["tos-fixture-a", "tos-fixture-b"]
        client._uploaded_image_metadata = {
            key: {
                "uri": key,
                "cdn_url": f"https://image.test/{index}.png",
                "name": f"{index}.png",
                "format": "png",
                "width": 640,
                "height": 480,
                "_account_id": None,
            }
            for index, key in enumerate(image_keys)
        }
        calls = []

        async def fake_chat_completion(text, **kwargs):
            calls.append((text, kwargs))
            yield {"ack_client_meta": {"conversation_id": conversation_id}}

        client.chat_completion = fake_chat_completion
        chain = _native_chain_fixture()["response"]["json"]
        first_message = _native_chain_message(chain)
        first_message["ext"].update({
            "ai_creation_res_code": "710082041",
            "is_creation_clarifying": "1",
        })
        video_block = next(block for block in first_message["content_block"] if block.get("block_type") == 2074)
        video_block["content"]["creation_block"]["creations"][0]["video"]["status"] = 2
        client._read_video_creation_chain = AsyncMock(return_value=chain)
        client._poll_video_creation_result = AsyncMock(return_value={
            "videos": [{"video_url": "https://video.test/result.mp4"}],
            "conversation_id": conversation_id,
        })

        result = await client.generate_video(
            prompt=prompt,
            ref_image_key=image_keys[0],
            ref_image_keys=image_keys,
            save_history=True,
        )

        assert result["videos"][0]["video_url"] == "https://video.test/result.mp4"
        assert len(calls) == 2
        first_text, first_options = calls[0]
        assert first_text == prompt
        assert first_options["chat_ability"]["ability_type"] == 17
        ability_param = json.loads(first_options["chat_ability"]["ability_param"])
        assert ability_param == {
            "ratio": "auto",
            "model": "seedance_v2.0_mini",
            "duration": 10,
            "input_box_content": {
                "user_input_content": prompt,
                "reply_message_format": "生成视频：%s",
            },
        }
        assert first_options["separate_attachment_messages"] is True
        assert [item["uri"] for item in first_options["image_attachments"]] == image_keys
        assert first_options["memory_type"] == 0

        second_text, second_options = calls[1]
        assert second_text == "确认，按上述参数开始生成视频。"
        assert second_options["conversation_id"] == conversation_id
        assert "chat_ability" not in second_options
        assert "image_attachments" not in second_options
        assert second_options["memory_type"] == 0
        client._read_video_creation_chain.assert_awaited_once_with(conversation_id)
        poll_kwargs = client._poll_video_creation_result.await_args.kwargs
        assert poll_kwargs["timeout"] == 600
        assert poll_kwargs["confirmation_sent"] is True
        assert poll_kwargs["confirmation_callback"] is None

    caplog.set_level(logging.INFO, logger="doubao2api.browser_client")
    asyncio.run(_run())
    assert "private user video prompt" not in caplog.text
    assert "tos-fixture" not in caplog.text
    assert "video.test/result" not in caplog.text


def test_browser_client_video_skips_confirmation_and_polls_current_conversation(monkeypatch, tmp_path):
    import asyncio
    import copy
    from unittest.mock import AsyncMock
    from doubao2api.browser_client import BrowserClient

    async def _run():
        client = BrowserClient(headless=True, user_data_dir=str(tmp_path))
        conversation_id = "current-video-conversation"
        calls = []

        async def fake_chat_completion(text, **kwargs):
            calls.append((text, kwargs))
            yield {"ack_client_meta": {"conversation_id": conversation_id}}

        client.chat_completion = fake_chat_completion
        fixture = _native_chain_fixture()["response"]["json"]
        responses = []
        for state in (2, 2, 3):
            response = copy.deepcopy(fixture)
            video_block = next(
                block for block in _native_chain_message(response)["content_block"]
                if block.get("block_type") == 2074
            )
            video_block["content"]["creation_block"]["creations"][0]["video"]["status"] = state
            responses.append(response)
        client._read_video_creation_chain = AsyncMock(side_effect=responses)

        result = await client.generate_video(prompt="pending to complete")

        assert result["videos"][0]["video_url"] == "https://video.test/native-result.mp4"
        assert len(calls) == 1
        assert calls[0][1]["memory_type"] == 2
        assert client._read_video_creation_chain.await_count == 3
        assert all(call.args[0] == conversation_id for call in client._read_video_creation_chain.await_args_list)

    monkeypatch.setattr("doubao2api.browser_client.asyncio.sleep", AsyncMock())
    asyncio.run(_run())


def test_video_confirms_once_when_gate_arrives_after_initial_pending(monkeypatch, tmp_path):
    import asyncio
    import copy
    from unittest.mock import AsyncMock
    from doubao2api.browser_client import BrowserClient

    async def _run():
        client = BrowserClient(headless=True, user_data_dir=str(tmp_path))
        conversation_id = "current-video-conversation"
        calls = []

        async def fake_chat_completion(text, **kwargs):
            calls.append((text, kwargs))
            yield {"ack_client_meta": {"conversation_id": conversation_id}}

        client.chat_completion = fake_chat_completion
        fixture = _native_chain_fixture()["response"]["json"]
        empty = {
            "status_code": 0,
            "downlink_body": {"pull_singe_chain_downlink_body": {"messages": []}},
        }
        clarification = copy.deepcopy(fixture)
        clarifying_message = _native_chain_message(clarification)
        clarifying_message["content_block"] = []
        clarifying_message["ext"] = {
            "ai_creation_res_code": "710082041",
            "is_creation_clarifying": "1",
        }
        stale = copy.deepcopy(clarification)
        complete = copy.deepcopy(fixture)
        completed_message = _native_chain_message(complete)
        completed_message["ext"] = {"ai_creation_res_code": "0"}
        client._read_video_creation_chain = AsyncMock(side_effect=[empty, clarification, stale, complete])

        result = await client.generate_video(prompt="eventual clarification")

        assert result["videos"][0]["video_url"] == "https://video.test/native-result.mp4"
        assert len(calls) == 2
        assert calls[1][0] == "确认，按上述参数开始生成视频。"
        assert calls[1][1]["conversation_id"] == conversation_id
        assert "chat_ability" not in calls[1][1]
        assert "image_attachments" not in calls[1][1]
        assert client._read_video_creation_chain.await_count == 4

    monkeypatch.setattr("doubao2api.browser_client.asyncio.sleep", AsyncMock())
    asyncio.run(_run())


def test_video_poll_cancellation_releases_browser_operation_gate(tmp_path):
    import asyncio
    from doubao2api.browser_client import BrowserClient

    async def _run():
        client = BrowserClient(headless=True, user_data_dir=str(tmp_path))
        conversation_id = "current-video-conversation"
        read_started = asyncio.Event()
        read_count = 0

        async def fake_chat_completion(text, **kwargs):
            yield {"ack_client_meta": {"conversation_id": conversation_id}}

        async def fake_read(_conversation_id, *, timeout=30):
            nonlocal read_count
            read_count += 1
            if read_count == 1:
                return {
                    "status_code": 0,
                    "downlink_body": {"pull_singe_chain_downlink_body": {"messages": []}},
                }
            read_started.set()
            await asyncio.Event().wait()

        client.chat_completion = fake_chat_completion
        client._read_video_creation_chain = fake_read
        task = asyncio.create_task(client.generate_video(prompt="cancel while polling"))
        await asyncio.wait_for(read_started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        async with client.operation_gate.lease():
            pass
        assert read_count == 2

    asyncio.run(_run())


def test_browser_client_video_rejects_unowned_or_incomplete_reference_metadata(tmp_path):
    import asyncio
    from unittest.mock import AsyncMock
    from doubao2api.browser_client import BrowserClient

    async def _run():
        client = BrowserClient(headless=True, user_data_dir=str(tmp_path))
        client._uploaded_image_metadata = {
            "tos-fixture": {
                "uri": "tos-fixture",
                "cdn_url": "https://image.test/ref.png",
                "name": "ref.png",
                "format": "png",
                "width": 0,
                "height": 480,
                "_account_id": None,
            }
        }
        client.chat_completion = AsyncMock()
        with pytest.raises(RuntimeError, match="metadata is incomplete"):
            await client.generate_video(prompt="invalid fixture", ref_image_key="tos-fixture")
        client.chat_completion.assert_not_called()

    asyncio.run(_run())


def test_browser_client_video_rejects_reference_from_another_account(tmp_path):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from doubao2api.browser_client import BrowserClient

    async def _run():
        client = BrowserClient(headless=True, user_data_dir=str(tmp_path))
        client.account_manager = SimpleNamespace(active_account_id="account-b")
        client._uploaded_image_metadata = {
            "tos-fixture": {
                "uri": "tos-fixture",
                "cdn_url": "https://image.test/ref.png",
                "name": "ref.png",
                "format": "png",
                "width": 640,
                "height": 480,
                "_account_id": "account-a",
            }
        }
        client.chat_completion = AsyncMock()
        with pytest.raises(RuntimeError, match="belongs to another account"):
            await client.generate_video(prompt="account ownership", ref_image_key="tos-fixture")
        client.chat_completion.assert_not_called()

    asyncio.run(_run())

def test_chat_completion_video_layout_is_opt_in_and_matches_capture():
    import asyncio
    import json
    from unittest.mock import AsyncMock, MagicMock
    from doubao2api.browser_client import BrowserClient

    async def capture_payload(client, **kwargs):
        captured = []

        async def fake_fetch(url, payload, request_id):
            captured.append(payload)
            client._stream_queues[request_id].put_nowait(None)

        client._page = MagicMock()
        client._page.is_closed.return_value = False
        client._ready = True
        client.is_alive = AsyncMock(return_value=True)
        client._sign_url = AsyncMock(return_value="/chat/completion?fixture")
        client._browser_fetch_stream = fake_fetch
        async for _ in client.chat_completion("fixture prompt", **kwargs):
            pass
        assert len(captured) == 1
        return captured[0]

    async def _run():
        attachments = [
            {"uri": "tos-fixture-a", "cdn_url": "https://image.test/a.png", "name": "a.png", "format": "png", "width": 640, "height": 640},
            {"uri": "tos-fixture-b", "cdn_url": "https://image.test/b.png", "name": "b.png", "format": "png", "width": 640, "height": 640},
        ]
        normal_client = BrowserClient(headless=True)
        normal = await capture_payload(normal_client, image_attachments=attachments)
        assert len(normal["messages"]) == 1
        assert "chat_ability" not in normal
        assert len([b for b in normal["messages"][0]["content_block"] if b["block_type"] == 10052]) == 2

        video_client = BrowserClient(headless=True)
        ability = {
            "ability_type": 17,
            "ability_param": json.dumps({
                "ratio": "auto",
                "model": "seedance_v2.0_mini",
                "duration": 10,
                "input_box_content": {
                    "user_input_content": "fixture prompt",
                    "reply_message_format": "生成视频：%s",
                },
            }, ensure_ascii=False),
        }
        video = await capture_payload(
            video_client,
            image_attachments=attachments,
            chat_ability=ability,
            separate_attachment_messages=True,
        )
        assert video["chat_ability"] == ability
        assert len(video["messages"]) == 2
        image_blocks = video["messages"][0]["content_block"]
        assert [block["block_type"] for block in image_blocks] == [10052]
        assert len(image_blocks[0]["content"]["attachment_block"]["attachments"]) == 2
        first_image = image_blocks[0]["content"]["attachment_block"]["attachments"][0]["image"]
        assert first_image["uri"] == "tos-fixture-a"
        assert first_image["image_ori"]["width"] == 640
        assert first_image["image_ori"]["height"] == 640
        assert video["messages"][1]["content_block"][0]["block_type"] == 10000

    asyncio.run(_run())


def test_chat_completion_applies_video_memory_type_without_changing_default():
    import asyncio
    import json
    from unittest.mock import AsyncMock, MagicMock
    from doubao2api.browser_client import BrowserClient

    async def capture(client, *, memory_type=None):
        captured = []

        async def fake_fetch(url, payload, request_id):
            captured.append(payload)
            client._stream_queues[request_id].put_nowait(None)

        client._page = MagicMock()
        client._page.is_closed.return_value = False
        client._ready = True
        client.is_alive = AsyncMock(return_value=True)
        client._sign_url = AsyncMock(return_value="/chat/completion?fixture")
        client._browser_fetch_stream = fake_fetch
        async for _ in client.chat_completion("fixture", memory_type=memory_type):
            pass
        return captured[0]["option"]["message_storage_type"]

    async def _run():
        assert await capture(BrowserClient(headless=True)) == 0
        assert await capture(BrowserClient(headless=True), memory_type=2) == 2

    asyncio.run(_run())

def test_browser_client_no_conversation_error_logs_only_safe_video_metadata(tmp_path, caplog):
    import asyncio
    import logging
    from doubao2api.browser_client import BrowserClient

    async def _run():
        client = BrowserClient(headless=True, user_data_dir=str(tmp_path))

        async def fake_chat_completion(text, **kwargs):
            yield {"_event": "CHUNK_DELTA", "text": "private provider response text"}

        client.chat_completion = fake_chat_completion
        with pytest.raises(RuntimeError, match=r"Video generation failed \(no_conversation_id\)"):
            await client.generate_video(prompt="private user prompt")

    caplog.set_level(logging.INFO, logger="doubao2api.browser_client")
    asyncio.run(_run())
    assert "category=no_conversation_id" in caplog.text
    assert "private provider response text" not in caplog.text
    assert "private user prompt" not in caplog.text


def test_browser_client_rejects_malformed_creation_chain_without_poll(tmp_path, caplog):
    import asyncio
    from unittest.mock import AsyncMock
    from doubao2api.browser_client import BrowserClient

    async def _run():
        client = BrowserClient(headless=True, user_data_dir=str(tmp_path))

        async def fake_chat_completion(text, **kwargs):
            yield {"ack_client_meta": {"conversation_id": "current-video-conversation"}}
            yield {"_event": "CHUNK_DELTA", "text": "private provider response text"}

        client.chat_completion = fake_chat_completion
        client._read_video_creation_chain = AsyncMock(return_value={"downlink_body": {}})
        client._poll_video_creation_result = AsyncMock()
        with pytest.raises(RuntimeError, match=r"Video generation failed \(malformed_response\)"):
            await client.generate_video(prompt="private user prompt")
        client._poll_video_creation_result.assert_not_awaited()

    asyncio.run(_run())
    assert "private provider response text" not in caplog.text
    assert "private user prompt" not in caplog.text


def test_browser_client_sanitizes_native_video_stream_error(tmp_path, caplog):
    import asyncio
    import json
    import logging
    from doubao2api.browser_client import BrowserClient

    async def _run():
        client = BrowserClient(headless=True, user_data_dir=str(tmp_path))
        body = json.dumps({
            "code": 429,
            "log_id": "safe-native-log",
            "error_msg": "private challenge token must not escape",
        })

        async def fake_chat_completion(text, **kwargs):
            yield {"error": True, "status": 429, "body": body}

        client.chat_completion = fake_chat_completion
        with pytest.raises(UpstreamAPIError) as raised:
            await client.generate_video(prompt="private user prompt")
        assert raised.value.upstream_error.category == "rate_limit"
        assert raised.value.upstream_error.status_code == 429
        assert raised.value.upstream_error.log_id == "safe-native-log"
        assert "private challenge token" not in str(raised.value)

    caplog.set_level(logging.INFO, logger="doubao2api.browser_client")
    asyncio.run(_run())
    assert "category=rate_limit" in caplog.text
    assert "log_id=safe-native-log" in caplog.text
    assert "private challenge token" not in caplog.text
    assert "private user prompt" not in caplog.text


def test_browser_client_sanitizes_native_video_http_error_status(tmp_path, caplog):
    import asyncio
    import json
    import logging
    from doubao2api.browser_client import BrowserClient

    async def _run():
        client = BrowserClient(headless=True, user_data_dir=str(tmp_path))
        body = json.dumps({
            "code": 429,
            "log_id": "safe-native-http-log",
            "error_msg": "private challenge token",
        })

        async def fake_chat_completion(text, **kwargs):
            yield {"error": True, "status": 429, "body": body}

        client.chat_completion = fake_chat_completion
        with pytest.raises(UpstreamAPIError) as raised:
            await client.generate_video(prompt="private user prompt")
        assert raised.value.upstream_error.category == "rate_limit"
        assert raised.value.upstream_error.status_code == 429
        assert raised.value.upstream_error.log_id == "safe-native-http-log"
        assert "private challenge token" not in str(raised.value)

    caplog.set_level(logging.INFO, logger="doubao2api.browser_client")
    asyncio.run(_run())
    assert "status=429" in caplog.text
    assert "log_id=safe-native-http-log" in caplog.text
    assert "private challenge token" not in caplog.text
    assert "private user prompt" not in caplog.text


@pytest.mark.parametrize("event_name", ["CHUNK_DELTA", "FULL_MSG_NOTIFY"])
def test_native_video_quota_hint_marks_only_video_feature_and_raises(tmp_path, caplog, event_name):
    import asyncio
    import logging
    from unittest.mock import Mock
    from types import SimpleNamespace
    from doubao2api.browser_client import BrowserClient, VideoQuotaExceeded

    async def _run():
        client = BrowserClient(headless=True, user_data_dir=str(tmp_path))
        manager = SimpleNamespace(active_account_id="account-a", mark_quota_exceeded=Mock())
        client.account_manager = manager

        async def fake_chat_completion(text, **kwargs):
            if event_name == "CHUNK_DELTA":
                yield {"_event": event_name, "text": "今日生成次数已达"}
                yield {"_event": event_name, "text": "上限"}
            else:
                yield {
                    "_event": event_name,
                    "data": {"message": {"content_block": [{
                        "block_type": 10000,
                        "content": {"text_block": {"text": "今日生成次数已达上限"}},
                    }]}},
                }

        client.chat_completion = fake_chat_completion
        with pytest.raises(VideoQuotaExceeded):
            await client.generate_video(prompt="quota fixture")
        manager.mark_quota_exceeded.assert_called_once_with(
            account_id="account-a", feature="video"
        )

    caplog.set_level(logging.INFO, logger="doubao2api.browser_client")
    asyncio.run(_run())
    assert "category=video_quota" in caplog.text
    assert "今日生成次数" not in caplog.text
    assert "quota fixture" not in caplog.text


def test_unrecognized_video_text_does_not_mark_account_quota(tmp_path):
    import asyncio
    from unittest.mock import Mock
    from types import SimpleNamespace
    from doubao2api.browser_client import BrowserClient

    async def _run():
        client = BrowserClient(headless=True, user_data_dir=str(tmp_path))
        manager = SimpleNamespace(active_account_id="account-a", mark_quota_exceeded=Mock())
        client.account_manager = manager

        async def fake_chat_completion(text, **kwargs):
            yield {"_event": "CHUNK_DELTA", "text": "暂时没有确认视频额度状态"}
            yield {"ack_client_meta": {"conversation_id": "current-video-conversation"}}

        client.chat_completion = fake_chat_completion
        conversation_id = await client._collect_video_chat_turn(
            "fixture", phase="submit", memory_type=0, attachments_count=0
        )
        assert conversation_id == "current-video-conversation"
        manager.mark_quota_exceeded.assert_not_called()

    asyncio.run(_run())


def test_native_video_quota_signal_is_checked_before_long_chunk_truncation(tmp_path, caplog):
    import asyncio
    import logging
    from unittest.mock import Mock
    from types import SimpleNamespace
    from doubao2api.browser_client import BrowserClient, VideoQuotaExceeded

    async def _run():
        client = BrowserClient(headless=True, user_data_dir=str(tmp_path))
        manager = SimpleNamespace(active_account_id="account-a", mark_quota_exceeded=Mock())
        client.account_manager = manager
        provider_text = "今日生成次数已达上限" + ("tail" * 100)

        async def fake_chat_completion(text, **kwargs):
            yield {"_event": "CHUNK_DELTA", "text": provider_text}

        client.chat_completion = fake_chat_completion
        with pytest.raises(VideoQuotaExceeded):
            await client.generate_video(prompt="long quota fixture")
        manager.mark_quota_exceeded.assert_called_once_with(
            account_id="account-a", feature="video"
        )
        assert provider_text not in caplog.text

    caplog.set_level(logging.INFO, logger="doubao2api.browser_client")
    asyncio.run(_run())


def test_video_upstream_error_event_keeps_classification_in_api(test_app):
    app, client = test_app
    client.generate_video = AsyncMock(
        side_effect=UpstreamAPIError(parse_upstream_error({"code": 429}))
    )

    response = TestClient(app).post(
        "/v1/video/generations", json={"prompt": "bounded test"}
    )

    assert response.status_code == 429
    assert response.json()["error"]["code"] == "rate_limit"
    client.generate_video.assert_awaited_once()


@pytest.mark.parametrize(
    "video_url",
    ["https://video.test/polled.mp4", "http://video.test/polled.mp4"],
)
def test_browser_client_poll_parses_video_result_event(tmp_path, video_url):
    import asyncio
    import json
    from doubao2api.browser_client import BrowserClient

    async def _run():
        client = BrowserClient(headless=True, user_data_dir=str(tmp_path))
        event = {
            "event_type": 2001,
            "event_data": {
                "message": {
                    "content_type": 2021,
                    "content": json.dumps({
                        "data": [{"video_url": video_url}],
                    }),
                },
            },
        }
        client._samantha_request = AsyncMock(
            return_value=f"data: {json.dumps(event)}\n\n"
        )

        result = await client._poll_video_result("opaque-task", "poll result")

        assert result["videos"][0]["video_url"] == video_url
        assert result["prompt"] == "poll result"

    asyncio.run(_run())


def test_browser_client_poll_empty_video_result_is_categorized(tmp_path, caplog):
    import asyncio
    import json
    from doubao2api.browser_client import BrowserClient

    async def _run():
        client = BrowserClient(headless=True, user_data_dir=str(tmp_path))
        event = {
            "event_type": 2001,
            "event_data": {
                "message": {"content_type": 2021, "content": json.dumps({"data": []})},
            },
        }
        client._samantha_request = AsyncMock(
            return_value=f"data: {json.dumps(event)}\n\n"
        )

        with pytest.raises(RuntimeError, match=r"Video generation failed \(no_video_result\)"):
            await client._poll_video_result("opaque-task", "no result")

    asyncio.run(_run())
    assert "category=no_video_result" in caplog.text
    assert "opaque-task" not in caplog.text


@pytest.mark.parametrize(
    "video_url",
    [
        "javascript:alert(1)",
        "data:video/mp4;base64,AAAA",
        "ftp://video.test/result.mp4",
        "/relative/result.mp4",
        "https:///missing-host.mp4",
        "https://video.test:invalid-port/result.mp4",
    ],
)
def test_browser_video_parser_rejects_non_http_urls_in_legacy_events(tmp_path, video_url):
    import asyncio
    import json
    from doubao2api.browser_client import BrowserClient

    async def _run():
        client = BrowserClient(headless=True, user_data_dir=str(tmp_path))
        event = {
            "event_type": 2001,
            "event_data": {
                "message": {
                    "content_type": 2021,
                    "content": json.dumps({
                        "data": [{"video_url": video_url}],
                    }),
                },
            },
        }
        client._samantha_request = AsyncMock(
            return_value=f"data: {json.dumps(event)}\n\n"
        )

        with pytest.raises(RuntimeError, match=r"Video generation failed \(malformed_result\)"):
            await client._poll_video_result("opaque-task", "no network probe")

    asyncio.run(_run())


def _native_chain_fixture():
    from pathlib import Path

    return json.loads(
        (Path(__file__).parent / "fixtures" / "video_creation_block_2074.json").read_text(encoding="utf-8")
    )


def _native_chain_message(response):
    return response["downlink_body"]["pull_singe_chain_downlink_body"]["messages"][0]


def test_native_creation_block_2074_fixture_returns_completed_video():
    from doubao2api.browser_client import BrowserClient

    fixture = _native_chain_fixture()
    result = BrowserClient._parse_video_creation_chain(fixture["response"]["json"])

    assert result["state"] == "complete"
    assert result["videos"] == [{
        "video_url": "https://video.test/native-result.mp4",
        "cover_url": "",
        "width": 720,
        "height": 720,
        "duration": 10.08,
    }]


def test_native_creation_chain_empty_messages_are_pending_for_new_conversation():
    from doubao2api.browser_client import BrowserClient

    result = BrowserClient._parse_video_creation_chain({
        "status_code": 0,
        "downlink_body": {"pull_singe_chain_downlink_body": {"messages": []}},
    }, conversation_id="current-api-conversation")

    assert result["state"] == "pending"


def test_native_creation_chain_result_beats_stale_clarification_in_either_order():
    import copy
    from doubao2api.browser_client import BrowserClient

    fixture = _native_chain_fixture()
    response = copy.deepcopy(fixture["response"]["json"])
    completed = copy.deepcopy(_native_chain_message(response))
    completed["ext"] = {"ai_creation_res_code": "0"}
    stale_confirmation = copy.deepcopy(completed)
    stale_confirmation["content_block"] = []
    stale_confirmation["ext"] = {
        "ai_creation_res_code": "710082041",
        "is_creation_clarifying": "1",
    }

    chain = response["downlink_body"]["pull_singe_chain_downlink_body"]
    for messages in ([stale_confirmation, completed], [completed, stale_confirmation]):
        chain["messages"] = copy.deepcopy(messages)
        parsed = BrowserClient._parse_video_creation_chain(
            response, conversation_id="current-api-conversation"
        )
        assert parsed["state"] == "complete"

    # Without a completed video block, a resolved code 0 still suppresses
    # a stale clarification gate regardless of history ordering.
    resolved_pending = copy.deepcopy(completed)
    resolved_pending["content_block"] = []
    chain["messages"] = [stale_confirmation, resolved_pending]
    parsed = BrowserClient._parse_video_creation_chain(
        response, conversation_id="current-api-conversation"
    )
    assert parsed["state"] == "pending"


def test_native_creation_chain_ignores_non_video_creation_types():
    import copy
    from doubao2api.browser_client import BrowserClient

    fixture = _native_chain_fixture()
    response = copy.deepcopy(fixture["response"]["json"])
    message = _native_chain_message(response)
    video_block = next(block for block in message["content_block"] if block.get("block_type") == 2074)
    video_block["content"]["creation_block"]["creations"][0]["type"] = 1

    result = BrowserClient._parse_video_creation_chain(response)
    assert result["state"] == "pending"
    assert result["videos"] == []


def test_video_chain_diagnostics_keep_numeric_code_and_hash_conversation_id(caplog):
    import hashlib
    import logging
    from doubao2api.browser_client import BrowserClient

    conversation_id = "private-conversation-id"
    response = {
        "downlink_body": {"pull_singe_chain_downlink_body": {"messages": [
            {
                "conversation_id": conversation_id,
                "ext": {"ai_creation_res_code": "710082041"},
                "content_block": [{
                    "block_type": 2074,
                    "content": {"creation_block": {"creations": [{
                        "type": 2, "video": {"status": 2},
                    }]}},
                }],
            },
            {
                "conversation_id": "another-private-conversation-id",
                "ext": {"ai_creation_res_code": "private-error-code"},
                "content_block": [],
            },
        ]}},
    }
    client = BrowserClient(headless=True)
    caplog.set_level(logging.WARNING, logger="doubao2api.browser_client")
    client._log_video_chain_diagnostic({
        "state": "upstream_error",
        "http_status": 200,
        "response": response,
        "conversation_id": conversation_id,
    })

    alias = hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()[:12]
    assert f"conversation_alias={alias}" in caplog.text
    assert "business_codes=['710082041']" in caplog.text
    assert conversation_id not in caplog.text
    assert "another-private-conversation-id" not in caplog.text
    assert "private-error-code" not in caplog.text


@pytest.mark.parametrize("status,expected_state", [(2, "pending"), (3, "malformed_result")])
def test_native_creation_result_requires_status_three_and_valid_url(status, expected_state):
    import copy
    from doubao2api.browser_client import BrowserClient

    fixture = _native_chain_fixture()
    response = copy.deepcopy(fixture["response"]["json"])
    message = _native_chain_message(response)
    block = next(item for item in message["content_block"] if item.get("block_type") == 2074)
    video = block["content"]["creation_block"]["creations"][0]["video"]
    video["status"] = status
    if status == 3:
        video["download_url"] = "javascript:alert(1)"

    result = BrowserClient._parse_video_creation_chain(response)
    assert result["state"] == expected_state
    assert result["videos"] == []


def test_native_creation_result_is_bound_to_the_requested_conversation():
    import copy
    from doubao2api.browser_client import BrowserClient

    fixture = _native_chain_fixture()
    response = copy.deepcopy(fixture["response"]["json"])
    _native_chain_message(response)["conversation_id"] = "older-conversation"

    result = BrowserClient._parse_video_creation_chain(
        response, conversation_id="current-api-conversation"
    )

    assert result["state"] == "conversation_mismatch"
    assert result["videos"] == []

    response = copy.deepcopy(fixture["response"]["json"])
    current_message = copy.deepcopy(_native_chain_message(response))
    foreign_message = copy.deepcopy(current_message)
    foreign_message["conversation_id"] = "older-conversation"
    response["downlink_body"]["pull_singe_chain_downlink_body"]["messages"] = [
        current_message,
        foreign_message,
    ]
    result = BrowserClient._parse_video_creation_chain(
        response, conversation_id="current-api-conversation"
    )
    assert result["state"] == "conversation_mismatch"
    assert result["videos"] == []


def test_video_confirmation_classifier_uses_native_enums_only():
    from doubao2api.browser_client import BrowserClient

    message = {
        "ext": {"ai_creation_res_code": "710082041", "is_creation_clarifying": "1"},
        "content_block": [],
    }
    assert BrowserClient._is_video_confirmation_required(message)
    message["ext"]["is_creation_clarifying"] = "0"
    assert not BrowserClient._is_video_confirmation_required(message)
    message["ext"] = {"ai_creation_res_code": "0", "is_creation_clarifying": "1"}
    assert not BrowserClient._is_video_confirmation_required(message)


def test_native_chain_parser_reports_confirmation_only_for_captured_gate():
    import copy
    from doubao2api.browser_client import BrowserClient

    fixture = _native_chain_fixture()
    response = copy.deepcopy(fixture["response"]["json"])
    message = _native_chain_message(response)
    message["ext"]["ai_creation_res_code"] = "710082041"
    message["ext"]["is_creation_clarifying"] = "1"
    video_block = next(block for block in message["content_block"] if block.get("block_type") == 2074)
    video_block["content"]["creation_block"]["creations"][0]["video"]["status"] = 2
    assert BrowserClient._parse_video_creation_chain(response)["state"] == "confirmation_required"

    message["ext"]["is_creation_clarifying"] = "0"
    assert BrowserClient._parse_video_creation_chain(response)["state"] != "confirmation_required"


def test_chain_reader_request_matches_captured_protocol_shape():
    import asyncio
    from unittest.mock import AsyncMock
    from doubao2api.browser_client import BrowserClient

    async def _run():
        client = BrowserClient(headless=True)
        client._browser_json_post = AsyncMock(return_value={"status_code": 0})

        await client._read_video_creation_chain("current-api-conversation")

        call = client._browser_json_post.await_args
        assert call.args[0] == "/im/chain/single"
        payload = call.args[1]
        assert payload["channel"] == 2
        assert payload["cmd"] == 3100
        assert payload["version"] == "1"
        assert payload["sequence_id"]
        body = payload["uplink_body"]["pull_singe_chain_uplink_body"]
        assert body == {
            "conversation_id": "current-api-conversation",
            "anchor_index": 9007199254740991,
            "conversation_type": 3,
            "direction": 1,
            "limit": 20,
            "ext": {},
            "filter": {"index_list": []},
            "evaluate_ab_params": "",
            "evaluate_common_params": "",
            "option": {"lazy_load_strategy": 0},
        }

    asyncio.run(_run())


def test_video_creation_poller_waits_for_status_three(tmp_path):
    import asyncio
    import copy
    from unittest.mock import AsyncMock
    from doubao2api.browser_client import BrowserClient

    async def _run():
        fixture = _native_chain_fixture()
        pending = copy.deepcopy(fixture["response"]["json"])
        complete = copy.deepcopy(fixture["response"]["json"])
        pending_block = next(
            item for item in _native_chain_message(pending)["content_block"]
            if item.get("block_type") == 2074
        )
        pending_block["content"]["creation_block"]["creations"][0]["video"]["status"] = 2
        client = BrowserClient(headless=True, user_data_dir=str(tmp_path))
        client._read_video_creation_chain = AsyncMock(side_effect=[pending, complete])

        result = await client._poll_video_creation_result(
            "current-api-conversation", timeout=1, poll_interval=0
        )

        assert result["videos"][0]["video_url"] == "https://video.test/native-result.mp4"
        assert client._read_video_creation_chain.await_count == 2
        assert all(
            call.args[0] == "current-api-conversation"
            for call in client._read_video_creation_chain.await_args_list
        )

    asyncio.run(_run())


def test_video_creation_poller_stops_at_timeout(tmp_path):
    import asyncio
    import copy
    from unittest.mock import AsyncMock
    from doubao2api.browser_client import BrowserClient

    async def _run():
        fixture = _native_chain_fixture()
        pending = copy.deepcopy(fixture["response"]["json"])
        block = next(item for item in _native_chain_message(pending)["content_block"] if item.get("block_type") == 2074)
        block["content"]["creation_block"]["creations"][0]["video"]["status"] = 2
        client = BrowserClient(headless=True, user_data_dir=str(tmp_path))
        client._read_video_creation_chain = AsyncMock(return_value=pending)

        with pytest.raises(RuntimeError, match=r"Video generation failed \(pending_timeout\)"):
            await client._poll_video_creation_result(
                "current-api-conversation", timeout=0, poll_interval=0
            )
        client._read_video_creation_chain.assert_not_awaited()

    asyncio.run(_run())


def test_browser_client_generate_image_attachments_payload():
    import asyncio
    import json
    from doubao2api.browser_client import BrowserClient

    async def _run():
        client = BrowserClient(headless=True)
        captured_payload = {}

        async def fake_samantha_request(payload, timeout=60):
            nonlocal captured_payload
            captured_payload = payload
            inner_content = json.dumps({
                "data": [
                    {"image_ori": {"url": "https://img.test", "width": 1024, "height": 1024}}
                ]
            })
            sse_event = {
                "event_type": 2001,
                "event_data": {
                    "message": {
                        "content_type": 2010,
                        "content": inner_content,
                    }
                },
            }
            return f"data: {json.dumps(sse_event)}\n\n"

        client._samantha_request = fake_samantha_request

        res = await client.generate_image(
            prompt="测试生图多图",
            ratio="1:1",
            ref_image_keys=["key_alpha", "key_beta"],
        )
        assert res["images"][0]["url"] == "https://img.test"
        assert len(captured_payload["messages"]) == 1
        msg = captured_payload["messages"][0]
        assert msg["attachments"] == [
            {"type": "image", "key": "key_alpha", "extra": {"refer_types": "overall"}},
            {"type": "image", "key": "key_beta", "extra": {"refer_types": "overall"}},
        ]

    asyncio.run(_run())


def test_doubao_chat_client_generate_video_attachments_payload():
    import asyncio
    import json
    from unittest.mock import MagicMock
    from doubao2api.client import DoubaoChatClient

    async def _run():
        client = DoubaoChatClient(cookies={"sessionid": "mock_cookie"}, ms_token="dummy")
        captured_data = {}

        class DummyResponse:
            status = 200
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def read(self):
                sse_event = {
                    "event_type": 2001,
                    "event_data": json.dumps({
                        "fin_reason": {"reason": 1, "async_task": {"id": "task_123"}}
                    }),
                }
                return f"data: {json.dumps(sse_event)}\n\n".encode("utf-8")

        calls = []
        def fake_post(url, data=None, headers=None, timeout=None):
            calls.append((url, json.loads(data) if data else {}))
            return DummyResponse()

        client._session = MagicMock()
        client._session.post = fake_post
        client._poll_video_result = AsyncMock(return_value=MagicMock())

        await client.generate_video(
            prompt="测试HTTP多图参考",
            ref_image_keys=["key1", "key2"],
        )
        assert len(calls) >= 1
        captured_data = calls[0][1]
        assert len(captured_data["messages"]) == 1
        msg = captured_data["messages"][0]
        assert msg["content_type"] == 2020
        assert msg["attachments"] == [
            {"type": "image", "key": "key1"},
            {"type": "image", "key": "key2"},
        ]

    asyncio.run(_run())


