"""
Tests for multimodal image & file support in chat_completions (Issue #7).
Validates image resolution extraction, attachment block construction,
and OpenAI format compatibility for both streaming and non-streaming.
"""
import asyncio
import base64
import io
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from PIL import Image
from fastapi.testclient import TestClient

from doubao2api.browser_client import BrowserClient
from doubao2api.unified_server import create_app, _extract_prompt_and_file_refs, _Message


def _create_test_image_bytes(width: int = 800, height: int = 600) -> bytes:
    """Generate a valid PNG image with specified dimensions."""
    img = Image.new("RGB", (width, height), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.unit
def test_extract_prompt_and_multimodal_refs():
    """Verify prompt, image_url, and file_url extraction logic."""
    # 1. Standard OpenAI image_url
    messages = [
        _Message(
            role="user",
            content=[
                {"type": "text", "text": "Describe this image:"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
            ],
        )
    ]
    prompt, file_refs, image_refs = _extract_prompt_and_file_refs(messages)
    assert prompt == "Describe this image:"
    assert len(image_refs) == 1
    assert image_refs[0]["url"] == "data:image/png;base64,iVBORw0KGgo="
    assert len(file_refs) == 0

    # 2. file_url referencing an image file -> auto-classified as image_refs
    messages_img_file = [
        _Message(
            role="user",
            content=[
                {"type": "file_url", "file_url": {"url": "https://example.com/pic.jpeg", "name": "pic.jpeg"}},
            ],
        )
    ]
    prompt, file_refs, image_refs = _extract_prompt_and_file_refs(messages_img_file)
    assert len(image_refs) == 1
    assert image_refs[0]["url"] == "https://example.com/pic.jpeg"
    assert len(file_refs) == 0

    # 3. file_url referencing a document -> classified as file_refs
    messages_doc = [
        _Message(
            role="user",
            content=[
                {"type": "text", "text": "Analyze this document:"},
                {"type": "file_url", "file_url": {"url": "https://example.com/doc.pdf", "name": "doc.pdf"}},
            ],
        )
    ]
    prompt, file_refs, image_refs = _extract_prompt_and_file_refs(messages_doc)
    assert prompt == "Analyze this document:"
    assert len(file_refs) == 1
    assert file_refs[0]["name"] == "doc.pdf"
    assert len(image_refs) == 0


def test_browser_client_upload_image_dimensions():
    """Verify upload_image detects authentic dimensions instead of hardcoded 64x64."""
    async def _run():
        img_bytes = _create_test_image_bytes(800, 600)
        client = BrowserClient()
        client._ready = True
        client._page = AsyncMock()
        client._context = AsyncMock()
        client._context.cookies = AsyncMock(return_value=[])
        client._sign_url = AsyncMock(return_value="https://www.doubao.com/upload")

        mock_resp1 = MagicMock()
        mock_resp1.status_code = 200
        mock_resp1.json = MagicMock(return_value={
            "code": 0,
            "data": {
                "uri": "tos-cn-i-upload/test_img_key",
            },
        })

        mock_resp2 = MagicMock()
        mock_resp2.status_code = 200
        mock_resp2.json = MagicMock(return_value={
            "code": 0,
            "data": {
                "file_urls": [{"url": "https://p3.doubao.com/test_img_key~tplv.image"}],
            },
        })

        client._http = AsyncMock()
        client._http.post = AsyncMock(side_effect=[mock_resp1, mock_resp2])

        uploaded = await client.upload_image(image_bytes=img_bytes, filename="test.png")
        assert uploaded["width"] == 800
        assert uploaded["height"] == 600
        assert uploaded["uri"] == "tos-cn-i-upload/test_img_key"
        assert uploaded["cdn_url"] == "https://p3.doubao.com/test_img_key~tplv.image"
        assert uploaded["format"] == "png"

    asyncio.run(_run())


def test_chat_completion_multimodal_block_structure():
    """Verify chat_completion builds type: 1 Attachment for images with authentic dimensions."""
    async def _run():
        client = BrowserClient()
        client._ready = True
        client._page = AsyncMock()
        client._page.is_closed = MagicMock(return_value=False)
        client.is_alive = AsyncMock(return_value=True)
        client._sign_url = AsyncMock(return_value="/chat/completion?sig=123")

        sent_payload = None

        async def mock_evaluate(script, payload=None, *args, **kwargs):
            nonlocal sent_payload
            if isinstance(payload, list) and len(payload) >= 3:
                sent_payload = json.loads(payload[1])
                req_id = payload[2]
                if req_id in client._stream_queues:
                    client._stream_queues[req_id].put_nowait(None)
            return {"done": True, "events": []}

        client._page.evaluate = mock_evaluate

        image_attachments = [{
            "name": "photo.png",
            "uri": "tos-test-key",
            "cdn_url": "https://img.doubao.com/test.png",
            "width": 1920,
            "height": 1080,
            "format": "png",
        }]

        async for _ in client.chat_completion(
            text="What is in this picture?",
            image_attachments=image_attachments,
        ):
            pass

        assert sent_payload is not None
        blocks = sent_payload["messages"][0]["content_block"]
        assert len(blocks) == 2  # 1 image attachment block + 1 text block

        # Verify attachment block
        att_block = blocks[0]
        assert att_block["block_type"] == 10052
        att = att_block["content"]["attachment_block"]["attachments"][0]
        assert att["type"] == 1  # image
        assert att["image"]["image_ori"]["width"] == 1920
        assert att["image"]["image_ori"]["height"] == 1080
        assert att["image"]["image_ori"]["url"] == "https://img.doubao.com/test.png"

        # Verify text block
        text_block = blocks[1]
        assert text_block["block_type"] == 10000
        assert text_block["content"]["text_block"]["text"] == "What is in this picture?"

    asyncio.run(_run())


def test_unified_server_multimodal_chat_completions():
    """Test unified server /v1/chat/completions endpoint with image_url (OpenAI format)."""
    img_bytes = _create_test_image_bytes(400, 300)
    b64_data = base64.b64encode(img_bytes).decode("ascii")
    data_uri = f"data:image/png;base64,{b64_data}"

    mock_client = AsyncMock()
    mock_client.needs_captcha = False
    mock_client.is_ready = True
    mock_client.is_alive = AsyncMock(return_value=True)
    mock_client.upload_image = AsyncMock(return_value={
        "uri": "tos-cn-test",
        "cdn_url": "https://p3.doubao.com/test",
        "name": "image.png",
        "width": 400,
        "height": 300,
        "format": "png",
    })

    async def mock_chat_completion(*args, **kwargs):
        assert kwargs.get("image_attachments") is not None
        assert kwargs["image_attachments"][0]["width"] == 400
        yield {
            "_event": "CHUNK_DELTA",
            "text": "这是一张蓝灰色的测试图片。",
            "conversation_id": "conv-test-123",
        }

    mock_client.chat_completion = mock_chat_completion
    mock_client.record_success = MagicMock()
    mock_client.extract_conversation_id = MagicMock(return_value="conv-test-123")

    app = create_app(browser_client=mock_client)
    test_client = TestClient(app)

    # 1. Non-streaming request
    payload = {
        "model": "doubao",
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "描述这张图片"},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
    }

    resp = test_client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "这是一张蓝灰色的测试图片。" in data["choices"][0]["message"]["content"]

    # 2. Streaming request with image
    payload["stream"] = True
    resp_stream = test_client.post("/v1/chat/completions", json=payload)
    assert resp_stream.status_code == 200
    assert "text/event-stream" in resp_stream.headers["content-type"]
    assert "这是一张蓝灰色的测试图片。" in resp_stream.text
