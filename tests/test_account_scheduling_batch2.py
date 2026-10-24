"""Account scheduling and retry coverage using local request mocks only."""

import asyncio
import json
import threading
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from doubao2api.account_manager import AccountManager
from doubao2api.browser_client import BrowserClient, VideoQuotaExceeded
from doubao2api.browser_operation import BrowserOperationGate
from doubao2api.unified_server import create_app


async def _asgi_request(app, path, payload, *, method="POST", disconnect_event=None):
    body = json.dumps(payload).encode("utf-8")
    disconnect_event = disconnect_event or asyncio.Event()
    request_sent = False
    sent = []

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnect_event.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"testserver"), (b"content-type", b"application/json")],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    await app(scope, receive, send)
    return sent


def _client_with_accounts(tmp_path, strategy="failover", count=2):
    manager = AccountManager(filepath=str(tmp_path / "accounts.json"), strategy=strategy)
    accounts = [
        manager.add_or_update_account(f"Account {i + 1}", {"sessionid": f"session-{i + 1}"}, account_id=f"a{i + 1}")
        for i in range(count)
    ]
    if accounts:
        manager.set_active_account(accounts[0].id)
    client = BrowserClient(headless=True, user_data_dir=str(tmp_path))
    client.account_manager = manager
    client._ready = True
    client.is_captcha_visible = AsyncMock(return_value=False)
    client.try_auto_solve_captcha = AsyncMock(return_value=False)

    async def switch_to_account(account_id):
        manager.set_active_account(account_id)
        client._needs_captcha = False
        return True

    client.switch_to_account = AsyncMock(side_effect=switch_to_account)
    return client, manager, accounts


@pytest.mark.parametrize(
    "strategy,expected_accounts,expected_switches",
    [
        ("failover", ["a1", "a1", "a1"], []),
        ("round_robin", ["a2", "a1", "a2"], ["a2", "a1", "a2"]),
    ],
)
def test_chat_image_and_video_endpoints_use_one_scheduled_account_per_request(
    tmp_path, strategy, expected_accounts, expected_switches
):
    client, manager, _ = _client_with_accounts(tmp_path, strategy=strategy)
    used_accounts = []

    async def mock_chat(*args, **kwargs):
        used_accounts.append(manager.active_account_id)
        yield {"_event": "CHUNK_DELTA", "text": "chat ok"}

    async def mock_image(**kwargs):
        used_accounts.append(manager.active_account_id)
        return {"images": [{"url": "https://image.test/result.png"}]}

    async def mock_video(**kwargs):
        used_accounts.append(manager.active_account_id)
        return {"videos": [{"video_url": "https://video.test/result.mp4"}]}

    client.chat_completion = mock_chat
    client.extract_conversation_id = MagicMock(return_value=None)
    client.generate_image = AsyncMock(side_effect=mock_image)
    client.generate_video = AsyncMock(side_effect=mock_video)
    test_client = TestClient(create_app(browser_client=client))

    chat = test_client.post(
        "/v1/chat/completions",
        json={"model": "doubao", "messages": [{"role": "user", "content": "hi"}]},
    )
    image = test_client.post("/v1/images/generations", json={"prompt": "image"})
    video = test_client.post("/v1/video/generations", json={"prompt": "video", "ratio": "16:9"})

    assert [chat.status_code, image.status_code, video.status_code] == [200, 200, 200]
    assert used_accounts == expected_accounts
    assert [call.args[0] for call in client.switch_to_account.await_args_list] == expected_switches


def test_known_conversation_stays_on_owner_and_unknown_id_uses_current_account(tmp_path):
    client, manager, _ = _client_with_accounts(tmp_path, strategy="round_robin")
    manager._round_robin_last_id = None
    conversation_ids = iter(["conversation-a", "legacy-conversation"])
    calls = 0

    async def mock_chat(*args, **kwargs):
        nonlocal calls
        calls += 1
        yield {"_event": "CHUNK_DELTA", "text": "ok"}

    client.chat_completion = mock_chat
    client.extract_conversation_id = MagicMock(side_effect=lambda _event: next(conversation_ids, None))
    test_client = TestClient(create_app(browser_client=client))
    first = test_client.post(
        "/v1/chat/completions",
        json={"model": "doubao", "messages": [{"role": "user", "content": "first"}]},
    )
    owner_id = manager.active_account_id
    other_id = "a2" if owner_id == "a1" else "a1"
    assert first.status_code == 200
    assert owner_id == "a1"

    assert asyncio.run(client.switch_to_account(other_id)) is True
    calls_before = calls
    known = test_client.post(
        "/v1/chat/completions",
        json={
            "model": "doubao",
            "conversation_id": "conversation-a",
            "messages": [{"role": "user", "content": "continue"}],
        },
    )
    assert known.status_code == 409
    assert calls == calls_before

    legacy = test_client.post(
        "/v1/chat/completions",
        json={
            "model": "doubao",
            "conversation_id": "external-id",
            "messages": [{"role": "user", "content": "legacy"}],
        },
    )
    assert legacy.status_code == 200
    assert manager.active_account_id == other_id


def test_chat_raw_tos_reference_is_rejected_before_round_robin_switch(tmp_path):
    client, manager, _ = _client_with_accounts(tmp_path, strategy="round_robin")
    client.chat_completion = AsyncMock()
    response = TestClient(create_app(browser_client=client)).post(
        "/v1/chat/completions",
        json={
            "model": "doubao",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "use this"},
                    {"type": "image_url", "image_url": {"url": "tos-owner-a1/image.png"}},
                ],
            }],
        },
    )

    assert response.status_code == 409
    assert manager.active_account_id == "a1"
    client.switch_to_account.assert_not_awaited()
    client.chat_completion.assert_not_awaited()


def test_uploaded_file_reference_owner_is_checked_on_later_chat(tmp_path):
    client, manager, _ = _client_with_accounts(tmp_path, strategy="failover")
    client.upload_file = AsyncMock(return_value={"uri": "tos-a1-file", "name": "n.txt", "size": 1})
    client.chat_completion = AsyncMock()
    test_client = TestClient(create_app(browser_client=client))

    uploaded = test_client.post("/v1/files", files={"file": ("n.txt", b"x", "text/plain")})
    assert uploaded.status_code == 200
    manager.set_active_account("a2")
    response = test_client.post(
        "/v1/chat/completions",
        json={
            "model": "doubao",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "summarize"},
                    {"type": "file_url", "file_url": {"url": "tos-a1-file", "name": "n.txt"}},
                ],
            }],
        },
    )

    assert response.status_code == 409
    client.chat_completion.assert_not_called()


@pytest.mark.parametrize("quota_source", ["event", "full_text"])
@pytest.mark.parametrize("history_kwargs", [{"memory_type": 0}, {"save_history": True}])
def test_video_quota_failover_reuploads_all_sources_and_preserves_options(tmp_path, history_kwargs, quota_source):
    client, manager, _ = _client_with_accounts(tmp_path, strategy="failover", count=3)
    uri_counts = {account_id: 0 for account_id in manager.accounts}
    upload_calls = []
    generation_calls = []

    async def mock_upload_image(image_bytes, filename):
        account_id = manager.active_account_id
        uri_counts[account_id] += 1
        uri = f"tos-{account_id}-image-{uri_counts[account_id]}"
        upload_calls.append((account_id, image_bytes, filename, uri))
        return {
            "uri": uri,
            "cdn_url": f"https://image.test/{uri}.png",
            "name": filename,
            "format": "png",
            "width": 640,
            "height": 480,
        }

    async def mock_generate_video(**kwargs):
        account_id = manager.active_account_id
        generation_calls.append((account_id, dict(kwargs)))
        if account_id in ("a1", "a2"):
            manager.mark_quota_exceeded(account_id, "video")
            raise VideoQuotaExceeded(f"{quota_source} quota test")
        return {"videos": [{"video_url": "https://video.test/final.mp4"}]}

    async def mock_chat(*args, **kwargs):
        yield {"_event": "CHUNK_DELTA", "text": "unused"}

    client.upload_image = AsyncMock(side_effect=mock_upload_image)
    client.generate_video = AsyncMock(side_effect=mock_generate_video)
    client.chat_completion = mock_chat
    client.extract_conversation_id = MagicMock(return_value=None)
    test_client = TestClient(create_app(browser_client=client))
    payload = {
        "prompt": "a scene",
        "ratio": "16:9",
        "images": [
            "data:image/png;base64,YQ==",
            "data:image/png;base64,Yg==",
        ],
        **history_kwargs,
    }

    response = test_client.post("/v1/video/generations", json=payload)

    assert response.status_code == 200, response.text
    assert [account_id for account_id, _ in generation_calls] == ["a1", "a2", "a3"]
    assert len(upload_calls) == 6
    assert [entry[0] for entry in upload_calls] == ["a1", "a1", "a2", "a2", "a3", "a3"]
    expected_history = history_kwargs
    for account_id, kwargs in generation_calls:
        assert kwargs["prompt"] == "a scene"
        assert kwargs["ratio"] == "16:9"
        assert kwargs["ref_image_keys"] == [f"tos-{account_id}-image-1", f"tos-{account_id}-image-2"]
        assert kwargs["ref_image_key"] == f"tos-{account_id}-image-1"
        assert [item["uri"] for item in kwargs["image_attachments"]] == kwargs["ref_image_keys"]
        assert all(item["width"] == 640 and item["height"] == 480 for item in kwargs["image_attachments"])
        assert {key: kwargs[key] for key in expected_history} == expected_history


def test_multipart_image_bytes_are_retained_after_upload_consumes_stream(tmp_path):
    client, manager, _ = _client_with_accounts(tmp_path, strategy="failover", count=2)
    uploads = []
    generated = []

    async def upload_image(image_bytes, filename):
        account_id = manager.active_account_id
        uri = f"tos-{account_id}-frame"
        uploads.append((account_id, image_bytes, filename, uri))
        return {
            "uri": uri,
            "cdn_url": f"https://image.test/{uri}.png",
            "name": filename,
            "format": "png",
            "width": 320,
            "height": 240,
        }

    async def generate_video(**kwargs):
        account_id = manager.active_account_id
        generated.append((account_id, dict(kwargs)))
        if account_id == "a1":
            manager.mark_quota_exceeded(account_id, "video")
            raise VideoQuotaExceeded("multipart quota")
        return {"videos": [{"video_url": "https://video.test/multipart.mp4"}]}

    client.upload_image = AsyncMock(side_effect=upload_image)
    client.generate_video = AsyncMock(side_effect=generate_video)
    response = TestClient(create_app(browser_client=client)).post(
        "/v1/video/generations",
        data={"prompt": "multipart", "ratio": "16:9", "save_history": "true"},
        files={"images": ("frame.png", b"frame-bytes", "image/png")},
    )

    assert response.status_code == 200, response.text
    assert [item[0] for item in uploads] == ["a1", "a2"]
    assert [item[1] for item in uploads] == [b"frame-bytes", b"frame-bytes"]
    assert generated[0][1]["save_history"] is True
    assert generated[1][1]["save_history"] is True
    assert generated[1][1]["ref_image_key"] == "tos-a2-frame"


def test_video_quota_switch_failure_skips_failed_candidate(tmp_path):
    client, manager, _ = _client_with_accounts(tmp_path, strategy="failover", count=3)
    generation_accounts = []
    switches = []
    uploads = 0

    async def switch(account_id):
        switches.append(account_id)
        if account_id == "a2":
            return False
        manager.set_active_account(account_id)
        return True

    async def upload(image_bytes, filename):
        nonlocal uploads
        uploads += 1
        uri = f"tos-{manager.active_account_id}-{uploads}"
        return {
            "uri": uri,
            "cdn_url": f"https://image.test/{uri}.png",
            "name": filename,
            "format": "png",
            "width": 640,
            "height": 480,
        }

    async def generate(**kwargs):
        account_id = manager.active_account_id
        generation_accounts.append(account_id)
        if account_id == "a1":
            manager.mark_quota_exceeded(account_id, "video")
            raise VideoQuotaExceeded("quota")
        return {"videos": [{"video_url": "https://video.test/final.mp4"}]}

    client.switch_to_account = AsyncMock(side_effect=switch)
    client.upload_image = AsyncMock(side_effect=upload)
    client.generate_video = AsyncMock(side_effect=generate)
    response = TestClient(create_app(browser_client=client)).post(
        "/v1/video/generations",
        json={
            "prompt": "quota fallback",
            "ratio": "9:16",
            "images": ["data:image/png;base64,YQ=="],
            "save_history": True,
        },
    )

    assert response.status_code == 200, response.text
    assert generation_accounts == ["a1", "a3"]
    assert switches == ["a2", "a3"]
    assert client.generate_video.await_args_list[0].kwargs["save_history"] is True
    assert client.generate_video.await_args_list[1].kwargs["save_history"] is True
    assert client.generate_video.await_args_list[1].kwargs["ratio"] == "9:16"


def test_video_quota_exhaustion_attempts_each_account_once(tmp_path):
    client, manager, _ = _client_with_accounts(tmp_path, strategy="failover", count=2)
    generation_accounts = []

    async def generate(**kwargs):
        account_id = manager.active_account_id
        generation_accounts.append(account_id)
        manager.mark_quota_exceeded(account_id, "video")
        raise VideoQuotaExceeded("quota")

    client.generate_video = AsyncMock(side_effect=generate)
    response = TestClient(create_app(browser_client=client)).post(
        "/v1/video/generations", json={"prompt": "exhaust all"}
    )

    assert response.status_code == 429
    assert generation_accounts == ["a1", "a2"]
    assert client.generate_video.await_count == 2


def test_video_empty_upstream_result_is_gateway_error_without_echoing_message(tmp_path):
    client, _, _ = _client_with_accounts(tmp_path, strategy="failover", count=1)
    client.generate_video = AsyncMock(
        return_value={"videos": [], "message": "provider-internal-detail"}
    )

    response = TestClient(create_app(browser_client=client)).post(
        "/v1/video/generations", json={"prompt": "single bounded attempt"}
    )

    assert response.status_code == 502
    assert response.json()["error"]["message"] == "Video generation failed (upstream_no_result)"
    assert "provider-internal-detail" not in response.text
    client.generate_video.assert_awaited_once()


@pytest.mark.parametrize("quota_branch", ["event", "full_text"])
def test_browser_video_quota_payloads_raise_without_internal_failover(tmp_path, quota_branch):
    async def run():
        client, manager, accounts = _client_with_accounts(tmp_path, strategy="failover", count=2)
        quota_message = "今日生成次数已达上限"
        if quota_branch == "event":
            async def mock_chat(*args, **kwargs):
                yield {
                    "error": True,
                    "status": 429,
                    "body": json.dumps({"code": 429, "error_msg": quota_message}, ensure_ascii=False),
                }
        else:
            async def mock_chat(*args, **kwargs):
                yield {"_event": "CHUNK_DELTA", "text": quota_message}

        client.chat_completion = mock_chat
        client.switch_to_account = AsyncMock(return_value=True)

        with pytest.raises(VideoQuotaExceeded):
            await client.generate_video(prompt="no real call")

        assert manager.get_account(accounts[0].id).video_quota_exceeded is True
        assert manager.get_account(accounts[0].id).status == "active"
        client.switch_to_account.assert_not_awaited()

    asyncio.run(run())


def test_initial_round_robin_rejects_raw_tos_reference_before_switch(tmp_path):
    client, manager, _ = _client_with_accounts(tmp_path, strategy="round_robin", count=2)
    client.generate_video = AsyncMock(return_value={"videos": [{"video_url": "unused"}]})
    response = TestClient(create_app(browser_client=client)).post(
        "/v1/video/generations",
        json={"prompt": "raw ref", "ref_image_key": "tos-owner-a1/image.png"},
    )

    assert response.status_code == 409
    assert manager.active_account_id == "a1"
    client.switch_to_account.assert_not_awaited()
    client.generate_video.assert_not_awaited()


def test_video_quota_raw_tos_reference_returns_explicit_conflict(tmp_path):
    client, manager, _ = _client_with_accounts(tmp_path, strategy="failover", count=2)

    async def generate(**kwargs):
        manager.mark_quota_exceeded(manager.active_account_id, "video")
        raise VideoQuotaExceeded("quota")

    client.generate_video = AsyncMock(side_effect=generate)
    response = TestClient(create_app(browser_client=client)).post(
        "/v1/video/generations",
        json={"prompt": "raw ref", "ref_image_key": "tos-owner-a1/image.png"},
    )

    assert response.status_code == 409
    assert "metadata is unavailable" in response.json()["error"]["message"]
    client.switch_to_account.assert_not_awaited()


def test_chat_recovery_reuploads_retained_image_and_file_bytes(tmp_path):
    client, manager, _ = _client_with_accounts(tmp_path, strategy="failover", count=2)
    image_uploads = []
    file_uploads = []
    attempts = 0

    async def upload_image(image_bytes, filename):
        uri = f"tos-{manager.active_account_id}-image"
        image_uploads.append((manager.active_account_id, image_bytes, filename, uri))
        return {"uri": uri, "cdn_url": f"https://cdn.test/{uri}", "name": filename}

    async def upload_file(file_data, filename):
        uri = f"tos-{manager.active_account_id}-file"
        file_uploads.append((manager.active_account_id, file_data, filename, uri))
        return {"uri": uri, "name": filename, "size": len(file_data)}

    async def switch(account_id):
        manager.set_active_account(account_id)
        client._needs_captcha = False
        return True

    async def chat(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            yield {
                "error_code": 710022004,
                "error_msg": "rate limited",
                "extra": {"decision": {"type": "verify", "subtype": "semantic_reasoning"}},
            }
        else:
            assert kwargs["image_attachments"][0]["uri"] == "tos-a2-image"
            assert kwargs["file_attachments"][0]["uri"] == "tos-a2-file"
            yield {"_event": "CHUNK_DELTA", "text": "recovered"}

    client.upload_image = AsyncMock(side_effect=upload_image)
    client.upload_file = AsyncMock(side_effect=upload_file)
    client.switch_to_account = AsyncMock(side_effect=switch)
    client.is_captcha_visible = AsyncMock(return_value=False)
    client.chat_completion = chat
    client.extract_conversation_id = MagicMock(return_value="conversation-a2")
    payload = {
        "model": "doubao",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "read these"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
                {"type": "file_url", "file_url": {"url": "data:text/plain;base64,SGVsbG8=", "name": "note.txt"}},
            ],
        }],
    }
    response = TestClient(create_app(browser_client=client)).post("/v1/chat/completions", json=payload)

    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "recovered"
    assert attempts == 2
    assert [call[0] for call in image_uploads] == ["a1", "a2"]
    assert [call[0] for call in file_uploads] == ["a1", "a2"]
    assert [call[1] for call in image_uploads] == [b"hello", b"hello"]
    assert [call[1] for call in file_uploads] == [b"Hello", b"Hello"]
    assert manager.active_account_id == "a2"


def test_switch_failure_restores_prior_account_and_manager_identity(tmp_path):
    async def run():
        client, manager, accounts = _client_with_accounts(tmp_path, strategy="failover", count=2)
        context = MagicMock()
        context.clear_cookies = AsyncMock(return_value=None)
        client._context = context
        client._page = MagicMock()
        client.is_alive = AsyncMock(return_value=True)
        client.switch_to_account = BrowserClient.switch_to_account.__get__(client, BrowserClient)
        client.inject_cookies_and_reload = AsyncMock(side_effect=[RuntimeError("target failed"), True])

        result = await client.switch_to_account(accounts[1].id)

        assert result is False
        assert manager.active_account_id == accounts[0].id
        assert client.is_ready is True
        assert client.inject_cookies_and_reload.await_args_list[0].args[0] == accounts[1].cookies
        assert client.inject_cookies_and_reload.await_args_list[1].args[0] == accounts[0].cookies
        assert context.clear_cookies.await_count == 2

    asyncio.run(run())


def test_switch_clear_exception_marks_browser_not_ready_without_relabeling_account(tmp_path):
    async def run():
        client, manager, accounts = _client_with_accounts(tmp_path, strategy="failover", count=2)
        context = MagicMock()
        context.clear_cookies = AsyncMock(side_effect=RuntimeError("clear failed"))
        client._context = context
        client._page = MagicMock()
        client.is_alive = AsyncMock(return_value=True)
        client.switch_to_account = BrowserClient.switch_to_account.__get__(client, BrowserClient)
        client.inject_cookies_and_reload = AsyncMock()

        result = await client.switch_to_account(accounts[1].id)

        assert result is False
        assert manager.active_account_id == accounts[0].id
        assert client.is_ready is False
        client.inject_cookies_and_reload.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.parametrize("cancel_at", ["clear", "inject"])
def test_cancelled_switch_leaves_not_ready_and_does_not_spawn_restore(tmp_path, cancel_at):
    async def run():
        client, manager, accounts = _client_with_accounts(tmp_path, strategy="failover", count=2)
        context = MagicMock()
        client._context = context
        client._page = MagicMock()
        client.is_alive = AsyncMock(return_value=True)
        client.switch_to_account = BrowserClient.switch_to_account.__get__(client, BrowserClient)
        if cancel_at == "clear":
            context.clear_cookies = AsyncMock(side_effect=asyncio.CancelledError())
            client.inject_cookies_and_reload = AsyncMock()
        else:
            context.clear_cookies = AsyncMock(return_value=None)
            client.inject_cookies_and_reload = AsyncMock(side_effect=asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await client.switch_to_account(accounts[1].id)

        assert manager.active_account_id == accounts[0].id
        assert client.is_ready is False
        assert client.inject_cookies_and_reload.await_count == (0 if cancel_at == "clear" else 1)
        assert context.clear_cookies.await_count == 1

    asyncio.run(run())


def test_stale_contextvar_lease_cannot_bypass_a_new_owner(tmp_path):
    async def run():
        gate = BrowserOperationGate()
        allow_stale_child = asyncio.Event()
        stale_child_entered = asyncio.Event()
        second_owner_entered = asyncio.Event()

        async def stale_child():
            await allow_stale_child.wait()
            async with gate.lease():
                stale_child_entered.set()

        async with gate.lease():
            child_task = asyncio.create_task(stale_child())

        async with gate.lease():
            second_owner_entered.set()
            allow_stale_child.set()
            await asyncio.sleep(0.02)
            assert not stale_child_entered.is_set()

        await asyncio.wait_for(child_task, timeout=1)
        assert second_owner_entered.is_set()
        assert stale_child_entered.is_set()

    asyncio.run(run())


@pytest.mark.parametrize("disconnect", [False, True])
def test_asgi_stream_holds_browser_gate_until_finish_or_cancel(tmp_path, disconnect):
    async def run():
        client, manager, accounts = _client_with_accounts(tmp_path, strategy="failover", count=2)
        upstream_started = asyncio.Event()
        release_upstream = asyncio.Event()
        upstream_cancelled = asyncio.Event()

        async def mock_chat(*args, **kwargs):
            upstream_started.set()
            try:
                await release_upstream.wait()
                if not disconnect:
                    yield {"_event": "CHUNK_DELTA", "text": "stream done"}
            finally:
                upstream_cancelled.set()

        async def switch_to_account(account_id):
            manager.set_active_account(account_id)
            return True

        client.chat_completion = mock_chat
        client.extract_conversation_id = MagicMock(return_value=None)
        client.switch_to_account = AsyncMock(side_effect=switch_to_account)
        client.switch_mode = AsyncMock(return_value=True)
        app = create_app(browser_client=client)
        disconnect_event = asyncio.Event()
        stream_task = asyncio.create_task(_asgi_request(
            app,
            "/v1/chat/completions",
            {"model": "doubao", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
            disconnect_event=disconnect_event,
        ))
        await asyncio.wait_for(upstream_started.wait(), timeout=1)

        switch_task = asyncio.create_task(_asgi_request(
            app, "/admin/api/accounts/switch", {"account_id": accounts[1].id}
        ))
        window_task = asyncio.create_task(_asgi_request(
            app, "/admin/api/browser/force-window", {}
        ))
        await asyncio.sleep(0.05)
        assert not client.switch_to_account.await_count
        assert not client.switch_mode.await_count
        assert not switch_task.done()
        assert not window_task.done()

        if disconnect:
            disconnect_event.set()
        else:
            release_upstream.set()
        stream_messages = await asyncio.wait_for(stream_task, timeout=2)
        await asyncio.wait_for(asyncio.gather(switch_task, window_task), timeout=2)

        assert any(message.get("type") == "http.response.start" for message in stream_messages)
        assert client.switch_to_account.await_count == 1
        assert client.switch_mode.await_count == 1
        assert upstream_cancelled.is_set()

    asyncio.run(run())


def test_background_login_wait_does_not_hold_browser_gate(tmp_path):
    client, _, _ = _client_with_accounts(tmp_path, strategy="failover", count=1)
    client._ready = False
    login_started = threading.Event()
    finish_login = threading.Event()

    async def wait_for_login(timeout=180):
        async with client.operation_gate.lease():
            login_started.set()
        while not finish_login.is_set():
            await asyncio.sleep(0.01)
        return False

    client.wait_for_login = wait_for_login
    client.switch_mode = AsyncMock(return_value=True)
    client.headless = False
    page = MagicMock()
    page.is_closed.return_value = False
    page.bring_to_front = AsyncMock()
    client._page = page
    app = create_app(browser_client=client)

    with TestClient(app) as test_client:
        response = test_client.post("/auth/login")
        assert response.status_code == 200
        assert login_started.wait(1)
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending_window = executor.submit(test_client.post, "/admin/api/browser/force-window")
            timely = False
            try:
                window = pending_window.result(timeout=1)
                timely = True
            except TimeoutError:
                pass
            finally:
                finish_login.set()
            if not timely:
                pending_window.result(timeout=1)
            assert timely
            assert window.status_code == 200
            client.switch_mode.assert_awaited_once_with(headless=False)


def test_delayed_delete_skips_after_owner_account_changes(tmp_path):
    client, manager, accounts = _client_with_accounts(tmp_path, strategy="failover", count=2)
    client.extract_conversation_id = MagicMock(return_value="conversation-a1")
    client.delete_conversation = AsyncMock(return_value=True)
    delayed_cleanup_finished = threading.Event()
    original_fresh_lease = client.operation_gate.fresh_lease

    @asynccontextmanager
    async def tracked_fresh_lease():
        async with original_fresh_lease():
            yield
        delayed_cleanup_finished.set()

    client.operation_gate.fresh_lease = tracked_fresh_lease

    async def mock_chat(*args, **kwargs):
        yield {"_event": "CHUNK_DELTA", "text": "ephemeral"}

    async def switch_to_account(account_id):
        manager.set_active_account(account_id)
        return True

    client.chat_completion = mock_chat
    client.switch_to_account = AsyncMock(side_effect=switch_to_account)
    app = create_app(browser_client=client)
    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/chat/completions",
            headers={"x-auto-delete": "true"},
            json={"model": "doubao", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 200
        assert manager.active_account_id == accounts[0].id

        switched = test_client.post(
            "/admin/api/accounts/switch", json={"account_id": accounts[1].id}
        )
        assert switched.status_code == 200
        assert delayed_cleanup_finished.wait(2)

    client.delete_conversation.assert_not_awaited()
