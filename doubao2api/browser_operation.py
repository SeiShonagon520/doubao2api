"""A request-scoped gate for the single Playwright browser session."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar, Token, copy_context
from typing import Any, AsyncIterator, Optional


_CURRENT_LEASE: ContextVar[Optional[tuple["BrowserOperationGate", object]]] = ContextVar(
    "doubao_browser_operation_lease", default=None
)


class BrowserOperationGate:
    """Serialize browser operations while allowing valid nested request work."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._active_token: Optional[object] = None

    def _reset_context(self, token: Token) -> None:
        try:
            _CURRENT_LEASE.reset(token)
        except ValueError:
            # Async generators may be closed from a different task than the one
            # that first advanced them. The active-token check still invalidates
            # the original inherited lease immediately.
            _CURRENT_LEASE.set(None)

    def spawn_detached(self, coroutine: Any) -> asyncio.Task:
        """Create background browser work without inheriting an HTTP lease token."""
        context = copy_context()
        context.run(_CURRENT_LEASE.set, None)
        return context.run(asyncio.create_task, coroutine)

    @asynccontextmanager
    async def lease(self) -> AsyncIterator[None]:
        inherited = _CURRENT_LEASE.get()
        if inherited is not None and inherited[0] is self and inherited[1] is self._active_token:
            yield
            return

        await self._lock.acquire()
        active_token = object()
        self._active_token = active_token
        context_token = _CURRENT_LEASE.set((self, active_token))
        try:
            yield
        finally:
            self._active_token = None
            self._reset_context(context_token)
            self._lock.release()

    @asynccontextmanager
    async def fresh_lease(self) -> AsyncIterator[None]:
        inherited_token = _CURRENT_LEASE.set(None)
        try:
            async with self.lease():
                yield
        finally:
            self._reset_context(inherited_token)


class BrowserOperationMiddleware:
    """Pure ASGI gate that stays held through the final streamed response frame."""

    def __init__(self, app: Any, *, get_client: Any, paths: set[str]) -> None:
        self.app = app
        self.get_client = get_client
        self.paths = paths

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("path") not in self.paths:
            await self.app(scope, receive, send)
            return

        client = self.get_client()
        gate = getattr(client, "operation_gate", None) if client is not None else None
        if not isinstance(gate, BrowserOperationGate):
            await self.app(scope, receive, send)
            return

        async with gate.lease():
            await self.app(scope, receive, send)
