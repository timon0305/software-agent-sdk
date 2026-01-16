"""Minimal sync helpers on top of fastmcp.Client, preserving original behavior."""

import asyncio
import inspect
from collections.abc import Callable
from typing import Any

from fastmcp import Client as AsyncMCPClient

from openhands.sdk.logger import get_logger
from openhands.sdk.utils.async_executor import AsyncExecutor


logger = get_logger(__name__)


class MCPClient(AsyncMCPClient):
    """
    Behaves exactly like fastmcp.Client (same constructor & async API),
    but owns a background event loop and offers:
      - call_async_from_sync(awaitable_or_fn, *args, timeout=None, **kwargs)
      - call_sync_from_async(fn, *args, **kwargs)  # await this from async code

    Additionally tracks session state for persistence:
      - _session_id: The current MCP session ID (set after connection)
      - _server_url: The server URL for session tracking
      - _connection_count: Number of active context manager entries (for reentrant use)
    """

    _executor: AsyncExecutor
    _session_id: str | None
    _server_url: str | None
    _connection_count: int

    def __init__(self, *args, **kwargs):
        # Extract server_url if provided for session tracking
        self._server_url = kwargs.pop("server_url", None)
        super().__init__(*args, **kwargs)
        self._executor = AsyncExecutor()
        self._session_id = None
        self._connection_count = 0

    def call_async_from_sync(
        self,
        awaitable_or_fn: Callable[..., Any] | Any,
        *args,
        timeout: float,
        **kwargs,
    ) -> Any:
        """
        Run a coroutine or async function on this client's loop from sync code.

        Usage:
            mcp.call_async_from_sync(async_fn, arg1, kw=...)
            mcp.call_async_from_sync(coro)
        """
        return self._executor.run_async(
            awaitable_or_fn, *args, timeout=timeout, **kwargs
        )

    async def call_sync_from_async(
        self, fn: Callable[..., Any], *args, **kwargs
    ) -> Any:
        """
        Await running a blocking function in the default threadpool from async code.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    @property
    def session_id(self) -> str | None:
        """Get the current session ID."""
        return self._session_id

    @property
    def server_url(self) -> str | None:
        """Get the server URL for this client."""
        return self._server_url

    def set_session_id(self, session_id: str | None) -> None:
        """Set the session ID (called after connection established)."""
        self._session_id = session_id
        if session_id:
            logger.debug(f"Session ID set: {session_id[:8]}...")

    async def __aenter__(self):
        """Enter the async context manager with connection reuse support.

        Uses reference counting to support reentrant context managers.
        Only establishes a new connection on the first entry.
        """
        self._connection_count += 1
        if self._connection_count == 1:
            # First entry - establish connection
            result = await super().__aenter__()
            # Try to capture session ID from transport after connection
            self._capture_session_id()
            return result
        else:
            # Reentrant entry - connection already established
            return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context manager with connection reuse support.

        Only closes the connection when the last context exits.
        """
        self._connection_count -= 1
        if self._connection_count == 0:
            # Last exit - close connection
            return await super().__aexit__(exc_type, exc_val, exc_tb)
        # Not the last exit - keep connection open
        return None

    def _capture_session_id(self) -> None:
        """Try to capture session ID from the underlying transport."""
        try:
            # The fastmcp Client may have a transport with session ID
            transport = getattr(self, "_transport", None)
            if transport is not None:
                get_session_id_fn = getattr(transport, "get_session_id", None)
                if get_session_id_fn is not None:
                    session_id = get_session_id_fn()
                    if session_id:
                        self.set_session_id(session_id)
        except Exception as e:
            logger.debug(f"Could not capture session ID: {e}")

    def sync_close(self) -> None:
        """
        Synchronously close the MCP client and cleanup resources.

        This will attempt to call the async close() method if available,
        then shutdown the background event loop.
        """
        # Reset connection count
        self._connection_count = 0

        # Best-effort: try async close if parent provides it
        if hasattr(self, "close") and inspect.iscoroutinefunction(self.close):
            try:
                self._executor.run_async(self.close, timeout=10.0)
            except Exception:
                pass  # Ignore close errors during cleanup

        # Always cleanup the executor
        self._executor.close()

        logger.debug(f"MCP client closed for {self._server_url}")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.sync_close()
        except Exception:
            pass  # Ignore cleanup errors during deletion
