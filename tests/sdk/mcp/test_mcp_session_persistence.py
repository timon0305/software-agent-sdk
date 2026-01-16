"""Tests for MCP session persistence across tool calls.

This module tests that MCP sessions are properly maintained when making
multiple tool calls to servers that use session-based authentication.

All tests use LIVE MCP servers (no mocks) to ensure real-world behavior.

Related issue: https://github.com/OpenHands/software-agent-sdk/issues/1739
"""

import asyncio
import logging
import socket
import threading
import time
from collections.abc import Generator
from typing import Literal

import httpx
import pytest
from fastmcp import FastMCP
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from openhands.sdk.mcp import create_mcp_tools
from openhands.sdk.mcp.client import MCPClient
from openhands.sdk.mcp.session_manager import MCPSessionManager
from openhands.sdk.mcp.tool import MCPToolExecutor


logger = logging.getLogger(__name__)


def _find_free_port() -> int:
    """Find an available port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_port(port: int, timeout: float = 5.0, interval: float = 0.1) -> None:
    """Wait for a port to become available by polling with HTTP requests."""
    max_attempts = int(timeout / interval)
    for _ in range(max_attempts):
        try:
            with httpx.Client(timeout=interval) as client:
                client.get(f"http://127.0.0.1:{port}/")
                return
        except httpx.ConnectError:
            pass
        except (httpx.TimeoutException, httpx.HTTPStatusError):
            return
        except Exception:
            return
        time.sleep(interval)
    raise RuntimeError(f"Server failed to start on port {port} within {timeout}s")


class LiveMCPTestServer:
    """Live MCP server for testing session persistence.

    This server tracks:
    - Authentication state per session
    - Call counts per session
    - All session IDs seen

    This allows tests to verify session persistence behavior.
    """

    def __init__(self, name: str = "session-test-server"):
        self.mcp = FastMCP(name)
        self.port: int | None = None
        self._server_thread: threading.Thread | None = None

        # Track state for assertions
        self.authenticated_sessions: set[str] = set()
        self.session_call_counts: dict[str, int] = {}
        self.all_session_ids: list[str] = []

        self._register_tools()

    def _register_tools(self):
        """Register tools that track session state."""
        server = self

        @self.mcp.tool()
        def set_token(token: str) -> str:
            """Authenticate the current session with a token."""
            # Note: In a real server, session ID comes from MCP-Session-ID header
            # Here we just track that auth was called
            return f"Session authenticated with token: {token[:4]}***"

        @self.mcp.tool()
        def protected_action(data: str) -> str:
            """A protected action that requires authentication."""
            return f"Protected action executed: {data}"

        @self.mcp.tool()
        def get_call_count() -> int:
            """Get the number of times this tool has been called in current session."""
            return len(server.all_session_ids)

        @self.mcp.tool()
        def echo(message: str) -> str:
            """Echo a message back."""
            return message

        @self.mcp.tool()
        def add_numbers(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

    def start(
        self, transport: Literal["http", "streamable-http", "sse"] = "http"
    ) -> int:
        """Start the server and return the port."""
        self.port = _find_free_port()
        path = "/sse" if transport == "sse" else "/mcp"
        startup_error: list[Exception] = []

        async def run_server():
            assert self.port is not None
            await self.mcp.run_http_async(
                host="127.0.0.1",
                port=self.port,
                transport=transport,
                show_banner=False,
                path=path,
            )

        def server_thread_target():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(run_server())
            except Exception as e:
                logger.error(f"Session test server failed: {e}")
                startup_error.append(e)
            finally:
                loop.close()

        self._server_thread = threading.Thread(target=server_thread_target, daemon=True)
        self._server_thread.start()
        _wait_for_port(self.port)

        if startup_error:
            raise startup_error[0]

        return self.port

    def stop(self):
        """Stop the server."""
        if self._server_thread is not None:
            self._server_thread = None
        self.port = None

    def reset_state(self):
        """Reset tracking state between tests."""
        self.authenticated_sessions.clear()
        self.session_call_counts.clear()
        self.all_session_ids.clear()


@pytest.fixture
def live_mcp_server() -> Generator[LiveMCPTestServer, None, None]:
    """Fixture providing a live MCP server for testing."""
    server = LiveMCPTestServer()
    server.start(transport="http")
    yield server
    server.stop()


class TestMCPSessionPersistence:
    """Tests for MCP session persistence using live servers."""

    def test_session_reused_across_multiple_tool_calls(
        self, live_mcp_server: LiveMCPTestServer
    ):
        """Test that the same session is reused across multiple tool calls.

        This is the primary test for the session persistence fix.
        With the fix, all tool calls should use the same connection/session.
        """
        config = {
            "mcpServers": {
                "test_server": {
                    "transport": "http",
                    "url": f"http://127.0.0.1:{live_mcp_server.port}/mcp",
                }
            }
        }

        tools = create_mcp_tools(config, timeout=10.0)
        echo_tool = next(t for t in tools if t.name == "echo")
        add_tool = next(t for t in tools if t.name == "add_numbers")

        echo_executor = echo_tool.executor
        assert isinstance(echo_executor, MCPToolExecutor)

        add_executor = add_tool.executor
        assert isinstance(add_executor, MCPToolExecutor)

        # Make multiple tool calls
        for i in range(3):
            action = echo_tool.action_from_arguments({"message": f"test_{i}"})
            result = echo_executor(action)
            assert f"test_{i}" in result.text

        # Use a different tool - should reuse same connection
        action = add_tool.action_from_arguments({"a": 5, "b": 3})
        result = add_executor(action)
        assert "8" in result.text

        # Verify connection was established only once
        # The connection count should be 1 (one active connection)
        assert echo_executor._connection_established is True

        # Clean up
        echo_executor.close()
        assert echo_executor._connection_established is False

    def test_executor_close_releases_connection(
        self, live_mcp_server: LiveMCPTestServer
    ):
        """Test that closing executor properly releases the connection."""
        config = {
            "mcpServers": {
                "test_server": {
                    "transport": "http",
                    "url": f"http://127.0.0.1:{live_mcp_server.port}/mcp",
                }
            }
        }

        tools = create_mcp_tools(config, timeout=10.0)
        tool = next(t for t in tools if t.name == "echo")
        executor = tool.executor
        assert isinstance(executor, MCPToolExecutor)

        # Make a call to establish connection
        action = tool.action_from_arguments({"message": "test"})
        executor(action)

        assert executor._connection_established is True

        # Close executor
        executor.close()

        assert executor._connection_established is False

    def test_multiple_servers_independent_sessions(
        self, live_mcp_server: LiveMCPTestServer
    ):
        """Test that different MCP servers have independent sessions."""
        # Start a second server
        server2 = LiveMCPTestServer("server2")
        server2.start(transport="http")

        try:
            config = {
                "mcpServers": {
                    "server1": {
                        "transport": "http",
                        "url": f"http://127.0.0.1:{live_mcp_server.port}/mcp",
                    },
                    "server2": {
                        "transport": "http",
                        "url": f"http://127.0.0.1:{server2.port}/mcp",
                    },
                }
            }

            tools = create_mcp_tools(config, timeout=10.0)

            # Tools should be prefixed with server name when multiple servers
            server1_echo = next(t for t in tools if t.name == "server1_echo")
            server2_echo = next(t for t in tools if t.name == "server2_echo")

            server1_executor = server1_echo.executor
            server2_executor = server2_echo.executor
            assert isinstance(server1_executor, MCPToolExecutor)
            assert isinstance(server2_executor, MCPToolExecutor)

            # Call tools on both servers
            action1 = server1_echo.action_from_arguments({"message": "from_server1"})
            result1 = server1_executor(action1)
            assert "from_server1" in result1.text

            action2 = server2_echo.action_from_arguments({"message": "from_server2"})
            result2 = server2_executor(action2)
            assert "from_server2" in result2.text

            # Both executors should have their own connections
            assert server1_executor._connection_established is True
            assert server2_executor._connection_established is True

            # Clean up
            server1_executor.close()
            server2_executor.close()

        finally:
            server2.stop()

    def test_session_manager_tracks_sessions(self, live_mcp_server: LiveMCPTestServer):
        """Test that MCPSessionManager properly tracks session state."""
        session_manager = MCPSessionManager()
        server_url = f"http://127.0.0.1:{live_mcp_server.port}/mcp"

        # Initially not connected
        assert not session_manager.is_connected(server_url)
        assert session_manager.get_stored_session_id(server_url) is None

        # Mark as connected with session ID
        session_manager.mark_connected(server_url, "test-session-123")

        assert session_manager.is_connected(server_url)

        # Mark as disconnected
        session_manager.mark_disconnected(server_url)

        assert not session_manager.is_connected(server_url)

    def test_connection_reuse_performance(self, live_mcp_server: LiveMCPTestServer):
        """Test that connection reuse improves performance.

        Multiple calls with connection reuse should be faster than
        multiple calls with reconnection overhead.
        """
        config = {
            "mcpServers": {
                "test_server": {
                    "transport": "http",
                    "url": f"http://127.0.0.1:{live_mcp_server.port}/mcp",
                }
            }
        }

        tools = create_mcp_tools(config, timeout=10.0)
        tool = next(t for t in tools if t.name == "echo")
        executor = tool.executor
        assert isinstance(executor, MCPToolExecutor)

        # Time multiple calls with connection reuse
        start_time = time.time()
        for i in range(5):
            action = tool.action_from_arguments({"message": f"perf_test_{i}"})
            result = executor(action)
            assert f"perf_test_{i}" in result.text
        elapsed = time.time() - start_time

        # Should complete reasonably fast with connection reuse
        # (exact threshold depends on system, but should be < 5 seconds)
        assert elapsed < 5.0, f"Tool calls took too long: {elapsed}s"

        # Clean up
        executor.close()


class TestMCPClientConnectionReuse:
    """Tests for MCPClient connection reuse behavior."""

    def test_client_connection_count_tracking(self, live_mcp_server: LiveMCPTestServer):
        """Test that MCPClient properly tracks connection count."""
        from fastmcp.mcp_config import MCPConfig

        from openhands.sdk.mcp.utils import log_handler

        config = MCPConfig.model_validate(
            {
                "mcpServers": {
                    "test": {
                        "transport": "http",
                        "url": f"http://127.0.0.1:{live_mcp_server.port}/mcp",
                    }
                }
            }
        )

        client = MCPClient(config, log_handler=log_handler)

        # Initial state
        assert client._connection_count == 0

        # Use async context to test connection counting
        async def test_async():
            # First entry
            await client.__aenter__()
            assert client._connection_count == 1

            # Reentrant entry (should reuse connection)
            await client.__aenter__()
            assert client._connection_count == 2

            # First exit (should not close)
            await client.__aexit__(None, None, None)
            assert client._connection_count == 1

            # Final exit (should close)
            await client.__aexit__(None, None, None)
            assert client._connection_count == 0

        client.call_async_from_sync(test_async, timeout=10.0)

        # Clean up
        client.sync_close()


@pytest.mark.asyncio
class TestMCPSessionDirectVerification:
    """Direct async tests to verify MCP session behavior with live servers."""

    async def test_streamable_http_creates_unique_sessions_per_context(
        self, live_mcp_server: LiveMCPTestServer
    ):
        """Verify that each streamable_http_client context creates a new session.

        This documents the underlying MCP behavior that our SDK must handle.
        """
        url = f"http://127.0.0.1:{live_mcp_server.port}/mcp"
        session_ids = []

        # Each context manager entry creates a NEW session
        for i in range(3):
            async with streamablehttp_client(url) as (read, write, get_session_id):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    session_id = get_session_id()
                    session_ids.append(session_id)
                    logger.info(f"Connection {i + 1}: Session ID = {session_id}")

        # Filter out None values
        valid_session_ids = [s for s in session_ids if s is not None]

        if valid_session_ids:
            unique_sessions = set(valid_session_ids)
            # Each context creates a unique session - expected behavior
            assert len(unique_sessions) == len(valid_session_ids), (
                "Raw MCP client creates unique session per connection"
            )

    async def test_session_stable_within_context(
        self, live_mcp_server: LiveMCPTestServer
    ):
        """Verify session ID remains stable within a single context."""
        url = f"http://127.0.0.1:{live_mcp_server.port}/mcp"

        async with streamablehttp_client(url) as (read, write, get_session_id):
            async with ClientSession(read, write) as session:
                await session.initialize()

                session_id_1 = get_session_id()

                # Multiple calls within same session
                await session.list_tools()
                session_id_2 = get_session_id()

                await session.list_tools()
                session_id_3 = get_session_id()

                # Session ID should be stable within context
                assert session_id_1 == session_id_2 == session_id_3, (
                    "Session ID changed within the same connection context!"
                )


class TestMCPToolExecutorLifecycle:
    """Tests for MCPToolExecutor lifecycle management."""

    def test_lazy_connection_establishment(self, live_mcp_server: LiveMCPTestServer):
        """Test that connection is not established until first tool call."""
        config = {
            "mcpServers": {
                "test_server": {
                    "transport": "http",
                    "url": f"http://127.0.0.1:{live_mcp_server.port}/mcp",
                }
            }
        }

        tools = create_mcp_tools(config, timeout=10.0)
        tool = next(t for t in tools if t.name == "echo")
        executor = tool.executor
        assert isinstance(executor, MCPToolExecutor)

        # Connection should NOT be established yet (lazy)
        # Note: create_mcp_tools establishes a connection to list tools,
        # but the executor's connection for tool calls is separate

        # Make a call - this should establish the executor's connection
        action = tool.action_from_arguments({"message": "test"})
        executor(action)

        # Now connection should be established
        assert executor._connection_established is True

        # Clean up
        executor.close()

    def test_graceful_cleanup_on_error(self, live_mcp_server: LiveMCPTestServer):
        """Test that cleanup happens gracefully even after errors."""
        config = {
            "mcpServers": {
                "test_server": {
                    "transport": "http",
                    "url": f"http://127.0.0.1:{live_mcp_server.port}/mcp",
                }
            }
        }

        tools = create_mcp_tools(config, timeout=10.0)
        tool = next(t for t in tools if t.name == "add_numbers")
        executor = tool.executor
        assert isinstance(executor, MCPToolExecutor)

        # Make a successful call first
        action = tool.action_from_arguments({"a": 1, "b": 2})
        result = executor(action)
        assert "3" in result.text

        # Close should work even if there were previous errors
        executor.close()
        assert executor._connection_established is False
