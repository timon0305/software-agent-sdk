"""Utility functions for MCP integration."""

import logging

import mcp.types
from fastmcp.client.logging import LogMessage
from fastmcp.mcp_config import MCPConfig

from openhands.sdk.logger import get_logger
from openhands.sdk.mcp.client import MCPClient
from openhands.sdk.mcp.exceptions import MCPTimeoutError
from openhands.sdk.mcp.tool import MCPToolDefinition
from openhands.sdk.tool.tool import ToolDefinition


logger = get_logger(__name__)
LOGGING_LEVEL_MAP = logging.getLevelNamesMapping()


async def log_handler(message: LogMessage):
    """
    Handles incoming logs from the MCP server and forwards them
    to the standard Python logging system.
    """
    msg = message.data.get("msg")
    extra = message.data.get("extra")

    # Convert the MCP log level to a Python log level
    level = LOGGING_LEVEL_MAP.get(message.level.upper(), logging.INFO)

    # Log the message using the standard logging library
    logger.log(level, msg, extra=extra)


async def _list_mcp_tools(client: MCPClient) -> list[ToolDefinition]:
    """List tools from a connected MCP client."""
    mcp_type_tools: list[mcp.types.Tool] = await client.list_tools()
    tools: list[ToolDefinition] = []
    for mcp_tool in mcp_type_tools:
        tool_sequence = MCPToolDefinition.create(mcp_tool=mcp_tool, mcp_client=client)
        tools.extend(tool_sequence)
    return tools


async def _create_stateful_toolset(client: MCPClient) -> list[ToolDefinition]:
    """Connect to MCP server and create tools sharing the connection.

    Establishes a persistent connection that remains open for the lifetime
    of the tools, allowing them to maintain session state across calls.
    """
    await client.connect()
    if not client.is_connected():
        raise RuntimeError("MCP client failed to connect")
    return await _list_mcp_tools(client)


def create_mcp_tools(
    config: dict | MCPConfig,
    timeout: float = 30.0,
) -> list[MCPToolDefinition]:
    """Create MCP tools with a persistent connection for session state.

    Returns tools that share a single MCP client connection, enabling stateful
    operations across multiple tool calls (e.g., browser sessions, auth tokens).

    The connection is cleaned up when:
    - Conversation.close() is called (automatically closes all tool executors)
    - executor.close() is called on any tool (closes the shared client)
    - The client is garbage collected

    Args:
        config: MCP configuration dict or MCPConfig object
        timeout: Timeout for connecting and listing tools (default 30s)

    Returns:
        List of MCP tools sharing a persistent connection
    """
    if isinstance(config, dict):
        config = MCPConfig.model_validate(config)
    client = MCPClient(config, log_handler=log_handler)

    try:
        tools: list[MCPToolDefinition] = client.call_async_from_sync(
            _create_stateful_toolset, timeout=timeout, client=client
        )
    except TimeoutError as e:
        client.sync_close()
        server_names = (
            list(config.mcpServers.keys()) if config.mcpServers else ["unknown"]
        )
        error_msg = (
            f"MCP tool listing timed out after {timeout} seconds.\n"
            f"MCP servers configured: {', '.join(server_names)}\n\n"
            "Possible solutions:\n"
            "  1. Increase the timeout value (default is 30 seconds)\n"
            "  2. Check if the MCP server is running and responding\n"
            "  3. Verify network connectivity to the MCP server\n"
        )
        raise MCPTimeoutError(
            error_msg, timeout=timeout, config=config.model_dump()
        ) from e
    except Exception:
        try:
            client.sync_close()
        except Exception as close_exc:
            logger.warning(
                "Failed to close MCP client during error cleanup", exc_info=close_exc
            )
        raise

    logger.info(f"Created {len(tools)} MCP tools: {[t.name for t in tools]}")
    return tools
