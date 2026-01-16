"""MCP Session Manager for handling persistent MCP connections.

This module provides session management for MCP clients, enabling:
- Persistent connections across tool calls
- Session ID tracking for reconnection
- Proper cleanup when conversations end

Related issue: https://github.com/OpenHands/software-agent-sdk/issues/1739
"""

from typing import TYPE_CHECKING, Any

from openhands.sdk.logger import get_logger
from openhands.sdk.mcp.client import MCPClient


if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState

logger = get_logger(__name__)


class MCPSessionManager:
    """Manages MCP client sessions with persistence support.
    
    This class maintains persistent MCP connections and tracks session IDs
    to enable session resumption after disconnects.
    
    Attributes:
        conversation_state: Reference to conversation state for session persistence
        _clients: Dictionary of active MCP clients keyed by server URL
        _connected: Set of server URLs with established connections
    """
    
    def __init__(self, conversation_state: "ConversationState | None" = None):
        """Initialize the session manager.
        
        Args:
            conversation_state: Optional conversation state for session persistence.
                If provided, session IDs will be stored/retrieved from state.
        """
        self._conversation_state = conversation_state
        self._clients: dict[str, MCPClient] = {}
        self._connected: set[str] = set()
    
    def register_client(self, server_url: str, client: MCPClient) -> None:
        """Register an MCP client for a server URL.
        
        Args:
            server_url: The MCP server URL
            client: The MCP client instance
        """
        self._clients[server_url] = client
        logger.debug(f"Registered MCP client for {server_url}")
    
    def get_client(self, server_url: str) -> MCPClient | None:
        """Get the MCP client for a server URL.
        
        Args:
            server_url: The MCP server URL
            
        Returns:
            The MCP client if registered, None otherwise
        """
        return self._clients.get(server_url)
    
    def is_connected(self, server_url: str) -> bool:
        """Check if a server connection is established.
        
        Args:
            server_url: The MCP server URL
            
        Returns:
            True if connected, False otherwise
        """
        return server_url in self._connected
    
    def mark_connected(self, server_url: str, session_id: str | None = None) -> None:
        """Mark a server as connected and optionally store session ID.
        
        Args:
            server_url: The MCP server URL
            session_id: Optional session ID to store for reconnection
        """
        self._connected.add(server_url)
        if session_id and self._conversation_state:
            self._conversation_state.mcp_sessions = {
                **self._conversation_state.mcp_sessions,
                server_url: session_id,
            }
            logger.info(f"Stored MCP session ID for {server_url}: {session_id[:8]}...")
    
    def mark_disconnected(self, server_url: str) -> None:
        """Mark a server as disconnected.
        
        Args:
            server_url: The MCP server URL
        """
        self._connected.discard(server_url)
    
    def get_stored_session_id(self, server_url: str) -> str | None:
        """Get stored session ID for a server URL.
        
        Args:
            server_url: The MCP server URL
            
        Returns:
            The stored session ID if available, None otherwise
        """
        if self._conversation_state:
            return self._conversation_state.mcp_sessions.get(server_url)
        return None
    
    def close_all(self) -> None:
        """Close all MCP client connections and cleanup resources.
        
        This should be called when the conversation ends or is paused.
        Session IDs are preserved in conversation state for later resumption.
        """
        for server_url, client in list(self._clients.items()):
            try:
                logger.info(f"Closing MCP connection for {server_url}")
                client.sync_close()
            except Exception as e:
                logger.warning(f"Error closing MCP client for {server_url}: {e}")
            finally:
                self._connected.discard(server_url)
        
        self._clients.clear()
        logger.info("All MCP connections closed")
    
    def close(self, server_url: str) -> None:
        """Close a specific MCP client connection.
        
        Args:
            server_url: The MCP server URL to close
        """
        client = self._clients.pop(server_url, None)
        if client:
            try:
                logger.info(f"Closing MCP connection for {server_url}")
                client.sync_close()
            except Exception as e:
                logger.warning(f"Error closing MCP client for {server_url}: {e}")
            finally:
                self._connected.discard(server_url)


# Global session manager instance (will be set per conversation)
_session_manager: MCPSessionManager | None = None


def get_session_manager() -> MCPSessionManager | None:
    """Get the current global session manager."""
    return _session_manager


def set_session_manager(manager: MCPSessionManager | None) -> None:
    """Set the global session manager."""
    global _session_manager
    _session_manager = manager
