"""Plugin module for OpenHands SDK.

This module provides support for loading and managing plugins that bundle
skills, hooks, MCP configurations, agents, and commands together.
"""

from openhands.sdk.plugin.fetch import PluginFetchError
from openhands.sdk.plugin.plugin import Plugin
from openhands.sdk.plugin.types import (
    AgentDefinition,
    CommandDefinition,
    PluginAuthor,
    PluginManifest,
)
from openhands.sdk.plugin.utils import merge_mcp_configs, merge_skills


__all__ = [
    "Plugin",
    "PluginFetchError",
    "PluginManifest",
    "PluginAuthor",
    "AgentDefinition",
    "CommandDefinition",
    "merge_mcp_configs",
    "merge_skills",
]
