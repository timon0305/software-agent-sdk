"""Utilities for plugin operations."""

from __future__ import annotations

from typing import Any

from openhands.sdk.context import AgentContext
from openhands.sdk.context.skills import Skill


def merge_skills(
    agent_context: AgentContext | None,
    plugin_skills: list[Skill],
) -> AgentContext:
    """Merge plugin skills into agent context.

    Plugin skills override existing skills with the same name.

    Args:
        agent_context: Existing agent context (or None)
        plugin_skills: Skills to merge in

    Returns:
        New AgentContext with merged skills
    """
    existing_skills = agent_context.skills if agent_context else []

    skills_by_name = {s.name: s for s in existing_skills}
    for skill in plugin_skills:
        skills_by_name[skill.name] = skill

    merged_skills = list(skills_by_name.values())

    if agent_context:
        return agent_context.model_copy(update={"skills": merged_skills})
    return AgentContext(skills=merged_skills)


def merge_mcp_configs(
    base_config: dict[str, Any] | None,
    plugin_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge MCP configurations.

    Plugin config takes precedence for same keys.

    Args:
        base_config: Base MCP configuration
        plugin_config: Plugin MCP configuration

    Returns:
        Merged configuration (empty dict if both are None)
    """
    if base_config is None and plugin_config is None:
        return {}
    if base_config is None:
        return plugin_config or {}
    if plugin_config is None:
        return base_config
    return {**base_config, **plugin_config}
