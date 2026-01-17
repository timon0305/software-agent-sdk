"""Tests for AgentContext plugin loading functionality."""

from pathlib import Path

import pytest

from openhands.sdk.context.agent_context import AgentContext
from openhands.sdk.context.skills import Skill


def create_test_plugin(
    tmp_path: Path,
    name: str = "test-plugin",
    *,
    skills: list[dict] | None = None,
    hooks: dict | None = None,
    mcp_config: dict | None = None,
) -> Path:
    """Create a test plugin directory structure.

    Args:
        tmp_path: Temporary path for the plugin
        name: Plugin name
        skills: List of skill dicts with 'name' and 'content' keys
        hooks: Hooks configuration to write to hooks/hooks.json
        mcp_config: MCP configuration to write to .mcp.json

    Returns:
        Path to the plugin directory
    """
    plugin_dir = tmp_path / name
    plugin_dir.mkdir(parents=True)

    # Create manifest
    manifest_dir = plugin_dir / ".plugin"
    manifest_dir.mkdir()
    manifest_file = manifest_dir / "plugin.json"
    manifest_file.write_text(f'{{"name": "{name}", "version": "1.0.0"}}')

    # Create skills if specified
    if skills:
        skills_dir = plugin_dir / "skills"
        skills_dir.mkdir()
        for skill_data in skills:
            skill_dir = skills_dir / skill_data["name"]
            skill_dir.mkdir()
            skill_md = skill_dir / "SKILL.md"
            skill_md.write_text(
                f"""---
name: {skill_data["name"]}
description: {skill_data.get("description", "Test skill")}
---

{skill_data.get("content", "Test content")}
"""
            )

    # Create hooks if specified
    if hooks:
        import json

        hooks_dir = plugin_dir / "hooks"
        hooks_dir.mkdir()
        hooks_json = hooks_dir / "hooks.json"
        hooks_json.write_text(json.dumps(hooks))

    # Create MCP config if specified
    if mcp_config:
        import json

        mcp_json = plugin_dir / ".mcp.json"
        mcp_json.write_text(json.dumps(mcp_config))

    return plugin_dir


def test_plugin_loading_basic(tmp_path: Path):
    """Test basic plugin loading via AgentContext."""
    plugin_dir = create_test_plugin(
        tmp_path,
        skills=[{"name": "test-skill", "content": "Test skill content"}],
    )

    context = AgentContext(plugin_source=str(plugin_dir))

    # Skill should be merged
    assert len(context.skills) == 1
    assert context.skills[0].name == "test-skill"
    assert "Test skill content" in context.skills[0].content


def test_plugin_loading_skills_merged_with_existing(tmp_path: Path):
    """Test that plugin skills are merged with existing skills."""
    plugin_dir = create_test_plugin(
        tmp_path,
        skills=[
            {"name": "plugin-skill", "content": "Plugin skill content"},
        ],
    )

    # Create existing skill
    existing_skill = Skill(
        name="existing-skill",
        content="Existing skill content",
        source="test.md",
    )

    context = AgentContext(
        skills=[existing_skill],
        plugin_source=str(plugin_dir),
    )

    # Both skills should be present
    assert len(context.skills) == 2
    skill_names = {s.name for s in context.skills}
    assert "existing-skill" in skill_names
    assert "plugin-skill" in skill_names


def test_plugin_loading_skill_override(tmp_path: Path):
    """Test that plugin skills override existing skills with same name."""
    plugin_dir = create_test_plugin(
        tmp_path,
        skills=[
            {"name": "shared-skill", "content": "Plugin version content"},
        ],
    )

    # Create existing skill with same name
    existing_skill = Skill(
        name="shared-skill",
        content="Existing version content",
        source="test.md",
    )

    context = AgentContext(
        skills=[existing_skill],
        plugin_source=str(plugin_dir),
    )

    # Only one skill with the name, and it should be the plugin version
    assert len(context.skills) == 1
    assert context.skills[0].name == "shared-skill"
    assert "Plugin version content" in context.skills[0].content


def test_plugin_loading_mcp_config(tmp_path: Path):
    """Test that MCP config is exposed via property."""
    mcp_config = {
        "mcpServers": {
            "test-server": {
                "command": "echo",
                "args": ["test"],
            }
        }
    }
    plugin_dir = create_test_plugin(
        tmp_path,
        mcp_config=mcp_config,
    )

    context = AgentContext(plugin_source=str(plugin_dir))

    assert context.plugin_mcp_config is not None
    assert "mcpServers" in context.plugin_mcp_config
    assert "test-server" in context.plugin_mcp_config["mcpServers"]


def test_plugin_loading_hooks(tmp_path: Path):
    """Test that hooks are exposed via property."""
    hooks = {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "*",
                    "hooks": [{"type": "command", "command": "echo test"}],
                }
            ]
        }
    }
    plugin_dir = create_test_plugin(
        tmp_path,
        hooks=hooks,
    )

    context = AgentContext(plugin_source=str(plugin_dir))

    assert context.plugin_hooks is not None
    assert len(context.plugin_hooks.pre_tool_use) == 1


def test_plugin_loading_path_validation_absolute(tmp_path: Path):
    """Test that absolute plugin_path is rejected."""
    plugin_dir = create_test_plugin(tmp_path)

    with pytest.raises(ValueError, match="plugin_path must be a relative path"):
        AgentContext(
            plugin_source=str(plugin_dir),
            plugin_path="/absolute/path",
        )


def test_plugin_loading_path_validation_parent_traversal(tmp_path: Path):
    """Test that parent directory traversal in plugin_path is rejected."""
    plugin_dir = create_test_plugin(tmp_path)

    with pytest.raises(ValueError, match="plugin_path must be a relative path"):
        AgentContext(
            plugin_source=str(plugin_dir),
            plugin_path="../sibling/path",
        )


def test_plugin_loading_no_plugin_source():
    """Test that AgentContext works without plugin_source."""
    context = AgentContext()

    assert context.plugin_source is None
    assert context.plugin_mcp_config is None
    assert context.plugin_hooks is None


def test_plugin_loading_graceful_failure(tmp_path: Path, caplog):
    """Test graceful handling of plugin fetch failures."""
    # Use a non-existent path
    nonexistent_path = tmp_path / "nonexistent-plugin"

    # Should not raise, but should log warning
    context = AgentContext(plugin_source=str(nonexistent_path))

    # Plugin outputs should be None
    assert context.plugin_mcp_config is None
    assert context.plugin_hooks is None

    # Should have logged a warning
    assert "Failed to load plugin" in caplog.text


def test_plugin_loading_empty_plugin(tmp_path: Path):
    """Test loading a plugin with no skills, hooks, or MCP config."""
    plugin_dir = create_test_plugin(tmp_path)

    context = AgentContext(plugin_source=str(plugin_dir))

    # Should work but have empty outputs
    assert context.skills == []
    assert context.plugin_mcp_config is None
    assert context.plugin_hooks is None


def test_plugin_loading_multiple_skills(tmp_path: Path):
    """Test loading a plugin with multiple skills."""
    plugin_dir = create_test_plugin(
        tmp_path,
        skills=[
            {"name": "skill-1", "content": "Content 1"},
            {"name": "skill-2", "content": "Content 2"},
            {"name": "skill-3", "content": "Content 3"},
        ],
    )

    context = AgentContext(plugin_source=str(plugin_dir))

    assert len(context.skills) == 3
    skill_names = {s.name for s in context.skills}
    assert skill_names == {"skill-1", "skill-2", "skill-3"}


def test_plugin_loading_preserves_other_fields(tmp_path: Path):
    """Test that plugin loading doesn't affect other AgentContext fields."""
    plugin_dir = create_test_plugin(
        tmp_path,
        skills=[{"name": "plugin-skill", "content": "Plugin content"}],
    )

    context = AgentContext(
        plugin_source=str(plugin_dir),
        system_message_suffix="Custom suffix",
        user_message_suffix="User suffix",
    )

    assert context.system_message_suffix == "Custom suffix"
    assert context.user_message_suffix == "User suffix"
    assert len(context.skills) == 1
