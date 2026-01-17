"""Example: Plugin Loading via AgentContext

This example demonstrates the recommended pattern for loading plugins:
pass plugin_source to AgentContext instead of explicit Plugin.load() calls.

With this pattern:
- Skills are automatically merged into AgentContext.skills
- MCP config is automatically merged during Agent initialization
- Hooks are automatically extracted and applied to the Conversation

This is the same pattern used by the agent-server API, ensuring consistency
between local SDK usage and remote agent-server usage.

Usage:
    export LLM_API_KEY=your-api-key  # Optional, demo runs without
    python main.py
"""

import os
import sys
import tempfile
from pathlib import Path

from pydantic import SecretStr

from openhands.sdk import LLM, Agent, AgentContext, Conversation
from openhands.sdk.tool import Tool
from openhands.tools.terminal import TerminalTool


# Get path to example plugin
script_dir = Path(__file__).parent
example_plugin_path = (
    script_dir.parent / "02_loading_plugins" / "example_plugins" / "code-quality"
)

# =============================================================================
# Part 1: Create AgentContext with plugin_source
# =============================================================================
print("=" * 80)
print("Part 1: Creating AgentContext with Plugin Source")
print("=" * 80)

# The recommended pattern: pass plugin_source to AgentContext
# The plugin is automatically fetched and loaded during initialization
agent_context = AgentContext(
    plugin_source=str(example_plugin_path),  # Local path, or "github:owner/repo"
    # plugin_ref="v1.0.0",  # Optional: specific version/branch/commit
    # plugin_path="plugins/sub",  # Optional: subdirectory within repo
)

print(f"\nPlugin source: {example_plugin_path}")
print(f"Skills loaded: {len(agent_context.skills)}")
for skill in agent_context.skills:
    print(f"  - {skill.name}")

print(f"\nMCP config available: {agent_context.plugin_mcp_config is not None}")
if agent_context.plugin_mcp_config:
    servers = agent_context.plugin_mcp_config.get("mcpServers", {})
    for server_name in servers:
        print(f"  - {server_name}")

print(f"\nHooks available: {agent_context.plugin_hooks is not None}")
if agent_context.plugin_hooks:
    hooks = agent_context.plugin_hooks
    if hooks.pre_tool_use:
        print(f"  - PreToolUse: {len(hooks.pre_tool_use)} matcher(s)")
    if hooks.post_tool_use:
        print(f"  - PostToolUse: {len(hooks.post_tool_use)} matcher(s)")

# =============================================================================
# Part 2: Create Agent - MCP config is automatically merged
# =============================================================================
print("\n" + "=" * 80)
print("Part 2: Creating Agent with AgentContext")
print("=" * 80)

# Check for API key
api_key = os.getenv("LLM_API_KEY")
if not api_key:
    print("\nSkipping agent demo (LLM_API_KEY not set)")
    print("\nTo run the full demo, set the LLM_API_KEY environment variable:")
    print("  export LLM_API_KEY=your-api-key")
    print("\nBut you can see the plugin was loaded successfully above!")
    sys.exit(0)

# Configure LLM
model = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-5-20250929")
llm = LLM(
    usage_id="plugin-demo",
    model=model,
    api_key=SecretStr(api_key),
)

# Create agent with the agent_context
# MCP config from plugin is automatically merged during initialization
agent = Agent(
    llm=llm,
    tools=[Tool(name=TerminalTool.name)],
    agent_context=agent_context,
    # No need to pass mcp_config - it's merged from plugin automatically
)

print(f"Agent created with {len(agent_context.skills)} skills from plugin")

# =============================================================================
# Part 3: Create Conversation - Hooks are automatically extracted
# =============================================================================
print("\n" + "=" * 80)
print("Part 3: Creating Conversation")
print("=" * 80)

with tempfile.TemporaryDirectory() as tmpdir:
    # Create conversation - hooks from plugin are automatically applied
    # because LocalConversation extracts them from agent.agent_context.plugin_hooks
    conversation = Conversation(
        agent=agent,
        workspace=tmpdir,
        # No need to pass hook_config - it's extracted from agent_context automatically
    )

    print("Conversation created!")
    print("  - Skills: loaded from plugin via agent_context")
    print("  - MCP config: merged during agent initialization")
    print("  - Hooks: extracted from agent_context.plugin_hooks")

    # =============================================================================
    # Part 4: Use the conversation
    # =============================================================================
    print("\n" + "=" * 80)
    print("Part 4: Running Demo")
    print("=" * 80)

    # The skill should be triggered by "lint" keyword
    print("\nSending message with 'lint' keyword to trigger skill...")
    conversation.send_message("How do I lint Python code? Brief explanation please.")
    conversation.run()

    print(f"\nTotal cost: ${llm.metrics.accumulated_cost:.4f}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 80)
print("Summary: The Recommended Pattern")
print("=" * 80)

print("""
The recommended way to load plugins is via AgentContext:

    # Create AgentContext with plugin source
    agent_context = AgentContext(
        plugin_source="github:owner/repo",  # or local path
        plugin_ref="v1.0.0",  # optional
        plugin_path="plugins/sub",  # optional
    )

    # Create Agent - MCP config is merged automatically
    agent = Agent(
        llm=llm,
        tools=[...],
        agent_context=agent_context,
    )

    # Create Conversation - hooks are extracted automatically
    conversation = Conversation(
        agent=agent,
        workspace="./workspace",
    )

This pattern:
- Is consistent with the agent-server API
- Automatically handles all plugin components (skills, MCP, hooks)
- Reduces boilerplate code
""")
