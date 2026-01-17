# Plugin Loading via AgentContext

This example demonstrates the **recommended pattern** for loading plugins in the OpenHands SDK.

## The Pattern

Instead of manually loading plugins with `Plugin.load()` and merging components, use `AgentContext.plugin_source`:

```python
from openhands.sdk import LLM, Agent, AgentContext, Conversation

# Create AgentContext with plugin source - plugin loads automatically
agent_context = AgentContext(
    plugin_source="github:owner/repo",  # or local path, or git URL
    plugin_ref="v1.0.0",  # optional: branch, tag, or commit
    plugin_path="plugins/sub",  # optional: subdirectory within repo
)

# Create Agent - MCP config from plugin is merged automatically
agent = Agent(
    llm=llm,
    tools=[...],
    agent_context=agent_context,
)

# Create Conversation - hooks from plugin are applied automatically
conversation = Conversation(
    agent=agent,
    workspace="./workspace",
)
```

## Why This Pattern?

1. **Consistency**: This is the same pattern used by the agent-server API
2. **Automatic merging**: Skills, MCP config, and hooks are handled automatically
3. **Less boilerplate**: No need to manually call `Plugin.load()`, `plugin.merge_into()`, etc.

## What Happens Internally

When you set `plugin_source` on `AgentContext`:

1. **During AgentContext initialization** (`_load_plugin` validator):
   - Plugin is fetched from source (git clone, local path, etc.)
   - Skills are merged into `AgentContext.skills`
   - MCP config and hooks are stored in private attributes

2. **During Agent initialization** (`_initialize` method):
   - MCP config from `agent_context.plugin_mcp_config` is merged with `agent.mcp_config`

3. **During Conversation creation** (`LocalConversation.__init__`):
   - Hooks are extracted from `agent.agent_context.plugin_hooks`
   - Hooks are automatically applied to the conversation

## Plugin Sources

You can specify plugins in multiple ways:

```python
# GitHub shorthand
plugin_source="github:OpenHands/example-plugin"

# Any git URL
plugin_source="https://gitlab.com/org/plugin.git"
plugin_source="git@bitbucket.org:team/plugin.git"

# Local path
plugin_source="/path/to/local/plugin"
plugin_source="./relative/plugin/path"
```

## Running the Example

```bash
# Without API key (shows plugin loading only)
python main.py

# With API key (full demo with conversation)
export LLM_API_KEY=your-api-key
python main.py
```

## Comparison with Manual Loading

The manual approach (still supported but not recommended):

```python
# Manual approach - more verbose
plugin = Plugin.load(plugin_path)
new_context, new_mcp = plugin.merge_into(agent.agent_context, agent.mcp_config)
agent = agent.model_copy(update={"agent_context": new_context, "mcp_config": new_mcp})
conversation = Conversation(agent=agent, hook_config=plugin.hooks)

# Recommended approach - cleaner
agent_context = AgentContext(plugin_source=str(plugin_path))
agent = Agent(llm=llm, agent_context=agent_context)
conversation = Conversation(agent=agent)  # hooks extracted automatically
```

See `02_loading_plugins/` for the manual approach if you need fine-grained control.
