import json
from typing import Annotated, Any

from pydantic import BeforeValidator, Field
from rich.text import Text

from openhands.sdk.event.base import N_CHAR_PREVIEW, LLMConvertibleEvent
from openhands.sdk.event.types import SourceType
from openhands.sdk.llm import Message, TextContent
from openhands.sdk.tool import ToolDefinition


def _validate_tool_item(item: Any) -> ToolDefinition | dict[str, Any]:
    """Validate a single tool item, accepting both ToolDefinition and OpenAI format."""
    if isinstance(item, ToolDefinition):
        return item
    if isinstance(item, dict):
        # Check if it's OpenAI format (has "type": "function")
        if item.get("type") == "function" and "function" in item:
            # Keep as dict - we'll handle it in visualize
            return item
        elif "kind" in item:
            # ToolDefinition format - validate it
            return ToolDefinition.model_validate(item)
        else:
            # Unknown format - keep as is
            return item
    return item


def _validate_tools_list(v: Any) -> list[ToolDefinition | dict[str, Any]]:
    """Validate the tools list, accepting both ToolDefinition and OpenAI format."""
    if not isinstance(v, list):
        return v
    return [_validate_tool_item(item) for item in v]


# Type alias for tools field with custom validation
# We use list[Any] as the base type to avoid Pydantic trying to validate
# the union type before our BeforeValidator runs
ToolsList = Annotated[list[Any], BeforeValidator(_validate_tools_list)]


class SystemPromptEvent(LLMConvertibleEvent):
    """System prompt added by the agent."""

    source: SourceType = "agent"
    system_prompt: TextContent = Field(..., description="The system prompt text")
    tools: ToolsList = Field(
        ...,
        description=(
            "List of tools as ToolDefinition objects or OpenAI function format dicts"
        ),
    )

    @property
    def visualize(self) -> Text:
        """Return Rich Text representation of this system prompt event."""
        content = Text()
        content.append("System Prompt:\n", style="bold")
        content.append(self.system_prompt.text)
        content.append(f"\n\nTools Available: {len(self.tools)}")
        for tool in self.tools:
            if isinstance(tool, ToolDefinition):
                # Use ToolDefinition properties directly
                description = tool.description.split("\n")[0][:100]
                if len(description) < len(tool.description):
                    description += "..."

                content.append(f"\n  - {tool.name}: {description}\n")

                # Get parameters from the action type schema
                try:
                    params_dict = tool.action_type.to_mcp_schema()
                    params_str = json.dumps(params_dict)
                    if len(params_str) > 200:
                        params_str = params_str[:197] + "..."
                    content.append(f"  Parameters: {params_str}")
                except Exception:
                    content.append("  Parameters: <unavailable>")
            elif isinstance(tool, dict):
                # Handle OpenAI function format
                func = tool.get("function", {})
                name = func.get("name", "unknown")
                description = func.get("description", "")
                description_preview = description.split("\n")[0][:100]
                if len(description_preview) < len(description):
                    description_preview += "..."

                content.append(f"\n  - {name}: {description_preview}\n")

                # Get parameters from the function schema
                params = func.get("parameters", {})
                if params:
                    params_str = json.dumps(params)
                    if len(params_str) > 200:
                        params_str = params_str[:197] + "..."
                    content.append(f"  Parameters: {params_str}")
                else:
                    content.append("  Parameters: <unavailable>")
        return content

    def to_llm_message(self) -> Message:
        return Message(role="system", content=[self.system_prompt])

    def __str__(self) -> str:
        """Plain text string representation for SystemPromptEvent."""
        base_str = f"{self.__class__.__name__} ({self.source})"
        prompt_preview = (
            self.system_prompt.text[:N_CHAR_PREVIEW] + "..."
            if len(self.system_prompt.text) > N_CHAR_PREVIEW
            else self.system_prompt.text
        )
        tool_count = len(self.tools)
        return (
            f"{base_str}\n  System: {prompt_preview}\n  Tools: {tool_count} available"
        )
