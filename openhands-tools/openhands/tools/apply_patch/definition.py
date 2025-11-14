"""ApplyPatch ToolDefinition and executor integrating the cookbook implementation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import Field

from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from openhands.sdk.tool.tool import FunctionToolParam

from .core import Commit, DiffError, process_patch


if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState


from pydantic import AliasChoices


class ApplyPatchAction(Action):
    # Accept both OpenAI server-known param name 'patch' and local 'patch_text'
    patch_text: str = Field(
        validation_alias=AliasChoices("patch", "patch_text"),
        serialization_alias="patch",
        description=(
            "Patch content following the '*** Begin Patch' ... '*** End Patch' "
            "format as described in OpenAI GPT-5.1 prompting guide."
        ),
    )


class ApplyPatchObservation(Observation):
    message: str = ""
    fuzz: int = 0
    commit: Commit | None = None


class ApplyPatchExecutor(ToolExecutor[ApplyPatchAction, ApplyPatchObservation]):
    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root).resolve()

    def _resolve_path(self, p: str) -> Path:
        pth = (
            (self.workspace_root / p).resolve()
            if not p.startswith("/")
            else Path(p).resolve()
        )
        if not str(pth).startswith(str(self.workspace_root)):
            raise DiffError("Absolute or escaping paths are not allowed")
        return pth

    def __call__(
        self,
        action: ApplyPatchAction,
        conversation=None,  # noqa: ARG002
    ) -> ApplyPatchObservation:
        def open_file(path: str) -> str:
            fp = self._resolve_path(path)
            with open(fp, encoding="utf-8") as f:
                return f.read()

        def write_file(path: str, content: str) -> None:
            fp = self._resolve_path(path)
            fp.parent.mkdir(parents=True, exist_ok=True)
            with open(fp, "w", encoding="utf-8") as f:
                f.write(content)

        def remove_file(path: str) -> None:
            fp = self._resolve_path(path)
            fp.unlink(missing_ok=False)

        try:
            msg, fuzz, commit = process_patch(
                action.patch_text, open_file, write_file, remove_file
            )
            return ApplyPatchObservation(message=msg, fuzz=fuzz, commit=commit)
        except DiffError as e:
            return ApplyPatchObservation.from_text(text=str(e), is_error=True)


_DESCRIPTION = (
    "Apply unified text patches to files in the workspace. "
    "Input must start with '*** Begin Patch' and end with '*** End Patch'."
)


class ApplyPatchTool(ToolDefinition[ApplyPatchAction, ApplyPatchObservation]):
    @classmethod
    def create(cls, conv_state: ConversationState) -> list[ApplyPatchTool]:
        executor = ApplyPatchExecutor(workspace_root=conv_state.workspace.working_dir)
        return [
            cls(
                description=_DESCRIPTION,
                action_type=ApplyPatchAction,
                observation_type=ApplyPatchObservation,
                annotations=ToolAnnotations(
                    title="apply_patch",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
                executor=executor,
            )
        ]

    # For OpenAI Responses API with GPT-5.1 models, the tool is server-known.
    # Return a minimal function spec so the provider wires its own definition.
    def to_responses_tool(
        self,
        add_security_risk_prediction: bool = False,  # noqa: ARG002 - signature match
        action_type: type | None = None,  # noqa: ARG002 - signature match
    ) -> FunctionToolParam:  # type: ignore[override]
        # Minimal spec: name only; let OpenAI server resolve params/description
        return {"type": "function", "name": self.name}  # type: ignore[return-value]


register_tool(ApplyPatchTool.name, ApplyPatchTool)
