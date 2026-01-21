"""
Probe: conversation restore behavior when tools change.

This is a temporary investigation script meant to answer, with evidence:

1) What gets persisted to disk when a conversation is created and run?
   - `base_state.json` (ConversationState snapshot, excluding events)
   - `events/event-*.json` (append-only EventLog)

2) What changes on disk when a user restores a conversation with a *different*
   runtime Agent configuration (specifically: adding a tool)?
   - Does `base_state.json` update to reflect the runtime agent's tool *specs*?
   - Do we create or overwrite a `SystemPromptEvent` on restore?

3) What is actually sent to the LLM after restore?
   - Even if the persisted `SystemPromptEvent` isn't rewritten, do we send the
     *new* tool definitions to the LLM on the next call?

How this script works
---------------------
We run two phases against the same persisted conversation ID:

Phase A (initial conversation)
  - Create a conversation with an Agent that has exactly one non-builtin tool:
    `file_editor` (FileEditorTool). Persistence is enabled.
  - Ensure `SystemPromptEvent` exists (created during `Agent.init_state()`).
    For readability in the report, we only display the first paragraph of the
    system prompt text, but we validate that the full prompt is persisted.
  - Send a user message and run the agent once.

Phase B (restore with an additional tool)
  - Restore the *same* conversation (same `conversation_id`, same persistence
    directory) but provide a runtime Agent configured with:
      - `file_editor` (same as Phase A)
      - `terminal` (TerminalTool) ADDED
  - Send a user message and run the agent once.

LLM + telemetry capture
-----------------------
This script enables completion telemetry logging via `LLM(log_completions=True)`.
To keep this script runnable without API keys and without network calls, it
patches `LLM._transport_call` with a deterministic stub response that always
invokes the builtin `finish` tool. This guarantees:
  - The SDK still builds the request payload (messages + tools)
  - The telemetry logger writes the request content to disk
  - The run terminates quickly (FinishTool sets execution_status=FINISHED)

Run
---
    uv run python .pr/restore_tool_change_persistence_probe.py

Artifacts are written under:
    .pr/artifacts/<timestamp>/
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import time
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

from litellm.types.utils import ModelResponse
from pydantic import SecretStr

from openhands.sdk import LLM, Agent, Conversation, Tool
from openhands.sdk.event import Event, LLMConvertibleEvent, SystemPromptEvent
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.terminal import TerminalTool


@dataclass(frozen=True)
class Snapshot:
    label: str
    root: Path
    persistence_dir: Path
    telemetry_dir: Path
    persistence_hashes: dict[str, str]
    base_state: dict[str, Any]
    event_files: list[Path]
    event_types: list[str]
    system_prompt_event: SystemPromptEvent
    system_prompt_first_paragraph: str
    system_prompt_tool_names: list[str]
    telemetry_files: list[Path]
    telemetry_hashes: dict[str, str]
    last_telemetry_payload: dict[str, Any] | None


def _now_slug() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_paragraph(text: str) -> str:
    for chunk in text.split("\n\n"):
        chunk = chunk.strip()
        if chunk:
            return chunk
    return ""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_hashes(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            out[rel] = _sha256_bytes(p.read_bytes())
    return out


def _diff_hashes(before: dict[str, str], after: dict[str, str]) -> dict[str, list[str]]:
    before_keys = set(before)
    after_keys = set(after)

    added = sorted(after_keys - before_keys)
    removed = sorted(before_keys - after_keys)
    changed = sorted(k for k in (before_keys & after_keys) if before[k] != after[k])
    unchanged = sorted(k for k in (before_keys & after_keys) if before[k] == after[k])
    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "unchanged": unchanged,
    }


def _render_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)


def _render_unified_diff(name: str, a: str, b: str, limit_lines: int = 200) -> str:
    diff = list(
        difflib.unified_diff(
            a.splitlines(),
            b.splitlines(),
            fromfile=f"{name} (before)",
            tofile=f"{name} (after)",
            lineterm="",
        )
    )
    if not diff:
        return ""
    if len(diff) > limit_lines:
        diff = diff[:limit_lines] + ["... (diff truncated)"]
    return "\n".join(diff)


def _fake_transport_call(
    _self: Any, *, messages: list[dict[str, Any]], **_: Any
) -> ModelResponse:
    _ = messages  # kept for debugging; telemetry already captures the full prompt
    tool_call = {
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {
            "name": "finish",
            "arguments": json.dumps({"message": "ok (stubbed LLM response)"}),
        },
    }
    assistant_message = {"role": "assistant", "content": "", "tool_calls": [tool_call]}
    return ModelResponse(
        id=f"resp_{uuid.uuid4().hex[:10]}",
        model="gpt-4o-mini",
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": assistant_message,
            }
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


def _snapshot(
    label: str, *, root: Path, persistence_dir: Path, telemetry_dir: Path
) -> Snapshot:
    base_state_path = persistence_dir / "base_state.json"
    base_state = _read_json(base_state_path)

    events_dir = persistence_dir / "events"
    event_files = sorted(events_dir.glob("event-*.json"))
    if not event_files:
        raise RuntimeError(f"[{label}] no event files found under {events_dir}")

    events: list[Event] = [
        Event.model_validate_json(p.read_text(encoding="utf-8")) for p in event_files
    ]
    event_types = [type(e).__name__ for e in events]

    # We expect the first LLM-convertible event to be the SystemPromptEvent.
    llm_events = [e for e in events if isinstance(e, LLMConvertibleEvent)]
    if not llm_events or not isinstance(llm_events[0], SystemPromptEvent):
        raise RuntimeError(
            f"[{label}] expected first LLMConvertibleEvent to be SystemPromptEvent; "
            f"got {[type(e).__name__ for e in llm_events[:3]]}"
        )
    system_event = llm_events[0]
    assert isinstance(system_event, SystemPromptEvent)
    system_first_para = _first_paragraph(system_event.system_prompt.text)
    system_tool_names = sorted({t.name for t in system_event.tools})

    telemetry_files = sorted(telemetry_dir.glob("*.json"))
    last_payload = _read_json(telemetry_files[-1]) if telemetry_files else None

    return Snapshot(
        label=label,
        root=root,
        persistence_dir=persistence_dir,
        telemetry_dir=telemetry_dir,
        persistence_hashes=_file_hashes(persistence_dir),
        base_state=base_state,
        event_files=event_files,
        event_types=event_types,
        system_prompt_event=system_event,
        system_prompt_first_paragraph=system_first_para,
        system_prompt_tool_names=system_tool_names,
        telemetry_files=telemetry_files,
        telemetry_hashes=_file_hashes(telemetry_dir) if telemetry_dir.exists() else {},
        last_telemetry_payload=last_payload,
    )


def _tool_spec_names_from_base_state(base_state: dict[str, Any]) -> list[str]:
    agent = base_state.get("agent") or {}
    tools = agent.get("tools") or []
    names: list[str] = []
    for t in tools:
        if isinstance(t, dict) and "name" in t:
            names.append(str(t["name"]))
    return names


def _tool_names_from_telemetry(payload: dict[str, Any] | None) -> list[str]:
    if not payload:
        return []
    tools = payload.get("tools") or []
    names: list[str] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        # ToolDefinition serializes its effective name under the computed field `title`.
        # (The `name` attribute is a ClassVar and is not included in model dumps.)
        if "title" in t:
            names.append(str(t["title"]))
        elif "name" in t:
            names.append(str(t["name"]))
    return sorted(set(names))


def _system_and_last_user_from_telemetry(
    payload: dict[str, Any] | None,
) -> tuple[str, str]:
    if not payload:
        return ("", "")
    messages = payload.get("messages") or []
    system_text = ""
    last_user = ""
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "system" and not system_text:
            content = m.get("content")
            if isinstance(content, str):
                system_text = content
            elif isinstance(content, list) and content and isinstance(content[0], dict):
                system_text = str(content[0].get("text", ""))
        if role == "user":
            content = m.get("content")
            if isinstance(content, str):
                last_user = content
            elif isinstance(content, list) and content and isinstance(content[0], dict):
                last_user = str(content[0].get("text", ""))
    return (system_text, last_user)


def _print_snapshot_md(s: Snapshot) -> None:
    print(f"## {s.label}")
    print()
    print(f"- Persistence dir: `{s.persistence_dir}`")
    print(f"- Telemetry dir: `{s.telemetry_dir}`")
    print()

    tool_specs = _tool_spec_names_from_base_state(s.base_state)
    print("### base_state.json")
    print()
    print(f"- Agent tool specs (`agent.tools`): `{tool_specs}`")
    print(f"- execution_status: `{s.base_state.get('execution_status')}`")
    print()

    print("### events/")
    print()
    print(f"- Event files: `{len(s.event_files)}`")
    print(f"- Event types: `{s.event_types}`")
    print()
    print("#### SystemPromptEvent (persisted)")
    print()
    print(f"- Tools in persisted SystemPromptEvent: `{s.system_prompt_tool_names}`")
    print("- System prompt (first paragraph):")
    print()
    print("```")
    print(s.system_prompt_first_paragraph)
    print("```")
    print()

    print("### telemetry logs")
    print()
    print(f"- Telemetry files: `{len(s.telemetry_files)}`")
    if s.telemetry_files:
        print(f"- Latest telemetry file: `{s.telemetry_files[-1].name}`")
        tool_names = _tool_names_from_telemetry(s.last_telemetry_payload)
        system_text, last_user = _system_and_last_user_from_telemetry(
            s.last_telemetry_payload
        )
        print(f"- Tools sent to LLM (per telemetry): `{tool_names}`")
        print("- Prompt sent to LLM (system first paragraph):")
        print()
        print("```")
        print(_first_paragraph(system_text))
        print("```")
        print()
        print("- Prompt sent to LLM (last user message):")
        print()
        print("```")
        print(last_user)
        print("```")
    else:
        print("- No telemetry logs found (did an LLM call occur?)")
    print()


def main() -> None:
    # Keep stdout as Markdown (suppress SDK/tool info logs and noisy warnings).
    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    repo_root = Path(__file__).resolve().parents[1]
    artifacts_root = repo_root / ".pr" / "artifacts" / _now_slug()
    artifacts_root.mkdir(parents=True, exist_ok=True)

    workspace_dir = artifacts_root / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    persistence_base_dir = artifacts_root / "conversations"
    persistence_base_dir.mkdir(parents=True, exist_ok=True)

    conversation_id = uuid.uuid4()
    persistence_dir = persistence_base_dir / conversation_id.hex

    print("# Restore tool-change persistence probe")
    print()
    print(f"- conversation_id: `{conversation_id}`")
    print(f"- persistence_dir: `{persistence_dir}`")
    print(f"- artifacts_root: `{artifacts_root}`")
    print()

    # ---- Phase A ----
    telemetry_a = artifacts_root / "telemetry_phase_a"
    llm_a = LLM(
        # Use a stable model string; we stub the transport so no network calls occur.
        model="gpt-4o-mini",
        api_key=SecretStr("dummy"),
        usage_id="probe",
        log_completions=True,
        log_completions_folder=str(telemetry_a),
    )
    agent_a = Agent(
        llm=llm_a,
        tools=[Tool(name=FileEditorTool.name)],
    )
    convo_a = Conversation(
        agent=agent_a,
        workspace=str(workspace_dir),
        persistence_dir=str(persistence_base_dir),
        conversation_id=conversation_id,
        visualizer=None,
    )

    # Ensure we have a system prompt event persisted on conversation creation.
    snap_a0 = _snapshot(
        "Phase A (after create, before user message)",
        root=artifacts_root,
        persistence_dir=persistence_dir,
        telemetry_dir=telemetry_a,
    )

    convo_a.send_message("Hello. Please confirm you can see my tools.")
    snap_a1 = _snapshot(
        "Phase A (after send_message, before run)",
        root=artifacts_root,
        persistence_dir=persistence_dir,
        telemetry_dir=telemetry_a,
    )

    with (
        patch.object(LLM, "uses_responses_api", return_value=False),
        patch.object(LLM, "_transport_call", new=_fake_transport_call),
    ):
        convo_a.run()

    snap_a2 = _snapshot(
        "Phase A (after run)",
        root=artifacts_root,
        persistence_dir=persistence_dir,
        telemetry_dir=telemetry_a,
    )

    # ---- Phase B ----
    telemetry_b = artifacts_root / "telemetry_phase_b"
    llm_b = LLM(
        model="gpt-4o-mini",
        api_key=SecretStr("dummy"),
        usage_id="probe",
        log_completions=True,
        log_completions_folder=str(telemetry_b),
    )
    agent_b = Agent(
        llm=llm_b,
        tools=[Tool(name=FileEditorTool.name), Tool(name=TerminalTool.name)],
    )
    convo_b = Conversation(
        agent=agent_b,
        workspace=str(workspace_dir),
        persistence_dir=str(persistence_base_dir),
        conversation_id=conversation_id,
        visualizer=None,
    )

    snap_b0 = _snapshot(
        "Phase B (after restore with terminal added, before user message)",
        root=artifacts_root,
        persistence_dir=persistence_dir,
        telemetry_dir=telemetry_b,
    )

    convo_b.send_message("Now I added another tool. Please confirm you can see it.")
    snap_b1 = _snapshot(
        "Phase B (after send_message, before run)",
        root=artifacts_root,
        persistence_dir=persistence_dir,
        telemetry_dir=telemetry_b,
    )

    with (
        patch.object(LLM, "uses_responses_api", return_value=False),
        patch.object(LLM, "_transport_call", new=_fake_transport_call),
    ):
        convo_b.run()

    snap_b2 = _snapshot(
        "Phase B (after run)",
        root=artifacts_root,
        persistence_dir=persistence_dir,
        telemetry_dir=telemetry_b,
    )

    # ---- Markdown report ----
    print("## Snapshots")
    print()
    _print_snapshot_md(snap_a0)
    _print_snapshot_md(snap_a1)
    _print_snapshot_md(snap_a2)
    _print_snapshot_md(snap_b0)
    _print_snapshot_md(snap_b1)
    _print_snapshot_md(snap_b2)

    print("## What changed on disk?")
    print()
    diff = _diff_hashes(snap_a2.persistence_hashes, snap_b2.persistence_hashes)

    print("### Files changed (persistence_dir)")
    print()
    print(f"- added: `{diff['added']}`")
    print(f"- removed: `{diff['removed']}`")
    print(f"- changed: `{diff['changed']}`")
    print()

    base_before = _render_json(snap_a2.base_state)
    base_after = _render_json(snap_b2.base_state)
    base_diff = _render_unified_diff("base_state.json", base_before, base_after)
    if base_diff:
        print("### base_state.json diff (after Phase A run -> after Phase B run)")
        print()
        print("```diff")
        print(base_diff)
        print("```")
        print()

    # Key assertions for the behavior we care about.
    tool_specs_a = _tool_spec_names_from_base_state(snap_a2.base_state)
    tool_specs_b = _tool_spec_names_from_base_state(snap_b2.base_state)
    persisted_system_tools = snap_b2.system_prompt_tool_names
    telemetry_tools_b = _tool_names_from_telemetry(snap_b2.last_telemetry_payload)

    print("## Key checks")
    print()
    print(
        "- `base_state.json` agent.tools updated on restore: "
        f"`{tool_specs_a}` -> `{tool_specs_b}`"
    )
    print(
        "- Persisted `SystemPromptEvent.tools` did not get rewritten on restore: "
        f"`{snap_a2.system_prompt_tool_names}` == `{persisted_system_tools}`"
    )
    print(
        "- Tools sent to LLM after restore (per telemetry) include the added tool: "
        f"`{telemetry_tools_b}`"
    )
    print()


if __name__ == "__main__":
    main()
