"""Tests for AgentContext serialization and deserialization."""

import json

from openhands.sdk.context.agent_context import AgentContext
from openhands.sdk.context.skills import (
    KeywordTrigger,
    Skill,
    TaskTrigger,
)
from openhands.sdk.context.skills.types import InputMetadata


def test_agent_context_serialization_roundtrip():
    """Ensure AgentContext round-trips through dict and JSON serialization."""

    repo_skill = Skill(
        name="repo-guidelines",
        content="Repository guidelines",
        source="repo.md",
        trigger=None,
    )
    knowledge_skill = Skill(
        name="python-help",
        content="Use type hints in Python code",
        source="knowledge.md",
        trigger=KeywordTrigger(keywords=["python"]),
    )
    task_skill = Skill(
        name="run-task",
        content="Execute the task with ${param}",
        source="task.md",
        trigger=TaskTrigger(triggers=["run"]),
        inputs=[InputMetadata(name="param", description="Task parameter")],
    )

    context = AgentContext(
        skills=[repo_skill, knowledge_skill, task_skill],
        system_message_suffix="System suffix",
        user_message_suffix="User suffix",
    )

    serialized = context.model_dump()
    assert serialized["system_message_suffix"] == "System suffix"
    assert serialized["user_message_suffix"] == "User suffix"
    # First skill has trigger=None (always-active), others have specific triggers
    assert serialized["skills"][0]["trigger"] is None
    assert serialized["skills"][1]["trigger"]["type"] == "keyword"
    assert serialized["skills"][2]["trigger"]["type"] == "task"

    json_str = context.model_dump_json()
    parsed = json.loads(json_str)
    assert parsed["system_message_suffix"] == "System suffix"
    assert parsed["user_message_suffix"] == "User suffix"
    assert parsed["skills"][2]["inputs"][0]["name"] == "param"

    deserialized_from_dict = AgentContext.model_validate(serialized)
    assert isinstance(deserialized_from_dict.skills[0], Skill)
    assert deserialized_from_dict.skills[0].trigger is None
    assert deserialized_from_dict.skills[0] == repo_skill
    assert isinstance(deserialized_from_dict.skills[1], Skill)
    assert isinstance(deserialized_from_dict.skills[1].trigger, KeywordTrigger)
    assert deserialized_from_dict.skills[1] == knowledge_skill
    assert isinstance(deserialized_from_dict.skills[2], Skill)
    assert isinstance(deserialized_from_dict.skills[2].trigger, TaskTrigger)
    assert deserialized_from_dict.skills[2] == task_skill
    assert deserialized_from_dict.system_message_suffix == "System suffix"
    assert deserialized_from_dict.user_message_suffix == "User suffix"

    deserialized_from_json = AgentContext.model_validate_json(json_str)
    assert isinstance(deserialized_from_json.skills[0], Skill)
    assert deserialized_from_json.skills[0].trigger is None
    assert deserialized_from_json.skills[0] == repo_skill
    assert isinstance(deserialized_from_json.skills[1], Skill)
    assert isinstance(deserialized_from_json.skills[1].trigger, KeywordTrigger)
    assert deserialized_from_json.skills[1] == knowledge_skill
    assert isinstance(deserialized_from_json.skills[2], Skill)
    assert isinstance(deserialized_from_json.skills[2].trigger, TaskTrigger)
    assert deserialized_from_json.skills[2] == task_skill
    assert deserialized_from_json.model_dump() == serialized


def test_agent_context_filters_null_secrets():
    """Test that AgentContext filters out null secrets during deserialization.

    Regression test for issue #1877: When secrets cannot be decrypted
    (e.g., cipher key changed or unavailable), they may have null values.
    The model validator should filter them out to prevent validation errors.
    """
    # Simulate data with a null secret value
    # Note: secrets use "kind" field for discriminated union (not "type")
    data_with_null_secret = {
        "secrets": {
            "VALID_SECRET": {"kind": "StaticSecret", "value": "test-value"},
            "NULL_SECRET": None,  # This would cause ValidationError
        },
    }

    context = AgentContext.model_validate(data_with_null_secret)

    # The null secret should be filtered out
    assert context.secrets is not None
    assert "VALID_SECRET" in context.secrets
    assert "NULL_SECRET" not in context.secrets


def test_agent_context_filters_masked_secrets():
    """Test that AgentContext filters out masked secrets during deserialization.

    Regression test for issue #1877: When secrets are serialized with a cipher
    that later becomes unavailable, the value field becomes null. The model
    validator should filter them out to prevent validation errors.
    """
    # Simulate data with a masked secret (value is null)
    data_with_masked_secret = {
        "secrets": {
            "VALID_SECRET": {"kind": "StaticSecret", "value": "test-value"},
            "MASKED_SECRET": {"kind": "StaticSecret", "value": None},  # Masked
        },
    }

    context = AgentContext.model_validate(data_with_masked_secret)

    # The masked secret should be filtered out
    assert context.secrets is not None
    assert "VALID_SECRET" in context.secrets
    assert "MASKED_SECRET" not in context.secrets


def test_agent_context_handles_all_secrets_null():
    """Test that AgentContext handles case where all secrets are null/masked."""
    data_with_all_null = {
        "secrets": {
            "SECRET1": None,
            "SECRET2": {"kind": "StaticSecret", "value": None},
        },
    }

    context = AgentContext.model_validate(data_with_all_null)

    # secrets should be None when all are filtered out
    assert context.secrets is None
