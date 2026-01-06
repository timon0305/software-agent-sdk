from unittest.mock import create_autospec

from openhands.sdk.context.view import EventMappings
from openhands.sdk.event.llm_convertible import (
    ActionEvent,
    MessageEvent,
    ObservationEvent,
)


def test_event_mappings_empty_list() -> None:
    """Test EventMappings with empty event list."""
    events = []
    mappings = EventMappings.from_events(events)

    assert mappings.event_id_to_index == {}
    assert mappings.batches == {}
    assert mappings.action_id_to_response_id == {}
    assert mappings.action_id_to_tool_call_id == {}
    assert mappings.observation_id_to_tool_call_id == {}
    assert mappings.tool_call_id_to_observation_id == {}
    assert mappings.action_tool_call_ids == set()
    assert mappings.observation_tool_call_ids == set()


def test_event_mappings_single_message_event() -> None:
    """Test EventMappings with a single non-action/observation event."""
    message_event = create_autospec(MessageEvent, instance=True)
    message_event.id = "msg_1"

    events = [message_event]
    mappings = EventMappings.from_events(events)

    assert mappings.event_id_to_index == {"msg_1": 0}
    assert mappings.batches == {}
    assert mappings.action_tool_call_ids == set()
    assert mappings.observation_tool_call_ids == set()


def test_event_mappings_action_events() -> None:
    """Test EventMappings extracts action tool_call_ids correctly."""
    message_event = create_autospec(MessageEvent, instance=True)
    message_event.id = "msg_1"

    action_event_1 = create_autospec(ActionEvent, instance=True)
    action_event_1.id = "action_1"
    action_event_1.tool_call_id = "call_1"
    action_event_1.llm_response_id = "response_1"

    action_event_2 = create_autospec(ActionEvent, instance=True)
    action_event_2.id = "action_2"
    action_event_2.tool_call_id = "call_2"
    action_event_2.llm_response_id = "response_1"

    action_event_none = create_autospec(ActionEvent, instance=True)
    action_event_none.id = "action_3"
    action_event_none.tool_call_id = None
    action_event_none.llm_response_id = "response_2"

    observation_event = create_autospec(ObservationEvent, instance=True)
    observation_event.id = "obs_1"
    observation_event.tool_call_id = "call_3"

    events = [
        message_event,
        action_event_1,
        action_event_2,
        action_event_none,
        observation_event,
    ]

    mappings = EventMappings.from_events(events)

    # Should only include tool_call_ids from ActionEvents with non-None tool_call_id
    assert mappings.action_tool_call_ids == {"call_1", "call_2"}

    # Check batches
    assert "response_1" in mappings.batches
    assert set(mappings.batches["response_1"]) == {"action_1", "action_2"}
    assert "response_2" in mappings.batches
    assert mappings.batches["response_2"] == ["action_3"]

    # Check action mappings
    assert mappings.action_id_to_response_id["action_1"] == "response_1"
    assert mappings.action_id_to_response_id["action_2"] == "response_1"
    assert mappings.action_id_to_response_id["action_3"] == "response_2"

    assert mappings.action_id_to_tool_call_id["action_1"] == "call_1"
    assert mappings.action_id_to_tool_call_id["action_2"] == "call_2"
    assert "action_3" not in mappings.action_id_to_tool_call_id

    # Check event indices
    assert mappings.event_id_to_index["msg_1"] == 0
    assert mappings.event_id_to_index["action_1"] == 1
    assert mappings.event_id_to_index["action_2"] == 2
    assert mappings.event_id_to_index["action_3"] == 3
    assert mappings.event_id_to_index["obs_1"] == 4


def test_event_mappings_observation_events() -> None:
    """Test EventMappings extracts observation tool_call_ids correctly."""
    message_event = create_autospec(MessageEvent, instance=True)
    message_event.id = "msg_1"

    observation_event_1 = create_autospec(ObservationEvent, instance=True)
    observation_event_1.id = "obs_1"
    observation_event_1.tool_call_id = "call_1"

    observation_event_2 = create_autospec(ObservationEvent, instance=True)
    observation_event_2.id = "obs_2"
    observation_event_2.tool_call_id = "call_2"

    observation_event_none = create_autospec(ObservationEvent, instance=True)
    observation_event_none.id = "obs_3"
    observation_event_none.tool_call_id = None

    action_event = create_autospec(ActionEvent, instance=True)
    action_event.id = "action_1"
    action_event.tool_call_id = "call_3"
    action_event.llm_response_id = "response_1"

    events = [
        message_event,
        observation_event_1,
        observation_event_2,
        observation_event_none,
        action_event,
    ]

    mappings = EventMappings.from_events(events)

    # Should only include tool_call_ids from ObservationEvents with non-None
    # tool_call_id
    assert mappings.observation_tool_call_ids == {"call_1", "call_2"}

    # Check observation mappings
    assert mappings.observation_id_to_tool_call_id["obs_1"] == "call_1"
    assert mappings.observation_id_to_tool_call_id["obs_2"] == "call_2"
    assert "obs_3" not in mappings.observation_id_to_tool_call_id

    assert mappings.tool_call_id_to_observation_id["call_1"] == "obs_1"
    assert mappings.tool_call_id_to_observation_id["call_2"] == "obs_2"
    assert "call_3" not in mappings.tool_call_id_to_observation_id

    # Check event indices
    assert mappings.event_id_to_index["msg_1"] == 0
    assert mappings.event_id_to_index["obs_1"] == 1
    assert mappings.event_id_to_index["obs_2"] == 2
    assert mappings.event_id_to_index["obs_3"] == 3
    assert mappings.event_id_to_index["action_1"] == 4


def test_event_mappings_action_observation_pairs() -> None:
    """Test EventMappings correctly maps action-observation pairs."""
    action_event_1 = create_autospec(ActionEvent, instance=True)
    action_event_1.id = "action_1"
    action_event_1.tool_call_id = "call_1"
    action_event_1.llm_response_id = "response_1"

    observation_event_1 = create_autospec(ObservationEvent, instance=True)
    observation_event_1.id = "obs_1"
    observation_event_1.tool_call_id = "call_1"

    action_event_2 = create_autospec(ActionEvent, instance=True)
    action_event_2.id = "action_2"
    action_event_2.tool_call_id = "call_2"
    action_event_2.llm_response_id = "response_2"

    observation_event_2 = create_autospec(ObservationEvent, instance=True)
    observation_event_2.id = "obs_2"
    observation_event_2.tool_call_id = "call_2"

    events = [
        action_event_1,
        observation_event_1,
        action_event_2,
        observation_event_2,
    ]

    mappings = EventMappings.from_events(events)

    # Check both action and observation tool_call_ids are present
    assert mappings.action_tool_call_ids == {"call_1", "call_2"}
    assert mappings.observation_tool_call_ids == {"call_1", "call_2"}

    # Check tool_call_id mappings
    assert mappings.action_id_to_tool_call_id["action_1"] == "call_1"
    assert mappings.action_id_to_tool_call_id["action_2"] == "call_2"
    assert mappings.observation_id_to_tool_call_id["obs_1"] == "call_1"
    assert mappings.observation_id_to_tool_call_id["obs_2"] == "call_2"

    # Check bidirectional observation mapping
    assert mappings.tool_call_id_to_observation_id["call_1"] == "obs_1"
    assert mappings.tool_call_id_to_observation_id["call_2"] == "obs_2"


def test_event_mappings_multiple_batches() -> None:
    """Test EventMappings correctly groups actions into batches."""
    action_1a = create_autospec(ActionEvent, instance=True)
    action_1a.id = "action_1a"
    action_1a.tool_call_id = "call_1a"
    action_1a.llm_response_id = "response_1"

    action_1b = create_autospec(ActionEvent, instance=True)
    action_1b.id = "action_1b"
    action_1b.tool_call_id = "call_1b"
    action_1b.llm_response_id = "response_1"

    action_2a = create_autospec(ActionEvent, instance=True)
    action_2a.id = "action_2a"
    action_2a.tool_call_id = "call_2a"
    action_2a.llm_response_id = "response_2"

    events = [action_1a, action_1b, action_2a]

    mappings = EventMappings.from_events(events)

    # Check batches are correctly grouped
    assert "response_1" in mappings.batches
    assert "response_2" in mappings.batches
    assert set(mappings.batches["response_1"]) == {"action_1a", "action_1b"}
    assert mappings.batches["response_2"] == ["action_2a"]

    # Check all actions are mapped to their response IDs
    assert mappings.action_id_to_response_id["action_1a"] == "response_1"
    assert mappings.action_id_to_response_id["action_1b"] == "response_1"
    assert mappings.action_id_to_response_id["action_2a"] == "response_2"
