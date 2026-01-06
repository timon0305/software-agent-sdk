from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from pydantic import BaseModel

from openhands.sdk.event.base import Event, EventID
from openhands.sdk.event.llm_convertible import (
    ActionEvent,
    ObservationBaseEvent,
)
from openhands.sdk.event.types import ToolCallID


class EventMappings(BaseModel):
    """Consolidated mappings for all events in a view.

    This utility class builds all necessary mappings in a single scan over events,
    including action batches, tool call relationships, and index lookups.
    """

    event_id_to_index: dict[EventID, int]
    """dict mapping any event ID to its index in the event list"""

    batches: dict[EventID, list[EventID]]
    """dict mapping llm_response_id to list of ActionEvent IDs"""

    action_id_to_response_id: dict[EventID, EventID]
    """dict mapping ActionEvent ID to llm_response_id"""

    action_id_to_tool_call_id: dict[EventID, ToolCallID]
    """dict mapping ActionEvent ID to tool_call_id"""

    observation_id_to_tool_call_id: dict[EventID, ToolCallID]
    """dict mapping ObservationEvent ID to tool_call_id"""

    tool_call_id_to_observation_id: dict[ToolCallID, EventID]
    """dict mapping tool_call_id to ObservationEvent ID"""

    action_tool_call_ids: set[ToolCallID]
    """set of all tool_call_ids from ActionEvents"""

    observation_tool_call_ids: set[ToolCallID]
    """set of all tool_call_ids from ObservationEvents"""

    @staticmethod
    def from_events(
        events: Sequence[Event],
    ) -> EventMappings:
        """Build all mappings in a single scan over events."""
        event_id_to_index: dict[EventID, int] = {}
        batches: dict[EventID, list[EventID]] = defaultdict(list)
        action_id_to_response_id: dict[EventID, EventID] = {}
        action_id_to_tool_call_id: dict[EventID, ToolCallID] = {}
        observation_id_to_tool_call_id: dict[EventID, ToolCallID] = {}
        tool_call_id_to_observation_id: dict[ToolCallID, EventID] = {}
        action_tool_call_ids: set[ToolCallID] = set()
        observation_tool_call_ids: set[ToolCallID] = set()

        for idx, event in enumerate(events):
            event_id_to_index[event.id] = idx

            if isinstance(event, ActionEvent):
                llm_response_id = event.llm_response_id
                batches[llm_response_id].append(event.id)
                action_id_to_response_id[event.id] = llm_response_id
                if event.tool_call_id is not None:
                    action_id_to_tool_call_id[event.id] = event.tool_call_id
                    action_tool_call_ids.add(event.tool_call_id)

            elif isinstance(event, ObservationBaseEvent):
                if event.tool_call_id is not None:
                    observation_id_to_tool_call_id[event.id] = event.tool_call_id
                    tool_call_id_to_observation_id[event.tool_call_id] = event.id
                    observation_tool_call_ids.add(event.tool_call_id)

        return EventMappings(
            event_id_to_index=event_id_to_index,
            batches=batches,
            action_id_to_response_id=action_id_to_response_id,
            action_id_to_tool_call_id=action_id_to_tool_call_id,
            observation_id_to_tool_call_id=observation_id_to_tool_call_id,
            tool_call_id_to_observation_id=tool_call_id_to_observation_id,
            action_tool_call_ids=action_tool_call_ids,
            observation_tool_call_ids=observation_tool_call_ids,
        )
