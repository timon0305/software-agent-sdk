from __future__ import annotations

from collections.abc import Sequence
from logging import getLogger
from typing import overload

from pydantic import BaseModel, Field

from openhands.sdk.context.view.event_mappings import EventMappings
from openhands.sdk.context.view.manipulation_indices import ManipulationIndices
from openhands.sdk.context.view.properties.batch_atomicity import (
    BatchAtomicityProperty,
)
from openhands.sdk.context.view.properties.tool_call_matching import (
    ToolCallMatchingProperty,
)
from openhands.sdk.context.view.properties.tool_loop_atomicity import (
    ToolLoopAtomicityProperty,
)
from openhands.sdk.event import (
    Condensation,
    CondensationRequest,
    CondensationSummaryEvent,
    LLMConvertibleEvent,
)
from openhands.sdk.event.base import Event, EventID
from openhands.sdk.event.llm_convertible import (
    ActionEvent,
    ObservationBaseEvent,
)
from openhands.sdk.event.types import ToolCallID


logger = getLogger(__name__)


class View(BaseModel):
    """Linearly ordered view of events.

    Produced by a condenser to indicate the included events are ready to process as LLM
    input. Also contains fields with information from the condensation process to aid
    in deciding whether further condensation is needed.
    """

    model_config = {"arbitrary_types_allowed": True}

    events: list[LLMConvertibleEvent]

    unhandled_condensation_request: bool = False
    """Whether there is an unhandled condensation request in the view."""

    condensations: list[Condensation] = []
    """A list of condensations that were processed to produce the view."""

    def __len__(self) -> int:
        return len(self.events)

    @property
    def most_recent_condensation(self) -> Condensation | None:
        """Return the most recent condensation, or None if no condensations exist."""
        return self.condensations[-1] if self.condensations else None

    @property
    def summary_event_index(self) -> int | None:
        """Return the index of the summary event, or None if no summary exists."""
        recent_condensation = self.most_recent_condensation
        if (
            recent_condensation is not None
            and recent_condensation.summary is not None
            and recent_condensation.summary_offset is not None
        ):
            return recent_condensation.summary_offset
        return None

    @property
    def summary_event(self) -> CondensationSummaryEvent | None:
        """Return the summary event, or None if no summary exists."""
        if self.summary_event_index is not None:
            event = self.events[self.summary_event_index]
            if isinstance(event, CondensationSummaryEvent):
                return event
        return None

    manipulation_indices: ManipulationIndices = Field(
        description=(
            "Manipulation indices for this view's events. "
            "These indices represent boundaries between atomic units where events can be "
            "safely manipulated (inserted or forgotten). An atomic unit is either: "
            "a tool loop (sequence of batches starting with thinking blocks), "
            "a batch of ActionEvents with the same llm_response_id, or "
            "a single event that is neither an ActionEvent nor an ObservationBaseEvent. "
            "Always includes 0 and len(events) as boundaries."
        )
    )

    # To preserve list-like indexing, we ideally support slicing and position-based
    # indexing. The only challenge with that is switching the return type based on the
    # input type -- we can mark the different signatures for MyPy with `@overload`
    # decorators.

    @overload
    def __getitem__(self, key: slice) -> list[LLMConvertibleEvent]: ...

    @overload
    def __getitem__(self, key: int) -> LLMConvertibleEvent: ...

    def __getitem__(
        self, key: int | slice
    ) -> LLMConvertibleEvent | list[LLMConvertibleEvent]:
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        elif isinstance(key, int):
            return self.events[key]
        else:
            raise ValueError(f"Invalid key type: {type(key)}")

    @staticmethod
    def _enforce_batch_atomicity(
        events: Sequence[Event],
        removed_event_ids: set[EventID],
    ) -> set[EventID]:
        """Ensure that if any ActionEvent in a batch is removed, all ActionEvents
        in that batch are removed.

        This prevents partial batches from being sent to the LLM, which can cause
        API errors when thinking blocks are separated from their tool calls.

        Args:
            events: The original list of events
            removed_event_ids: Set of event IDs that are being removed

        Returns:
            Updated set of event IDs that should be removed (including all
            ActionEvents in batches where any ActionEvent was removed)
        """
        mappings = EventMappings.from_events(events)

        if not mappings.batches:
            return removed_event_ids

        updated_removed_ids = set(removed_event_ids)

        for llm_response_id, batch_event_ids in mappings.batches.items():
            # Check if any ActionEvent in this batch is being removed
            if any(event_id in removed_event_ids for event_id in batch_event_ids):
                # If so, remove all ActionEvents in this batch
                updated_removed_ids.update(batch_event_ids)
                logger.debug(
                    f"Enforcing batch atomicity: removing entire batch "
                    f"with llm_response_id={llm_response_id} "
                    f"({len(batch_event_ids)} events)"
                )

        return updated_removed_ids

    @staticmethod
    def filter_unmatched_tool_calls(
        events: list[LLMConvertibleEvent],
    ) -> list[LLMConvertibleEvent]:
        """Filter out unmatched tool call events.

        Removes ActionEvents and ObservationEvents that have tool_call_ids
        but don't have matching pairs. Also enforces batch atomicity - if any
        ActionEvent in a batch is filtered out, all ActionEvents in that batch
        are also filtered out.
        """
        mappings = EventMappings.from_events(events)

        # First pass: identify which events would NOT be kept based on matching
        removed_event_ids: set[EventID] = set()
        for event in events:
            if not View._should_keep_event(
                event, mappings.action_tool_call_ids, mappings.observation_tool_call_ids
            ):
                removed_event_ids.add(event.id)

        # Second pass: enforce batch atomicity for ActionEvents
        # If any ActionEvent in a batch is removed, all ActionEvents in that
        # batch should also be removed
        removed_event_ids = View._enforce_batch_atomicity(events, removed_event_ids)

        # Third pass: also remove ObservationEvents whose ActionEvents were removed
        # due to batch atomicity
        tool_call_ids_to_remove: set[ToolCallID] = set()
        for action_id in removed_event_ids:
            if action_id in mappings.action_id_to_tool_call_id:
                tool_call_ids_to_remove.add(
                    mappings.action_id_to_tool_call_id[action_id]
                )

        # Filter out removed events
        result = []
        for event in events:
            if event.id in removed_event_ids:
                continue
            if isinstance(event, ObservationBaseEvent):
                if event.tool_call_id in tool_call_ids_to_remove:
                    continue
            result.append(event)

        return result

    @staticmethod
    def _should_keep_event(
        event: LLMConvertibleEvent,
        action_tool_call_ids: set[ToolCallID],
        observation_tool_call_ids: set[ToolCallID],
    ) -> bool:
        """Determine if an event should be kept based on tool call matching."""
        if isinstance(event, ObservationBaseEvent):
            return event.tool_call_id in action_tool_call_ids
        elif isinstance(event, ActionEvent):
            return event.tool_call_id in observation_tool_call_ids
        else:
            return True

    def find_next_manipulation_index(self, threshold: int, strict: bool = False) -> int:
        """Find the smallest manipulation index greater than (or equal to) a threshold.

        This is a helper method for condensation logic that needs to find safe
        boundaries for forgetting events. Uses the cached manipulation_indices property.

        Args:
            threshold: The threshold value to compare against
            strict: If True, finds index > threshold. If False, finds index >= threshold

        Returns:
            The smallest manipulation index that satisfies the condition, or the
            threshold itself if no such index exists
        """
        return self.manipulation_indices.find_next(threshold, strict)

    @staticmethod
    def from_events(events: Sequence[Event]) -> View:
        """Create a view from a list of events, respecting the semantics of any
        condensation events.
        """
        forgotten_event_ids: set[EventID] = set()
        condensations: list[Condensation] = []
        for event in events:
            if isinstance(event, Condensation):
                condensations.append(event)
                forgotten_event_ids.update(event.forgotten_event_ids)
                # Make sure we also forget the condensation action itself
                forgotten_event_ids.add(event.id)
            if isinstance(event, CondensationRequest):
                forgotten_event_ids.add(event.id)

        # Enforce batch atomicity: if any event in a multi-action batch is forgotten,
        # forget all events in that batch to prevent partial batches with thinking
        # blocks separated from their tool calls
        forgotten_event_ids = View._enforce_batch_atomicity(events, forgotten_event_ids)

        kept_events = [
            event
            for event in events
            if event.id not in forgotten_event_ids
            and isinstance(event, LLMConvertibleEvent)
        ]

        # If we have a summary, insert it at the specified offset.
        summary: str | None = None
        summary_offset: int | None = None

        # The relevant summary is always in the last condensation event (i.e., the most
        # recent one).
        for event in reversed(events):
            if isinstance(event, Condensation):
                if event.summary is not None and event.summary_offset is not None:
                    summary = event.summary
                    summary_offset = event.summary_offset
                    break

        if summary is not None and summary_offset is not None:
            logger.debug(f"Inserting summary at offset {summary_offset}")

            _new_summary_event = CondensationSummaryEvent(summary=summary)
            kept_events.insert(summary_offset, _new_summary_event)

        # Check for an unhandled condensation request -- these are events closer to the
        # end of the list than any condensation action.
        unhandled_condensation_request = False

        for event in reversed(events):
            if isinstance(event, Condensation):
                break

            if isinstance(event, CondensationRequest):
                unhandled_condensation_request = True
                break

        # Filter unmatched tool calls to get the final view events
        view_events = View.filter_unmatched_tool_calls(kept_events)

        # Calculate manipulation_indices using properties
        # Instantiate the three properties
        batch_atomicity = BatchAtomicityProperty()
        tool_loop_atomicity = ToolLoopAtomicityProperty()
        tool_call_matching = ToolCallMatchingProperty()

        # Call manipulation_indices() on each property
        # For empty views, return {0} as the single manipulation index
        if not view_events:
            final_indices = ManipulationIndices({0})
        else:
            batch_indices = batch_atomicity.manipulation_indices(view_events, events)
            tool_loop_indices = tool_loop_atomicity.manipulation_indices(
                view_events, events
            )
            tool_call_indices = tool_call_matching.manipulation_indices(
                view_events, events
            )

            # Take the intersection of all three sets
            final_indices = ManipulationIndices(
                batch_indices & tool_loop_indices & tool_call_indices
            )

        return View(
            events=view_events,
            unhandled_condensation_request=unhandled_condensation_request,
            condensations=condensations,
            manipulation_indices=final_indices,
        )
