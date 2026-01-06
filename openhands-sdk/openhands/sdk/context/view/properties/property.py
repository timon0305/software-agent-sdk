from abc import ABC, abstractmethod

from openhands.sdk.context.view.event_mappings import EventMappings
from openhands.sdk.context.view.manipulation_indices import ManipulationIndices
from openhands.sdk.event.base import LLMConvertibleEvent
from openhands.sdk.event.types import EventID


class Property(ABC):
    """Abstract base class for properties of a view."""

    @abstractmethod
    def enforce(
        self, events: list[LLMConvertibleEvent], event_mappings: EventMappings
    ) -> set[EventID]:
        """Enforce the property on a list of events.

        Args:
            events: A list of LLMConvertibleEvent objects to enforce the property on.

        Returns:
            A set of EventID objects to be removed to enforce the property.
        """
        pass

    @abstractmethod
    def manipulation_indices(
        self, events: list[LLMConvertibleEvent], event_mappings: EventMappings
    ) -> ManipulationIndices:
        """Get manipulation indices for the property on a list of events."""
        pass
