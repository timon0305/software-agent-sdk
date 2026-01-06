from abc import ABC, abstractmethod

from openhands.sdk.context.view.event_mappings import EventMappings
from openhands.sdk.context.view.manipulation_indices import ManipulationIndices
from openhands.sdk.event.base import LLMConvertibleEvent
from openhands.sdk.event.types import EventID


class ViewPropertyBase(ABC):
    """Abstract base class for properties of a view.
    
    Properties define rules that help maintain the integrity and coherence of the events
    in the view. The properties are maintained via two strategies:
    
    1. Enforcing the property by removing events that violate it.
    2. Defining manipulation indices that restrict where the view can be manipulated.

    In an ideal scenario, sticking to the manipulation indices should suffice to ensure
    the property holds. Enforcement is only intended as a fallback mechanism to handle
    edge cases, bad data, or unforeseen situations.
    """

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
