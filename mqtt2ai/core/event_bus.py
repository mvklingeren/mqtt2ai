"""Event Bus module for validation hooks.

Provides a simple pub/sub system for decoupled event tracking during
simulation runs. Events are collected and can be queried by the
scenario validator to check assertions.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional
import time


class EventType(Enum):
    """Types of events that can be published during simulation."""
    TRIGGER_FIRED = "trigger_fired"
    RULE_EXECUTED = "rule_executed"
    RULE_NOT_MATCHED = "rule_not_matched"
    AI_TOOL_CALLED = "ai_tool_called"
    SIMULATION_COMPLETE = "simulation_complete"


@dataclass
class Event:
    """A single event with type, data, and timestamp."""
    type: EventType
    data: dict
    timestamp: float = field(default_factory=time.time)


class EventBus:
    """Simple event bus for publishing and collecting events.

    Events are stored in a list and can be queried by type.
    Subscribers can be registered to receive events in real-time.
    """

    def __init__(self):
        self._events: List[Event] = []
        self._subscribers: dict[EventType, List[Callable[[Event], None]]] = {}

    def publish(self, event_type: EventType, data: dict) -> None:
        """Publish an event to the bus.

        Args:
            event_type: The type of event
            data: Event data dictionary
        """
        event = Event(type=event_type, data=data)
        self._events.append(event)

        # Notify subscribers
        for callback in self._subscribers.get(event_type, []):
            try:
                callback(event)
            except Exception:  # pylint: disable=broad-exception-caught
                pass  # Don't let subscriber errors affect the main flow

    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """Subscribe to events of a specific type.

        Args:
            event_type: The type of event to subscribe to
            callback: Function to call when event is published
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    def get_events(self, event_type: Optional[EventType] = None) -> List[Event]:
        """Get all collected events, optionally filtered by type.

        Args:
            event_type: If provided, only return events of this type

        Returns:
            List of matching events
        """
        if event_type is not None:
            return [e for e in self._events if e.type == event_type]
        return list(self._events)

    def clear(self) -> None:
        """Clear all collected events."""
        self._events.clear()

    def event_count(self, event_type: Optional[EventType] = None) -> int:
        """Get count of events, optionally filtered by type."""
        if event_type is not None:
            return sum(1 for e in self._events if e.type == event_type)
        return len(self._events)


# Global instance for easy access across modules
event_bus = EventBus()

