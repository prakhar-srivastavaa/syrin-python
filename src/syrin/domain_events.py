"""Domain events — typed events for key lifecycle moments.

Domain events complement the Hook system: Hooks carry EventContext dicts;
domain events are typed dataclasses that observability and other consumers
can subscribe to without parsing strings.

Usage:
    >>> from syrin.domain_events import BudgetThresholdReached, ContextCompacted, EventBus
    >>>
    >>> bus = EventBus()
    >>> bus.subscribe(BudgetThresholdReached, lambda e: print(f"Budget at {e.percentage}%"))
    >>> bus.subscribe(ContextCompacted, lambda e: print(f"Compacted: {e.method}"))
    >>>
    >>> agent = Agent(model=..., bus=bus)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

__all__ = [
    "BudgetThresholdReached",
    "ContextCompacted",
    "DomainEvent",
    "EventBus",
]

T = TypeVar("T", bound="DomainEvent")


@dataclass(frozen=True)
class DomainEvent:
    """Base for domain events. All domain events are immutable dataclasses.

    Use EventBus to subscribe and emit. Pass bus=EventBus() to Agent.
    """


@dataclass(frozen=True)
class BudgetThresholdReached(DomainEvent):
    """Emitted when a budget threshold is crossed (e.g. 80% of run budget)."""

    percentage: int
    """Utilization percentage (0-100) that triggered the threshold."""
    current_value: float
    """Current cost or token count."""
    limit_value: float
    """Limit or cap (e.g. run budget, max tokens)."""
    metric: str = "cost"
    """Metric type: 'cost' or 'tokens'."""
    action_taken: str | None = None
    """Optional: action executed by threshold (e.g. 'warn', 'switch_model')."""


@dataclass(frozen=True)
class ContextCompacted(DomainEvent):
    """Emitted when context window compaction runs (truncation/summarization)."""

    method: str
    """Compaction method (e.g. 'middle_out', 'truncate')."""
    tokens_before: int
    """Token count before compaction."""
    tokens_after: int
    """Token count after compaction."""
    messages_before: int = 0
    """Number of messages before compaction."""
    messages_after: int = 0
    """Number of messages after compaction."""


class EventBus(Generic[T]):
    """Event bus for typed domain events. Subscribe and emit typed events.

    Use this when you need typed, structured event handling (e.g. for metrics,
    observability, or custom pipelines).
    """

    def __init__(self) -> None:
        self._listeners: dict[type[DomainEvent], list[Callable[[DomainEvent], None]]] = {}

    def subscribe(
        self,
        event_type: type[T],
        handler: Callable[[T], None],
    ) -> None:
        """Subscribe to a domain event type."""
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(handler)  # type: ignore[arg-type]

    def emit(self, event: DomainEvent) -> None:
        """Emit a domain event to all subscribers."""
        t = type(event)
        for base in t.__mro__:
            if base in self._listeners:
                for h in self._listeners[base]:
                    h(event)
