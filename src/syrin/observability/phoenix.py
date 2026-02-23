"""Phoenix (Arize) exporter for local debugging and development.

Phoenix is an open-source observability platform for debugging,
testing, and iterating on LLM applications.

Example:
    >>> from syrin.observability.phoenix import PhoenixExporter
    >>>
    >>> exporter = PhoenixExporter(
    ...     endpoint="http://localhost:6006",
    ...     project_name="my-agent"
    ... )
    >>> trace.add_exporter(exporter)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from syrin.observability import Span, SpanExporter

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    import phoenix as phoenix_sdk
    from phoenix.trace import Span as PhoenixSpan

# Try to import phoenix, but make it optional
try:
    import phoenix as phoenix_sdk
    from phoenix.trace import Span as PhoenixSpan

    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False
    phoenix_sdk = None
    PhoenixSpan = None


class PhoenixExporter(SpanExporter):
    """Export spans to Phoenix for local debugging.

    Phoenix provides:
    - Local UI for trace visualization
    - Prompt experimentation
    - Evaluation and testing
    - No external service required

    Requires: pip install arize-phoenix

    Args:
        endpoint: Phoenix endpoint (default: http://localhost:6006)
        project_name: Project name for grouping traces
        auto_flush: Auto-flush spans to the UI

    Example:
        >>> exporter = PhoenixExporter(
        ...     project_name="my-agent"
        ... )
        >>> trace.add_exporter(exporter)

    Then open http://localhost:6006 to view traces.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:6006",
        project_name: str = "syrin-agent",
        auto_flush: bool = True,
    ) -> None:
        if not PHOENIX_AVAILABLE:
            raise ImportError("Phoenix not installed. Install with: pip install arize-phoenix")

        self.endpoint = endpoint
        self.project_name = project_name
        self.auto_flush = auto_flush

        # Initialize Phoenix
        try:
            self._session = phoenix_sdk.Session(
                endpoint=endpoint,
                project_name=project_name,
            )
            self._tracer = phoenix_sdk.trace.tracer(project_name)
        except Exception as e:
            _log.warning("Failed to initialize Phoenix: %s", e)
            self._session = None
            self._tracer = None

    def export(self, span: Span) -> None:
        """Export span to Phoenix."""
        if not PHOENIX_AVAILABLE or self._tracer is None:
            return

        try:
            # Only export root spans as Phoenix handles nesting
            if span.parent_span_id is not None:
                return

            self._export_trace(span)

            if self.auto_flush:
                self.flush()

        except Exception as e:
            _log.warning("Failed to export span to Phoenix: %s", e)

    def _export_trace(self, span: Span) -> None:
        """Export a trace to Phoenix."""
        with self._tracer.start_as_root_span(span.name) as phoenix_span:
            # Set attributes
            for key, value in span.attributes.items():
                phoenix_span.set_attribute(key, value)

            # Add events
            for event in span.events:
                phoenix_span.add_event(
                    name=event["name"],
                    attributes=event.get("attributes", {}),
                )

            # Set status
            if span.status.value == "error":
                phoenix_span.set_status(phoenix_sdk.trace.StatusCode.ERROR)
            else:
                phoenix_span.set_status(phoenix_sdk.trace.StatusCode.OK)

            # Add child spans
            for child in span.children:
                self._export_child_span(phoenix_span, child)

    def _export_child_span(self, parent: PhoenixSpan, span: Span) -> None:
        """Export a child span to Phoenix."""
        with parent.span(span.name) as child:
            # Set attributes
            for key, value in span.attributes.items():
                child.set_attribute(key, value)

            # Add events
            for event in span.events:
                child.add_event(
                    name=event["name"],
                    attributes=event.get("attributes", {}),
                )

            # Set status
            if span.status.value == "error":
                child.set_status(phoenix_sdk.trace.StatusCode.ERROR)
            else:
                child.set_status(phoenix_sdk.trace.StatusCode.OK)

            # Recursively add children
            for grandchild in span.children:
                self._export_child_span(child, grandchild)

    def flush(self) -> None:
        """Flush any pending data to Phoenix."""
        if self._session:
            try:
                self._session.flush()
            except Exception as e:
                _log.warning("Failed to flush Phoenix: %s", e)

    def close(self) -> None:
        """Close the Phoenix session."""
        if self._session:
            try:
                self._session.close()
            except Exception as e:
                _log.warning("Failed to close Phoenix session: %s", e)


class PhoenixInlineExporter(PhoenixExporter):
    """Phoenix exporter that works without a running Phoenix server.

    This exporter prints traces to console in a format compatible
    with Phoenix's visualization.
    """

    def __init__(self) -> None:
        # Don't call parent __init__ as we don't need Phoenix
        self._spans: list[Span] = []

    def export(self, span: Span) -> None:
        """Store span for later analysis."""
        if span.parent_span_id is None:
            self._spans.append(span)

    def get_traces(self) -> list[Span]:
        """Get all captured traces."""
        return self._spans

    def print_summary(self) -> None:
        """Print a summary of captured traces."""
        print("\n" + "=" * 60)
        print("Phoenix Traces Summary")
        print("=" * 60)

        for i, span in enumerate(self._spans, 1):
            print(f"\nTrace {i}: {span.name}")
            print(f"  Trace ID: {span.trace_id}")
            print(f"  Duration: {span.duration_ms:.2f}ms")
            print(f"  Spans: {len(list(span.walk()))}")

            # Count by kind
            kinds: dict[str, int] = {}
            for s in span.walk():
                kinds[s.kind.value] = kinds.get(s.kind.value, 0) + 1

            print(f"  Kinds: {kinds}")

        print("\n" + "=" * 60)

    def clear(self) -> None:
        """Clear captured traces."""
        self._spans.clear()


__all__ = [
    "PhoenixExporter",
    "PhoenixInlineExporter",
]
