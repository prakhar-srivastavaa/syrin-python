"""Default context manager implementation."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from syrin.context.compactors import CompactionResult, ContextCompactor
from syrin.context.config import Context, ContextBudget, ContextStats
from syrin.context.counter import TokenCounter, get_counter
from syrin.enums import Hook
from syrin.threshold import ThresholdContext


@dataclass
class ContextPayload:
    """The prepared context for an LLM call."""

    messages: list[dict[str, Any]]
    system_prompt: str
    tools: list[dict[str, Any]]
    tokens: int


class _NullSpan:
    """Null context manager for when no tracer is set."""

    def __enter__(self) -> "_NullSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        pass


class _ContextSpan:
    """Wrapper for tracer spans."""

    def __init__(self, tracer: Any, name: str, **attrs: Any):
        self._tracer = tracer
        self._name = name
        self._attrs = attrs
        self._span: Any | None = None

    def __enter__(self) -> Any:
        if hasattr(self._tracer, "span"):
            self._span = self._tracer.span(self._name)
            for k, v in self._attrs.items():
                if self._span is not None:
                    self._span.set_attribute(k, v)
            if self._span is not None:
                return self._span.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._span is not None:
            self._span.__exit__(*args)

    def set_attribute(self, key: str, value: Any) -> None:
        if self._span is not None:
            self._span.set_attribute(key, value)


class ContextManager(Protocol):
    """Protocol for custom context management strategies.

    Implement this protocol to create custom context management strategies.
    """

    def prepare(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list[dict[str, Any]],
        memory_context: str,
        budget: ContextBudget,
    ) -> ContextPayload:
        """Prepare context for LLM call."""
        ...

    def on_compact(self, _event: CompactionResult) -> None:
        """Called after compaction."""
        ...


@dataclass
class DefaultContextManager:
    """Default context manager with automatic compaction.

    Features:
    - Automatic token counting
    - Middle-out truncation (keep start/end of conversation)
    - Auto-compaction at threshold
    - Full observability via events and spans
    - Lifecycle integration
    """

    context: Context = field(default_factory=Context)
    _counter: TokenCounter = field(default_factory=get_counter)
    _compactor: ContextCompactor = field(default_factory=ContextCompactor)
    _stats: ContextStats = field(default_factory=ContextStats)
    _compaction_count: int = 0
    _emit_fn: Callable[[str, dict[str, Any]], None] | None = field(default=None, repr=False)
    _tracer: Any = field(default=None, repr=False)

    def _emit(self, event: Hook | str, ctx: dict[str, Any]) -> None:
        """Emit an event if emit_fn is configured."""
        if self._emit_fn:
            event_str = event.value if hasattr(event, "value") else str(event)
            self._emit_fn(event_str, ctx)

    def _span(self, name: str, **attrs: Any) -> _ContextSpan | _NullSpan:
        """Create a span if tracer is configured."""
        if self._tracer is None:
            return _NullSpan()
        return _ContextSpan(self._tracer, name, **attrs)

    def set_emit_fn(self, emit_fn: Callable[[str, dict[str, Any]], None]) -> None:
        """Set the event emit function for lifecycle hooks."""
        self._emit_fn = emit_fn

    def set_tracer(self, tracer: Any) -> None:
        """Set the tracer for observability."""
        self._tracer = tracer

    def prepare(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list[dict[str, Any]],
        memory_context: str = "",
        budget: ContextBudget | None = None,
    ) -> ContextPayload:
        """Prepare context for LLM call with automatic management.

        Args:
            messages: Conversation messages
            system_prompt: System prompt
            tools: Tool definitions
            memory_context: Injected memory context
            budget: Context budget (auto-created if not provided)

        Returns:
            ContextPayload ready for LLM call
        """
        budget = budget or self.context.get_budget()

        with self._span("context.prepare") as span:
            if span:
                span.set_attribute("context.max_tokens", budget.max_tokens)
                span.set_attribute("context.available", budget.available)

            memory_msg: dict[str, Any] | None = None
            if memory_context:
                memory_msg = {
                    "role": "system",
                    "content": f"[Memory]\n{memory_context}",
                }

            final_messages = list(messages)

            if memory_msg:
                final_messages = [memory_msg] + final_messages

            system_msg: dict[str, Any] | None = None
            if system_prompt:
                system_msg = {"role": "system", "content": system_prompt}

            all_messages = final_messages
            if system_msg:
                all_messages = [system_msg] + all_messages

            tokens_before = self._counter.count_messages(all_messages).total
            tools_tokens = self._counter.count_tools(tools)
            available_for_messages = budget.available - tools_tokens

            if available_for_messages <= 0:
                return ContextPayload(
                    messages=final_messages,
                    system_prompt=system_prompt,
                    tools=tools,
                    tokens=budget.available,
                )

            result = self._compactor.compact(all_messages, available_for_messages)

            tokens_after = self._counter.count_messages(result.messages).total

            if result.method != "none":
                self._compaction_count += 1

                compact_event = {
                    "method": result.method,
                    "tokens_before": tokens_before,
                    "tokens_after": tokens_after,
                    "messages_before": len(all_messages),
                    "messages_after": len(result.messages),
                }
                self._emit("context.compact", compact_event)

                if span:
                    span.set_attribute("context.compacted", True)
                    span.set_attribute("context.compaction_method", result.method)
                    span.set_attribute("context.tokens_saved", tokens_before - tokens_after)

            final_msgs = result.messages

            if system_msg and system_msg not in final_msgs:
                final_msgs = [system_msg] + final_msgs

            tokens_used = (
                self._counter.count_messages(
                    [m for m in final_msgs if m.get("role") != "system"],
                    system_prompt,
                ).total
                + tools_tokens
            )

            budget.used_tokens = tokens_used

            thresholds_triggered = self._check_thresholds(budget)

            self._stats = ContextStats(
                total_tokens=tokens_used,
                max_tokens=budget.max_tokens,
                utilization=budget.utilization,
                compacted=result.method != "none",
                compaction_count=self._compaction_count,
                compaction_method=result.method if result.method != "none" else None,
                thresholds_triggered=thresholds_triggered,
            )

            if span:
                span.set_attribute("context.tokens", tokens_used)
                span.set_attribute("context.utilization", budget.utilization)
                if thresholds_triggered:
                    span.set_attribute("context.thresholds_triggered", thresholds_triggered)

            return ContextPayload(
                messages=final_msgs if memory_msg else final_messages,
                system_prompt=system_prompt,
                tools=tools,
                tokens=tokens_used,
            )

    def on_compact(self, _event: CompactionResult) -> None:
        """Hook called after compaction."""
        self._compaction_count += 1

    def _check_thresholds(self, budget: ContextBudget) -> list[str]:
        """Check and trigger thresholds.

        Returns list of triggered threshold metrics.
        """
        triggered = []
        percent = budget.utilization_percent

        for threshold in self.context.thresholds:
            if percent >= threshold.at:
                triggered.append(threshold.metric)

                threshold_event = {
                    "at": threshold.at,
                    "percent": percent,
                    "metric": threshold.metric,
                    "tokens": budget.used_tokens,
                    "max_tokens": budget.max_tokens,
                }
                self._emit("context.threshold", threshold_event)

                # Execute the threshold action with context
                ctx = ThresholdContext(
                    percentage=percent,
                    metric=threshold.metric,
                    current_value=float(budget.used_tokens),
                    limit_value=float(budget.max_tokens),
                )
                threshold.execute(ctx)

        return triggered

    @property
    def stats(self) -> ContextStats:
        """Get context statistics from last call."""
        return self._stats

    @property
    def current_tokens(self) -> int:
        """Get current token count."""
        return self._stats.total_tokens


def create_context_manager(
    context: Context | None = None,
    emit_fn: Callable[[str, dict[str, Any]], None] | None = None,
    tracer: Any = None,
) -> DefaultContextManager:
    """Create a default context manager from config.

    Args:
        context: Context configuration (creates default if None)
        emit_fn: Optional event emit function for lifecycle hooks
        tracer: Optional tracer for observability

    Returns:
        Configured DefaultContextManager
    """
    if context is None:
        context = Context()

    manager = DefaultContextManager(context=context)
    if emit_fn:
        manager.set_emit_fn(emit_fn)
    if tracer:
        manager.set_tracer(tracer)
    return manager


__all__ = ["ContextManager", "ContextPayload", "DefaultContextManager", "create_context_manager"]
