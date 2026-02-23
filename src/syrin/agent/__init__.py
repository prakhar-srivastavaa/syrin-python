"""Agent base class and response loop with tool execution and budget."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any, cast

from syrin.budget import (
    Budget,
    BudgetStatus,
    BudgetTracker,
    OnExceeded,
)
from syrin.budget_store import BudgetStore
from syrin.checkpoint import CheckpointConfig, Checkpointer
from syrin.context import Context, DefaultContextManager
from syrin.context.config import ContextStats
from syrin.cost import calculate_cost
from syrin.enums import (
    GuardrailStage,
    Hook,
    LoopStrategy,
    MemoryBackend,
    MemoryType,
    MessageRole,
    RateLimitAction,
    StopReason,
)
from syrin.events import EventContext, Events
from syrin.exceptions import BudgetExceededError, BudgetThresholdError, ToolExecutionError
from syrin.guardrails import Guardrail, GuardrailChain, GuardrailResult
from syrin.loop import Loop, ReactLoop
from syrin.memory import ConversationMemory
from syrin.memory.backends import InMemoryBackend, get_backend
from syrin.memory.config import Memory as MemoryConfig
from syrin.memory.config import MemoryEntry
from syrin.model import Model
from syrin.observability import (
    ConsoleExporter,
    SemanticAttributes,
    SpanKind,
    SpanStatus,
    get_tracer,
)
from syrin.output import Output
from syrin.providers.base import Provider
from syrin.ratelimit import (
    APIRateLimit,
    RateLimitManager,
    RateLimitStats,
    create_rate_limit_manager,
)
from syrin.response import (
    AgentReport,
    Response,
    StreamChunk,
    StructuredOutput,
    TraceStep,
)
from syrin.types import CostInfo, Message, ModelConfig, ProviderResponse, TokenUsage, ToolSpec

DEFAULT_MAX_TOOL_ITERATIONS = 10
_UNSET: Any = object()
_log = logging.getLogger(__name__)

_agent_loop: asyncio.AbstractEventLoop | None = None


def _get_agent_loop() -> asyncio.AbstractEventLoop:
    """Get or create a persistent event loop for the agent."""
    global _agent_loop
    if _agent_loop is None or _agent_loop.is_closed():
        _agent_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_agent_loop)
        _agent_loop.set_exception_handler(lambda _loop, _ctx: None)
    return _agent_loop


def _merge_class_attrs(mro: tuple[type, ...], name: str, merge: bool) -> Any:
    """From MRO: for 'merge' (e.g. tools) concatenate lists; else first defined."""
    if merge:
        out: list[Any] = []
        for cls in mro:
            if cls is object:
                continue
            val = cls.__dict__.get(name, _UNSET)
            if val is not _UNSET and val is not None:
                out.extend(val) if isinstance(val, list) else out.append(val)
        return out
    for cls in mro:
        if cls is object:
            continue
        if name in cls.__dict__:
            return cls.__dict__[name]
    return _UNSET


def _get_provider(provider_name: str) -> Provider:
    """Return the Provider instance for the given provider name."""
    if provider_name == "anthropic":
        from syrin.providers.anthropic import AnthropicProvider

        return AnthropicProvider()
    if provider_name == "openai":
        from syrin.providers.openai import OpenAIProvider

        return OpenAIProvider()
    if provider_name in ("ollama", "litellm"):
        from syrin.providers.litellm import LiteLLMProvider

        return LiteLLMProvider()
    from syrin.providers.litellm import LiteLLMProvider

    return LiteLLMProvider()


class Agent:
    """
    Base agent: model, system_prompt, optional tools, optional budget.
    Subclasses can set class-level model, system_prompt, tools, budget; child
    overrides parent for prompt/model/budget; tools are merged along MRO.
    response(input) runs the completion and a tool-call loop. When budget is set,
    cost is tracked and thresholds (e.g. switch model, stop) are applied.
    """

    _Syrin_default_model: Model | ModelConfig | None = None
    _Syrin_default_system_prompt: str = ""
    _Syrin_default_tools: list[ToolSpec] = []
    _Syrin_default_budget: Budget | None = None
    _Syrin_default_guardrails: list[Guardrail] = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        mro = cls.__mro__
        default_model = _merge_class_attrs(mro, "model", merge=False)
        default_prompt = _merge_class_attrs(mro, "system_prompt", merge=False)
        default_tools = _merge_class_attrs(mro, "tools", merge=True)
        default_budget = _merge_class_attrs(mro, "budget", merge=False)
        default_guardrails = _merge_class_attrs(mro, "guardrails", merge=True)
        cls._Syrin_default_model = default_model if default_model is not _UNSET else None
        cls._Syrin_default_system_prompt = default_prompt if default_prompt is not _UNSET else ""
        cls._Syrin_default_tools = list(default_tools) if default_tools is not _UNSET else []
        cls._Syrin_default_budget = default_budget if default_budget is not _UNSET else None
        cls._Syrin_default_guardrails = (
            list(default_guardrails) if default_guardrails is not _UNSET else []
        )

    def __init__(
        self,
        model: Model | ModelConfig | None = _UNSET,
        system_prompt: str | None = _UNSET,
        tools: list[ToolSpec] | None = _UNSET,
        budget: Budget | None = _UNSET,
        *,
        output: Output | None = None,
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
        budget_store: BudgetStore | None = None,
        budget_store_key: str = "default",
        memory: ConversationMemory | MemoryConfig | None = None,
        loop_strategy: LoopStrategy = LoopStrategy.REACT,
        loop: Loop | type[Loop] | None = None,
        guardrails: list[Guardrail] | GuardrailChain | None = _UNSET,
        context: Context | DefaultContextManager | None = None,
        rate_limit: APIRateLimit | RateLimitManager | None = None,
        checkpoint: CheckpointConfig | Checkpointer | None = None,
        debug: bool = False,
        tracer: Any = None,
    ) -> None:
        cls = self.__class__
        if model is _UNSET:
            model = getattr(cls, "_Syrin_default_model", None)
        if system_prompt is _UNSET:
            system_prompt = getattr(cls, "_Syrin_default_system_prompt", "") or ""
        if tools is _UNSET:
            tools = getattr(cls, "_Syrin_default_tools", None) or []
        if budget is _UNSET:
            budget = getattr(cls, "_Syrin_default_budget", None)
        if guardrails is _UNSET:
            guardrails = getattr(cls, "_Syrin_default_guardrails", None) or []
        if model is None:
            raise TypeError("Agent requires model (pass explicitly or set class-level model)")
        if isinstance(model, Model):
            self._model: Model | None = model
            self._model_config = model.to_config()
        else:
            self._model = None
            self._model_config = model

        # Handle output configuration
        self._output: Output | None = output
        if output is not None and self._model_config is not None and output.type is not None:
            self._model_config.output = output.type

        self._system_prompt = system_prompt or ""
        self._tools = list(tools) if tools else []
        self._max_tool_iterations = max_tool_iterations
        self._budget = budget
        self._budget_store = budget_store
        self._budget_store_key = budget_store_key
        self._conversation_memory: ConversationMemory | None = None
        self._persistent_memory: MemoryConfig | None = None
        self._memory_backend: InMemoryBackend | None = None
        self._parent_agent: Agent | None = None
        self._provider: Provider

        if memory is None:
            # Default: enable persistent memory with sensible defaults
            self._persistent_memory = MemoryConfig(
                types=[MemoryType.CORE, MemoryType.EPISODIC],
                top_k=10,
            )
            self._memory_backend = get_backend(MemoryBackend.MEMORY)
        elif isinstance(memory, MemoryConfig):
            self._persistent_memory = memory
            self._memory_backend = get_backend(memory.backend, path=memory.path)
        else:
            self._conversation_memory = memory
        if budget and budget_store:
            loaded = budget_store.load(budget_store_key)
            self._budget_tracker = loaded if loaded is not None else BudgetTracker()
        else:
            self._budget_tracker = BudgetTracker()
        self._provider = _get_provider(self._model_config.provider)
        self._agent_name = self.__class__.__name__
        self._loop_strategy = loop_strategy

        loop_instance: Loop
        if loop is not None:
            if isinstance(loop, type) and hasattr(loop, "run") and callable(loop.run):
                loop_instance = loop()
            elif hasattr(loop, "run") and callable(loop.run):
                loop_instance = loop  # type: ignore[assignment]
            else:
                loop_instance = ReactLoop()
        else:
            loop_instance = ReactLoop()
        self._loop = loop_instance

        # Guardrails setup
        if guardrails is None or (isinstance(guardrails, list) and len(guardrails) == 0):
            self._guardrails = GuardrailChain()
        elif isinstance(guardrails, GuardrailChain):
            self._guardrails = guardrails
        else:
            self._guardrails = GuardrailChain(list(guardrails))

        # Observability setup (before context to pass tracer)
        self._debug = debug
        self._tracer = tracer or get_tracer()
        if debug and not any(isinstance(e, ConsoleExporter) for e in self._tracer._exporters):
            self._tracer.add_exporter(ConsoleExporter())
        if debug:
            self._tracer.set_debug_mode(True)

        # Context management setup
        if context is None:
            self._context = DefaultContextManager(Context())
        elif isinstance(context, Context):
            self._context = DefaultContextManager(context)
        else:
            self._context = context

        # Connect context to events and observability
        if hasattr(self._context, "set_emit_fn"):
            from typing import cast

            self._context.set_emit_fn(cast(Any, self._emit_event))
        if hasattr(self._context, "set_tracer"):
            self._context.set_tracer(self._tracer)

        # Rate limit management setup
        if rate_limit is None:
            self._rate_limit_manager: RateLimitManager | None = None
        elif isinstance(rate_limit, RateLimitManager):
            self._rate_limit_manager = rate_limit
        else:
            self._rate_limit_manager = cast(RateLimitManager, create_rate_limit_manager(rate_limit))

        # Validation settings from Output config
        if self._output is not None:
            self._validation_retries = self._output.validation_retries
            self._validation_context = self._output.context
            self._output_validator = self._output.validator
        else:
            self._validation_retries = 3
            self._validation_context = {}
            self._output_validator = None

        # Connect rate limit to events and observability
        if self._rate_limit_manager and hasattr(self._rate_limit_manager, "set_emit_fn"):
            self._rate_limit_manager.set_emit_fn(self._emit_event)
        if self._rate_limit_manager and hasattr(self._rate_limit_manager, "set_tracer"):
            self._rate_limit_manager.set_tracer(self._tracer)

        self.events = Events(self._emit_event)

        # Initialize run report for tracking metrics across a response() call
        self._run_report: AgentReport = AgentReport()

        # Checkpoint setup
        if checkpoint is None:
            self._checkpoint_config: CheckpointConfig | None = None
            self._checkpointer: Checkpointer | None = None
        elif isinstance(checkpoint, Checkpointer):
            self._checkpoint_config = None
            self._checkpointer = checkpoint
        else:
            self._checkpoint_config = checkpoint
            if checkpoint.enabled:
                from syrin.checkpoint import get_checkpoint_backend

                kwargs = {}
                if checkpoint.path is not None:
                    kwargs["path"] = checkpoint.path
                backend = get_checkpoint_backend(checkpoint.storage, **kwargs)
                self._checkpointer = Checkpointer(
                    max_checkpoints=checkpoint.max_checkpoints, backend=backend
                )
            else:
                self._checkpointer = None

    def save_checkpoint(self, name: str | None = None, reason: str | None = None) -> str | None:
        """Save a checkpoint of the agent's current state.

        Args:
            name: Optional name for the checkpoint. If not provided, uses the agent name.
            reason: Optional reason for the checkpoint (e.g., 'step', 'tool', 'budget', 'error').

        Returns:
            The checkpoint ID, or None if checkpointing is not enabled.
        """
        if self._checkpointer is None:
            return None

        agent_name = name or self._agent_name
        state = {
            "iteration": self._run_report.context.final_tokens,
            "messages": [],  # Could include conversation history
            "memory_data": {},
            "budget_state": {"remaining": self._budget.remaining, "spent": self._budget._spent}
            if self._budget
            else None,
            "checkpoint_reason": reason,
        }

        checkpoint_id = self._checkpointer.save(agent_name, state)
        self._run_report.checkpoints.saves += 1
        self._emit_event(Hook.CHECKPOINT_SAVE, EventContext(checkpoint_id=checkpoint_id))
        return checkpoint_id

    def _maybe_checkpoint(self, reason: str) -> None:
        """Automatically checkpoint based on trigger configuration.

        Args:
            reason: The reason for checkpointing ('step', 'tool', 'error', 'budget')
        """
        if self._checkpointer is None or self._checkpoint_config is None:
            return

        trigger = self._checkpoint_config.trigger.value if self._checkpoint_config else None
        if trigger == "manual":
            return

        should_checkpoint = (trigger == "step" and reason in ("step", "tool")) or (
            trigger == reason
        )
        if should_checkpoint:
            self.save_checkpoint(reason=reason)

    def load_checkpoint(self, checkpoint_id: str) -> bool:
        """Load a checkpoint to restore agent state.

        Args:
            checkpoint_id: The checkpoint ID to load.

        Returns:
            True if checkpoint was loaded successfully, False otherwise.
        """
        if self._checkpointer is None:
            return False

        state = self._checkpointer.load(checkpoint_id)
        if state is None:
            return False

        self._run_report.checkpoints.loads += 1
        self._emit_event(Hook.CHECKPOINT_LOAD, EventContext(checkpoint_id=checkpoint_id))
        return True

    def list_checkpoints(self, name: str | None = None) -> list[str]:
        """List available checkpoints for this agent.

        Args:
            name: Optional name to filter checkpoints. If not provided, uses the agent name.

        Returns:
            List of checkpoint IDs.
        """
        if self._checkpointer is None:
            return []

        agent_name = name or self._agent_name
        return self._checkpointer.list_checkpoints(agent_name)

    def get_checkpoint_report(self) -> AgentReport:
        """Get the checkpoint operations report for the last response() call.

        Returns:
            AgentReport with checkpoint statistics.
        """
        return self._run_report

    def _emit_event(self, hook: Hook, ctx: EventContext) -> None:
        """Internal: trigger a hook through the events system.

        Args:
            hook: Hook enum value (e.g., Hook.AGENT_RUN_START)
            ctx: EventContext with hook-specific data
        """
        # Print event to console when debug=True
        if self._debug:
            self._print_event(hook.value, ctx)

        # Trigger before/on/after handlers
        self.events._trigger_before(hook, ctx)
        self.events._trigger(hook, ctx)
        self.events._trigger_after(hook, ctx)

    def _print_event(self, event: str, ctx: EventContext) -> None:
        """Print event to console when debug=True."""
        import sys
        from datetime import datetime

        # Check if terminal supports colors
        is_tty = sys.stdout.isatty()

        # Colors
        RESET = "\033[0m" if is_tty else ""
        GREEN = "\033[92m" if is_tty else ""
        BLUE = "\033[94m" if is_tty else ""
        YELLOW = "\033[93m" if is_tty else ""
        CYAN = "\033[96m" if is_tty else ""
        RED = "\033[91m" if is_tty else ""

        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # Determine event type for coloring
        event_str = str(event)
        if "start" in event_str or "init" in event_str:
            color = GREEN
            symbol = "▶"
        elif "end" in event_str or "complete" in event_str:
            color = BLUE
            symbol = "✓"
        elif "tool" in event_str:
            color = YELLOW
            symbol = "🔧"
        elif "llm" in event_str or "request" in event_str:
            color = CYAN
            symbol = "💬"
        elif "error" in event_str:
            color = RED
            symbol = "✗"
        else:
            color = ""
            symbol = "•"

        print(f"{color}{symbol} {timestamp} {event}{RESET}")

        # Print key context info
        indent = "     "

        # Input/task
        if "input" in ctx:
            task = ctx["input"]
            if isinstance(task, str) and len(task) > 60:
                task = task[:57] + "..."
            print(f"{indent}Input: {task}")

        # Model
        if "model" in ctx:
            print(f"{indent}Model: {ctx['model']}")

        # Cost
        if "cost" in ctx and ctx["cost"] is not None:
            cost_val = float(ctx["cost"])
            if cost_val > 0:
                print(f"{indent}Cost: ${cost_val:.6f}")

        # Tokens
        if "tokens" in ctx and ctx["tokens"] is not None:
            print(f"{indent}Tokens: {ctx['tokens']}")

        # Iteration
        if "iteration" in ctx:
            print(f"{indent}Iteration: {ctx['iteration']}")

        # Tool name
        if "name" in ctx:
            print(f"{indent}Tool: {ctx['name']}")

        # Error
        if "error" in ctx:
            print(f"{indent}{RED}Error: {ctx['error']}{RESET}")

        # Duration
        if "duration" in ctx and ctx["duration"] is not None:
            duration_ms = float(ctx["duration"]) * 1000
            print(f"{indent}Duration: {duration_ms:.2f}ms")

        print()

    def switch_model(self, model: Model | ModelConfig) -> None:
        """Change the active model mid-execution (e.g. when a budget threshold triggers)."""
        if isinstance(model, Model):
            self._model = model
            self._model_config = model.to_config()
        else:
            self._model = None
            self._model_config = model
        self._provider = _get_provider(self._model_config.provider)

    @property
    def budget_summary(self) -> dict[str, Any]:
        """Current budget state (run cost, hourly/daily/weekly/monthly, entries count)."""
        return self._budget_tracker.get_summary().to_dict()

    @property
    def memory(self) -> ConversationMemory | MemoryConfig | None:
        """Memory configuration (conversation or persistent)."""
        return self._persistent_memory or self._conversation_memory

    @property
    def conversation_memory(self) -> ConversationMemory | None:
        """Conversation memory for current session."""
        return self._conversation_memory

    @property
    def persistent_memory(self) -> MemoryConfig | None:
        """Persistent memory configuration."""
        return self._persistent_memory

    @property
    def context(self) -> Context:
        """Context configuration for this agent."""
        if hasattr(self._context, "context"):
            return self._context.context
        return Context()

    @property
    def context_stats(self) -> ContextStats:
        """Context stats from last call."""
        if hasattr(self._context, "stats"):
            return self._context.stats
        return ContextStats()

    @property
    def _context_manager(self) -> DefaultContextManager:
        """Internal context manager."""
        return self._context

    @property
    def rate_limit(self) -> APIRateLimit | None:
        """Rate limit configuration for this agent."""
        if self._rate_limit_manager and hasattr(self._rate_limit_manager, "config"):
            return cast(APIRateLimit, self._rate_limit_manager.config)
        return None

    @property
    def rate_limit_stats(self) -> RateLimitStats:
        """Rate limit stats from last call."""
        if self._rate_limit_manager and hasattr(self._rate_limit_manager, "stats"):
            return cast(RateLimitStats, self._rate_limit_manager.stats)
        return RateLimitStats()

    @property
    def _rate_limit_manager_internal(self) -> RateLimitManager | None:
        """Internal rate limit manager."""
        return self._rate_limit_manager

    @property
    def report(self) -> AgentReport:
        """Get the aggregated report of all agent operations.

        Includes guardrail results, context usage, memory operations,
        budget status, token usage, output validation, rate limits, and checkpoints.

        Example:
            agent.response("Hello")
            agent.report.guardrail       # GuardrailReport
            agent.report.context         # ContextReport
            agent.report.memory         # MemoryReport
            agent.report.budget         # BudgetStatus
            agent.report.tokens         # TokenReport
            agent.report.output         # OutputReport
            agent.report.ratelimits     # RateLimitReport
            agent.report.checkpoints    # CheckpointReport
        """
        return self._run_report

    def remember(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        importance: float = 1.0,
        **metadata: Any,
    ) -> str:
        """Store a memory in persistent storage.

        Args:
            content: The memory content to store
            memory_type: Type of memory (CORE, EPISODIC, SEMANTIC, PROCEDURAL)
            importance: Importance score (0.0-1.0)
            **metadata: Additional metadata to store with the memory

        Returns:
            Memory ID
        """
        import uuid

        if self._memory_backend is None:
            raise RuntimeError("No persistent memory configured")

        with self._tracer.span(
            "memory.store",
            kind=SpanKind.MEMORY,
            attributes={
                SemanticAttributes.MEMORY_OPERATION: "store",
                SemanticAttributes.MEMORY_TYPE: memory_type.value,
            },
        ) as mem_span:
            entry = MemoryEntry(
                id=str(uuid.uuid4()),
                content=content,
                type=memory_type,
                importance=importance,
                metadata=metadata,
            )
            self._memory_backend.add(entry)
            mem_span.set_attribute("memory.id", entry.id)
            # Track in report
            self._run_report.memory.stores += 1
            # Emit hook
            self._emit_event(
                Hook.MEMORY_STORE,
                EventContext(
                    memory_id=entry.id,
                    content=content[:100],  # Truncate for event
                    memory_type=memory_type.value,
                    importance=importance,
                ),
            )
            return entry.id

    def recall(
        self,
        query: str | None = None,
        memory_type: MemoryType | None = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Retrieve memories from persistent storage.

        Args:
            query: Optional search query
            memory_type: Filter by memory type
            limit: Maximum number of memories to return

        Returns:
            List of MemoryEntry objects
        """
        if self._memory_backend is None:
            raise RuntimeError("No persistent memory configured")

        with self._tracer.span(
            "memory.recall",
            kind=SpanKind.MEMORY,
            attributes={
                SemanticAttributes.MEMORY_OPERATION: "recall",
                SemanticAttributes.MEMORY_TYPE: memory_type.value if memory_type else "all",
                SemanticAttributes.MEMORY_QUERY: query or "",
            },
        ) as mem_span:
            if query:
                results = self._memory_backend.search(query, memory_type, limit)
            else:
                results = self._memory_backend.list(memory_type, limit=limit)

            mem_span.set_attribute(
                SemanticAttributes.MEMORY_RESULTS_COUNT,
                len(results),
            )
            # Track in report
            self._run_report.memory.recalls += 1
            # Emit hook
            self._emit_event(
                Hook.MEMORY_RECALL,
                EventContext(
                    query=query,
                    memory_type=memory_type.value if memory_type else "all",
                    results_count=len(results),
                    limit=limit,
                ),
            )
            return results

    def forget(
        self,
        memory_id: str | None = None,
        query: str | None = None,
        memory_type: MemoryType | None = None,
    ) -> int:
        """Remove memories from persistent storage.

        Args:
            memory_id: Specific memory ID to delete
            query: Delete all memories matching query
            memory_type: Delete all memories of this type

        Returns:
            Number of memories deleted
        """
        if self._memory_backend is None:
            raise RuntimeError("No persistent memory configured")

        with self._tracer.span(
            "memory.forget",
            kind=SpanKind.MEMORY,
            attributes={
                SemanticAttributes.MEMORY_OPERATION: "forget",
                SemanticAttributes.MEMORY_TYPE: memory_type.value if memory_type else "all",
                SemanticAttributes.MEMORY_QUERY: query or "",
            },
        ) as mem_span:
            deleted = 0
            if memory_id:
                self._memory_backend.delete(memory_id)
                deleted = 1
            elif query or memory_type:
                memories = self._memory_backend.list(memory_type)
                for mem in memories:
                    if query and query.lower() in mem.content.lower():
                        self._memory_backend.delete(mem.id)
                        deleted += 1

            mem_span.set_attribute("memory.deleted_count", deleted)
            # Track in report
            self._run_report.memory.forgets += deleted
            # Emit hook
            self._emit_event(
                Hook.MEMORY_FORGET,
                EventContext(
                    memory_id=memory_id,
                    query=query,
                    memory_type=memory_type.value if memory_type else "all",
                    deleted_count=deleted,
                ),
            )
            return deleted

    def _run_guardrails(
        self,
        text: str,
        stage: GuardrailStage,
    ) -> GuardrailResult:
        """Run guardrails on text with observability."""
        if len(self._guardrails) == 0:
            return GuardrailResult(passed=True)

        # Emit start hook
        self._emit_event(
            Hook.GUARDRAIL_INPUT if stage == GuardrailStage.INPUT else Hook.GUARDRAIL_OUTPUT,
            EventContext(
                text=text[:200],  # Truncate for event
                stage=stage.value,
                guardrail_count=len(self._guardrails),
            ),
        )

        with self._tracer.span(
            f"guardrails.{stage.value}",
            kind=SpanKind.GUARDRAIL,
            attributes={
                SemanticAttributes.GUARDRAIL_STAGE: stage.value,
            },
        ) as guardrail_span:
            result = self._guardrails.check(text, stage)

            guardrail_span.set_attribute(
                SemanticAttributes.GUARDRAIL_PASSED,
                result.passed,
            )

            if not result.passed:
                guardrail_span.set_attribute(
                    SemanticAttributes.GUARDRAIL_VIOLATION,
                    result.reason or "Unknown",
                )
                guardrail_span.set_status(SpanStatus.ERROR, result.reason)

                # Emit blocked hook
                self._emit_event(
                    Hook.GUARDRAIL_BLOCKED,
                    EventContext(
                        stage=stage.value,
                        reason=result.reason,
                        guardrail_names=[g.name for g in self._guardrails],
                    ),
                )

                # Store in report
                guardrail_names = [g.name for g in self._guardrails]
                if stage == GuardrailStage.INPUT:
                    self._run_report.guardrail.input_passed = False
                    self._run_report.guardrail.input_reason = result.reason
                    self._run_report.guardrail.input_guardrails = guardrail_names
                    self._run_report.guardrail.blocked = True
                    self._run_report.guardrail.blocked_stage = stage.value
                else:
                    self._run_report.guardrail.output_passed = False
                    self._run_report.guardrail.output_reason = result.reason
                    self._run_report.guardrail.output_guardrails = guardrail_names
                    self._run_report.guardrail.blocked = True
                    self._run_report.guardrail.blocked_stage = stage.value
            else:
                # Store guardrail names that passed
                guardrail_names = [g.name for g in self._guardrails]
                if stage == GuardrailStage.INPUT:
                    self._run_report.guardrail.input_passed = True
                    self._run_report.guardrail.input_guardrails = guardrail_names
                else:
                    self._run_report.guardrail.output_passed = True
                    self._run_report.guardrail.output_guardrails = guardrail_names

            return result

    def handoff(
        self,
        target_agent: type[Agent],
        task: str,
        *,
        transfer_context: bool = True,
        transfer_budget: bool = False,
    ) -> Response[str]:
        """Hand off control to another agent.

        Args:
            target_agent: The agent class to hand off to
            task: The task description for the target agent
            transfer_context: Whether to transfer persistent memories to target
            transfer_budget: Whether to transfer remaining budget to target

        Returns:
            Response from the target agent
        """
        target = target_agent()

        if transfer_budget and self._budget:
            target._budget = self._budget
            target._budget_tracker = self._budget_tracker

        if transfer_context:
            if self._memory_backend is None:
                _log.warning(
                    "handoff: transfer_context=True but source agent has no memory backend. "
                    "Did you set memory=Memory() on the source agent?"
                )
            else:
                memories = self._memory_backend.list()
                if memories:
                    if target._memory_backend is None:
                        target._persistent_memory = MemoryConfig(top_k=10, relevance_threshold=0.7)
                        target._memory_backend = get_backend(
                            MemoryConfig(top_k=10, relevance_threshold=0.7).backend
                        )
                    for mem in memories:
                        target.remember(
                            mem.content, memory_type=mem.type, importance=mem.importance
                        )
                    _log.debug("handoff: transferred %d memories to target agent", len(memories))

        return target.response(task)

    def spawn(
        self,
        agent_class: type[Agent],
        task: str | None = None,
        *,
        budget: Budget | None = None,
        max_children: int | None = None,
    ) -> Agent | Response[str]:
        """Spawn a sub-agent to execute a task.

        Budget Inheritance Rules:
        1. If parent has shared budget (shared=True), child borrows from parent's budget
        2. If child has its own budget (pocket money), it must not exceed parent's remaining
        3. Child's spending is tracked and deducted from parent's budget

        Args:
            agent_class: The agent class to spawn
            task: Optional task to run immediately
            budget: Optional budget for the sub-agent (pocket money)
            max_children: Maximum number of child agents (enforces limit)

        Returns:
            If task provided, returns Response; otherwise returns the spawned agent instance
        """
        use_instance_limit = max_children is None
        _max_children = getattr(self, "_max_children", 10) if use_instance_limit else max_children

        current_children = getattr(self, "_child_count", 0)

        if _max_children and current_children >= _max_children:
            raise RuntimeError(f"Cannot spawn: max children ({_max_children}) reached")

        if not hasattr(self, "_child_count"):
            self._child_count = 0
        if use_instance_limit:
            self._child_count += 1

        agent_kwargs: dict[str, Any] = {}

        # Handle budget inheritance with borrow mechanism
        if budget is not None:
            # Child has pocket money - validate it doesn't exceed parent's budget
            if self._budget is not None and self._budget.run is not None:
                parent_remaining = self._budget.remaining
                if (
                    parent_remaining is not None
                    and budget.run is not None
                    and budget.run > parent_remaining
                ):
                    raise ValueError(
                        f"Child budget (${budget.run:.2f}) cannot exceed parent's "
                        f"remaining budget (${parent_remaining:.2f}). "
                        "Pocket money must be less than or equal to parent's available funds."
                    )
            # Use child's pocket money
            agent_kwargs["budget"] = budget
        elif self._budget is not None and self._budget.shared:
            # Parent has shared budget - child borrows from parent
            # Create a borrowed budget that shares the parent's budget tracker
            borrowed_budget = Budget(
                run=self._budget.remaining,  # Child can spend up to parent's remaining
                per=self._budget.per,
                on_exceeded=self._budget.on_exceeded,
                thresholds=self._budget.thresholds,
                shared=True,  # Child can also share with its children
            )
            borrowed_budget._parent_budget = self._budget  # Track parent for updates
            agent_kwargs["budget"] = borrowed_budget

        child_agent = agent_class(**agent_kwargs)

        # Track parent for budget updates
        if hasattr(agent_kwargs.get("budget"), "_parent_budget"):
            child_agent._parent_agent = self

        if task:
            result = child_agent.response(task)
            # Update parent's budget with child's spending (borrow mechanism)
            self._update_parent_budget(result.cost)
            return result

        return child_agent

    def _update_parent_budget(self, cost: float) -> None:
        """Update parent's budget when child spends (borrow mechanism)."""
        if self._budget is not None:
            # Record the cost in parent's budget tracker
            from syrin.types import CostInfo

            cost_info = CostInfo(
                cost_usd=cost, model_name=getattr(self._model, "model_id", "unknown")
            )
            self._budget_tracker.record(cost_info)
            # Update budget's spent tracking
            current_spent = getattr(self._budget, "_spent", 0.0)
            self._budget._set_spent(current_spent + cost)

    def spawn_parallel(
        self,
        agents: list[tuple[type[Agent], str]],
    ) -> list[Response[str]]:
        """Spawn multiple sub-agents in parallel.

        Args:
            agents: List of (agent_class, task) tuples

        Returns:
            List of responses from all spawned agents
        """
        import asyncio

        async def spawn_one(
            agent_class: type[Agent],
            task: str,
        ) -> Response[str]:
            child = agent_class()
            return await child.arun(task)

        async def run_all() -> list[Response[str]]:
            tasks = [spawn_one(ac, t) for ac, t in agents]
            return await asyncio.gather(*tasks)

        loop = _get_agent_loop()
        return loop.run_until_complete(run_all())

    def _build_messages(self, user_input: str) -> list[Message]:
        messages: list[Message] = []

        # Build system prompt (memory context is handled by context manager)
        system_content = self._system_prompt or ""
        memory_context = ""

        # Auto-inject relevant persistent memories - pass to context manager
        if self._memory_backend is not None:
            with self._tracer.span(
                "memory.recall",
                kind=SpanKind.MEMORY,
                attributes={
                    SemanticAttributes.MEMORY_OPERATION: "recall",
                },
            ) as mem_span:
                # Search for relevant memories based on user input
                top_k = self._persistent_memory.top_k if self._persistent_memory else 10
                memories = self._memory_backend.search(user_input, None, top_k)

                mem_span.set_attribute(
                    SemanticAttributes.MEMORY_RESULTS_COUNT,
                    len(memories),
                )

                if memories:
                    # Format memories for context manager
                    memory_context = "## Relevant Memories:\n"
                    for mem in memories:
                        memory_context += f"- [{mem.type.value}] {mem.content}\n"

        # Add system prompt if present
        if system_content:
            messages.append(Message(role=MessageRole.SYSTEM, content=system_content))

        # Add conversation memory
        if self._conversation_memory is not None:
            # Memory recall span
            with self._tracer.span(
                "memory.recall",
                kind=SpanKind.MEMORY,
                attributes={
                    SemanticAttributes.MEMORY_OPERATION: "recall",
                },
            ) as mem_span:
                mem_messages = self._conversation_memory.get_messages()
                mem_span.set_attribute(
                    SemanticAttributes.MEMORY_RESULTS_COUNT,
                    len(mem_messages),
                )
                messages.extend(mem_messages)

        messages.append(Message(role=MessageRole.USER, content=user_input))

        # Apply context management (compaction if needed)
        msg_dicts = []
        for msg in messages:
            msg_dicts.append(
                {
                    "role": msg.role.value if hasattr(msg.role, "value") else str(msg.role),
                    "content": msg.content,
                }
            )

        # Get tool specs
        tool_dicts = []
        for tool in self._tools:
            if hasattr(tool, "to_tool_spec"):
                tool_dicts.append(tool.to_tool_spec())
            elif isinstance(tool, dict):
                tool_dicts.append(tool)

        # Use context manager to prepare context
        model_for_context = self._model if self._model is not None else None

        # Get budget - handle both DefaultContextManager and custom managers
        if hasattr(self._context, "context"):
            budget = self._context.context.get_budget(model_for_context)
        else:
            # Custom context manager - create default budget
            budget = Context().get_budget(model_for_context)

        payload = self._context.prepare(
            messages=msg_dicts,
            system_prompt=self._system_prompt,
            tools=tool_dicts,
            memory_context=memory_context,
            budget=budget,
        )

        # Rebuild messages from payload
        final_messages = []
        for msg_dict in payload.messages:
            msg_data: dict[str, Any] = msg_dict
            role = msg_data.get("role", "user")
            if hasattr(MessageRole, role.upper()):
                final_messages.append(
                    Message(role=MessageRole(role), content=msg_data.get("content", ""))
                )

        return final_messages

    def _build_output(
        self,
        content: str,
        validation_retries: int = 3,
        validation_context: dict[str, Any] | None = None,
        validator: Any = None,
    ) -> StructuredOutput | None:
        """Build structured output from response content with validation.

        Args:
            content: Raw response content from LLM
            validation_retries: Number of validation retries
            validation_context: Context for validation
            validator: Custom output validator
        """
        from syrin.validation import ValidationPipeline

        output_type = getattr(self._model_config, "output", None)
        if output_type is None:
            return None

        # Get the pydantic model from output_type
        pydantic_model = None

        # Check if output_type has _structured_pydantic (from @structured decorator)
        if hasattr(output_type, "_structured_pydantic"):
            pydantic_model = output_type._structured_pydantic

        # Also check if it's a direct BaseModel subclass
        if pydantic_model is None:
            try:
                from pydantic import BaseModel

                if isinstance(output_type, type) and issubclass(output_type, BaseModel):
                    pydantic_model = output_type
            except Exception:
                pass

        if pydantic_model is None:
            # Not a Pydantic model, just return raw
            import json

            try:
                data = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                return StructuredOutput(raw=content, parsed=None, _data={})
            return StructuredOutput(
                raw=content, parsed=data, _data=data if isinstance(data, dict) else {}
            )

        # Use ValidationPipeline for full validation with retries
        context = validation_context or {}

        # Get emit function for hooks
        emit_fn = getattr(self, "_emit_event", None)

        pipeline = ValidationPipeline(
            output_type=pydantic_model,
            max_retries=validation_retries,
            context=context,
            validator=validator,
            emit_fn=emit_fn,
        )

        parsed, attempts, error = pipeline.validate(content)

        # Track in report
        self._run_report.output.validated = True
        self._run_report.output.attempts = len(attempts)
        self._run_report.output.is_valid = error is None and parsed is not None
        self._run_report.output.final_error = str(error) if error else None

        return StructuredOutput(
            raw=content,
            parsed=parsed,
            _data=parsed.model_dump() if parsed else {},
            validation_attempts=attempts,
            final_error=error,
        )

    def _execute_tool(self, name: str, arguments: dict[str, Any]) -> str:
        for spec in self._tools:
            if spec.name == name:
                try:
                    result = spec.func(**arguments)
                    return str(result) if result is not None else ""
                except Exception as e:
                    raise ToolExecutionError(f"Tool {name!r} failed: {e}") from e
        raise ToolExecutionError(f"Unknown tool: {name!r}")

    async def execute_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool. Used by custom loops."""
        return self._execute_tool(name, arguments)

    def _check_and_apply_budget(self) -> None:
        """Raise if budget exceeded; apply threshold actions (switch, warn). Stop raises."""
        if self._budget is None:
            return
        status = self._budget_tracker.check_budget(self._budget)
        if status == BudgetStatus.EXCEEDED:
            run_cost = self._budget_tracker.current_run_cost
            limit = self._budget.run or 0
            on_exceeded = self._budget.on_exceeded
            if on_exceeded == OnExceeded.ERROR:
                raise BudgetExceededError(
                    f"Budget exceeded: run cost ${run_cost:.4f} >= ${limit:.4f}",
                    current_cost=run_cost,
                    limit=limit,
                    budget_type="run",
                )
            if on_exceeded == OnExceeded.WARN:
                _log.warning(
                    "Budget exceeded: run cost %.4f >= %.4f",
                    run_cost,
                    limit,
                )
                return
            if on_exceeded == OnExceeded.STOP:
                raise BudgetThresholdError(
                    f"Budget exceeded (stop): ${run_cost:.4f} >= ${limit:.4f}",
                    threshold_percent=100.0,
                    action_taken="stop",
                )

    def _check_and_apply_rate_limit(self) -> None:
        """Check rate limits and apply threshold actions (switch model, wait, warn, stop).

        Called before each LLM request. This is the main integration point that makes
        rate limiting work automatically during agent execution.
        """
        if self._rate_limit_manager is None:
            return

        manager = self._rate_limit_manager

        allowed, reason = manager.check()

        # Track in report
        self._run_report.ratelimits.checks += 1
        if not allowed:
            self._run_report.ratelimits.exceeded = True
            _log.error("Rate limit exceeded: %s", reason)
            raise RuntimeError(f"Rate limit exceeded: {reason}")

        triggered_action = manager.get_triggered_action()
        if triggered_action is None:
            return

        action = triggered_action.action

        if action == RateLimitAction.SWITCH_MODEL and triggered_action.switch_to_model:
            _log.warning(
                "Rate limit threshold %d%%: switching from %s to %s",
                triggered_action.at,
                self._model_config.model_id,
                triggered_action.switch_to_model,
            )
            self.switch_model(Model(triggered_action.switch_to_model))

        elif action == RateLimitAction.STOP:
            _log.error(
                "Rate limit threshold %d%%: stop",
                triggered_action.at,
            )
            raise RuntimeError(
                triggered_action.message or f"Rate limit threshold reached: {triggered_action.at}%"
            )

        elif action == RateLimitAction.ERROR:
            _log.error(
                "Rate limit threshold %d%%: error",
                triggered_action.at,
            )
            raise RuntimeError(
                triggered_action.message or f"Rate limit threshold reached: {triggered_action.at}%"
            )

        elif action == RateLimitAction.WARN:
            _log.warning(
                "Rate limit threshold %d%%: %s",
                triggered_action.at,
                triggered_action.message or "threshold reached",
            )

        elif action == RateLimitAction.WAIT:
            import asyncio

            wait_time = (
                triggered_action.wait_seconds or self._rate_limit_manager.config.wait_backoff
            )
            _log.warning(
                "Rate limit threshold %d%%: waiting %.2fs",
                triggered_action.at,
                wait_time,
            )
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(time.sleep, wait_time)
                        future.result()
                else:
                    time.sleep(wait_time)
            except RuntimeError:
                time.sleep(wait_time)

    def _record_rate_limit_usage(self, token_usage: TokenUsage) -> None:
        """Record token usage and re-check rate limits after LLM call.

        Called after each LLM request to track usage and trigger threshold checks
        for the next iteration.
        """
        if self._rate_limit_manager is None:
            return

        self._rate_limit_manager.record(tokens_used=token_usage.total_tokens)

    def _record_cost(self, token_usage: TokenUsage, model_id: str) -> None:
        """Compute cost, build CostInfo, record on tracker, then re-check thresholds."""
        pricing = getattr(self._model, "pricing", None) if self._model is not None else None
        cost_usd = calculate_cost(model_id, token_usage, pricing_override=pricing)
        cost_info = CostInfo(
            token_usage=token_usage,
            cost_usd=cost_usd,
            model_name=model_id,
        )
        self._budget_tracker.record(cost_info)
        if self._budget_store is not None:
            self._budget_store.save(self._budget_store_key, self._budget_tracker)
        self._check_and_apply_budget()

    async def _complete_async(
        self, messages: list[Message], tools: list[ToolSpec] | None
    ) -> ProviderResponse:
        """Async internal method to call provider."""
        return await self._provider.complete(
            messages=messages, model=self._model_config, tools=tools
        )

    async def complete(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> ProviderResponse:
        """Public method to call the LLM. Used by custom loops."""
        return await self._complete_async(messages, tools)

    def _run_response_async(self, user_input: str) -> Response[str]:
        """Run response synchronously using a persistent event loop."""
        loop = _get_agent_loop()
        return loop.run_until_complete(self._run_response(user_input))

    async def _run_response(self, user_input: str) -> Response[str]:
        with self._tracer.span(
            f"{self._agent_name}.response",
            kind=SpanKind.AGENT,
            attributes={
                SemanticAttributes.AGENT_NAME: self._agent_name,
                SemanticAttributes.AGENT_CLASS: self.__class__.__name__,
                "input": user_input if not self._debug else user_input[:1000],
                SemanticAttributes.LLM_MODEL: self._model_config.model_id,
                SemanticAttributes.LLM_PROVIDER: self._model_config.provider,
            },
        ) as agent_span:
            messages = self._build_messages(user_input)
            tools = self._tools if self._tools else None
            iteration = 0
            response = None
            trace: list[TraceStep] = []
            total_input = 0
            total_output = 0
            total_cost_usd = 0.0
            run_start = time.perf_counter()

            self._emit_event(
                Hook.AGENT_RUN_START,
                EventContext(
                    input=user_input,
                    model=self._model_config.model_id,
                    iteration=0,
                ),
            )

            # Input guardrails
            input_guardrail = self._run_guardrails(user_input, GuardrailStage.INPUT)
            if not input_guardrail.passed:
                return Response(
                    content="",
                    raw="",
                    cost=0.0,
                    tokens=TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0),
                    model=self._model_config.model_id,
                    duration=0.0,
                    trace=[],
                    tool_calls=[],
                    stop_reason=StopReason.GUARDRAIL,
                    budget_remaining=self._budget.remaining if self._budget else None,
                    budget_used=0.0,
                    structured=None,
                    report=self._run_report,
                )

            while iteration < self._max_tool_iterations:
                iteration += 1
                agent_span.set_attribute(SemanticAttributes.AGENT_ITERATION, iteration)
                self._check_and_apply_budget()
                self._check_and_apply_rate_limit()
                step_start = time.perf_counter()

                self._emit_event(
                    Hook.LLM_REQUEST_START,
                    EventContext(
                        messages=messages,
                        tools=[t.name for t in tools] if tools else [],
                        iteration=iteration,
                    ),
                )

                with self._tracer.span(
                    f"llm.iteration_{iteration}",
                    kind=SpanKind.LLM,
                    attributes={
                        SemanticAttributes.LLM_MODEL: self._model_config.model_id,
                        SemanticAttributes.LLM_PROVIDER: self._model_config.provider,
                        "iteration": iteration,
                    },
                ) as llm_span:
                    if self._debug:
                        llm_span.set_attribute(
                            SemanticAttributes.LLM_PROMPT,
                            json.dumps([m.model_dump() for m in messages], default=str)[:5000],
                        )

                    response = await self._provider.complete(
                        messages=messages,
                        model=self._model_config,
                        tools=tools,
                    )

                    step_ms = (time.perf_counter() - step_start) * 1000
                    u = response.token_usage
                    total_input += u.input_tokens
                    total_output += u.output_tokens
                    pricing = getattr(self._model, "pricing", None) if self._model else None
                    cost_usd = calculate_cost(
                        self._model_config.model_id, u, pricing_override=pricing
                    )
                    total_cost_usd += cost_usd

                    llm_span.set_attributes(
                        {
                            SemanticAttributes.LLM_TOKENS_INPUT: u.input_tokens,
                            SemanticAttributes.LLM_TOKENS_OUTPUT: u.output_tokens,
                            SemanticAttributes.LLM_TOKENS_TOTAL: u.total_tokens,
                            SemanticAttributes.LLM_COST: cost_usd,
                            SemanticAttributes.LLM_STOP_REASON: response.stop_reason
                            if response.stop_reason
                            else "end_turn",
                        }
                    )

                    if self._debug:
                        llm_span.set_attribute(
                            SemanticAttributes.LLM_COMPLETION,
                            response.content[:2000] if response.content else "",
                        )

                trace.append(
                    TraceStep(
                        step_type="llm",
                        timestamp=time.time(),
                        model=self._model_config.model_id,
                        tokens=u.total_tokens,
                        cost_usd=cost_usd,
                        latency_ms=step_ms,
                    )
                )

                stop_reason_val = getattr(response, "stop_reason", None)
                if stop_reason_val is None or not isinstance(stop_reason_val, StopReason):
                    stop_reason_val = StopReason.END_TURN

                self._emit_event(
                    Hook.LLM_REQUEST_END,
                    EventContext(
                        content=response.content,
                        tokens=u.total_tokens,
                        cost=cost_usd,
                        tool_calls=[tc.model_dump() for tc in response.tool_calls],
                        stop_reason=stop_reason_val.value
                        if hasattr(stop_reason_val, "value")
                        else str(stop_reason_val),
                        latency_ms=step_ms,
                        iteration=iteration,
                    ),
                )

                # Output guardrails (only on final response, not tool calls)
                if not response.tool_calls:
                    output_guardrail = self._run_guardrails(
                        response.content or "", GuardrailStage.OUTPUT
                    )
                    if not output_guardrail.passed:
                        latency_ms = (time.perf_counter() - run_start) * 1000
                        return Response(
                            content="",
                            raw="",
                            cost=round(total_cost_usd, 6),
                            tokens=TokenUsage(
                                input_tokens=total_input,
                                output_tokens=total_output,
                                total_tokens=total_input + total_output,
                            ),
                            model=self._model_config.model_id,
                            duration=latency_ms / 1000,
                            trace=trace,
                            tool_calls=[],
                            stop_reason=StopReason.GUARDRAIL,
                            budget_remaining=self._budget.remaining if self._budget else None,
                            budget_used=total_cost_usd if self._budget else None,
                            structured=None,
                            report=self._run_report,
                        )

                if self._budget is not None:
                    self._record_cost(u, self._model_config.model_id)
                if self._rate_limit_manager is not None:
                    self._record_rate_limit_usage(u)
                if not response.tool_calls:
                    latency_ms = (time.perf_counter() - run_start) * 1000
                    if self._conversation_memory is not None:
                        # Memory store span
                        with self._tracer.span(
                            "memory.store",
                            kind=SpanKind.MEMORY,
                            attributes={
                                SemanticAttributes.MEMORY_OPERATION: "store",
                                SemanticAttributes.MEMORY_RESULTS_COUNT: 2,
                            },
                        ):
                            self._conversation_memory.add(
                                Message(role=MessageRole.USER, content=user_input)
                            )
                            self._conversation_memory.add(
                                Message(role=MessageRole.ASSISTANT, content=response.content or "")
                            )
                    output = self._build_output(
                        response.content or "",
                        validation_retries=self._validation_retries,
                        validation_context=self._validation_context,
                        validator=getattr(self, "_output_validator", None),
                    )

                    self._emit_event(
                        Hook.AGENT_RUN_END,
                        EventContext(
                            content=response.content,
                            cost=total_cost_usd,
                            tokens=u.total_tokens,
                            duration=latency_ms / 1000,
                            stop_reason=stop_reason_val.value
                            if hasattr(stop_reason_val, "value")
                            else str(stop_reason_val),
                            iteration=iteration,
                        ),
                    )

                    agent_span.set_attributes(
                        {
                            "output": response.content[:500] if response.content else "",
                            "iterations": iteration,
                            SemanticAttributes.BUDGET_USED: total_cost_usd,
                            SemanticAttributes.BUDGET_REMAINING: self._budget.remaining
                            if self._budget
                            else None,
                        }
                    )
                    agent_span.set_status(SpanStatus.OK)

                    return Response(
                        content=response.content or "",
                        raw=response.content or "",
                        cost=round(total_cost_usd, 6),
                        tokens=TokenUsage(
                            input_tokens=total_input,
                            output_tokens=total_output,
                            total_tokens=total_input + total_output,
                        ),
                        model=self._model_config.model_id,
                        duration=latency_ms / 1000,
                        trace=trace,
                        tool_calls=[],
                        stop_reason=stop_reason_val,
                        budget_remaining=self._budget.remaining if self._budget else None,
                        budget_used=total_cost_usd if self._budget else None,
                        structured=output,
                        report=self._run_report,
                    )
            assert response is not None, "Provider returned None response"
            messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=response.content or "",
                    tool_calls=response.tool_calls,
                )
            )
            for tc in response.tool_calls:
                tool_start = time.perf_counter()
                with self._tracer.span(
                    f"tool.{tc.name}",
                    kind=SpanKind.TOOL,
                    attributes={
                        SemanticAttributes.TOOL_NAME: tc.name,
                        SemanticAttributes.TOOL_INPUT: json.dumps(tc.arguments, default=str)[:500],
                        "iteration": iteration,
                    },
                ) as tool_span:
                    try:
                        result = self._execute_tool(tc.name, tc.arguments)
                        tool_duration = time.perf_counter() - tool_start

                        tool_span.set_attributes(
                            {
                                SemanticAttributes.TOOL_OUTPUT: str(result)[:500],
                                SemanticAttributes.TOOL_DURATION_MS: tool_duration * 1000,
                            }
                        )
                        tool_span.set_status(SpanStatus.OK)
                    except Exception as e:
                        tool_span.record_exception(e)
                        tool_span.set_status(SpanStatus.ERROR, str(e))
                        self._emit_event(
                            Hook.TOOL_ERROR,
                            EventContext(
                                error=str(e),
                                tool_name=tc.name,
                                tool_input=tc.arguments,
                                iteration=iteration,
                            ),
                        )
                        raise

                tool_duration = time.perf_counter() - tool_start

                self._emit_event(
                    Hook.TOOL_CALL_END,
                    EventContext(
                        name=tc.name,
                        input=tc.arguments,
                        output=result,
                        duration=tool_duration,
                        iteration=iteration,
                    ),
                )

                messages.append(
                    Message(
                        role=MessageRole.TOOL,
                        content=result,
                        tool_call_id=tc.id,
                    )
                )
        latency_ms = (time.perf_counter() - run_start) * 1000
        content = response.content if response and response.content else ""
        if self._conversation_memory is not None:
            self._conversation_memory.add(Message(role=MessageRole.USER, content=user_input))
            self._conversation_memory.add(Message(role=MessageRole.ASSISTANT, content=content))
        output = self._build_output(
            content,
            validation_retries=self._validation_retries,
            validation_context=self._validation_context,
            validator=getattr(self, "_output_validator", None),
        )

        self._emit_event(
            Hook.AGENT_RUN_END,
            EventContext(
                content=content,
                cost=total_cost_usd,
                tokens=total_input + total_output,
                duration=latency_ms / 1000,
                stop_reason="max_iterations",
                iteration=iteration,
            ),
        )

        # Auto-store conversation if enabled
        if self._persistent_memory is not None and self._persistent_memory.auto_store:
            self.remember(
                content=f"User: {user_input}",
                memory_type=MemoryType.EPISODIC,
                importance=0.5,
            )
            self.remember(
                content=f"Agent: {content}",
                memory_type=MemoryType.EPISODIC,
                importance=0.5,
            )

        # Populate report with final data
        self._run_report.budget_remaining = self._budget.remaining if self._budget else None
        self._run_report.budget_used = total_cost_usd if self._budget else None
        self._run_report.tokens.input_tokens = total_input
        self._run_report.tokens.output_tokens = total_output
        self._run_report.tokens.total_tokens = total_input + total_output
        self._run_report.tokens.cost_usd = round(total_cost_usd, 6)

        return Response(
            content=content,
            raw=content,
            cost=round(total_cost_usd, 6),
            tokens=TokenUsage(
                input_tokens=total_input,
                output_tokens=total_output,
                total_tokens=total_input + total_output,
            ),
            model=self._model_config.model_id,
            duration=latency_ms / 1000,
            trace=trace,
            tool_calls=[tc.model_dump() for tc in response.tool_calls] if response else [],
            stop_reason=StopReason.END_TURN,
            budget_remaining=self._budget.remaining if self._budget else None,
            budget_used=total_cost_usd if self._budget else None,
            structured=output if output else None,
            report=self._run_report,
        )

    def _stream_response(self, user_input: str) -> Iterator[StreamChunk]:
        """Stream response chunks synchronously. Returns StreamChunk with accumulated text."""
        messages = self._build_messages(user_input)
        tools = self._tools if self._tools else None
        accumulated = ""
        total_cost = 0.0
        total_tokens = TokenUsage()

        try:
            for chunk in self._provider.stream_sync(messages, self._model_config, tools):
                content = chunk.content or ""
                accumulated += content
                total_cost += chunk.cost_usd if hasattr(chunk, "cost_usd") else 0.0
                total_tokens = TokenUsage(
                    input_tokens=total_tokens.input_tokens
                    + (chunk.token_usage.input_tokens if hasattr(chunk, "token_usage") else 0),
                    output_tokens=total_tokens.output_tokens
                    + (chunk.token_usage.output_tokens if hasattr(chunk, "token_usage") else 0),
                    total_tokens=total_tokens.total_tokens
                    + (chunk.token_usage.total_tokens if hasattr(chunk, "token_usage") else 0),
                )
                yield StreamChunk(
                    text=content,
                    accumulated_text=accumulated,
                    cost_so_far=total_cost,
                    tokens_so_far=total_tokens,
                )
        except Exception as e:
            raise ToolExecutionError(f"Streaming failed: {e}") from e

    def response(self, user_input: str) -> Response[str]:
        """
        Run completion and tool-call loop. Returns Response with content, cost, trace.
        str(response) returns response.content. When budget is set, checks limits
        and records cost; threshold actions (switch model, stop, warn) are applied.
        """
        # Reset report for new run
        self._run_report = AgentReport()
        if self._budget is not None:
            self._budget_tracker.reset_run()
        try:
            return self._run_loop_response(user_input)
        except (BudgetThresholdError, BudgetExceededError):
            # Checkpoint before re-raising budget errors
            self._maybe_checkpoint("budget")
            raise
        except Exception:
            # Checkpoint on error
            self._maybe_checkpoint("error")
            raise

    def _run_loop_response(self, user_input: str) -> Response[str]:
        """Run using the configured loop strategy with full observability."""
        from syrin.observability import SemanticAttributes, SpanKind
        from syrin.response import Response as ResponseClass
        from syrin.types import TokenUsage

        with self._tracer.span(
            f"{self._agent_name}.response",
            kind=SpanKind.AGENT,
            attributes={
                SemanticAttributes.AGENT_NAME: self._agent_name,
                SemanticAttributes.AGENT_CLASS: self.__class__.__name__,
                "input": user_input if not self._debug else user_input[:1000],
                SemanticAttributes.LLM_MODEL: self._model_config.model_id,
                SemanticAttributes.LLM_PROVIDER: self._model_config.provider,
            },
        ) as agent_span:
            # Input guardrails check
            input_guardrail = self._run_guardrails(user_input, GuardrailStage.INPUT)
            if not input_guardrail.passed:
                return ResponseClass(
                    content="",
                    raw="",
                    cost=0.0,
                    tokens=TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0),
                    model=self._model_config.model_id,
                    duration=0.0,
                    trace=[],
                    tool_calls=[],
                    stop_reason=StopReason.GUARDRAIL,
                    budget_remaining=self._budget.remaining if self._budget else None,
                    budget_used=0.0,
                    structured=None,
                    report=self._run_report,
                )

            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()

            result = loop.run_until_complete(self._loop.run(self, user_input))

            # Auto-checkpoint after step completion
            self._maybe_checkpoint("step")

            token_usage = result.token_usage
            tokens = TokenUsage(
                input_tokens=token_usage.get("input", 0),
                output_tokens=token_usage.get("output", 0),
                total_tokens=token_usage.get("total", 0),
            )

            if self._budget is not None:
                self._record_cost(tokens, self._model_config.model_id)

            agent_span.set_attribute(SemanticAttributes.LLM_TOKENS_TOTAL, tokens.total_tokens)
            agent_span.set_attribute("cost.usd", result.cost_usd)

            if self._budget is not None:
                agent_span.set_attribute("budget.remaining", self._budget.remaining)
                agent_span.set_attribute("budget.spent", self._budget._spent)

            tool_calls_list = []
            if result.tool_calls:
                from syrin.types import ToolCall

                # Auto-checkpoint after tool call
                self._maybe_checkpoint("tool")

                for tc in result.tool_calls:
                    tool_calls_list.append(
                        ToolCall(
                            id=tc.get("id", ""),
                            name=tc.get("name", ""),
                            arguments=tc.get("arguments", {}),
                        )
                    )

            # Output guardrails check (only if no tool calls)
            if not result.tool_calls:
                output_guardrail = self._run_guardrails(result.content or "", GuardrailStage.OUTPUT)
                if not output_guardrail.passed:
                    return ResponseClass(
                        content="",
                        raw="",
                        cost=result.cost_usd,
                        tokens=tokens,
                        model=self._model_config.model_id,
                        duration=result.latency_ms / 1000,
                        trace=[],
                        tool_calls=tool_calls_list,
                        stop_reason=StopReason.GUARDRAIL,
                        budget_remaining=self._budget.remaining if self._budget else None,
                        budget_used=self._budget._spent if self._budget else 0.0,
                        structured=None,
                        report=self._run_report,
                    )

            # Build structured output with validation
            structured = self._build_output(
                result.content,
                validation_retries=self._validation_retries,
                validation_context=self._validation_context,
                validator=getattr(self, "_output_validator", None),
            )

            # Populate report with final data
            self._run_report.budget_remaining = self._budget.remaining if self._budget else None
            self._run_report.budget_used = self._budget._spent if self._budget else 0.0
            self._run_report.tokens.input_tokens = tokens.input_tokens
            self._run_report.tokens.output_tokens = tokens.output_tokens
            self._run_report.tokens.total_tokens = tokens.total_tokens
            self._run_report.tokens.cost_usd = result.cost_usd

            return ResponseClass(
                content=result.content,
                cost=result.cost_usd,
                tokens=tokens,
                model=self._model_config.model_id,
                duration=result.latency_ms / 1000,
                tool_calls=tool_calls_list,
                stop_reason=StopReason(result.stop_reason)
                if isinstance(result.stop_reason, str)
                else result.stop_reason,
                budget_remaining=self._budget.remaining if self._budget else None,
                budget_used=self._budget._spent if self._budget else 0.0,
                iterations=result.iterations,
                structured=structured,
                report=self._run_report,
            )

    async def arun(self, user_input: str) -> Response[str]:
        """Async version of response(). Returns Response with content, cost, trace."""
        if self._budget is not None:
            self._budget_tracker.reset_run()
        return await self._run_response(user_input)

    def stream(self, user_input: str) -> Iterator[StreamChunk]:
        """Stream response chunks synchronously."""
        if self._budget is not None:
            self._budget_tracker.reset_run()
        messages = self._build_messages(user_input)
        tools = self._tools if self._tools else None
        accumulated = ""
        total_cost = 0.0
        total_tokens = TokenUsage()

        try:
            for chunk in self._provider.stream_sync(messages, self._model_config, tools):
                content = chunk.content or ""
                accumulated += content
                total_cost += chunk.cost_usd if hasattr(chunk, "cost_usd") else 0.0
                total_tokens = TokenUsage(
                    input_tokens=total_tokens.input_tokens
                    + (chunk.token_usage.input_tokens if hasattr(chunk, "token_usage") else 0),
                    output_tokens=total_tokens.output_tokens
                    + (chunk.token_usage.output_tokens if hasattr(chunk, "token_usage") else 0),
                    total_tokens=total_tokens.total_tokens
                    + (chunk.token_usage.total_tokens if hasattr(chunk, "token_usage") else 0),
                )
                yield StreamChunk(
                    text=content,
                    accumulated_text=accumulated,
                    cost_so_far=total_cost,
                    tokens_so_far=total_tokens,
                )
        except Exception as e:
            raise ToolExecutionError(f"Streaming failed: {e}") from e

    async def astream(self, user_input: str) -> AsyncIterator[StreamChunk]:
        """Stream response chunks asynchronously."""
        if self._budget is not None:
            self._budget_tracker.reset_run()
        messages = self._build_messages(user_input)
        tools = self._tools if self._tools else None
        accumulated = ""
        total_cost = 0.0
        total_tokens = TokenUsage()

        try:
            async for chunk in self._provider.stream(messages, self._model_config, tools):
                content = chunk.content or ""
                accumulated += content
                total_cost += chunk.cost_usd if hasattr(chunk, "cost_usd") else 0.0
                total_tokens = TokenUsage(
                    input_tokens=total_tokens.input_tokens
                    + (chunk.token_usage.input_tokens if hasattr(chunk, "token_usage") else 0),
                    output_tokens=total_tokens.output_tokens
                    + (chunk.token_usage.output_tokens if hasattr(chunk, "token_usage") else 0),
                    total_tokens=total_tokens.total_tokens
                    + (chunk.token_usage.total_tokens if hasattr(chunk, "token_usage") else 0),
                )
                yield StreamChunk(
                    text=content,
                    accumulated_text=accumulated,
                    cost_so_far=total_cost,
                    tokens_so_far=total_tokens,
                )
        except Exception as e:
            raise ToolExecutionError(f"Streaming failed: {e}") from e
