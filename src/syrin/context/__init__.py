"""Context management for intelligent LLM context window handling.

Provides automatic context management with:
- Token counting and budget tracking
- Automatic compaction (middle-out truncation, summarization)
- Full observability via events and spans
- Protocol for custom implementations

Example:
    >>> from Syrin import Agent, Model
    >>> from syrin.context import Context
    >>>
    >>> agent = Agent(
    ...     model=Model("openai/gpt-4o"),
    ...     context=Context(max_tokens=80000, auto_compact_at=0.75)
    ... )
    >>> result = agent.run("Hello")
    >>> print(agent.context.stats)
"""

from syrin.context._manager import (
    ContextManager,
    ContextPayload,
    DefaultContextManager,
    create_context_manager,
)
from syrin.context.compactors import (
    CompactionResult,
    Compactor,
    ContextCompactor,
    MiddleOutTruncator,
    Summarizer,
)
from syrin.context.config import Context, ContextBudget, ContextStats
from syrin.context.counter import TokenCount, TokenCounter, get_counter
from syrin.threshold import Threshold

__all__ = [
    # Config
    "Context",
    "ContextStats",
    "ContextBudget",
    "Threshold",
    # Counter
    "TokenCounter",
    "TokenCount",
    "get_counter",
    # Compactors
    "Compactor",
    "CompactionResult",
    "ContextCompactor",
    "MiddleOutTruncator",
    "Summarizer",
    # Manager
    "ContextManager",
    "ContextPayload",
    "DefaultContextManager",
    "create_context_manager",
]
