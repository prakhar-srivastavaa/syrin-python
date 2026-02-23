"""Context configuration and stats."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from syrin.threshold import ContextThreshold

if TYPE_CHECKING:
    from syrin.model import Model


@dataclass
class ContextStats:
    """Statistics about context usage after an LLM call."""

    total_tokens: int = 0
    max_tokens: int = 0
    utilization: float = 0.0
    compacted: bool = False
    compaction_count: int = 0
    compaction_method: str | None = None
    thresholds_triggered: list[str] = field(default_factory=list)


@dataclass
class ContextBudget:
    """Internal budget tracking for context management."""

    max_tokens: int
    reserve_for_response: int = 2000
    auto_compact_at: float = 0.75
    _used_tokens: int = 0

    @property
    def available(self) -> int:
        """Tokens available for context (excluding response reserve)."""
        return max(0, self.max_tokens - self.reserve_for_response)

    @property
    def used_tokens(self) -> int:
        return self._used_tokens

    @used_tokens.setter
    def used_tokens(self, value: int) -> None:
        self._used_tokens = value

    @property
    def utilization(self) -> float:
        """Current utilization as a fraction (0-1)."""
        if self.available <= 0:
            return 0.0
        return self._used_tokens / self.available

    @property
    def should_compact(self) -> bool:
        """Whether compaction should be triggered."""
        return self.utilization >= self.auto_compact_at

    @property
    def utilization_percent(self) -> int:
        """Current utilization as percentage (0-100)."""
        return int(self.utilization * 100)

    def reset(self) -> None:
        """Reset used tokens for new call."""
        self._used_tokens = 0


@dataclass
class Context:
    """Context management configuration.

    Provides intelligent context window management with automatic
    compaction when approaching token limits.

    Args:
        max_tokens: Maximum tokens in context window. Auto-detected from model if not specified.
        auto_compact_at: Threshold (0-1) at which to trigger automatic compaction.
                        Default 0.75 (75%).
        thresholds: List of ContextThreshold (only ContextThreshold allowed)

    Example:
        >>> from syrin import Agent, Model, Context
        >>> from syrin.threshold import ContextThreshold
        >>>
        >>> agent = Agent(
        ...     model=Model("openai/gpt-4o"),
        ...     context=Context(
        ...         max_tokens=80000,
        ...         thresholds=[
        ...             ContextThreshold(at=80, action=lambda ctx: print(f"Tokens at {ctx.percentage}%"))
        ...         ]
        ...     )
        ... )
    """

    max_tokens: int | None = None
    auto_compact_at: float = 0.75
    thresholds: list[ContextThreshold] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.auto_compact_at < 0 or self.auto_compact_at > 1:
            raise ValueError("auto_compact_at must be between 0 and 1")
        self._validate_thresholds()

    def _validate_thresholds(self) -> None:
        """Validate that only ContextThreshold is used."""
        for th in self.thresholds:
            if not isinstance(th, ContextThreshold):
                raise ValueError(
                    f"Context thresholds only accept ContextThreshold, got {type(th).__name__}"
                )

    def get_budget(self, model: "Model | None" = None) -> ContextBudget:
        """Get a ContextBudget for this configuration.

        Args:
            model: Optional model to auto-detect context window from.

        Returns:
            ContextBudget configured for this context.
        """
        max_tokens = self.max_tokens

        if max_tokens is None and model is not None:
            from syrin.model import Model as ModelClass
            from syrin.model.core import ModelSettings

            if isinstance(model, ModelClass):
                settings = model.settings
                if isinstance(settings, ModelSettings) and settings.context_window:
                    max_tokens = settings.context_window

        if max_tokens is None:
            max_tokens = 128000

        return ContextBudget(
            max_tokens=max_tokens,
            auto_compact_at=self.auto_compact_at,
        )


__all__ = ["Context", "ContextStats", "ContextBudget"]
