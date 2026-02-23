"""GuardrailContext - Rich context for guardrail evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from syrin.guardrails.enums import GuardrailStage


@dataclass(frozen=True)
class GuardrailContext:
    """Rich context for guardrail evaluation.

    Provides all necessary information for guardrails to make informed decisions.
    Immutable to ensure thread safety and prevent accidental modifications.

    Example:
        >>> context = GuardrailContext(
        ...     text="Transfer $500",
        ...     stage=GuardrailStage.ACTION,
        ...     user=user_obj,
        ...     budget=budget_obj
        ... )
    """

    # Core fields
    text: str
    """The text/content being evaluated."""

    stage: GuardrailStage = field(default=GuardrailStage.INPUT)
    """Which stage of the pipeline this is (input/action/output)."""

    # Context objects (optional)
    conversation: Any | None = field(default=None)
    """Conversation history/object if available."""

    user: Any | None = field(default=None)
    """User object with identity and permissions."""

    agent: Any | None = field(default=None)
    """Agent instance being guarded."""

    budget: Any | None = field(default=None)
    """Budget tracker for this session."""

    action: Any | None = field(default=None)
    """Action being evaluated (for action-stage guardrails)."""

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    """Arbitrary metadata for this evaluation."""

    def copy(self, **kwargs: Any) -> GuardrailContext:
        """Create a copy with optional modifications.

        Args:
            **kwargs: Fields to override in the copy.

        Returns:
            New GuardrailContext with specified modifications.
        """
        # Build kwargs for new instance
        new_kwargs: dict[str, Any] = {
            "text": kwargs.get("text", self.text),
            "stage": kwargs.get("stage", self.stage),
            "conversation": kwargs.get("conversation", self.conversation),
            "user": kwargs.get("user", self.user),
            "agent": kwargs.get("agent", self.agent),
            "budget": kwargs.get("budget", self.budget),
            "action": kwargs.get("action", self.action),
            "metadata": kwargs.get("metadata", self.metadata.copy()),
        }

        return GuardrailContext(**new_kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Convert context to dictionary for serialization.

        Returns:
            Dictionary representation of the context.
        """
        return {
            "text": self.text,
            "stage": self.stage.value,
            "metadata": self.metadata,
            # Note: complex objects are not serialized, just their type
            "has_conversation": self.conversation is not None,
            "has_user": self.user is not None,
            "has_agent": self.agent is not None,
            "has_budget": self.budget is not None,
            "has_action": self.action is not None,
        }
