"""Base Guardrail class."""

from __future__ import annotations

from abc import ABC, abstractmethod

from syrin.guardrails.context import GuardrailContext
from syrin.guardrails.decision import GuardrailDecision


class Guardrail(ABC):
    """Abstract base class for all guardrails.

    All guardrails must inherit from this class and implement the
    `evaluate` method. Guardrails are evaluated asynchronously to
    enable parallel execution.

    Example:
        >>> class MyGuardrail(Guardrail):
        ...     async def evaluate(self, context: GuardrailContext) -> GuardrailDecision:
        ...         if "forbidden" in context.text:
        ...             return GuardrailDecision(
        ...                 passed=False,
        ...                 rule="forbidden_word",
        ...                 reason="Contains forbidden word"
        ...             )
        ...         return GuardrailDecision(passed=True)
    """

    def __init__(self, name: str | None = None) -> None:
        """Initialize guardrail.

        Args:
            name: Optional custom name. Defaults to class name.
        """
        self.name = name or self.__class__.__name__
        self.budget_cost: float = 0.0
        """Cost in USD to run this guardrail. Override in subclasses."""

    @abstractmethod
    async def evaluate(self, context: GuardrailContext) -> GuardrailDecision:
        """Evaluate the guardrail against the given context.

        This method must be implemented by all concrete guardrail classes.
        It should return a GuardrailDecision indicating whether the check
        passed or failed.

        Args:
            context: The context to evaluate against.

        Returns:
            GuardrailDecision with the evaluation result.
        """
        pass

    def __repr__(self) -> str:
        """String representation of the guardrail."""
        return f"{self.__class__.__name__}(name='{self.name}')"
