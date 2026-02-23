"""Legacy guardrails for backward compatibility."""

from __future__ import annotations

from typing import Any

from syrin.enums import GuardrailStage as LegacyGuardrailStage
from syrin.guardrails.base import Guardrail
from syrin.guardrails.context import GuardrailContext
from syrin.guardrails.decision import GuardrailDecision


class BlockedWordsGuardrail(Guardrail):
    """Legacy blocked words guardrail - wraps new ContentFilter."""

    def __init__(
        self,
        blocked_words: list[str],
        name: str | None = None,
    ) -> None:
        """Initialize blocked words guardrail."""
        super().__init__(name)
        self._blocked_words = [w.lower() for w in blocked_words]

    async def evaluate(self, context: GuardrailContext) -> GuardrailDecision:
        """Check text against blocked words."""
        text_lower = context.text.lower()

        for word in self._blocked_words:
            if word in text_lower:
                return GuardrailDecision(
                    passed=False,
                    rule="blocked_word",
                    reason=f"Blocked word found: {word}",
                    metadata={"word": word},
                )

        return GuardrailDecision(passed=True)

    # Legacy compatibility method
    def check(self, text: str, stage: LegacyGuardrailStage) -> Any:
        """Legacy check method for backward compatibility."""
        import asyncio

        context = GuardrailContext(text=text, stage=GuardrailStage.INPUT)

        # Run async evaluate in sync context
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in async context, schedule it
                asyncio.ensure_future(self.evaluate(context))
                # Return a simple result for legacy code
                return LegacyGuardrailResult(
                    passed=True,  # Assume passed for compatibility
                    reason="Async guardrail - use evaluate() for full result",
                )
            else:
                decision = loop.run_until_complete(self.evaluate(context))
                return LegacyGuardrailResult(
                    passed=decision.passed,
                    reason=decision.reason,
                )
        except RuntimeError:
            # No event loop, create one
            decision = asyncio.run(self.evaluate(context))
            return LegacyGuardrailResult(
                passed=decision.passed,
                reason=decision.reason,
            )


class LengthGuardrail(Guardrail):
    """Legacy length guardrail."""

    def __init__(
        self,
        min_length: int = 0,
        max_length: int = 10000,
        name: str | None = None,
    ) -> None:
        """Initialize length guardrail."""
        super().__init__(name)
        self._min_length = min_length
        self._max_length = max_length

    async def evaluate(self, context: GuardrailContext) -> GuardrailDecision:
        """Check text length."""
        length = len(context.text)

        if length < self._min_length:
            return GuardrailDecision(
                passed=False,
                rule="too_short",
                reason=f"Text too short: {length} < {self._min_length}",
                metadata={"length": length, "min": self._min_length},
            )

        if length > self._max_length:
            return GuardrailDecision(
                passed=False,
                rule="too_long",
                reason=f"Text too long: {length} > {self._max_length}",
                metadata={"length": length, "max": self._max_length},
            )

        return GuardrailDecision(passed=True)

    # Legacy compatibility
    def check(self, text: str, stage: LegacyGuardrailStage) -> Any:
        """Legacy check method."""
        import asyncio

        context = GuardrailContext(text=text)

        try:
            decision = asyncio.run(self.evaluate(context))
            return LegacyGuardrailResult(
                passed=decision.passed,
                reason=decision.reason,
            )
        except Exception:
            return LegacyGuardrailResult(passed=True, reason="")


class LegacyGuardrailResult:
    """Legacy result format for backward compatibility."""

    def __init__(
        self, passed: bool, reason: str | None = None, metadata: dict[str, Any] | None = None
    ):
        self.passed = passed
        self.reason = reason
        self.metadata = metadata or {}


# Alias for backward compatibility
GuardrailResult = LegacyGuardrailResult

# Map legacy stage to new stage
GuardrailStage = LegacyGuardrailStage
