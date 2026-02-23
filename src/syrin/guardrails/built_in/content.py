"""Content filter guardrail."""

from __future__ import annotations

from syrin.guardrails.base import Guardrail
from syrin.guardrails.context import GuardrailContext
from syrin.guardrails.decision import GuardrailDecision


class ContentFilter(Guardrail):
    """Guardrail that filters content based on blocked words.

    Checks if the text contains any blocked words and returns a
    failed decision if found. Case-insensitive matching.

    Example:
        >>> guardrail = ContentFilter(
        ...     blocked_words=["password", "secret", "api_key"]
        ... )
        >>> result = await guardrail.evaluate(context)
    """

    def __init__(
        self, blocked_words: list[str], name: str | None = None, case_sensitive: bool = False
    ):
        """Initialize content filter.

        Args:
            blocked_words: List of words to block.
            name: Optional custom name.
            case_sensitive: If True, matching is case sensitive.
        """
        super().__init__(name)
        self.case_sensitive = case_sensitive

        # Normalize blocked words
        if case_sensitive:
            self._blocked_words = blocked_words
        else:
            self._blocked_words = [w.lower() for w in blocked_words]

    async def evaluate(self, context: GuardrailContext) -> GuardrailDecision:
        """Check if text contains blocked words.

        Args:
            context: Guardrail context with text to check.

        Returns:
            GuardrailDecision indicating if blocked words were found.
        """
        text = context.text

        # Normalize text for comparison
        if not self.case_sensitive:
            text = text.lower()

        # Check each blocked word
        for word in self._blocked_words:
            if word in text:
                return GuardrailDecision(
                    passed=False,
                    rule="blocked_word",
                    reason=f"Blocked word found: {word}",
                    confidence=1.0,
                    metadata={"word": word, "case_sensitive": self.case_sensitive},
                    alternatives=[
                        f"Remove the word '{word}' from your message",
                        "Rephrase without using restricted terms",
                    ],
                )

        # No blocked words found
        return GuardrailDecision(
            passed=True,
            rule="content_check",
            metadata={"blocked_words_checked": len(self._blocked_words)},
        )
