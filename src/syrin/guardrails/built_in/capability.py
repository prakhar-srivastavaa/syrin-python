"""Capability token-based guardrail."""

from __future__ import annotations

from contextlib import suppress

from syrin.enums import DecisionAction
from syrin.guardrails.base import Guardrail
from syrin.guardrails.context import GuardrailContext
from syrin.guardrails.decision import GuardrailDecision


class CapabilityGuardrail(Guardrail):
    """Guardrail checking capability tokens.

    Validates that the user has a valid capability token for the
    requested action and consumes budget from the token.

    Example:
        >>> guardrail = CapabilityGuardrail(
        ...     required_capability="finance:transfer"
        ... )
        >>> result = await guardrail.evaluate(context)
    """

    def __init__(
        self,
        required_capability: str,
        consume_budget: bool = True,
        name: str | None = None,
    ):
        """Initialize capability guardrail.

        Args:
            required_capability: Required capability scope.
            consume_budget: Whether to consume token budget.
            name: Optional custom name.
        """
        super().__init__(name)
        self.required_capability = required_capability
        self.consume_budget = consume_budget

    async def evaluate(self, context: GuardrailContext) -> GuardrailDecision:
        """Check if user has valid capability token.

        Args:
            context: Guardrail context with user.

        Returns:
            GuardrailDecision indicating token status.
        """
        # Check for user
        if context.user is None:
            return GuardrailDecision(
                passed=False,
                rule="no_user",
                reason="No user in context",
                action=DecisionAction.BLOCK,
            )

        # Get capability token from user
        token = None
        if hasattr(context.user, "get_capability_token"):
            with suppress(Exception):
                token = context.user.get_capability_token()
        elif hasattr(context.user, "capability_token"):
            token = context.user.capability_token

        if token is None:
            return GuardrailDecision(
                passed=False,
                rule="no_capability_token",
                reason="No capability token found for user",
                action=DecisionAction.BLOCK,
                alternatives=[
                    "Request a capability token from administrator",
                    "Use a different authentication method",
                ],
            )

        # Check if token is valid
        if not token.is_valid():
            if token.is_expired():
                return GuardrailDecision(
                    passed=False,
                    rule="token_expired",
                    reason="Capability token has expired",
                    action=DecisionAction.BLOCK,
                )
            else:
                return GuardrailDecision(
                    passed=False,
                    rule="capability_exhausted",
                    reason="Capability token budget exhausted",
                    action=DecisionAction.BLOCK,
                    metadata={"scope": token.scope, "budget": token.budget},
                )

        # Check if token permits the action
        if not token.can(self.required_capability):
            return GuardrailDecision(
                passed=False,
                rule="insufficient_capability",
                reason=f"Token scope '{token.scope}' does not permit '{self.required_capability}'",
                action=DecisionAction.BLOCK,
                metadata={
                    "token_scope": token.scope,
                    "required_capability": self.required_capability,
                },
                alternatives=[
                    f"Request a token with '{self.required_capability}' scope",
                    "Use an action permitted by your token",
                ],
            )

        # Consume budget if enabled
        if self.consume_budget:
            success = token.consume(1)
            if not success:
                return GuardrailDecision(
                    passed=False,
                    rule="capability_exhausted",
                    reason="Failed to consume capability budget",
                    action=DecisionAction.BLOCK,
                )

        # All checks passed
        return GuardrailDecision(
            passed=True,
            rule="capability_valid",
            metadata={
                "token_id": token.token_id,
                "scope": token.scope,
                "remaining_budget": token.budget,
                "token_consumed": self.consume_budget,
            },
        )
