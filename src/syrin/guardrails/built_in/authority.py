"""Authority check guardrail for permission validation."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from syrin.enums import DecisionAction
from syrin.guardrails.base import Guardrail
from syrin.guardrails.context import GuardrailContext
from syrin.guardrails.decision import GuardrailDecision


class AuthorityCheck(Guardrail):
    """Guardrail that checks if user has required permissions.

    Validates that the user in the context has the required permissions
    to perform an action. Supports both single and multiple permissions
    with ALL or ANY logic.

    Example:
        >>> guardrail = AuthorityCheck(requires="finance:transfer")
        >>> result = await guardrail.evaluate(context)

        >>> guardrail = AuthorityCheck(
        ...     requires=["read", "write"],
        ...     logic="all"
        ... )
    """

    def __init__(
        self,
        requires: str | list[str] | None = None,
        logic: str = "all",
        scope: str | None = None,
        resource: Callable[[GuardrailContext], str] | None = None,
        allowed_hours: tuple[int, int] | None = None,
        condition: Callable[[GuardrailContext], bool] | None = None,
        name: str | None = None,
    ):
        """Initialize authority check.

        Args:
            requires: Permission(s) required. Can be single string or list.
            logic: "all" (default) or "any" for multiple permissions.
            scope: Resource scope to check (e.g., "documents").
            resource: Function to extract resource ID from context.
            allowed_hours: Tuple of (start_hour, end_hour) for time restrictions.
            condition: Optional condition function. If returns False, check is skipped.
            name: Optional custom name.
        """
        super().__init__(name)
        self.requires = requires
        self.logic = logic
        self.scope = scope
        self.resource = resource
        self.allowed_hours = allowed_hours
        self.condition = condition

    async def evaluate(self, context: GuardrailContext) -> GuardrailDecision:
        """Check if user has required permissions.

        Args:
            context: Guardrail context with user.

        Returns:
            GuardrailDecision indicating permission status.
        """
        # Check condition first
        if self.condition is not None and not self.condition(context):
            return GuardrailDecision(
                passed=True,
                rule="condition_not_met",
                reason="Condition not met, authority check skipped",
                metadata={"condition": True},
            )

        # Check for user
        if context.user is None:
            return GuardrailDecision(
                passed=False,
                rule="no_user",
                reason="No user in context",
                action=DecisionAction.BLOCK,
            )

        # Check time restrictions
        if self.allowed_hours is not None:
            current_hour = datetime.now().hour
            start, end = self.allowed_hours
            if not (start <= current_hour < end):
                return GuardrailDecision(
                    passed=False,
                    rule="outside_hours",
                    reason=f"Action not allowed outside business hours ({start}:00-{end}:00)",
                    action=DecisionAction.BLOCK,
                    metadata={"current_hour": current_hour, "allowed_hours": self.allowed_hours},
                )

        # If no permission required, allow
        if self.requires is None:
            return GuardrailDecision(
                passed=True, rule="no_permission_required", metadata={"requires": None}
            )

        # Convert single permission to list
        permissions = [self.requires] if isinstance(self.requires, str) else self.requires

        # Check each permission
        results = []
        for perm in permissions:
            has_perm = self._check_permission(context, perm)
            results.append((perm, has_perm))

        # Evaluate based on logic
        passed = all(r[1] for r in results) if self.logic == "all" else any(r[1] for r in results)

        if passed:
            return GuardrailDecision(
                passed=True,
                rule="permission_granted",
                metadata={
                    "permissions_checked": permissions,
                    "logic": self.logic,
                    "results": {r[0]: r[1] for r in results},
                },
            )
        else:
            # Build reason message
            if self.logic == "all":
                missing = [r[0] for r in results if not r[1]]
                reason = f"Missing required permissions: {', '.join(missing)}"
            else:
                reason = f"None of the required permissions granted: {', '.join(permissions)}"

            return GuardrailDecision(
                passed=False,
                rule="permission_denied",
                reason=reason,
                action=DecisionAction.BLOCK,
                metadata={
                    "permissions_required": permissions,
                    "logic": self.logic,
                    "results": {r[0]: r[1] for r in results},
                },
                alternatives=[
                    "Request additional permissions from administrator",
                    "Use an action that requires fewer permissions",
                ],
            )

    def _check_permission(self, context: GuardrailContext, permission: str) -> bool:
        """Check if user has a specific permission.

        Args:
            context: Guardrail context.
            permission: Permission to check.

        Returns:
            True if user has permission.
        """
        user = context.user

        # Try has_permission method first
        if user is not None and hasattr(user, "has_permission"):
            try:
                return bool(user.has_permission(permission))
            except Exception:
                pass

        # Fall back to permissions list
        if user is not None and hasattr(user, "permissions"):
            perms = user.permissions
            if isinstance(perms, (list, set, tuple)):
                return permission in perms

        return False
