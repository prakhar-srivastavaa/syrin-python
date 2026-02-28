"""Guardrail check result - return type for sync check() calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GuardrailCheckResult:
    """Result of guardrail check (sync API).

    Returned by GuardrailChain.check() when agent runs guardrails.

    Attributes:
        passed: Whether the guardrail check passed.
        reason: Human-readable reason if failed.
        metadata: Arbitrary metadata about the check.
    """

    passed: bool
    """Whether the guardrail check passed."""
    reason: str | None = None
    """Human-readable reason if failed."""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Arbitrary metadata about the check."""


__all__ = ["GuardrailCheckResult"]
