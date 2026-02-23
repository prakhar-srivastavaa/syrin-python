"""Guardrails system for input/output/action validation."""

from __future__ import annotations

# Import auth module
from syrin.guardrails.auth.capability import CapabilityIssuer, CapabilityToken
from syrin.guardrails.base import Guardrail

# Import built-in guardrails
from syrin.guardrails.built_in import (
    AuthorityCheck,
    BudgetEnforcer,
    CapabilityGuardrail,
    ContentFilter,
    HumanApproval,
    PIIScanner,
    ThresholdApproval,
)
from syrin.guardrails.chain import GuardrailChain
from syrin.guardrails.context import GuardrailContext
from syrin.guardrails.decision import GuardrailDecision
from syrin.guardrails.engine import EvaluationResult, ParallelEvaluationEngine
from syrin.guardrails.enums import DecisionAction, GuardrailStage

# Import intelligence layer
from syrin.guardrails.intelligence import (
    AdaptiveThresholdGuardrail,
    AttackSimulator,
    ContextAwareGuardrail,
    EscalationDetector,
    FuzzingEngine,
    RedTeamEvaluator,
)

# Keep old guardrails for backward compatibility
from syrin.guardrails.legacy import (
    BlockedWordsGuardrail,
    GuardrailResult,
    LengthGuardrail,
)

__all__ = [
    # Core classes
    "Guardrail",
    "GuardrailContext",
    "GuardrailDecision",
    "GuardrailChain",
    "ParallelEvaluationEngine",
    "EvaluationResult",
    # Enums
    "GuardrailStage",
    "DecisionAction",
    # Built-in guardrails
    "ContentFilter",
    "PIIScanner",
    # Authority layer
    "AuthorityCheck",
    "BudgetEnforcer",
    "ThresholdApproval",
    "HumanApproval",
    "CapabilityGuardrail",
    # Auth
    "CapabilityToken",
    "CapabilityIssuer",
    # Intelligence layer
    "ContextAwareGuardrail",
    "EscalationDetector",
    "AdaptiveThresholdGuardrail",
    "AttackSimulator",
    "RedTeamEvaluator",
    "FuzzingEngine",
    # Legacy (backward compatibility)
    "BlockedWordsGuardrail",
    "LengthGuardrail",
    "GuardrailResult",
]
