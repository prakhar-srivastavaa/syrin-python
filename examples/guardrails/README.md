# Syrin Guardrails Examples

Complete examples demonstrating the Syrin Guardrails system.

## Quick Start

```bash
# Run all examples in order
python examples/guardrails/01_foundation.py
python examples/guardrails/02_authority.py
python examples/guardrails/03_intelligence.py
python examples/guardrails/04_complete_workflows.py
```

## File Overview

### `01_foundation.py` - Foundation Layer
Basic guardrails for input/output validation:
- **Content Filtering**: Block specific words/phrases
- **PII Detection**: Detect and redact personal information
- **Parallel Evaluation**: Run multiple guardrails concurrently
- **Guardrail Chains**: Sequential evaluation with early exit
- **Custom Guardrails**: Build your own validation logic

### `02_authority.py` - Authority Layer
Permission-based authorization and enforcement:
- **Authority Check**: Permission-based access control
- **Budget Enforcer**: Per-action and time-windowed budget limits
- **Threshold Approval**: K-of-N multi-party consensus
- **Human Approval**: Human-in-the-loop with 2FA and justification
- **Capability Tokens**: Consumable permissions with TTL and budget

### `03_intelligence.py` - Intelligence Layer
Context-aware protection and adaptive security:
- **Context Awareness**: Multi-turn conversation tracking
- **Escalation Detection**: Detect violation spikes and progressive tactics
- **Adaptive Thresholds**: Auto-tune based on false positive/negative rates
- **Attack Simulation**: Generate jailbreak and injection attacks
- **Red Team Evaluation**: Automated security testing
- **Fuzzing Engine**: Mutation-based edge case discovery

### `04_complete_workflows.py` - Complete Workflows
End-to-end examples combining all layers:
- **Financial Services**: Complete transaction validation
- **Content Moderation**: Adaptive filtering with feedback
- **Production Deployment**: Multi-approval deployment workflow
- **Multi-Tenant SaaS**: Per-customer guardrail configuration
- **Complete Agent**: All three layers working together

## Usage

```python
from Syrin.guardrails import (
    ContentFilter,
    PIIScanner,
    AuthorityCheck,
    BudgetEnforcer,
)
from Syrin.guardrails.intelligence import (
    ContextAwareGuardrail,
    EscalationDetector,
)

# Simple content filtering
guardrail = ContentFilter(blocked_words=["password", "secret"])

# Permission check
guardrail = AuthorityCheck(requires="finance:transfer")

# Context-aware protection
guardrail = ContextAwareGuardrail(max_history_turns=5)
```

## Key Concepts

### GuardrailContext
Provides all information needed for evaluation:
- Text being evaluated
- User identity and permissions
- Budget state
- Action being performed
- Conversation history
- Custom metadata

### GuardrailDecision
Rich evaluation result:
- Passed/failed status
- Rule that triggered
- Human-readable reason
- Confidence score
- Suggested alternatives
- Latency and cost tracking

### Parallel vs Sequential
- **ParallelEvaluationEngine**: All guardrails run concurrently (fastest)
- **GuardrailChain**: Sequential evaluation, stops on first failure

## Next Steps

- Read the full [Guardrails Documentation](../../docs/guardrails.md)
- Check [API Reference](../../docs/guardrails.md#api-reference)
- Review [Feature Comparison](../../docs/GUARDRAILS_FEATURE_AVAILABILITY.md)
