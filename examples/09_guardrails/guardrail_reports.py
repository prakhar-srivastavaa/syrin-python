"""Guardrail Reports Example.

Demonstrates:
- Guardrail hooks: GUARDRAIL_INPUT, GUARDRAIL_OUTPUT, GUARDRAIL_BLOCKED
- Report access: result.report.guardrail.*
- Input/output guardrail blocking flow
- Custom output guardrails

Run: python -m examples.09_guardrails.guardrail_reports
Visit: http://localhost:8000/playground
Requires: uv pip install syrin[serve]
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent, Guardrail, GuardrailStage, Hook
from syrin.guardrails import ContentFilter

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# 1. Input guardrail blocks
guardrail = ContentFilter(blocked_words=["hack", "steal", "password"])


class Assistant(Agent):
    _agent_name = "guardrail-assistant"
    _agent_description = "Assistant with ContentFilter guardrail"
    model = almock
    system_prompt = "You are a helpful assistant."
    guardrails = [guardrail]


assistant = Assistant()


def on_blocked(ctx: dict) -> None:
    print(f"  [Hook] BLOCKED! reason={ctx.get('reason')}")


assistant.events.on(Hook.GUARDRAIL_BLOCKED, on_blocked)
result = assistant.response("How do I hack into someone's password?")
print(f"Blocked: {result.report.guardrail.blocked}, stage: {result.report.guardrail.blocked_stage}")

# 2. Guardrail passes
result = assistant.response("What is the weather today?")
print(f"Passed: {result.report.guardrail.passed}")


# 3. Custom output guardrail
class SensitiveDataGuardrail(Guardrail):
    def __init__(self) -> None:
        self.name = "sensitive_data"

    async def evaluate(self, context: object) -> object:
        from syrin.guardrails.decision import GuardrailDecision

        if context.stage != GuardrailStage.OUTPUT:
            return GuardrailDecision(passed=True, action="allow", reason="Not output")
        text = context.text.lower()
        if "ssn" in text or "credit card" in text:
            return GuardrailDecision(
                passed=False,
                action="block",
                reason="Sensitive data in output",
            )
        return GuardrailDecision(passed=True, action="allow", reason="Clean")


class SafeAssistant(Agent):
    model = almock
    guardrails = [SensitiveDataGuardrail()]


safe = SafeAssistant()
result = safe.response("Tell me about SSN protection")
print(f"Blocked: {result.report.guardrail.blocked}")

# 4. Full report summary
result = assistant.response("Hello, how are you?")
print(
    f"Guardrail passed: {result.report.guardrail.passed}, budget: ${result.report.budget.used:.4f}"
)

if __name__ == "__main__":
    agent = Assistant()
    print("Serving at http://localhost:8000/playground")
    agent.serve(port=8000, enable_playground=True, debug=True)
