"""Input/Output Guardrails Example.

Demonstrates:
- ContentFilter for blocked words
- GuardrailChain combining multiple guardrails
- Agent with guardrails= parameter

Run: python -m examples.09_guardrails.input_output_guardrails
Visit: http://localhost:8000/playground
Requires: uv pip install syrin[serve]
"""

from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent
from syrin.enums import GuardrailStage
from syrin.guardrails import ContentFilter, GuardrailChain

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

chain = GuardrailChain(
    [
        ContentFilter(blocked_words=["spam", "scam"], name="NoSpam"),
    ]
)


class IOGuardrailAgent(Agent):
    _agent_name = "io-guardrail"
    _agent_description = "Agent with ContentFilter guardrail chain"
    model = almock
    system_prompt = "You are helpful."
    guardrails = chain


if __name__ == "__main__":
    result = chain.check("Hello, legitimate message", GuardrailStage.INPUT)
    print(f"Clean text: passed={result.passed}")
    result = chain.check("This is spam", GuardrailStage.INPUT)
    print(f"Blocked text: passed={result.passed}, reason={result.reason}")

    agent = IOGuardrailAgent()
    r = agent.response("Hello")
    print(f"Agent response: {r.content[:50]}...")
    print("Serving at http://localhost:8000/playground")
    agent.serve(port=8000, enable_playground=True, debug=True)
