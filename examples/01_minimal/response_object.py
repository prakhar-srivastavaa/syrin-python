"""Response Object Example.

Demonstrates:
- All Response object attributes: content, raw, cost, tokens, model, duration
- Trace steps (execution log)
- Budget information (budget_remaining, budget_used)
- Structured output (result.data, result.structured)
- Boolean check
- Serving to inspect response in playground

Run: python -m examples.01_minimal.response_object
Visit: http://localhost:8000/playground

Requires: uv pip install syrin[serve]
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent, Budget

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class ResponseDemoAgent(Agent):
    """Agent to demo response attributes in playground."""

    name = "response-demo"
    description = "Agent for exploring response object (content, cost, tokens, etc.)"
    model = almock
    system_prompt = "You are a helpful assistant. Be concise."
    budget = Budget(run=0.10)


if __name__ == "__main__":
    agent = ResponseDemoAgent()
    print("Serving at http://localhost:8000/playground")
    agent.serve(port=8000, enable_playground=True, debug=True)
    # assistant.serve(protocol=ServeProtocol.CLI)
