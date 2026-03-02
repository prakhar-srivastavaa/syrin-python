"""Agent with Budget Example.

Demonstrates:
- Creating an Agent with a Budget
- Budget tracking via response.cost and agent.budget_state
- Budget limits and exceeded handling

Run: python -m examples.01_minimal.agent_with_budget
Visit: http://localhost:8000/playground

Requires: uv pip install syrin[serve]
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent, Budget, ServeProtocol, warn_on_exceeded

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class Assistant(Agent):
    _agent_name = "assistant"
    _agent_description = "Assistant with budget control"
    model = almock
    system_prompt = "You are a helpful assistant."
    budget = Budget(run=0.10, on_exceeded=warn_on_exceeded)


if __name__ == "__main__":
    assistant = Assistant()
    print("Serving at http://localhost:8000/playground")
    # assistant.serve(port=8000, enable_playground=True, debug=True)
    assistant.serve(protocol=ServeProtocol.CLI)
