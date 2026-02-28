"""Quick Run Example.

Demonstrates:
- syrin.run() one-liner for quick LLM calls
- syrin.configure() for global settings
- Using run() with system prompt and budget
- Serving the equivalent agent

Run: python -m examples.01_minimal.quick_run
Visit: http://localhost:8000/playground

Requires: uv pip install syrin[serve]
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

import syrin
from examples.models.models import almock
from syrin import Agent

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class QuickRunAgent(Agent):
    """Agent equivalent to syrin.run() with system prompt and budget."""

    name = "quick-run"
    description = "Quick run demo agent"
    model = almock
    system_prompt = "Explain like I'm five years old."
    budget = syrin.Budget(run=0.10, on_exceeded=syrin.warn_on_exceeded)


if __name__ == "__main__":
    agent = QuickRunAgent()
    print("Serving at http://localhost:8000/playground")
    agent.serve(port=8000, enable_playground=True, debug=True)
