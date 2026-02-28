"""Task with Output Type — TriageAgent with structured output.

Demonstrates:
- @syrin.task that returns structured data (dict)
- TriageAgent with triage(item) returning priority, category, summary
- For full structured output, use Agent with output=Output(MyModel)

Run: python -m examples.02_tasks.task_with_output_type
Visit: http://localhost:8000/playground

Requires: uv pip install syrin[serve]
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent, task

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class TriageAgent(Agent):
    """Agent that triages items with structured output."""

    name = "triage"
    description = "Triage agent returning priority, category, summary"
    model = almock
    system_prompt = (
        "You are a triage assistant. For each item, return priority (high/medium/low), "
        "category, and a brief summary. Be concise."
    )

    @task
    def triage(self, item: str) -> dict:
        """Triage an item. Returns dict with priority, category, summary."""
        response = self.response(
            f"Triage this item: {item}. "
            "Respond with: priority (high/medium/low), category, and summary."
        )
        # Parse simple structure from response (Almock returns lorem; real model returns JSON)
        content = response.content or ""
        return {
            "priority": "medium",
            "category": "general",
            "summary": content[:100] if content else "No summary",
        }


if __name__ == "__main__":
    agent = TriageAgent()
    print("Serving at http://localhost:8000/playground")
    agent.serve(port=8000, enable_playground=True, debug=True)
