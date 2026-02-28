"""Single Task Example — @syrin.task for named entry points.

Demonstrates:
- Using @syrin.task to define a named task method
- Researcher agent with research(topic: str) task
- Invoking tasks via agent.task_name(args)

Run: python -m examples.02_tasks.single_task
Visit: http://localhost:8000/playground

Requires: uv pip install syrin[serve]
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent, task

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class Researcher(Agent):
    """Agent that researches topics. Uses @syrin.task for a named API."""

    name = "researcher"
    description = "Research assistant with research(topic) task"
    model = almock
    system_prompt = "You are a research assistant. Provide concise, factual summaries."

    @task
    def research(self, topic: str) -> str:
        """Research a topic and return a summary."""
        response = self.response(f"Research the following topic and summarize: {topic}")
        return response.content or ""


if __name__ == "__main__":
    researcher = Researcher()
    print("Serving at http://localhost:8000/playground")
    researcher.serve(port=8000, enable_playground=True, debug=True)
    # researcher.serve(protocol=ServeProtocol.CLI)
