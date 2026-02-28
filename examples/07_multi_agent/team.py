"""AgentTeam Example.

Demonstrates:
- Creating an AgentTeam with multiple specialized agents
- Shared budget across team members
- Automatic agent selection for tasks (team.select_agent)
- Running tasks via team.run_task()

Run: python -m examples.07_multi_agent.team
Visit: http://localhost:8000/playground
Requires: uv pip install syrin[serve]
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent, Budget, prompt
from syrin.agent.multi_agent import AgentTeam
from syrin.serve import AgentRouter

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@prompt
def researcher_prompt(domain: str) -> str:
    return f"You are a researcher specializing in {domain}."


@prompt
def writer_prompt(style: str) -> str:
    return f"You are a writer with a {style} style."


# 1. Basic team with shared budget
class Researcher(Agent):
    name = "researcher"
    description = "Researches topics (technology)"
    model = almock
    system_prompt = researcher_prompt(domain="technology")


class Writer(Agent):
    name = "writer"
    description = "Writes content in engaging style"
    model = almock
    system_prompt = writer_prompt(style="engaging")


# 2. Agent selection
class GeneralResearcher(Agent):
    name = "general-researcher"
    description = "General researcher"
    model = almock
    system_prompt = researcher_prompt(domain="general")


class GeneralWriter(Agent):
    name = "general-writer"
    description = "General writer"
    model = almock
    system_prompt = writer_prompt(style="general")


if __name__ == "__main__":
    team = AgentTeam(
        agents=[Researcher(), Writer()],
        budget=Budget(run=0.50, shared=True),
    )
    result = team.run_task("Research AI trends")
    print(f"Result: {result.content[:80]}...")
    print(f"Cost: ${result.cost:.6f}")

    team = AgentTeam(agents=[GeneralResearcher(), GeneralWriter()])
    selected = team.select_agent("research machine learning")
    print(f"Task 'research ML' → {selected.__class__.__name__}")
    selected = team.select_agent("write an article about AI")
    print(f"Task 'write article' → {selected.__class__.__name__}")

    router = AgentRouter(agents=[Researcher(), Writer()])
    print("Serving at http://localhost:8000/playground")
    router.serve(port=8000, enable_playground=True, debug=True)
