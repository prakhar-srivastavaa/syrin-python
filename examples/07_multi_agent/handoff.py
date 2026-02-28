"""Handoff Example.

Demonstrates:
- Agent handoff between specialized agents
- Context transfer via memory
- Budget transfer between agents

Run: python -m examples.07_multi_agent.handoff
Visit: http://localhost:8000/playground
Requires: uv pip install syrin[serve]
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent, prompt

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@prompt
def analyzer_prompt() -> str:
    return "You are an analyzer agent. Analyze information and provide key findings."


@prompt
def presenter_prompt() -> str:
    return "You are a presenter agent. Present information clearly and concisely."


class Analyzer(Agent):
    name = "analyzer"
    description = "Analyzes information and provides key findings"
    model = almock
    system_prompt = analyzer_prompt()


class Presenter(Agent):
    name = "presenter"
    description = "Presents information clearly and concisely"
    model = almock
    system_prompt = presenter_prompt()


if __name__ == "__main__":
    analyzer = Analyzer()
    result1 = analyzer.response("Analyze the benefits of renewable energy")
    print(f"Analyzer: {result1.content[:80]}...")
    result2 = analyzer.handoff(Presenter, "Present the analysis")
    print(f"Presenter: {result2.content[:80]}...")
    print("Serving at http://localhost:8000/playground")
    analyzer.serve(port=8000, enable_playground=True, debug=True)
