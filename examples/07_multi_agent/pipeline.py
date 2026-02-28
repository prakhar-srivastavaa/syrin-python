"""Pipeline Example.

Demonstrates:
- Sequential pipeline execution
- Parallel pipeline execution
- Pipeline with budget

Run: python -m examples.07_multi_agent.pipeline
Visit: http://localhost:8000/playground
Requires: uv pip install syrin[serve]
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent, Pipeline, prompt

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@prompt
def researcher_prompt(domain: str) -> str:
    return f"You are a researcher specializing in {domain}."


@prompt
def writer_prompt(style: str) -> str:
    return f"You are a writer with a {style} style."


class Researcher(Agent):
    name = "researcher"
    description = "Researches topics and gathers information"
    model = almock
    system_prompt = researcher_prompt(domain="technology")


class Writer(Agent):
    name = "writer"
    description = "Writes content in professional style"
    model = almock
    system_prompt = writer_prompt(style="professional")


pipeline = Pipeline()

if __name__ == "__main__":
    result = pipeline.run(
        [
            (Researcher, "Find information about renewable energy"),
            (Writer, "Write about renewable energy"),
        ]
    )
    print(f"Pipeline result: {result.content[:100]}...")
    print(f"Cost: ${result.cost:.6f}")
    print("Serving at http://localhost:8000/playground")
    pipeline.serve(port=8000, enable_playground=True, debug=True)
