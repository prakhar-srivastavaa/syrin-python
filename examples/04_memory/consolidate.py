"""Consolidate Example — Memory consolidation concepts.

Demonstrates:
- Memory consolidation: deduplicate by content
- Memory.consolidate(deduplicate=True)
- Keeps highest-importance entry when duplicates found

Run: python -m examples.04_memory.consolidate
Visit: http://localhost:8000/playground

Requires: uv pip install syrin[serve]
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent, Memory

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class ConsolidateAgent(Agent):
    """Agent with memory — use remember/recall in playground; consolidation runs on backend."""

    name = "consolidate"
    description = "Agent with memory consolidation (deduplicate by content)"
    model = almock
    system_prompt = "You are a helpful assistant. Use memory to remember user preferences."


if __name__ == "__main__":
    agent = ConsolidateAgent(memory=Memory())
    print("Serving at http://localhost:8000/playground")
    agent.serve(port=8000, enable_playground=True, debug=True)
