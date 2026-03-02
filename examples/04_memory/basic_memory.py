"""Basic Memory Example.

Demonstrates:
- Creating an agent with Memory (4-type persistent memory)
- remember() and recall() operations
- Memory types: CORE, EPISODIC, SEMANTIC, PROCEDURAL

Run: python -m examples.04_memory.basic_memory
Visit: http://localhost:8000/playground

Requires: uv pip install syrin[serve]
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent, Memory, MemoryType

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class Assistant(Agent):
    _agent_name = "assistant"
    _agent_description = "Assistant with persistent memory (remember/recall/forget)"
    model = almock
    system_prompt = "You are a helpful assistant that remembers user preferences."


if __name__ == "__main__":
    assistant = Assistant(memory=Memory())
    assistant.remember("The user's name is Alice.", memory_type=MemoryType.CORE, importance=1.0)
    assistant.remember("User asked about machine learning", memory_type=MemoryType.EPISODIC)
    print("Serving at http://localhost:8000/playground")
    assistant.serve(port=8000, enable_playground=True, debug=True)
