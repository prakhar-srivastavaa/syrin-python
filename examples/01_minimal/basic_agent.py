"""Basic Agent Example.

Demonstrates:
- Creating an Agent with a model
- Making a simple response call
- Accessing response properties (content, cost, tokens)

Run: python -m examples.01_minimal.basic_agent
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
from syrin import Agent

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class Assistant(Agent):
    name = "assistant"
    description = "Basic helpful assistant"
    model = almock
    system_prompt = "You are a helpful assistant."


if __name__ == "__main__":
    assistant = Assistant()
    print("Serving at http://localhost:8000/playground")
    assistant.serve(port=8000, enable_playground=True, debug=True)
    # assistant.serve(protocol=ServeProtocol.CLI)
