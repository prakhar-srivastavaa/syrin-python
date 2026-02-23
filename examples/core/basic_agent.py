"""Basic Agent Example.

Demonstrates:
- Creating an Agent with a model
- Making a simple response call
- Accessing response properties (content, cost, tokens)

Run: python -m examples.core.basic_agent
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from syrin import Agent, Model

logging.basicConfig(level=logging.ERROR)
# Suppress httpx async cleanup warnings
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("httpcore").setLevel(logging.CRITICAL)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODEL_ID = os.getenv("OPENAI_MODEL_NAME", "openai/gpt-4o-mini")


def example_basic_agent() -> None:
    """Simple agent with model and response."""
    print("\n" + "=" * 50)
    print("Basic Agent Example")
    print("=" * 50)

    class Assistant(Agent):
        model = Model(MODEL_ID)
        system_prompt = "You are a helpful assistant."

    assistant = Assistant()

    result = assistant.response("What is 2 + 2?")

    print(f"Question: What is 2 + 2?")
    print(f"Answer: {result.content}")
    print(f"Cost: ${result.cost:.6f}")
    print(f"Tokens: {result.tokens.total_tokens}")
    print(f"Model: {result.model}")


if __name__ == "__main__":
    example_basic_agent()
