"""OpenRouter integration test for Syrin.

Tests Model.OpenRouter with direct model.complete() and Agent.
Run from project root: python test_openrouter.py
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

# Syrin examples load from examples/.env
load_dotenv(Path(__file__).resolve().parent / "examples" / ".env")

from syrin import Agent, Model
from syrin.enums import MessageRole
from syrin.types import Message

API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = "stepfun/step-3.5-flash:free"

print(f"API key loaded: {'YES' if API_KEY else 'NO - check examples/.env'}\n")
print("=" * 50)


async def main() -> None:
    # ── Test 1: model.acomplete() directly ─────────────────
    print("Test 1: model.acomplete() directly")
    model = Model.OpenRouter(MODEL_NAME, api_key=API_KEY, temperature=0.5, max_tokens=256)
    messages = [Message(role=MessageRole.USER, content="What is 2 + 2? Answer in one line.")]
    response = await model.acomplete(messages)
    print(f"  Response : {response.content}")
    print(f"  Tokens   : {response.token_usage}")
    print()

    # ── Test 2: Agent.arun() ───────────────────────────────
    print("Test 2: Agent.arun()")
    agent = Agent(
        model=Model.OpenRouter(MODEL_NAME, api_key=API_KEY, temperature=0.5, max_tokens=256),
        system_prompt="You are a helpful assistant. Keep answers short.",
    )
    result = await agent.arun("Name the capital of India in one word.")
    print(f"  Response : {result.content}")
    print(f"  Cost     : ${result.cost:.6f}")
    print(f"  Model    : {result.model}")
    print()

    print("=" * 50)
    print("All tests passed!")


asyncio.run(main())