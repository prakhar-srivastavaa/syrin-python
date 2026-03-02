"""Stream Async — Async streaming with astream().

Demonstrates:
- agent.astream(input) for async token-by-token output
- async for chunk in agent.astream(...)

Run: python -m examples.08_streaming.stream_async
Visit: http://localhost:8000/playground
Requires: uv pip install syrin[serve]
"""

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class AsyncStreamAgent(Agent):
    _agent_name = "async-stream-agent"
    _agent_description = "Async token-by-token streaming"
    model = almock
    system_prompt = "You are a helpful assistant."


async def _run() -> None:
    agent = AsyncStreamAgent()
    full_text = ""
    async for chunk in agent.astream("Explain machine learning in one sentence"):
        full_text += chunk.text
    print(full_text)
    print(f"Length: {len(full_text)}")


if __name__ == "__main__":
    asyncio.run(_run())
    agent = AsyncStreamAgent()
    print("Serving at http://localhost:8000/playground")
    agent.serve(port=8000, enable_playground=True, debug=True)
