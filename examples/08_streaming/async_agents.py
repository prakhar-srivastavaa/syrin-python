"""Async Agents Example.

Demonstrates:
- agent.arun() for async execution
- asyncio.gather() for parallel agent calls
- Async patterns with syrin agents

Run: python -m examples.08_streaming.async_agents
Visit: http://localhost:8000/playground
Requires: uv pip install syrin[serve]
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


async def example_basic_arun() -> None:
    """Basic async agent call."""
    print("\n" + "=" * 50)
    print("1. agent.arun() — basic async call")
    print("=" * 50)

    agent = Agent(model=almock, system_prompt="You are a helpful assistant.")
    result = await agent.arun("What is Python?")
    print(f"Answer: {result.content[:80]}...")
    print(f"Cost: ${result.cost:.6f}")


async def example_parallel_agents() -> None:
    """Run multiple agents in parallel with asyncio.gather()."""
    print("\n" + "=" * 50)
    print("2. asyncio.gather() — parallel agents")
    print("=" * 50)

    class Researcher(Agent):
        model = almock
        system_prompt = "You are a researcher."

    class Writer(Agent):
        model = almock
        system_prompt = "You are a writer."

    class Reviewer(Agent):
        model = almock
        system_prompt = "You are a reviewer."

    researcher = Researcher()
    writer = Writer()
    reviewer = Reviewer()

    results = await asyncio.gather(
        researcher.arun("Research AI trends"),
        writer.arun("Write about machine learning"),
        reviewer.arun("Review the code quality"),
    )

    for i, result in enumerate(results):
        agent_name = ["Researcher", "Writer", "Reviewer"][i]
        print(f"{agent_name}: {result.content[:60]}...")
        print(f"  Cost: ${result.cost:.6f}, Tokens: {result.tokens.total_tokens}")


async def example_sequential_async() -> None:
    """Sequential async calls where each depends on the previous."""
    print("\n" + "=" * 50)
    print("3. Sequential async — dependent calls")
    print("=" * 50)

    agent = Agent(model=almock, system_prompt="You are a helpful assistant.")

    r1 = await agent.arun("What is Python?")
    print(f"Step 1: {r1.content[:60]}...")

    r2 = await agent.arun(f"Summarize this: {r1.content[:50]}")
    print(f"Step 2: {r2.content[:60]}...")

    total_cost = r1.cost + r2.cost
    print(f"Total cost: ${total_cost:.6f}")


async def example_async_with_timeout() -> None:
    """Async with timeout protection."""
    print("\n" + "=" * 50)
    print("4. Async with timeout")
    print("=" * 50)

    agent = Agent(model=almock)

    try:
        result = await asyncio.wait_for(agent.arun("Hello!"), timeout=10.0)
        print(f"Result: {result.content[:60]}...")
    except asyncio.TimeoutError:
        print("Timed out!")


class AsyncDemoAgent(Agent):
    _agent_name = "async-demo"
    _agent_description = "Agent with async arun() execution"
    model = almock
    system_prompt = "You are a helpful assistant."


async def _run() -> None:
    await example_basic_arun()
    await example_parallel_agents()
    await example_sequential_async()
    await example_async_with_timeout()


if __name__ == "__main__":
    asyncio.run(_run())
    agent = AsyncDemoAgent()
    print("Serving at http://localhost:8000/playground")
    agent.serve(port=8000, enable_playground=True, debug=True)
