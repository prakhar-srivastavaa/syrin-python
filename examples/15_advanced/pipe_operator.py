"""Pipe Operator Example.

Demonstrates:
- pipe() function for chaining functions
- Pipe class for fluent API (Pipe().then().result())
- | operator for piping
- Async pipe with result_async()
- Using pipe with Agent

Run: python -m examples.15_advanced.pipe_operator
Visit: http://localhost:8000/playground
Requires: uv pip install syrin[serve]
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent, Pipe, pipe

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def add_exclamation(text: str) -> str:
    return text + "!"


def uppercase(text: str) -> str:
    return text.upper()


def add_greeting(text: str) -> str:
    return f"Hello! {text}"


# 1. pipe() function
result = pipe("hello world", add_exclamation, uppercase).result()
print(f"pipe(): '{result}'")

# 2. Pipe class (fluent API)
result = Pipe("hello").then(add_exclamation).then(uppercase).then(add_greeting).result()
print(f"Pipe().then(): '{result}'")

# 3. | operator
result = (Pipe("hello world") | add_exclamation | uppercase | add_greeting).result()
print(f"| operator: '{result}'")


# 4. Pipe with Agent
class TransformAgent(Agent):
    name = "transform-agent"
    description = "Agent for pipe operator demo"
    model = almock
    system_prompt = "You are helpful."


agent = TransformAgent()


def process_with_agent(text: str) -> str:
    return agent.response(text).content


result = pipe("What is Python?", process_with_agent, lambda x: x.strip()).result()
print(f"Pipe + Agent: {result[:80]}...")

# 5. Multiple transformations
result = pipe(
    "hello world from python",
    lambda t: t.split(),
    lambda w: len(w),
    lambda c: f"Word count: {c}",
).result()
print(f"'{result}'")


# 6. Async pipe
async def async_double(x: int) -> int:
    await asyncio.sleep(0.01)
    return x * 2


async def async_add(x: int) -> int:
    await asyncio.sleep(0.01)
    return x + 10


async def run_async():
    return await Pipe(5).then(async_double).then(async_add).result_async()


result = asyncio.run(run_async())
print(f"Async: 5 → {result}")

# 7. Lambda pipe
result = pipe(
    "  hello world  ",
    lambda x: x.strip(),
    lambda x: x.upper(),
    lambda x: f"**{x}**",
).result()
print(f"Lambda: '{result}'")

# 8. Complex data pipe
data = {"name": "John", "age": 30, "city": "New York"}
result = pipe(
    data,
    lambda d: list(d.values()),
    lambda v: [str(x) for x in v],
    lambda s: " | ".join(s),
).result()
print(f"Data: '{result}'")

if __name__ == "__main__":
    agent = TransformAgent()
    print("Serving at http://localhost:8000/playground")
    agent.serve(port=8000, enable_playground=True, debug=True)
