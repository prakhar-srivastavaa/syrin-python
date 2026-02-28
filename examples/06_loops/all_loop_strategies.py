"""Loop Strategies Example.

Demonstrates:
- SingleShotLoop — one LLM call, no tool iteration
- ReactLoop — Think, Act, Observe (default)
- HumanInTheLoop — human approval for tool calls
- PlanExecuteLoop — plan upfront, then execute
- CodeActionLoop — LLM writes Python to execute
- Custom loop — implement your own strategy

Run: python -m examples.06_loops.all_loop_strategies
Visit: http://localhost:8000/playground
Requires: uv pip install syrin[serve]
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent
from syrin.loop import (
    CodeActionLoop,
    HumanInTheLoop,
    LoopResult,
    PlanExecuteLoop,
    ReactLoop,
    SingleShotLoop,
)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def example_single_shot() -> None:
    """SingleShotLoop — one call, no tools."""
    print("\n--- SingleShotLoop ---")
    print("Use case: Simple Q&A without tools")

    agent = Agent(model=almock, loop=SingleShotLoop())
    result = agent.response("What is the capital of France?")
    print("Q: What is the capital of France?")
    print(f"A: {result.content[:80]}...")


def example_react() -> None:
    """ReactLoop — Think → Act → Observe."""
    print("\n--- ReactLoop ---")
    print("Use case: Multi-step tasks with tools")

    agent = Agent(model=almock, loop=ReactLoop(max_iterations=5))
    result = agent.response("What is 5 + 3?")
    print("Q: What is 5 + 3?")
    print(f"A: {result.content[:80]}...")


def example_human_in_the_loop() -> None:
    """HumanInTheLoop — human approval for tools."""
    print("\n--- HumanInTheLoop ---")
    print("Use case: Safety-critical apps needing human oversight")

    approved_count = 0

    async def approve(tool_name: str, args: dict) -> bool:
        nonlocal approved_count
        approved_count += 1
        print(f"  Tool '{tool_name}' requested — auto-approved")
        return True

    agent = Agent(model=almock, loop=HumanInTheLoop(approve=approve, max_iterations=5))
    result = agent.response("What is the square root of 144?")
    print(f"A: {result.content[:80]}...")
    print(f"Tools evaluated: {approved_count}")


def example_plan_execute() -> None:
    """PlanExecuteLoop — plan first, then execute."""
    print("\n--- PlanExecuteLoop ---")
    print("Use case: Complex multi-step tasks")

    agent = Agent(
        model=almock,
        loop=PlanExecuteLoop(max_plan_iterations=3, max_execution_iterations=10),
    )
    result = agent.response("Research programming languages and summarize pros/cons")
    print(f"A: {result.content[:100]}...")
    print(f"Iterations: {result.iterations}")


def example_code_action() -> None:
    """CodeActionLoop — LLM writes Python code."""
    print("\n--- CodeActionLoop ---")
    print("Use case: Math, data processing, code execution")

    agent = Agent(model=almock, loop=CodeActionLoop(max_iterations=5, timeout_seconds=30))
    result = agent.response("What is the sum of even numbers from 1 to 100?")
    print(f"A: {result.content[:80]}...")
    print(f"Iterations: {result.iterations}")


def example_custom_loop() -> None:
    """Custom loop — your own strategy."""
    print("\n--- Custom Loop ---")
    print("Use case: Specialized execution patterns")

    class MyLoop:
        """Custom loop — just implement run()."""

        name = "my_custom"

        async def run(self, ctx: object, user_input: str) -> LoopResult:
            from syrin.types import Message, MessageRole

            messages = [Message(role=MessageRole.USER, content=user_input)]
            response = await ctx.complete(messages)

            return LoopResult(
                content=f"[Custom] {response.content}",
                stop_reason="end_turn",
                iterations=1,
            )

    agent = Agent(model=almock, loop=MyLoop())
    result = agent.response("Hello!")
    print(f"A: {result.content[:80]}...")


class LoopDemoAgent(Agent):
    name = "loop-demo"
    description = "Agent with ReactLoop (Think, Act, Observe)"
    model = almock
    system_prompt = "You are a helpful assistant. Use tools when needed."
    loop = ReactLoop(max_iterations=5)


if __name__ == "__main__":
    example_single_shot()
    example_react()
    example_human_in_the_loop()
    example_plan_execute()
    example_code_action()
    example_custom_loop()

    agent = LoopDemoAgent()
    print("Serving at http://localhost:8000/playground")
    agent.serve(port=8000, enable_playground=True, debug=True)
