"""Events and Hooks Example.

Demonstrates:
- Full Hook enum — all lifecycle events
- agent.events.on() / on_all() API
- EventContext with full state (budget, model, iteration)
- Chaining multiple event handlers
- Hook categories: agent, guardrail, budget, memory, output, context

Run: python -m examples.10_observability.events_and_hooks
Visit: http://localhost:8000/playground
Requires: uv pip install syrin[serve]
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent, Budget, Hook, Memory, warn_on_exceeded
from syrin.enums import MemoryType

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# 1. Basic event handlers
events_log: list[str] = []
agent = Agent(
    model=almock,
    budget=Budget(run=1.0, on_exceeded=warn_on_exceeded),
    memory=Memory(),
)
agent.events.on(Hook.AGENT_RUN_START, lambda _ctx: events_log.append("run_start"))
agent.events.on(Hook.AGENT_RUN_END, lambda _ctx: events_log.append("run_end"))
agent.events.on(Hook.LLM_REQUEST_START, lambda _ctx: events_log.append("llm_start"))
agent.events.on(Hook.LLM_REQUEST_END, lambda _ctx: events_log.append("llm_end"))
agent.response("Hello!")
print(f"Events fired: {events_log}")

# 2. Hook categories
agent_hooks = [h for h in Hook if h.value.startswith("agent")]
print(f"Agent hooks (sample): {[h.value for h in agent_hooks[:5]]}")
print(f"Total hooks: {len(list(Hook))}")

# 3. Multiple handlers on same event
agent2 = Agent(model=almock)
calls = []
agent2.events.on(Hook.AGENT_RUN_START, lambda _: calls.append("handler_1"))
agent2.events.on(Hook.AGENT_RUN_START, lambda _: calls.append("handler_2"))
agent2.events.on(Hook.AGENT_RUN_START, lambda _: calls.append("handler_3"))
agent2.response("Hi")
print(f"All 3 handlers fired: {calls}")

# 4. Cost tracking via events
total_cost = {"value": 0.0}
total_tokens = {"value": 0}


def track_cost(ctx: dict) -> None:
    total_cost["value"] += ctx.get("cost", 0)


def track_tokens(ctx: dict) -> None:
    tokens = ctx.get("tokens", {})
    total_tokens["value"] += tokens.get("total_tokens", 0) if isinstance(tokens, dict) else 0


agent3 = Agent(model=almock)
agent3.events.on(Hook.LLM_REQUEST_END, track_cost)
agent3.events.on(Hook.LLM_REQUEST_END, track_tokens)
for i in range(3):
    agent3.response(f"Question {i + 1}")
print(f"Total cost: ${total_cost['value']:.6f}, tokens: {total_tokens['value']}")

# 5. Memory events
memory_ops: list[str] = []
agent4 = Agent(model=almock, memory=Memory())
agent4.events.on(Hook.MEMORY_STORE, lambda _: memory_ops.append("store"))
agent4.events.on(Hook.MEMORY_RECALL, lambda _: memory_ops.append("recall"))
agent4.remember("Python is great", memory_type=MemoryType.CORE)
agent4.recall("Python")
print(f"Memory operations: {memory_ops}")


class EventsDemoAgent(Agent):
    name = "events-demo"
    description = "Agent with events and hooks"
    model = almock
    system_prompt = "You are a helpful assistant."
    budget = Budget(run=1.0, on_exceeded=warn_on_exceeded)
    memory = Memory()


if __name__ == "__main__":
    agent = EventsDemoAgent()
    print("Serving at http://localhost:8000/playground")
    agent.serve(port=8000, enable_playground=True, debug=True)
