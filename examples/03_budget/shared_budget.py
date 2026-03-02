"""Shared Budget Example.

Demonstrates:
- Budget(shared=True) for multi-agent shared budgets
- Spawn with shared budget (child borrows from parent)
- Budget tracking across agents

Run: python -m examples.03_budget.shared_budget
Visit: http://localhost:8000/playground

Requires: uv pip install syrin[serve]
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent, Budget, warn_on_exceeded

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# 1. Shared budget across agents
shared = Budget(run=10.0, shared=True, on_exceeded=warn_on_exceeded)
parent = Agent(model=almock, budget=shared)
result = parent.response("Hello from parent")
print(f"Parent cost: ${result.cost:.6f}")
print(f"Parent budget: {parent.budget_state}")


# 2. Spawn child that borrows from shared budget
class Child(Agent):
    model = almock


result = parent.spawn(Child, task="Do work")
print(f"Child result: {result.content[:60]}...")
print(f"Parent budget after child: {parent.budget_state}")

# 3. Multiple children sharing budget
parent2 = Agent(model=almock, budget=Budget(run=10.0, shared=True))
for i in range(3):
    parent2.spawn(Child, task=f"Task {i + 1}")
print(f"Budget after 3 children: {parent2.budget_state}")


class SharedBudgetParent(Agent):
    """Parent agent with shared budget for spawning children."""

    _agent_name = "shared-budget"
    _agent_description = "Agent with shared budget (spawn children that borrow)"
    model = almock
    budget = Budget(run=10.0, shared=True, on_exceeded=warn_on_exceeded)


if __name__ == "__main__":
    agent = SharedBudgetParent()
    print("Serving at http://localhost:8000/playground")
    agent.serve(port=8000, enable_playground=True, debug=True)
