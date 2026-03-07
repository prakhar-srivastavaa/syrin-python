"""Checkpoint Example.

Demonstrates:
- CheckpointConfig for configuring state persistence
- save_checkpoint() / load_checkpoint() / list_checkpoints()
- Trigger types: MANUAL, STEP, TOOL, ERROR, BUDGET
- Storage backends: memory, sqlite, filesystem
- Checkpoint report access

Run: python -m examples.12_checkpoints.basic_checkpoint
Visit: http://localhost:8000/playground
Requires: uv pip install syrin[serve]
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent, AgentConfig, CheckpointConfig, CheckpointTrigger

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# 1. Basic checkpointing
agent = Agent(model=almock, system_prompt="You are a research assistant.")
checkpoint_id = agent.save_checkpoint()
print(f"Saved checkpoint: {checkpoint_id}")
checkpoints = agent.list_checkpoints()
print(f"Available: {checkpoints}")

# 2. Auto checkpoint triggers
for trigger in [
    CheckpointTrigger.STEP,
    CheckpointTrigger.TOOL,
    CheckpointTrigger.ERROR,
    CheckpointTrigger.BUDGET,
    CheckpointTrigger.MANUAL,
]:
    agent = Agent(
        model=almock,
        config=AgentConfig(checkpoint=CheckpointConfig(storage="memory", trigger=trigger)),
    )
    print(f"  {trigger.value}")

# 3. Checkpoint configuration
config = CheckpointConfig(
    enabled=True,
    storage="sqlite",
    path="/tmp/research_agent.db",
    trigger=CheckpointTrigger.STEP,
    max_checkpoints=5,
    compress=False,
)
agent = Agent(model=almock, config=AgentConfig(checkpoint=config))

# 4. Save and restore
agent = Agent(model=almock)
cp_id = agent.save_checkpoint("my_research")
checkpoints = agent.list_checkpoints("my_research")
if checkpoints:
    agent.load_checkpoint(checkpoints[-1])

# 5. Storage backends
for backend in ["memory", "sqlite", "filesystem"]:
    CheckpointConfig(enabled=True, storage=backend)

# 6. Checkpoint report
agent = Agent(model=almock)
agent.save_checkpoint()
agent.save_checkpoint()
report = agent.get_checkpoint_report()
print(f"Saves: {report.checkpoints.saves}, Loads: {report.checkpoints.loads}")


class CheckpointDemoAgent(Agent):
    _agent_name = "checkpoint-demo"
    _agent_description = "Agent with checkpoint persistence"
    model = almock
    system_prompt = "You are a research assistant."


if __name__ == "__main__":
    agent = CheckpointDemoAgent()
    print("Serving at http://localhost:8000/playground")
    agent.serve(port=8000, enable_playground=True, debug=True)
