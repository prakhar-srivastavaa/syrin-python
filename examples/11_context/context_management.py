"""Context Management Example.

Demonstrates:
- Context(max_tokens=) for window limits
- ContextCompactor for manual compaction
- MiddleOutTruncator for intelligent truncation
- ContextThreshold with actions at usage levels
- context_stats for monitoring
- Custom ContextManager via Protocol
- CONTEXT_COMPRESS event

Run: python -m examples.11_context.context_management
Visit: http://localhost:8000/playground
Requires: uv pip install syrin[serve]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent
from syrin.context import (
    Context,
    ContextCompactor,
    ContextManager,
    ContextPayload,
    MiddleOutTruncator,
)
from syrin.context.counter import TokenCounter
from syrin.enums import Hook
from syrin.threshold import ContextThreshold

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# 1. Default context
agent = Agent(model=almock, system_prompt="You are a helpful assistant.")
print(f"Context config: {agent.context}, utilization: {agent.context_stats.utilization:.2%}")

# 2. Custom context limit
agent = Agent(model=almock, context=Context(max_tokens=80000))
print(f"Max tokens: {agent.context.max_tokens}")

# 3. Context stats
agent = Agent(
    model=almock,
    system_prompt="You are a helpful assistant.",
    context=Context(max_tokens=128000),
)
stats = agent.context_stats
print(f"Total tokens: {stats.total_tokens}, utilization: {stats.utilization:.4%}")

# 4. Manual compaction (MiddleOutTruncator)
counter = TokenCounter()
messages: list[dict[str, str]] = [{"role": "system", "content": "You are helpful."}]
for i in range(20):
    messages.append({"role": "user", "content": f"Message {i}: Tell me about topic {i}."})
    messages.append({"role": "assistant", "content": f"Response about topic {i}. " * 50})
before = counter.count_messages(messages).total
truncator = MiddleOutTruncator()
result = truncator.compact(messages, 2000, counter)
after = counter.count_messages(result.messages).total
print(f"Before: {before} tokens, after: {after} tokens")

# 5. ContextCompactor
messages = [{"role": "system", "content": "You are helpful."}]
for i in range(10):
    messages.append({"role": "user", "content": f"User message {i} - " + "x" * 100})
    messages.append({"role": "assistant", "content": f"Response {i} - " + "y" * 100})
before_tokens = counter.count_messages(messages).total
compact_result = ContextCompactor().compact(messages, 3000)
after_tokens = counter.count_messages(compact_result.messages).total
print(f"Compacted: {before_tokens} -> {after_tokens} tokens")

# 6. Context thresholds
threshold_events: list[str] = []
agent = Agent(
    model=almock,
    system_prompt="You are a helpful assistant.",
    context=Context(
        max_tokens=5000,
        thresholds=[
            ContextThreshold(at=50, action=lambda _: threshold_events.append("50%")),
            ContextThreshold(at=70, action=lambda _: threshold_events.append("70%")),
            ContextThreshold(at=100, action=lambda _: threshold_events.append("100%")),
        ],
    ),
)

# 7. CONTEXT_COMPRESS event
compact_events: list[str] = []
agent = Agent(
    model=almock,
    system_prompt="You are a helpful assistant.",
    context=Context(max_tokens=128000),
)
agent.events.on(Hook.CONTEXT_COMPRESS, lambda _: compact_events.append("compressed"))
agent.response("Hello!")


# 8. Custom ContextManager
class MyContextManager(ContextManager):
    def prepare(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list[dict[str, Any]],
        memory_context: str,
        budget: Any,
        context: Context | None = None,
    ) -> ContextPayload:
        counter = TokenCounter()
        all_tokens = counter.count_messages(messages).total
        return ContextPayload(
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            tokens=all_tokens,
        )

    def on_compact(self, event: Any) -> None:
        print(f"Custom compaction: {event}")


agent = Agent(model=almock, context=MyContextManager())
agent.response("Hello custom!")


class ContextDemoAgent(Agent):
    _agent_name = "context-demo"
    _agent_description = "Agent with context management"
    model = almock
    system_prompt = "You are a helpful assistant."
    context = Context(max_tokens=1000)


if __name__ == "__main__":
    agent = ContextDemoAgent()
    print("Serving at http://localhost:8000/playground")
    agent.serve(port=8000, enable_playground=True, debug=True)
