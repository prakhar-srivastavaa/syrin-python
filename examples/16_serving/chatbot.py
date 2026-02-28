"""Chatbot Example — Full-featured chatbot with context, memory, guardrails, and checkpoints.

Demonstrates:
- Context: Context(max_tokens=) for controlled context window
- Memory: Memory with 4 types, Decay, auto_store (stores user+assistant turns as episodic)
- Guardrails: ContentFilter + LengthGuardrail for input/output safety
- Checkpoints: CheckpointConfig with STEP trigger and memory storage
- Budget: Per-run and rate limits
- HTTP serving: agent.serve(port=8000)

Run: python -m examples.16_serving.chatbot
Visit: http://localhost:8000/chat (POST), http://localhost:8000/playground

Requires: uv pip install syrin[serve]
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dotenv import load_dotenv

from examples.models.models import gpt4_mini
from syrin import (
    Agent,
    Budget,
    CheckpointConfig,
    CheckpointTrigger,
    Decay,
    Memory,
    RateLimit,
)
from syrin.context import Context
from syrin.enums import DecayStrategy, MemoryType
from syrin.guardrails import ContentFilter, LengthGuardrail

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

guardrails = [
    ContentFilter(blocked_words=["spam", "scam", "phishing"], name="NoSpam"),
    LengthGuardrail(max_length=4000, name="ResponseLength"),
]

memory = Memory(
    types=[MemoryType.CORE, MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL],
    top_k=10,
    auto_store=True,
    decay=Decay(
        strategy=DecayStrategy.EXPONENTIAL,
        half_life_hours=24,
        reinforce_on_access=True,
        min_importance=0.2,
    ),
)

context = Context(max_tokens=16000)

checkpoint = CheckpointConfig(
    storage="memory",
    trigger=CheckpointTrigger.STEP,
    max_checkpoints=10,
)


class Chatbot(Agent):
    """Full-featured chatbot with context, memory, guardrails, and checkpoints."""

    name = "chatbot"
    description = "Conversational chatbot with memory, guardrails, and checkpoints"
    model = gpt4_mini
    system_prompt = (
        "You are a helpful, friendly chatbot. You recall past conversation turns automatically. "
        "Keep responses concise but informative. Be respectful and safe."
    )
    memory = memory
    context = context
    guardrails = guardrails
    checkpoint = checkpoint
    budget = Budget(run=0.50, per=RateLimit(hour=10, day=100))


if __name__ == "__main__":
    agent = Chatbot()
    print("Chatbot serving at http://localhost:8000")
    print("  POST /chat  — send messages")
    print("  GET /playground — web UI")
    agent.serve(port=8000, enable_playground=True, debug=True)
