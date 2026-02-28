"""Handoff Interception Example.

Demonstrates:
- HANDOFF_START / HANDOFF_END hooks — observe what is passed
- HANDOFF_BLOCKED — when handoff is blocked by before-handler
- HandoffBlockedError — raise in before-handler to block
- HandoffRetryRequested — target signals invalid data, caller retries

Run: python -m examples.07_multi_agent.handoff_intercept
Visit: http://localhost:8000/playground
Requires: uv pip install syrin[serve]
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent, HandoffBlockedError, HandoffRetryRequested, Hook, Response

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class SourceAgent(Agent):
    name = "source"
    description = "Researcher that hands off to presenter"
    model = almock
    system_prompt = "You are a researcher. Provide brief findings."


class TargetAgent(Agent):
    name = "target"
    description = "Presents findings clearly"
    model = almock
    system_prompt = "You present findings clearly."


def main_observability() -> None:
    """Log what is passed on handoff for debugging."""
    print("=== Observability: HANDOFF_START / HANDOFF_END ===\n")

    source = SourceAgent()

    def on_start(ctx) -> None:
        print(f"  HANDOFF_START: {ctx.source_agent} → {ctx.target_agent}")
        print(f"    task: {ctx.task[:50]}...")
        print(f"    mem_count: {ctx.mem_count}, xfer_budget: {ctx.transfer_budget}")

    def on_end(ctx) -> None:
        print(f"  HANDOFF_END: cost=${ctx.cost:.4f}, duration={ctx.duration:.3f}s")
        print(f"    preview: {ctx.response_preview[:60]}...\n")

    source.events.on(Hook.HANDOFF_START, on_start)
    source.events.on(Hook.HANDOFF_END, on_end)

    result = source.handoff(TargetAgent, "Present: renewable energy benefits")
    print(f"Result: {result.content[:80]}...\n")


def main_block() -> None:
    """Block handoff when task fails validation."""
    print("=== Block: before-handler raises HandoffBlockedError ===\n")

    class BlockSourceAgent(Agent):
        name = "block-source"
        description = "Source for block demo"
        model = almock
        system_prompt = "You research."

    class BlockTargetAgent(Agent):
        name = "block-target"
        description = "Target for block demo"
        model = almock
        system_prompt = "You present."

    source = BlockSourceAgent()
    source.events.on(Hook.HANDOFF_BLOCKED, lambda ctx: print(f"  BLOCKED: {ctx.reason}\n"))

    def block_if_invalid(ctx) -> None:
        if "forbidden" in (ctx.task or "").lower():
            raise HandoffBlockedError(
                "Task contains forbidden keyword",
                ctx.source_agent,
                ctx.target_agent,
                ctx.task,
            )

    source.events.before(Hook.HANDOFF_START, block_if_invalid)

    # Valid task — succeeds
    result = source.handoff(BlockTargetAgent, "Summarize topic X")
    print(f"Valid task: {result.content[:50]}...\n")

    # Invalid task — blocked
    try:
        source.handoff(BlockTargetAgent, "Handle forbidden content")
    except HandoffBlockedError as e:
        print(f"Blocked as expected: {e}\n")


def main_retry() -> None:
    """Target raises HandoffRetryRequested; caller retries with format hint."""
    print("=== Retry: HandoffRetryRequested with format_hint ===\n")

    class RetrySourceAgent(Agent):
        name = "retry-source"
        description = "Source for retry demo"
        model = almock
        system_prompt = "You format data."

    class RetryTargetAgent(Agent):
        name = "retry-target"
        description = "Target for retry demo"
        model = almock
        system_prompt = "You expect JSON with 'title' and 'items'."

        _retry_count = 0

        def response(self, user_input: str, **kwargs: object):
            # Simulate target detecting bad format on first attempt
            self._retry_count = getattr(self, "_retry_count", 0) + 1
            if self._retry_count == 1 and "title" not in (user_input or ""):
                raise HandoffRetryRequested(
                    "Invalid format",
                    format_hint='Use JSON: {"title": str, "items": [...]}',
                )
            from syrin.types import TokenUsage

            return Response(
                content=f"Accepted: {user_input[:50]}...",
                cost=0.001,
                tokens=TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30),
            )

    source = RetrySourceAgent()

    task = "plain text"
    for attempt in range(2):
        try:
            result = source.handoff(RetryTargetAgent, task)
            print(f"Attempt {attempt + 1}: success — {result.content[:50]}...\n")
            break
        except HandoffRetryRequested as e:
            print(f"Attempt {attempt + 1}: retry requested — {e.format_hint}")
            task = '{"title": "Fixed", "items": ["a", "b"]}'
    else:
        print("Retry exhausted\n")


if __name__ == "__main__":
    main_observability()
    main_block()
    main_retry()

    agent = SourceAgent()
    print("Serving at http://localhost:8000/playground")
    agent.serve(port=8000, enable_playground=True, debug=True)
