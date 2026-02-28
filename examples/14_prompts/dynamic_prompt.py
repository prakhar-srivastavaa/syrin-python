"""Dynamic System Prompt Example.

Demonstrates:
- @prompt with prompt_vars for runtime variable injection
- Per-call prompt_vars override
- effective_prompt_vars() and get_prompt_builtins() for introspection

Run: python -m examples.14_prompts.dynamic_prompt
Visit: http://localhost:8000/playground
Requires: uv pip install syrin[serve]
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent, prompt

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@prompt
def persona_prompt(
    user_name: str,
    tone: str = "professional",
) -> str:
    """Persona system prompt with runtime variables."""
    return f"You assist {user_name or 'the user'}. Be {tone}."


class PersonaAgent(Agent):
    name = "persona-agent"
    description = "Agent with dynamic prompt_vars"
    model = almock
    system_prompt = persona_prompt
    prompt_vars = {"tone": "friendly"}


# Instance prompt_vars override class
alice = PersonaAgent(prompt_vars={"user_name": "Alice", "tone": "casual"})
vars_ = alice.effective_prompt_vars()
print(f"Effective vars: user_name={vars_['user_name']}, tone={vars_['tone']}")

r1 = alice.response("What can you help me with?")
print(f"Alice: {r1.content[:80]}...")

# Per-call override (same agent, different user)
r2 = alice.response("Hi", prompt_vars={"user_name": "Bob"})
print(f"Bob (per-call): {r2.content[:80]}...")


if __name__ == "__main__":
    agent = PersonaAgent(prompt_vars={"user_name": "Demo", "tone": "concise"})
    print("Serving at http://localhost:8000/playground")
    agent.serve(port=8000, enable_playground=True, debug=True)
