"""Persona Prompt with @system_prompt In-Class Example.

Demonstrates:
- @system_prompt decorator: encapsulate prompt inside agent class
- One system prompt per agent
- Method signatures: (self), (self, **kwargs)

Run: python -m examples.14_prompts.persona_prompt
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent, system_prompt

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class PersonaAgent(Agent):
    model = almock

    @system_prompt
    def my_prompt(self, user_name: str = "", tone: str = "professional") -> str:
        """In-class system prompt. Receives prompt_vars."""
        return f"You assist {user_name or 'the user'}. Be {tone}."


agent = PersonaAgent(prompt_vars={"user_name": "Carol", "tone": "witty"})
r = agent.response("What's your personality?")
print(f"Carol: {r.content[:80]}...")
