"""Persona Prompt with @system_prompt In-Class Example.

Demonstrates:
- @system_prompt decorator: encapsulate prompt inside agent class
- One system prompt per agent
- Method signatures: (self), (self, **kwargs)

Run: python -m examples.14_prompts.persona_prompt
Visit: http://localhost:8000/playground
Requires: uv pip install syrin[serve]
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent, system_prompt

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class PersonaAgent(Agent):
    _agent_name = "persona-agent"
    _agent_description = "Agent with @system_prompt in-class"
    model = almock

    @system_prompt
    def my_prompt(self, user_name: str = "", tone: str = "professional") -> str:
        """In-class system prompt. Receives template_variables."""
        return f"You assist {user_name or 'the user'}. Be {tone}."


if __name__ == "__main__":
    agent = PersonaAgent(template_variables={"user_name": "Carol", "tone": "witty"})
    r = agent.response("What's your personality?")
    print(f"Carol: {r.content[:80]}...")
    print("Serving at http://localhost:8000/playground")
    agent.serve(port=8000, enable_playground=True, debug=True)
