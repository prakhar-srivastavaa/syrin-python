"""Prompt Decorator Example.

Demonstrates:
- @prompt decorator for parameterized system prompts
- Creating specialized agents from a single template
- Prompt composition via function calls

Run: python -m examples.14_prompts.prompt_decorator
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent, prompt

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@prompt
def expert_prompt(domain: str, tone: str = "professional") -> str:
    """Generate system prompt for expert agents."""
    return f"You are an expert in {domain}. Provide accurate, detailed, and {tone} responses."


@prompt
def role_prompt(role: str, specialization: str = "") -> str:
    """Generate role-based system prompt."""
    base = f"You are a {role}."
    if specialization:
        base += f" You specialize in {specialization}."
    return base


# 1. @prompt for parameterized prompts
class ScienceExpert(Agent):
    model = almock
    system_prompt = expert_prompt(domain="quantum physics", tone="academic")


class BusinessExpert(Agent):
    model = almock
    system_prompt = expert_prompt(domain="business strategy", tone="practical")


# 2. Different agents from same template
science = ScienceExpert()
business = BusinessExpert()
question = "What is innovation?"
r1 = science.response(question)
r2 = business.response(question)
print(f"Science: {r1.content[:60]}...")
print(f"Business: {r2.content[:60]}...")


# 3. Role-based prompts
class Researcher(Agent):
    model = almock
    system_prompt = role_prompt(role="researcher", specialization="machine learning")


class Writer(Agent):
    model = almock
    system_prompt = role_prompt(role="technical writer")


# 4. Dynamic prompt at runtime (pass prompt_vars; Prompt resolves per call)
for domain in ["Python", "JavaScript", "Rust"]:
    agent = Agent(
        model=almock,
        system_prompt=expert_prompt,
        prompt_vars={"domain": domain, "tone": "concise"},
    )
    result = agent.response(f"What is {domain} best for?")
    print(f"{domain}: {result.content[:50]}...")
