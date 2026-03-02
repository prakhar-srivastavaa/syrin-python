"""Agent with Tools Example.

Demonstrates:
- Creating an Agent with tools
- Tool execution during response
- Multiple tools working together

Run: python -m examples.01_minimal.agent_with_tools
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
from syrin import Agent, tool

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@tool
def calculate(a: float, b: float, operation: str = "add") -> str:
    """Perform basic arithmetic. operation: add, subtract, multiply, divide."""
    if operation == "add":
        return str(a + b)
    if operation == "subtract":
        return str(a - b)
    if operation == "multiply":
        return str(a * b)
    if operation == "divide":
        return str(a / b) if b != 0 else "Error: division by zero"
    return "Unknown operation"


@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get weather for a city. unit: celsius or fahrenheit."""
    return f"The weather in {city} is 22°{unit[0].upper()}"


class MathAssistant(Agent):
    _agent_name = "math-assistant"
    _agent_description = "Assistant with calculator and weather tools"
    model = almock
    system_prompt = "You are a helpful assistant. Use tools for calculations."
    tools = [calculate, get_weather]


if __name__ == "__main__":
    assistant = MathAssistant()
    print("Serving at http://localhost:8000/playground")
    assistant.serve(port=8000, enable_playground=True, debug=True)
    # assistant.serve(protocol=ServeProtocol.CLI)
