"""TOON Format Example.

Demonstrates:
- TOON (Token-Oriented Object Notation) tool schema format
- Why TOON uses ~40% fewer tokens than JSON
- ToolSpec.schema_to_toon() and ToolSpec.to_format()
- DocFormat.TOON vs DocFormat.JSON
- Efficiency comparison across multiple tools

Run: python -m examples.05_tools.toon_format
Visit: http://localhost:8000/playground
Requires: uv pip install syrin[serve]
"""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent, tool
from syrin.enums import DocFormat

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@tool
def calculate(a: float, b: float, operation: str = "add") -> str:
    """Perform basic arithmetic operations.

    Args:
        a: First number
        b: Second number
        operation: One of add, subtract, multiply, divide
    """
    ops = {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b if b else 0}
    return str(ops.get(operation, "Unknown"))


@tool
def search_web(query: str, max_results: int = 5) -> str:
    """Search the web for information.

    Args:
        query: The search query to execute
        max_results: Maximum number of results (1-10)
    """
    return f"Found {max_results} results for: {query}"


@tool
def send_email(to: str, subject: str, body: str, priority: str = "normal") -> str:
    """Send an email to a recipient.

    Args:
        to: Email address of recipient
        subject: Email subject line
        body: Email body content
        priority: Priority level (low, normal, high)
    """
    return f"Email sent to {to}"


# 1. TOON vs JSON comparison (use method on tool spec)
tool_spec = calculate
json_schema = json.dumps(tool_spec.parameters_schema, indent=2)
toon_schema = tool_spec.schema_to_toon()
savings = ((len(json_schema) - len(toon_schema)) / len(json_schema)) * 100
print(
    f"Tool: {tool_spec.name}; JSON {len(json_schema)} chars, TOON {len(toon_schema)} chars; savings {savings:.1f}%"
)

# 2. Format conversion (use method on tool spec)
for fmt in [DocFormat.TOON, DocFormat.JSON]:
    schema = search_web.to_format(fmt)
    print(f"{fmt.value}: {json.dumps(schema)[:80]}...")

# 3. Multi-tool efficiency
tools = [calculate, search_web, send_email]
total_json = sum(len(json.dumps(t.parameters_schema)) for t in tools)
total_toon = sum(len(t.schema_to_toon()) for t in tools)
total_sv = ((total_json - total_toon) / total_json) * 100
print(f"3 tools: JSON {total_json}ch, TOON {total_toon}ch, savings {total_sv:.1f}%")


class ToolDemoAgent(Agent):
    _agent_name = "tool-demo"
    _agent_description = "Agent with TOON-format tools (calculate, search_web, send_email)"
    model = almock
    system_prompt = "You are a helpful assistant. Use tools when needed."
    tools = [calculate, search_web, send_email]


if __name__ == "__main__":
    agent = ToolDemoAgent()
    print("Serving at http://localhost:8000/playground")
    agent.serve(port=8000, enable_playground=True, debug=True)
