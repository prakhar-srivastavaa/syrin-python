# Syrin

A Python library for building AI agents with built-in budget management, declarative agent definitions, and DSL-to-Python code generation. Companion to the [Syrin DSL](https://github.com/Syrin-labs/Syrin) programming language.

## Features

- **Budget control** — Per-run limits, rate limits, threshold-based model switching
- **Guardrails v2.0** — Three-layer safety system:
  - *Foundation*: Content filtering, PII detection, parallel evaluation
  - *Authority*: Permission-based authorization, budget enforcement, human approval
  - *Intelligence*: Context-aware tracking, escalation detection, adaptive thresholds, attack simulation
- **Agent inheritance** — Define agents via Python classes
- **DSL codegen** — Compile `.syrin` files to Python
- **Developer control** — Typed APIs, explicit configuration, no magic

## Requirements

- Python >= 3.11

## Install

```bash
uv add Syrin
# or
pip install Syrin
```

## Quick start

### Basic Agent

```python
from Syrin import Agent, Model

agent = Agent(
    model=Model("anthropic/claude-3-5-sonnet-20241022"),
    system_prompt="You are a helpful assistant.",
)
response = agent.response("Hello!")
print(response.content)
print(response.cost)
```

### With Budget

```python
from Syrin import Agent, Model, Budget, OnExceeded

agent = Agent(
    model=Model("anthropic/claude-3-5-sonnet-20241022"),
    system_prompt="You are a helpful assistant.",
    budget=Budget(
        run=1.00,  # $1 per run
        on_exceeded=OnExceeded.ERROR,  # Raise on budget exceeded
    ),
)
response = agent.response("What's 2+2?")
print(response.content)
print(response.cost)
```

### With Tools

```python
from Syrin import Agent, Model, tool, Budget

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

class MathAssistant(Agent):
    model = "anthropic/claude-3-5-sonnet-20241022"
    system_prompt = "You are a math assistant."
    tools = [calculate]

agent = MathAssistant(budget=Budget(run=0.50))
response = agent.response("What is 2 + 2?")
print(response.content)
```

### Class-based Agent

```python
from Syrin import Agent, Model, Budget, OnExceeded

class Assistant(Agent):
    model = "anthropic/claude-3-5-sonnet-20241022"
    system_prompt = "You are a helpful assistant."
    budget = Budget(run=1.00, on_exceeded=OnExceeded.WARN)

agent = Assistant()
response = agent.response("Explain quantum computing in one sentence.")
print(response.content)
```

### With Guardrails

```python
from Syrin import Agent, Model
from Syrin.guardrails import ContentFilter, PIIScanner

# Content filtering and PII protection
agent = Agent(
    model=Model("anthropic/claude-3-5-sonnet-20241022"),
    system_prompt="You are a helpful assistant.",
    guardrails=[
        ContentFilter(blocked_words=["password", "secret", "api_key"]),
        PIIScanner(redact=True),
    ],
)

# Guardrails automatically check input and output
response = agent.response("My email is user@example.com")
# PII is redacted in output
```

## Development

```bash
uv venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -e ".[dev]"
ruff check . && ruff format .
mypy src/Syrin
pytest
```

## License

MIT
