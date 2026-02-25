# Syrin examples

This directory contains example scripts showing how to use the Syrin library.

## Core Examples

- **hello_model.py** — Create a `Model` and inspect config (no API key).
- **hello_agent.py** — Agent with budget, prompt template (`@prompt`), and Response. Requires API key and `Syrin[anthropic]`.
- **hello_memory.py** — Agent with `BufferMemory`; multi-turn conversation with context.
- **agent_inheritance.py** — Base + Sub agent: class-level model/prompt/tools, merged tools, overridden prompt.

## Guardrails Examples

The `guardrails/` directory contains comprehensive examples for the Syrin Guardrails system:

- **`guardrails/01_foundation.py`** — Content filtering, PII detection, parallel evaluation
- **`guardrails/02_authority.py`** — Permissions, budget enforcement, human approval, capability tokens
- **`guardrails/03_intelligence.py`** — Context awareness, escalation detection, adaptive thresholds, red teaming
- **`guardrails/04_complete_workflows.py`** — End-to-end workflows combining all layers

See `guardrails/README.md` for details.

## Running examples

From the project root (with `OPENAI_API_KEY` or other provider keys in `examples/.env`):

```bash
# Run all examples (core, memory, multi_agent, advanced)
PYTHONPATH=. python examples/main.py

# Run a subset
PYTHONPATH=. python examples/main.py core
PYTHONPATH=. python examples/main.py memory

# Run a single example module
PYTHONPATH=. python -m examples.core.basic_agent
PYTHONPATH=. python -m examples.memory.basic_memory
```

## Setup

1. From project root with virtualenv activated:
   ```bash
   source .venv/bin/activate
   uv pip install -e ".[dev,anthropic]"
   ```

2. Add a `.env` file in this directory (`examples/.env`) with your API key:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ```
   Optional: `ANTHROPIC_MODEL_ID=anthropic/claude-3-7-sonnet-latest` (default; use any valid id from Anthropic’s API).

Examples load `.env` via `python-dotenv`. Run: `python examples/hello_agent.py`

**CLI (transport):** `Syrin serve -a examples.hello_agent:agent` (interactive REPL), `Syrin cost`, `Syrin version`, `Syrin run <script.py>`.
