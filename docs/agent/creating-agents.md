# Creating Agents

**Recommended order:** (1) Class-based, (2) Builder, (3) Presets, (4) Direct constructor.

## Class-based (Primary)

Subclass `Agent`, set `model`, `system_prompt`, `tools`, etc. on the class; instantiate with `MyAgent()`. Use when you have named agent types (e.g. `Researcher`, `Writer`):

```python
from syrin import Agent
from syrin.model import Model

class Assistant(Agent):
    # model = Model.OpenAI("gpt-4o-mini")
    model = Model.Almock()  # No API Key needed
    system_prompt = "You are a helpful assistant."

agent = Assistant()
response = agent.response("Hello")
```

Class attributes (`model`, `system_prompt`, `tools`, `budget`, `guardrails`) become defaults. Instance arguments override them.

## Builder (Secondary)

For agents with many options, the fluent **Builder** scales cleanly:

```python
from syrin import Agent, Budget
from syrin.model import Model

agent = (
    Agent.builder(Model.OpenAI("gpt-4o-mini"))
    .with_system_prompt("You are helpful.")
    .with_budget(Budget(run=0.50))
    .with_tools([search, calculate])
    .build()
)
```

**When to use:** Any agent with 3+ options. Keeps construction readable and avoids a long constructor call.

## Presets (Quick Paths)

For common patterns, use presets:

```python
agent = Agent.basic(Model.OpenAI("gpt-4o-mini"))           # Minimal: no memory, no budget
agent = Agent.with_memory(Model.OpenAI("gpt-4o-mini"))     # Multi-turn with memory
agent = Agent.with_budget(Model.OpenAI("gpt-4o-mini"))     # With cost budget
agent = Agent.presets.assistant()                          # Full assistant preset
agent = Agent.presets.research()                           # Research agent with tools
```

## Direct instantiation (Tertiary)

Call `Agent(model=..., system_prompt=..., tools=[...])` with no subclass. Use for one-off agents or scripts.

```python
from syrin import Agent
from syrin.model import Model

agent = Agent(
    model=Model.Almock(),  # No API Key needed
    system_prompt="You are helpful.",
)
response = agent.response("Hello")
```

All configuration is passed to `Agent()` as constructor arguments.

## Inheritance and MRO

Syrin uses `__init_subclass__` to merge or override class-level attributes along the MRO (Method Resolution Order).

### Merge vs Override

| Attribute        | Behavior | Description                                      |
|------------------|----------|--------------------------------------------------|
| `model`          | Override | First defined in MRO wins                        |
| `system_prompt`  | Override | First defined in MRO wins                        |
| `budget`         | Override | First defined in MRO wins                        |
| `tools`          | Merge    | All tools from the MRO are concatenated          |
| `guardrails`     | Merge    | All guardrails from the MRO are concatenated     |

### Example: Inheritance

```python
from syrin import Agent
from syrin.model import Model
from syrin.tool import tool

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results: {query}"

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

class BaseResearcher(Agent):
    # model = Model.OpenAI("gpt-4o")
    model = Model.Almock()  # No API Key needed
    system_prompt = "You are a researcher."
    tools = [search]

class MathResearcher(BaseResearcher):
    model = Model.OpenAI("gpt-4o-mini")  # Overrides parent (use real model when you have a key)
    tools = [calculate]  # Merged with [search] → [search, calculate]

agent = MathResearcher()
# agent has: model=gpt-4o-mini, system_prompt="You are a researcher.",
#            tools=[search, calculate]
```

## Overriding at Instantiation

Constructor arguments override class attributes:

```python
agent = MathResearcher(
    system_prompt="You are a math specialist.",  # Overrides class default
    tools=[calculate],  # Replaces merged tools for this instance
)
```

## Required: Model

`model` is required. It can be set on the class or passed to the constructor. If neither is provided, `TypeError` is raised:

```python
agent = Agent()  # TypeError: Agent requires model
agent = Agent(model=Model.Almock())  # OK (or Model.OpenAI("gpt-4o") when you have a key)
```

## Class-Level Defaults Summary

| Attribute       | Type                       | Default   | Merge? |
|----------------|----------------------------|-----------|--------|
| `model`        | `Model \| ModelConfig`     | None      | No     |
| `system_prompt`| `str`                      | `""`      | No     |
| `tools`        | `list[ToolSpec]`           | `[]`      | Yes    |
| `budget`       | `Budget \| None`           | None      | No     |
| `guardrails`   | `list[Guardrail]`          | `[]`      | Yes    |

## Agent Name and Description (Discovery + Routing)

Agents have `name` and `description` for discovery, routing, and Agent Cards. Set on the class as `name` or `_agent_name` (same thing; `_agent_name` avoids type-checker override warnings).

**Precedence (highest wins):**
1. Constructor: `Agent(name="my-agent", ...)`
2. Class: `name = "researcher"` or `_agent_name = "researcher"`
3. Fallback: lowercase class name (e.g. `ResearcherAgent` → `"researcheragent"`)

```python
class ResearcherAgent(Agent):
    _agent_name = "researcher"
    _agent_description = "Searches and summarizes information from the web"
    model = Model.OpenAI("gpt-4o")
    system_prompt = "You are a researcher."

agent = ResearcherAgent(name="custom")  # Uses "custom" (constructor overrides class)
```

`description` defaults to `""`.

## Next Steps

- [Constructor Reference](constructor.md) — All parameters in detail
- [Loop Strategies](loop-strategies.md) — Customizing execution
- [Multi-Agent Patterns](multi-agent-patterns.md) — Pipelines and teams
