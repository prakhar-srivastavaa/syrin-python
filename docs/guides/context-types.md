# Context Types: RunContext vs PromptContext

Syrin uses two "context" objects at different stages. Here's when each is used.

## Quick Reference

| Context | When | Who gets it | Purpose |
|---------|------|-------------|---------|
| **RunContext** | During tool execution | Tools with `ctx: RunContext[Deps]` | Dependency injection, agent info, budget state |
| **PromptContext** | During system prompt resolution | Callable prompts, `@system_prompt` methods | Dynamic prompts, built-in vars, memory/budget access |

## RunContext

**Location:** `syrin.RunContext[DepsT]`  
**When:** Passed to tools when the first parameter is `ctx: RunContext[YourDeps]`.  
**Used by:** `@syrin.tool` methods that need injected dependencies, conversation ID, or budget state.

```python
from syrin import Agent, RunContext

@dataclass
class MyDeps:
    db: Database
    user_id: str

class MyAgent(Agent):
    @syrin.tool
    def get_data(self, ctx: RunContext[MyDeps], query: str) -> str:
        return ctx.deps.db.query(user_id=ctx.deps.user_id, q=query)
```

**Fields:** `deps`, `agent_name`, `conversation_id`, `budget_state`, `retry_count`.

## PromptContext

**Location:** `syrin.prompt.context.PromptContext`  
**When:** Passed to callable system prompts or `@system_prompt` methods.  
**Used by:** Dynamic prompts that need date, agent_id, memory, or budget state.

```python
def my_prompt(ctx: PromptContext) -> str:
    return f"You are {ctx.agent_id}. Today is {ctx.date}. Remaining budget: {ctx.budget_state}."
```

**Fields:** `agent`, `agent_id`, `conversation_id`, `memory`, `budget_state`, `date`, `builtins`.

## Why Two Contexts?

- **RunContext** = tool runtime (execution). Generic over `Deps` for dependency injection.
- **PromptContext** = prompt resolution time (before LLM call). No deps; focused on prompt template vars and agent state.

They serve different phases of the agent lifecycle. Use RunContext in tools; use PromptContext in dynamic prompts.
