# Decision Guide

Quick answers to common "which do I use?" questions.

---

## Memory: MemoryPreset vs Memory vs None

| Value | Behavior | When to use |
|-------|----------|-------------|
| `memory=None` or `memory=MemoryPreset.DISABLED` | No memory. Stateless agent. | Simple one-shot Q&A, stateless APIs. |
| `memory=MemoryPreset.DEFAULT` | Defaults: core + episodic, top_k=10. | Quick multi-turn chat; you don't need custom config. |
| `memory=Memory(...)` | Full config: types, top_k, backend, etc. | When you need remember/recall/forget, or to tweak top_k, backends, injection. |

**Example:**

```python
from syrin.enums import MemoryPreset

# Stateless
agent = Agent(model=m, system_prompt="...", memory=MemoryPreset.DISABLED)
# Or: memory=None

# Multi-turn with defaults
agent = Agent(model=m, system_prompt="...", memory=MemoryPreset.DEFAULT)
# Or: Agent.with_memory(m)

# Full control (remember/recall/forget)
agent = Agent(model=m, system_prompt="...", memory=Memory(top_k=15))
```

---

## Budget vs Context.token_limits (TokenLimits)

**Budget** = cost limits ($). **Context** = token limits and formation policy (what goes in the window, compaction, mode).

**Why both have `run`, `per`, `on_exceeded`:** Same mental model—limit per run or per period, with an `on_exceeded` callback. Different units:

| Concern | Use | Units | Config |
|---------|-----|-------|--------|
| **Cost (USD)** | Budget | Dollars | `Agent(budget=Budget(run=1.0, per=RateLimit(day=50)))` |
| **Tokens** | Context.token_limits (TokenLimits) | Token count | `Agent(config=AgentConfig(context=Context(token_limits=TokenLimits(run=100_000))))` or `context=ContextConfig(token_limits=...)` for the simple case |

- **Budget** = how much money the agent can spend (input + output cost).
- **TokenLimits** = how many tokens can be used (run or per hour/day/week/month).
- **When to use which:** Use Budget for cost control. Use TokenLimits when you need hard token caps (e.g. provider limits, rate limiting by tokens).
- **Use both** when you want to cap both cost and tokens (e.g. $10/day and 100k tokens/day).

---

## When to use which loop strategy

**Recommended for most agents:** `REACT` (if you have tools) or `SINGLE_SHOT` (if no tools).

| Strategy | Behavior | Use when |
|----------|----------|----------|
| `REACT` | Reason → Act → Observe. Agent calls tools, sees results, continues. | Agent has tools (search, calculator, etc.). Default for tool-using agents. |
| `SINGLE_SHOT` | One LLM call, no tools. | Simple Q&A, no tools, fastest path. |
| `HITL` | Human-in-the-loop; requires approval before each tool call. | When tools need human approval. |

For plan-then-execute or code-generation loops, pass `loop=PlanExecuteLoop()` or `loop=CodeActionLoop()` directly (not via `loop_strategy`).

---

## Formation mode: PUSH vs PULL

| Mode | Behavior | When to use |
|------|----------|-------------|
| `PUSH` (default) | Use full conversation from memory (last N or all). | Short or linear chats. Simpler. |
| `PULL` | Retrieve segments by relevance to current query. | Long conversations with topic shifts; only include relevant prior turns. Requires Memory. |

---

## Quick agent creation paths

| Need | Use |
|------|-----|
| Minimal (no memory, no budget) | `Agent.basic(model, system_prompt="...")` |
| With memory | `Agent.with_memory(model)` or `Agent.presets.assistant()` |
| With budget | `Agent.with_budget(model, budget=Budget(run=0.50))` |
| Full control | `Agent(model=..., system_prompt=..., memory=..., budget=..., ...)` or `Agent.builder(model).with_*(...).build()` |
