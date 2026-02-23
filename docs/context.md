# Context Management

Syrin provides a comprehensive context management system that automatically handles token limits, compaction, and optimization. It's designed to prevent "context rot" - the degradation of LLM performance as conversation history grows.

## Installation

No additional dependencies required. The context system is built into Syrin and **enabled by default** on all agents.

## Quick Start

```python
from Syrin import Agent, Model

# Context is automatically enabled with smart defaults
agent = Agent(
    model=Model("openai/gpt-4o"),
    system_prompt="You are a helpful assistant.",
)

# Context is automatically managed
result = agent.response("Hello! Ask me anything.")
```

That's it! The context system:
- Counts tokens automatically
- Compacts context when approaching limits (75% by default)
- Emits events for observability
- Tracks statistics you can inspect

---

## Why Context Management Matters

LLMs have limited context windows (typically 8K-200K tokens). As conversations grow:

1. **Performance degrades** - "Context rot" causes LLM quality to drop
2. **Costs increase** - More tokens = more money
3. **Quality varies** - Start/end of context remembered better than middle

Syrin handles this automatically so you don't have to think about it.

---

## Basic Usage

### Default Context

```python
from Syrin import Agent, Model

agent = Agent(
    model=Model("openai/gpt-4o"),
    system_prompt="You are a helpful assistant.",
)

# Access context configuration
print(agent.context)
# Context(max_tokens=None, auto_compact_at=0.75, thresholds=[])

# After running, check statistics
result = agent.response("What is Python?")
print(agent.context_stats)
# ContextStats(total_tokens=45, max_tokens=128000, utilization=0.04%, ...)
```

### Custom Context

```python
from Syrin import Agent, Model
from Syrin.context import Context

agent = Agent(
    model=Model("openai/gpt-4o"),
    system_prompt="You are a helpful assistant.",
    context=Context(
        max_tokens=80000,      # Set context window limit
        auto_compact_at=0.75,  # Trigger compaction at 75%
    ),
)
```

---

## Configuration Options

### Context

```python
from Syrin.context import Context

context = Context(
    max_tokens=80000,           # Max tokens in context (None = auto-detect from model)
    auto_compact_at=0.75,       # Threshold (0-1) to trigger auto-compaction
    thresholds=[...],           # List of ContextThreshold (see below)
)
```

### ContextThreshold

Define actions at specific token percentages:

```python
from Syrin.context import ContextThreshold
from Syrin.enums import ContextAction

context = Context(
    max_tokens=80000,
    thresholds=[
        # At 50% - warn developers
        ContextThreshold(at=50, action=ContextAction.WARN),
        
        # At 70% - summarize oldest messages
        ContextThreshold(at=70, action=ContextAction.SUMMARIZE_OLDEST),
        
        # At 85% - compress context
        ContextThreshold(at=85, action=ContextAction.COMPRESS),
        
        # At 100% - raise error
        ContextThreshold(at=100, action=ContextAction.ERROR),
    ],
)
```

### ContextAction Options

| Action | Description |
|--------|-------------|
| `WARN` | Emit warning event |
| `SUMMARIZE_OLDEST` | Summarize oldest messages |
| `COMPRESS` | Compress context |
| `DROP_LOW_PRIORITY` | Drop lowest priority sections |
| `DROP_MEDIUM_PRIORITY` | Drop medium priority sections |
| `SWITCH_MODEL` | Switch to a different model |
| `STOP` | Stop execution |
| `ERROR` | Raise an error |
| `CUSTOM` | Call custom handler |

---

## Context Properties

### agent.context

Returns the `Context` configuration:

```python
agent = Agent(model=Model("openai/gpt-4o"), context=Context(max_tokens=80000))

print(agent.context.max_tokens)      # 80000
print(agent.context.auto_compact_at) # 0.75
```

### agent.context_stats

Returns `ContextStats` from the last call:

```python
agent = Agent(model=Model("openai/gpt-4o"), context=Context(max_tokens=80000))
result = agent.response("Hello!")

stats = agent.context_stats
print(stats.total_tokens)          # Tokens used in last call
print(stats.max_tokens)           # Max context window
print(stats.utilization)          # Fraction (0-1)
print(stats.utilization_percent)  # Percentage (0-100)
print(stats.compacted)            # Was context compacted?
print(stats.compaction_count)     # How many times compacted
print(stats.compaction_method)    # Method used: "middle_out_truncate", "summarize", etc.
print(stats.thresholds_triggered)  # List of triggered threshold actions
```

---

## Events

The context system emits events you can listen to:

### context.compact

Emitted when context is compacted:

```python
from Syrin import Agent, Model
from Syrin.context import Context

agent = Agent(
    model=Model("openai/gpt-4o"),
    context=Context(max_tokens=5000),  # Small limit to force compaction
)

def on_compact(ctx):
    print(f"Context compacted!")
    print(f"  Method: {ctx['method']}")
    print(f"  Tokens: {ctx['tokens_before']} -> {ctx['tokens_after']}")
    print(f"  Messages: {ctx['messages_before']} -> {ctx['messages_after']}")

agent.events.on("context.compact", on_compact)
```

### context.threshold

Emitted when a threshold is crossed:

```python
from Syrin.context import ContextThreshold
from Syrin.enums import ContextAction

agent = Agent(
    model=Model("openai/gpt-4o"),
    context=Context(
        max_tokens=5000,
        thresholds=[
            ContextThreshold(at=50, action=ContextAction.WARN),
            ContextThreshold(at=100, action=ContextAction.ERROR),
        ],
    ),
)

def on_threshold(ctx):
    print(f"Threshold reached: {ctx['at']}%")
    print(f"  Action: {ctx['action']}")
    print(f"  Tokens: {ctx['tokens']}/{ctx['max_tokens']}")

agent.events.on("context.threshold", on_threshold)
```

---

## Observability

Context operations are automatically traced when debug mode is enabled:

```python
agent = Agent(
    model=Model("openai/gpt-4o"),
    context=Context(max_tokens=80000),
    debug=True,  # Enables trace output
)
```

Output shows:
- `context.prepare` spans with token counts
- Compaction events
- Threshold triggers

---

## Custom Context Manager

For advanced use cases, implement the `ContextManager` Protocol:

```python
from Syrin import Agent, Model
from Syrin.context import ContextManager, ContextBudget, ContextPayload
from Syrin.context.counter import TokenCounter
from typing import Any

class MyContextManager(ContextManager):
    """Custom context strategy - keeps only recent messages."""
    
    def prepare(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list[dict[str, Any]],
        memory_context: str,
        budget: ContextBudget,
    ) -> ContextPayload:
        counter = TokenCounter()
        
        # Keep only last 5 messages
        recent = messages[-5:] if len(messages) > 5 else messages
        
        tokens = counter.count_messages(recent).total
        
        return ContextPayload(
            messages=recent,
            system_prompt=system_prompt,
            tools=tools,
            tokens=tokens,
        )
    
    def on_compact(self, event: Any) -> None:
        print(f"Compacted: {event.method}")

# Use custom manager
agent = Agent(
    model=Model("openai/gpt-4o"),
    context=MyContextManager(),
)
```

---

## Compaction Methods

Syrin uses smart default compaction strategies:

### Middle-Out Truncate

Keeps the beginning and end of conversation (LLMs have better recall at start/end):

```
Before: [S][M1][M2][M3][M4][M5][M6][M7][M8]
                 ↓
After:  [S][M1][M2][M7][M8]
```

### Summarize

For significant overages, summarizes older messages into a summary message.

---

## Token Counting

The context system uses `tiktoken` for accurate token counting:

```python
from Syrin.context import TokenCounter

counter = TokenCounter()

# Count tokens in text
tokens = counter.count("Hello, world!")
print(tokens)  # 4

# Count tokens in messages
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"},
]
result = counter.count_messages(messages)
print(result.total)      # Total tokens
print(result.system)    # System prompt tokens
print(result.messages)  # Message tokens

# Count tool definitions
tools = [{"type": "function", "name": "search", "description": "Search the web"}]
tool_tokens = counter.count_tools(tools)
```

---

## Integration with Memory

Context works seamlessly with Syrin memory:

```python
from Syrin import Agent, Model
from Syrin.context import Context
from Syrin.memory import Memory
from Syrin.enums import MemoryType

# Memory is auto-injected into context
agent = Agent(
    model=Model("openai/gpt-4o"),
    memory=Memory(types=[MemoryType.CORE, MemoryType.EPISODIC]),
    context=Context(max_tokens=80000),
)

# Memories are automatically retrieved and injected
result = agent.response("What do you know about me?")

# Check memory usage in context stats
print(agent.context_stats.thresholds_triggered)
```

---

## Complete Example

```python
from Syrin import Agent, Model
from Syrin.context import Context, ContextThreshold
from Syrin.enums import ContextAction
from Syrin.memory import Memory, MemoryType

# Create agent with full context configuration
agent = Agent(
    model=Model("openai/gpt-4o"),
    system_prompt="You are a helpful assistant with long-term memory.",
    memory=Memory(types=[MemoryType.CORE, MemoryType.EPISODIC]),
    context=Context(
        max_tokens=128000,
        auto_compact_at=0.75,
        thresholds=[
            ContextThreshold(at=50, action=ContextAction.WARN),
            ContextThreshold(at=70, action=ContextAction.SUMMARIZE_OLDEST),
            ContextThreshold(at=100, action=ContextAction.ERROR),
        ],
    ),
)

# Listen to events
def on_compact(ctx):
    print(f"Context compacted: {ctx['method']}")

def on_threshold(ctx):
    print(f"Threshold: {ctx['at']}% -> {ctx['action']}")

agent.events.on("context.compact", on_compact)
agent.events.on("context.threshold", on_threshold)

# Run conversation
result = agent.response("Hello! I'm John.")
print(result.content)

# Store some memories
agent.remember("User's name is John", memory_type=MemoryType.CORE)
agent.remember("We discussed Python programming", memory_type=MemoryType.EPISODIC)

# Continue conversation - memories are auto-injected
result = agent.response("What is my name?")

# Check context stats
print(f"Tokens used: {agent.context_stats.total_tokens}")
print(f"Utilization: {agent.context_stats.utilization:.1%}")
print(f"Compacted: {agent.context_stats.compacted}")
print(f"Compaction count: {agent.context_stats.compaction_count}")
```

---

## API Reference

### Classes

| Class | Description |
|-------|-------------|
| `Context` | Main configuration for context management |
| `ContextStats` | Statistics from last context preparation |
| `ContextBudget` | Internal budget tracking |
| `ContextThreshold` | Threshold + action definition |
| `TokenCounter` | Token counting with tiktoken |
| `ContextCompactor` | Default compaction implementation |
| `MiddleOutTruncator` | Keep start/end of conversation |
| `DefaultContextManager` | Default context manager |

### ContextManager Protocol

```python
class ContextManager(Protocol):
    def prepare(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list[dict[str, Any]],
        memory_context: str,
        budget: ContextBudget,
    ) -> ContextPayload:
        ...
    
    def on_compact(self, event: CompactionResult) -> None:
        ...
```

---

## Troubleshooting

### Context not being compacted

- Check `agent.context_stats.utilization` - compaction only triggers when utilization exceeds `auto_compact_at`
- Ensure messages are actually exceeding the budget

### Tokens still too high

- Reduce `max_tokens` in Context
- Lower `auto_compact_at` to trigger compaction earlier

### Custom handler not called

- Ensure threshold action is `ContextAction.CUSTOM`
- Pass handler function to `ContextThreshold`

### Memory not being injected

- Check that `memory` is configured on Agent
- Verify memory backend is working
