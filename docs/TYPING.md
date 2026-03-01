# Type Safety Standards

Syrin follows strict typing practices for reliability and maintainability.

## Configuration

- **mypy**: `strict = true` in `pyproject.toml`. All src modules must pass.
- **Pyright**: `pyrightconfig.json` enables strict mode for IDE support.

## Principles

1. **Avoid `Any`** — Use `object` or concrete types. Reserve `Any` only where truly needed (e.g., Pydantic `**kwargs` passthrough).
2. **Prefer `TypedDict`** — For structured dicts (JSON, config kwargs), use `TypedDict` with `Unpack` for `**kwargs`.
3. **Minimize `# type: ignore`** — Use only for known limitations (e.g., decorator return inference) with clear comments.
4. **Explicit public API types** — All public functions, methods, and constructors must have type annotations.

## Patterns

### TypedDict for kwargs

```python
from typing import TypedDict, Unpack

class MyConfigKwargs(TypedDict, total=False):
    host: str
    port: int

def serve(**kwargs: Unpack[MyConfigKwargs]) -> None:
    ...
```

### Protocol for extension points

```python
from typing import Protocol

class MyBackend(Protocol):
    def save(self, data: dict[str, object]) -> None: ...
    def load(self, key: str) -> dict[str, object] | None: ...
```

### Cast when type checker cannot infer

```python
from typing import cast

# Decorator returns ToolSpec but mypy loses type
plan_tool: ToolSpec = cast(ToolSpec, plan_agents)
```

## CI

`mypy src/syrin` must pass. Enable `disallow_any_explicit = true` once remaining `Any` usage is reduced.
