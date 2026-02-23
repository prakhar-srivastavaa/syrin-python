"""Task decorator for agent tasks."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

from syrin.types import TaskSpec


def task(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
) -> TaskSpec | Callable[..., TaskSpec]:
    """
    Decorator that marks a method as an agent task. The task can be invoked
    from outside as agent.task_name(args). Parameter types are validated from
    type hints; execution is wrapped with tracing metadata (no-op for now).
    """

    def decorator(f: Callable[..., Any]) -> TaskSpec:
        task_name = name or f.__name__
        hints = get_type_hints(f) if hasattr(f, "__annotations__") else {}
        sig = inspect.signature(f)
        parameters: dict[str, Any] = {}
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            parameters[param_name] = str(hints.get(param_name, param.annotation))
        return_type = hints.get("return", None)
        return TaskSpec(
            name=task_name,
            parameters=parameters,
            return_type=return_type,
            func=f,
        )

    if func is not None:
        return decorator(func)
    return decorator
