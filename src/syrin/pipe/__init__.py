"""Pipe operator: apply functions in sequence (sync and async)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, Generic, TypeVar

T = TypeVar("T")


def _run_maybe_async(val: Any) -> Any:
    """If val is a coroutine, run it and return result; else return val."""
    if asyncio.iscoroutine(val):
        return asyncio.run(val)
    return val


class Pipe(Generic[T]):
    """
    Fluent pipe: Pipe(value).then(fn1).then(fn2).result().
    Supports async: if a then() callable returns a coroutine, result() runs it.
    """

    def __init__(self, value: T) -> None:
        self._value = value
        self._steps: list[Callable[[Any], Any]] = []

    def then(self, fn: Callable[[T], Any]) -> Pipe[Any]:
        """Add a step to the pipe. Returns a new Pipe for chaining."""
        out: Pipe[Any] = Pipe.__new__(Pipe)
        out._value = self._value
        out._steps = list(self._steps) + [fn]
        return out

    def result(self) -> Any:
        """Run the chain and return the final value. Handles async steps via asyncio.run."""
        val: Any = self._value
        for step in self._steps:
            val = step(val)
            val = _run_maybe_async(val)
        return val

    async def result_async(self) -> Any:
        """Run the chain and return the final value; awaits async steps."""
        val: Any = self._value
        for step in self._steps:
            val = step(val)
            if asyncio.iscoroutine(val):
                val = await val
        return val

    def __or__(self, other: Callable[[T], Any]) -> Pipe[Any]:
        """Pipe value through callable: pipe(x) | fn is pipe(x).then(fn)."""
        return self.then(other)

    def __call__(self) -> T:
        """Return current value (before running steps). For compatibility."""
        return self._value


def pipe(
    value: T,
    *fns: Callable[[Any], Any],
) -> Pipe[Any]:
    """
    Wrap a value for piping. pipe(x) returns Pipe(x); pipe(x, f, g) returns Pipe(x).then(f).then(g).
    Use .result() to run the chain.
    """
    p: Pipe[Any] = Pipe(value)
    for fn in fns:
        p = p.then(fn)
    return p
