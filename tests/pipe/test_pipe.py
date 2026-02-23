"""Tests for pipe operator (pipe.py)."""

from __future__ import annotations

import pytest

from syrin.pipe import pipe


def test_pipe_wrap() -> None:
    p = pipe(10)
    assert p() == 10
    assert p.result() == 10


def test_pipe_then_sync() -> None:
    p = pipe(5).then(lambda x: x + 1).then(lambda x: x * 2)
    assert p.result() == 12


def test_pipe_single_fn() -> None:
    p = pipe(3, lambda x: x * 2)
    assert p.result() == 6


def test_pipe_multi_fn() -> None:
    p = pipe(1, lambda x: x + 1, lambda x: x * 2, lambda x: x - 1)
    assert p.result() == 3


def test_pipe_or_operator() -> None:
    p = pipe(10) | (lambda x: x + 5) | (lambda x: x * 2)
    assert p.result() == 30


def test_pipe_async_step() -> None:
    async def double(x: int) -> int:
        return x * 2

    p = pipe(7).then(double)
    assert p.result() == 14


@pytest.mark.asyncio
async def test_pipe_result_async() -> None:
    async def add_one(x: int) -> int:
        return x + 1

    p = pipe(10).then(add_one).then(lambda x: x * 2)
    assert await p.result_async() == 22


# =============================================================================
# PIPE EDGE CASES - TRY TO BREAK FUNCTIONALITY
# =============================================================================


def test_pipe_with_none() -> None:
    """Pipe with None value."""
    p = pipe(None).then(lambda x: x)
    assert p.result() is None


def test_pipe_with_empty_string() -> None:
    """Pipe with empty string."""
    p = pipe("").then(lambda x: x + "test")
    assert p.result() == "test"


def test_pipe_chaining_many_steps() -> None:
    """Pipe with many chained steps."""
    p = pipe(0)
    for i in range(100):
        p = p.then(lambda x, _n=i: x + 1)
    assert p.result() == 100


def test_pipe_with_exception() -> None:
    """Pipe with function that raises exception."""

    def failing_fn(_x):
        raise ValueError("Intentional error")

    p = pipe(10).then(failing_fn)
    with pytest.raises(ValueError):
        p.result()


def test_pipe_with_complex_types() -> None:
    """Pipe with complex data types."""
    p = pipe({"a": 1}).then(lambda d: {**d, "b": 2}).then(lambda d: list(d.values()))
    assert p.result() == [1, 2]


def test_pipe_or_operator_chain() -> None:
    """Pipe with multiple | operators."""
    p = pipe(1) | (lambda x: x + 1) | (lambda x: x * 3) | (lambda x: x - 1)
    assert p.result() == 5


def test_pipe_nested_pipes() -> None:
    """Nested pipe operations."""
    inner = pipe(5).then(lambda x: x * 2)
    outer = pipe(inner.result()).then(lambda x: x + 10)
    assert outer.result() == 20


def test_pipe_with_list() -> None:
    """Pipe with list operations."""
    p = pipe([1, 2, 3]).then(lambda lst: [x * 2 for x in lst]).then(sum)
    assert p.result() == 12


def test_pipe_with_unicode() -> None:
    """Pipe with unicode strings."""
    p = pipe("Hello ").then(lambda x: x + "🌍").then(lambda x: x.upper())
    assert "🌍" in p.result()


def test_pipe_zero_steps() -> None:
    """Pipe with no transformation steps."""
    p = pipe(42)
    assert p.result() == 42


def test_pipe_identity_function() -> None:
    """Pipe with identity function."""
    p = pipe("test").then(lambda x: x)
    assert p.result() == "test"


def test_pipe_with_floats() -> None:
    """Pipe with floating point numbers."""
    p = pipe(0.1).then(lambda x: x + 0.2)
    assert abs(p.result() - 0.3) < 1e-10


@pytest.mark.asyncio
async def test_pipe_async_exception() -> None:
    """Async pipe with exception."""

    async def failing_async(_x):
        raise RuntimeError("Async error")

    p = pipe(10).then(failing_async)
    with pytest.raises(RuntimeError):
        await p.result_async()


def test_pipe_result_called_multiple_times() -> None:
    """Calling result() multiple times."""
    p = pipe(5).then(lambda x: x * 2)
    assert p.result() == 10
    assert p.result() == 10  # Should be idempotent


def test_pipe_with_large_number() -> None:
    """Pipe with very large number."""
    p = pipe(10**100).then(lambda x: x + 1)
    assert p.result() == 10**100 + 1
