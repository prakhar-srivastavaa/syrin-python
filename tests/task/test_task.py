"""Tests for @syrin.task decorator (task.py)."""

from __future__ import annotations

from syrin.task import task
from syrin.types import TaskSpec


def test_task_decorator_returns_task_spec() -> None:
    @task
    def my_task(x: int) -> str:
        """Do something."""
        return str(x)

    assert isinstance(my_task, TaskSpec)
    assert my_task.name == "my_task"
    assert "int" in str(my_task.parameters.get("x", ""))
    assert my_task.func is not None
    assert my_task.func(42) == "42"


def test_task_with_name() -> None:
    @task(name="custom_name")
    def my_task() -> None:
        pass

    assert my_task.name == "custom_name"


# =============================================================================
# TASK EDGE CASES - TRY TO BREAK FUNCTIONALITY
# =============================================================================


def test_task_with_no_parameters() -> None:
    """Task with no parameters."""

    @task
    def no_params() -> str:
        """No parameters."""
        return "done"

    assert no_params.name == "no_params"
    assert no_params.func() == "done"


def test_task_with_many_parameters() -> None:
    """Task with many parameters."""

    @task
    def many_params(a: str, b: str, c: str, d: str, e: str) -> str:
        return f"{a}{b}{c}{d}{e}"

    assert many_params.func("1", "2", "3", "4", "5") == "12345"


def task_for_test():
    """Helper for testing."""
    pass


def test_task_with_complex_return() -> None:
    """Task returns complex data."""

    @task
    def complex_task() -> dict:
        return {"key": "value"}

    result = complex_task.func()
    assert result["key"] == "value"


def test_task_with_default_values() -> None:
    """Task with default parameter values."""

    @task
    def default_task(x: str, y: int = 10) -> str:
        return f"{x}-{y}"

    assert default_task.func("hello") == "hello-10"
    assert default_task.func("hello", 20) == "hello-20"


def test_task_with_optional_params() -> None:
    """Task with optional parameters."""

    @task
    def optional_task(x: str, y: int | None = None) -> str:
        return f"{x}-{y}"

    assert optional_task.func("a") == "a-None"
    assert optional_task.func("a", 5) == "a-5"


def test_task_with_various_types() -> None:
    """Task with various parameter types."""

    @task
    def types_task(a: int, b: float, c: bool, d: list) -> str:
        return f"{a}-{b}-{c}-{len(d)}"

    assert types_task.func(1, 2.5, True, [1, 2]) == "1-2.5-True-2"


def test_task_with_unicode() -> None:
    """Task with unicode in parameters."""

    @task
    def unicode_task(text: str) -> str:
        return text

    result = unicode_task.func("Hello 🌍")
    assert "🌍" in result


def test_task_func_attribute() -> None:
    """Task preserves original function in func attribute."""

    @task
    def preserved_task() -> str:
        return "preserved"

    # Function is stored in func attribute
    assert preserved_task.func() == "preserved"


def test_task_with_special_characters_in_name() -> None:
    """Task with special characters in name."""

    @task
    def task_123_test() -> str:
        return "ok"

    assert task_123_test.name == "task_123_test"
