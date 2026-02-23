"""Tests for @syrin.tool decorator (tool.py)."""

from __future__ import annotations

import pytest

from syrin.tool import tool
from syrin.types import ToolSpec


def test_tool_decorator_without_args() -> None:
    @tool
    def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"weather in {city}"

    assert isinstance(get_weather, ToolSpec)
    assert get_weather.name == "get_weather"
    assert "city" in get_weather.parameters_schema.get("properties", {})
    assert get_weather.parameters_schema.get("required") == ["city"]
    assert get_weather.func is not None
    assert get_weather.func("Paris") == "weather in Paris"


def test_tool_decorator_with_name_and_description() -> None:
    @tool(name="fetch_weather", description="Fetch current weather by city")
    def get_weather(city: str) -> str:  # noqa: ARG001
        return "sunny"

    assert get_weather.name == "fetch_weather"
    assert get_weather.description == "Fetch current weather by city"


def test_tool_schema_types() -> None:
    @tool
    def mixed(a: str, b: int, c: float, d: bool) -> None:
        """Mixed params."""
        pass

    props = mixed.parameters_schema.get("properties", {})
    assert props.get("a") == {"type": "string"}
    assert props.get("b") == {"type": "integer"}
    assert props.get("c") == {"type": "number"}
    assert props.get("d") == {"type": "boolean"}
    assert set(mixed.parameters_schema.get("required", [])) == {"a", "b", "c", "d"}


def test_tool_optional_param() -> None:
    @tool
    def with_optional(x: str, y: int | None = None) -> str:  # noqa: ARG001
        """Optional y."""
        return x

    props = with_optional.parameters_schema.get("properties", {})
    assert "x" in props
    assert "y" in props
    assert with_optional.parameters_schema.get("required") == ["x"]


# =============================================================================
# TOOL EDGE CASES - TRY TO BREAK FUNCTIONALITY
# =============================================================================


def test_tool_with_no_parameters() -> None:
    """Tool with no parameters."""

    @tool
    def no_params() -> str:
        """No parameters."""
        return "done"

    assert no_params.name == "no_params"
    assert no_params.func() == "done"
    assert no_params.parameters_schema.get("required") == []


def test_tool_with_many_parameters() -> None:
    """Tool with many parameters."""

    @tool
    def many_params(a: str, b: str, c: str, d: str, e: str) -> str:
        """Many params."""
        return f"{a}{b}{c}{d}{e}"

    props = many_params.parameters_schema.get("properties", {})
    assert len(props) == 5
    assert many_params.func("1", "2", "3", "4", "5") == "12345"


def test_tool_with_default_values() -> None:
    """Tool with default values."""

    @tool
    def with_defaults(a: str, b: int = 10, c: float = 1.5) -> str:
        """Defaults."""
        return f"{a}-{b}-{c}"

    assert with_defaults.func("x") == "x-10-1.5"
    assert with_defaults.func("x", 20) == "x-20-1.5"
    assert with_defaults.func("x", 20, 3.0) == "x-20-3.0"


def test_tool_with_list_type() -> None:
    """Tool with list type."""

    @tool
    def list_param(items: list[str]) -> int:
        """List param."""
        return len(items)

    props = list_param.parameters_schema.get("properties", {})
    assert props.get("items", {}).get("type") == "array"


def test_tool_with_dict_type() -> None:
    """Tool with dict type."""

    @tool
    def dict_param(data: dict) -> str:
        """Dict param."""
        return str(data)

    props = dict_param.parameters_schema.get("properties", {})
    assert props.get("data", {}).get("type") == "object"


def test_tool_returns_none() -> None:
    """Tool that returns None."""

    @tool
    def returns_none(_x: str) -> None:
        """Returns None."""
        return None

    assert returns_none.func("test") is None


def test_tool_with_complex_return() -> None:
    """Tool that returns complex data."""

    @tool
    def complex_return() -> dict:
        """Returns complex."""
        return {"key": "value", "nested": {"a": 1}}

    result = complex_return.func()
    assert result["key"] == "value"
    assert result["nested"]["a"] == 1


def test_tool_preserves_function_metadata() -> None:
    """Tool preserves function metadata."""

    @tool
    def documented() -> str:
        """This is the docstring."""
        return "result"

    assert "docstring" in documented.description.lower() or documented.description != ""


def test_tool_with_special_characters_in_name() -> None:
    """Tool with underscores and numbers."""

    @tool
    def tool_123_test(x: int) -> int:
        """Test tool."""
        return x * 2

    assert tool_123_test.name == "tool_123_test"
    assert tool_123_test.func(5) == 10


def test_tool_execution_error_handling() -> None:
    """Tool that raises error during execution."""

    @tool
    def failing_tool() -> str:
        raise ValueError("Intentional error")

    # Tool execution should raise
    with pytest.raises(ValueError):
        failing_tool.func()


def test_tool_with_union_types() -> None:
    """Tool with union types."""

    @tool
    def union_param(x: int | str) -> str:
        """Union type."""
        return str(x)

    result = union_param.func(5)
    assert result == "5"
    result = union_param.func("hello")
    assert result == "hello"
