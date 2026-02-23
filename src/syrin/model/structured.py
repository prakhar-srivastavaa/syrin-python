"""Structured output system - Simple decorator-based API.

Usage:
    from syrin.model import structured, output, OutputType

    @structured
    class Sentiment:
        sentiment: str
        confidence: float

    model = Model.OpenAI("gpt-4o", output=Sentiment)
"""

from __future__ import annotations

from typing import Annotated, Any, TypeVar, cast, get_type_hints

T = TypeVar("T")


class StructuredOutput:
    """Container for structured output schema."""

    _schema: dict[str, Any]
    _pydantic_model: type | None = None

    def __init__(self, schema: dict[str, Any] | type) -> None:
        if isinstance(schema, type):
            self._schema = self._class_to_schema(schema)
            self._pydantic_model = self._create_pydantic_model(schema)
        else:
            self._schema = schema
            self._pydantic_model = self._create_pydantic_model_from_schema(schema)

    def _class_to_schema(self, cls: type) -> dict[str, Any]:
        """Convert a Python class to JSON schema."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

        for name, hint in get_type_hints(cls).items():
            if hasattr(hint, "__args__"):
                args = hint.__args__
                origin = getattr(hint, "__origin__", None)

                if origin is list:
                    item_type = self._python_type_to_json_type(args[0] if args else str)
                    schema["properties"][name] = {"type": "array", "items": item_type}
                elif origin is dict:
                    schema["properties"][name] = {"type": "object"}
                elif origin is Annotated:
                    meta = hint.__metadata__[0] if hint.__metadata__ else {}
                    json_type = self._python_type_to_json_type(args[0] if args else str)
                    schema["properties"][name] = {**json_type, **meta}
                else:
                    schema["properties"][name] = self._python_type_to_json_type(hint)
            else:
                schema["properties"][name] = self._python_type_to_json_type(hint)

            default = getattr(cls, name, None)
            if default is None:
                schema["required"].append(name)

        return schema

    def _python_type_to_json_type(self, py_type: type) -> dict[str, str]:
        """Convert Python type to JSON Schema type."""
        type_map = {
            str: {"type": "string"},
            int: {"type": "integer"},
            float: {"type": "number"},
            bool: {"type": "boolean"},
            list: {"type": "array"},
            dict: {"type": "object"},
        }

        if py_type in type_map:
            return type_map[py_type]

        if hasattr(py_type, "__origin__") and py_type.__origin__ is None:
            return {"type": "string"}

        return {"type": "string"}

    def _create_pydantic_model(self, cls: type) -> type:
        """Create a Pydantic model from a Python class."""
        try:
            from typing import get_type_hints

            from pydantic import BaseModel, create_model

            # Get the type hints for the class
            hints = get_type_hints(cls) if hasattr(cls, "__annotations__") else {}

            if not hints:
                # Fallback: create fields from class attributes
                hints = {}
                for name in dir(cls):
                    if not name.startswith("_"):
                        val = getattr(cls, name, None)
                        if val is not None:
                            hints[name] = type(val)

            if hints:
                return create_model(cls.__name__, **hints, __base__=BaseModel)
            return cls
        except Exception:
            return cls

    def _create_pydantic_model_from_schema(self, schema: dict[str, Any]) -> type:
        """Create a Pydantic model from JSON schema."""
        try:
            from pydantic import create_model

            properties = schema.get("properties", {})
            required = schema.get("required", [])

            field_definitions: dict[str, Any] = {}
            for name, prop in properties.items():
                prop_type = self._json_type_to_python_type(prop)
                if name in required:
                    field_definitions[name] = (prop_type, ...)
                else:
                    field_definitions[name] = (prop_type | None, None)

            return create_model("StructuredOutput", **field_definitions)  # type: ignore[no-any-return]
        except Exception:
            return type

    def _json_type_to_python_type(self, prop: dict[str, Any]) -> type:
        """Convert JSON Schema type to Python type."""
        json_type = prop.get("type", "string")
        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        return type_map.get(json_type, str)

    @property
    def schema(self) -> dict[str, Any]:
        """Get the JSON schema."""
        return self._schema

    @property
    def pydantic_model(self) -> type | None:
        """Get the Pydantic model (if available)."""
        return self._pydantic_model


def structured(cls: type[T]) -> type[T]:
    """Decorator to define structured output schema.

    Usage:
        @structured
        class Sentiment:
            sentiment: str  # "positive", "negative", "neutral"
            confidence: float  # 0.0 to 1.0
            explanation: str = ""  # Optional field

        # Use with model
        model = Model.OpenAI("gpt-4o", output=Sentiment)
    """
    obj = cast(Any, cls)
    obj._is_structured = True
    obj._structured_schema = StructuredOutput(cls).schema
    obj._structured_pydantic = StructuredOutput(cls).pydantic_model
    return cls


class OutputType:
    """Wrapper for output type that can be used with Agent/Model."""

    def __init__(self, output_cls: type | StructuredOutput) -> None:
        if isinstance(output_cls, StructuredOutput):
            self._output_cls = output_cls.pydantic_model or output_cls._schema
        else:
            self._output_cls = output_cls
        self._structured = (
            StructuredOutput(output_cls)
            if not isinstance(output_cls, StructuredOutput)
            else output_cls
        )

    @property
    def schema(self) -> dict[str, Any]:
        return self._structured.schema

    @property
    def pydantic_model(self) -> type | None:
        return self._structured.pydantic_model

    @property
    def model_class(self) -> type:
        if isinstance(self._output_cls, type):
            return self._output_cls
        return self._structured.pydantic_model or type(self._structured)

    def __repr__(self) -> str:
        name = getattr(self._output_cls, "__name__", "Schema")
        return f"OutputType({name})"


def output(output_cls: type[T]) -> OutputType:
    """Shorthand decorator to create an output type.

    Usage:
        @output
        class Sentiment:
            sentiment: str
            confidence: float
    """
    return OutputType(output_cls)


__all__ = [
    "StructuredOutput",
    "structured",
    "OutputType",
    "output",
]
