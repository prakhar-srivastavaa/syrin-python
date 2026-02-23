"""Tool decorator and schema generation for agent tools."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from typing import Any, Union, get_type_hints

from syrin.enums import DocFormat
from syrin.types import ToolSpec

_TYPE_TO_JSON: dict[type[Any], str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def schema_to_toon(schema: dict[str, Any], indent: int = 0) -> str:
    """Convert a JSON schema to TOON format (token-efficient).

    TOON (Token-Oriented Object Notation) examples:
    - "type": "string" -> type: string
    - "required": ["query"] -> required[1]: query
    - "properties": {...} -> properties:

    Args:
        schema: JSON schema dict
        indent: Current indentation level

    Returns:
        TOON-formatted string
    """
    lines: list[str] = []
    prefix = "  " * indent

    if not isinstance(schema, dict):
        return f"{prefix}{json.dumps(schema)}"

    for key, value in schema.items():
        if key == "type" and isinstance(value, str):
            lines.append(f"{prefix}{key}: {value}")
        elif key == "properties" and isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            for prop_name, prop_schema in value.items():
                lines.append(f"{prefix}  {prop_name}:")
                prop_lines = schema_to_toon(prop_schema, indent + 2).split("\n")
                for line in prop_lines:
                    if line.strip():
                        lines.append(line)
        elif key == "required" and isinstance(value, list):
            if value:
                lines.append(f"{prefix}{key}[{len(value)}]: {','.join(value)}")
            else:
                lines.append(f"{prefix}{key}: []")
        elif key == "items" and isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            item_lines = schema_to_toon(value, indent + 1).split("\n")
            for line in item_lines:
                if line.strip():
                    lines.append(line)
        elif isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            sub_lines = schema_to_toon(value, indent + 1).split("\n")
            for line in sub_lines:
                if line.strip():
                    lines.append(line)
        elif isinstance(value, list):
            if value:
                lines.append(f"{prefix}{key}: {json.dumps(value)}")
            else:
                lines.append(f"{prefix}{key}: []")
        elif isinstance(value, str):
            if value:
                lines.append(f"{prefix}{key}: {value}")
            else:
                lines.append(f"{prefix}{key}:")
        elif value is None:
            lines.append(f"{prefix}{key}:")
        else:
            lines.append(f"{prefix}{key}: {json.dumps(value)}")

    return "\n".join(lines)


def tool_schema_to_format(
    tool_spec: ToolSpec, format: DocFormat = DocFormat.TOON
) -> dict[str, Any]:
    """Convert a ToolSpec to the specified format for LLM providers.

    Args:
        tool_spec: The tool specification
        format: Output format (TOON, JSON, YAML)

    Returns:
        Provider-specific tool schema dict
    """
    if format == DocFormat.TOON:
        toon_params = schema_to_toon(tool_spec.parameters_schema or {})
        return {
            "type": "function",
            "function": {
                "name": tool_spec.name,
                "description": tool_spec.description or "",
                "parameters": toon_params,
            },
        }
    elif format == DocFormat.YAML:
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            import json

            return {
                "type": "function",
                "function": {
                    "name": tool_spec.name,
                    "description": tool_spec.description or "",
                    "parameters": json.dumps(tool_spec.parameters_schema or {}),
                },
            }

        return {
            "type": "function",
            "function": {
                "name": tool_spec.name,
                "description": tool_spec.description or "",
                "parameters": yaml.dump(tool_spec.parameters_schema or {}),
            },
        }
    else:
        return {
            "type": "function",
            "function": {
                "name": tool_spec.name,
                "description": tool_spec.description or "",
                "parameters": tool_spec.parameters_schema or {"type": "object", "properties": {}},
            },
        }


def _annotation_to_json_schema(annotation: Any) -> dict[str, Any]:
    """Convert a type annotation to a JSON schema fragment."""
    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", ())

    if annotation is type(None):
        return {"type": "null"}
    if origin is Union or (hasattr(annotation, "__args__") and type(None) in (args or ())):
        non_none = [a for a in (args or ()) if a is not type(None)]
        if len(non_none) == 1:
            return {"oneOf": [_annotation_to_json_schema(non_none[0]), {"type": "null"}]}
        if non_none:
            return {"oneOf": [_annotation_to_json_schema(a) for a in non_none]}
    if origin is list and args:
        return {"type": "array", "items": _annotation_to_json_schema(args[0])}
    if origin is dict and len(args) >= 2:
        value_ann = args[1]  # type: ignore[misc]  # safe after len check; mypy tuple narrow
        if value_ann is not type(None):
            return {
                "type": "object",
                "additionalProperties": _annotation_to_json_schema(value_ann),
            }
    if annotation in _TYPE_TO_JSON:
        return {"type": _TYPE_TO_JSON[annotation]}
    return {"type": "string"}


def _parameters_schema_from_function(func: Callable[..., Any]) -> dict[str, Any]:
    """Build a JSON schema for the function's parameters from type hints."""
    hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}
    sig = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        if name == "self" or name == "cls":
            continue
        ann = hints.get(name, param.annotation)
        if ann is inspect.Parameter.empty:
            ann = Any
        properties[name] = _annotation_to_json_schema(ann)
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "object", "properties": properties, "required": required}


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[..., Any] | ToolSpec:
    """
    Decorator to register a function as a Syrin tool. Extracts name, docstring,
    and builds a JSON schema from type hints (str, int, float, bool, list, dict).
    """

    def decorator(f: Callable[..., Any]) -> ToolSpec:
        tool_name = name or f.__name__
        desc = description or (inspect.getdoc(f) or "").strip().split("\n")[0] or ""
        params_schema = _parameters_schema_from_function(f)
        return ToolSpec(
            name=tool_name,
            description=desc,
            parameters_schema=params_schema,
            func=f,
        )

    if func is not None:
        return decorator(func)
    return decorator


__all__ = ["tool", "schema_to_toon", "tool_schema_to_format"]
