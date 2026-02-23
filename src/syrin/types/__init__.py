"""Core types and Pydantic models for the Syrin library."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from syrin.enums import MessageRole
from syrin.types.validation import (
    OutputValidator,
    ToolOutput,
    ValidationAction,
    ValidationAttempt,
    ValidationContext,
    ValidationResult,
)


class ModelConfig(BaseModel):
    """Configuration for an LLM model."""

    name: str = Field(..., description="Human-readable model name")
    provider: str = Field(
        ..., description="Provider identifier (anthropic, openai, ollama, litellm)"
    )
    model_id: str = Field(..., description="Model identifier (e.g. anthropic/claude-3-5-sonnet)")
    api_key: str | None = Field(default=None, description="Optional API key (can use env)")
    base_url: str | None = Field(default=None, description="Optional base URL for API")
    output: type | None = Field(default=None, description="Structured output type")


class ToolSpec(BaseModel):
    """Specification for a tool callable by an agent."""

    name: str = Field(..., description="Tool name")
    description: str = Field(default="", description="Tool description for the model")
    parameters_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON schema for parameters"
    )
    func: Callable[..., Any] = Field(..., description="Python function to invoke")

    model_config = {"arbitrary_types_allowed": True}


class TaskSpec(BaseModel):
    """Specification for an agent task."""

    name: str = Field(..., description="Task name")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Parameter schema or metadata"
    )
    return_type: type[Any] | None = Field(default=None, description="Declared return type")
    func: Callable[..., Any] | None = Field(default=None, description="Bound callable")

    model_config = {"arbitrary_types_allowed": True}


class ToolCall(BaseModel):
    """A single tool call from the model."""

    id: str = Field(..., description="Tool call id for matching tool results")
    name: str = Field(..., description="Tool name to invoke")
    arguments: dict[str, Any] = Field(default_factory=dict, description="JSON arguments")


class Message(BaseModel):
    """A single message in a conversation."""

    role: MessageRole = Field(..., description="One of: system, user, assistant, tool")
    content: str = Field(default="", description="Message content")
    tool_call_id: str | None = Field(default=None, description="ID of tool call (for tool role)")
    tool_calls: list[ToolCall] = Field(
        default_factory=list, description="Tool calls (for assistant role)"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata")


class TokenUsage(BaseModel):
    """Token usage for a completion."""

    input_tokens: int = Field(default=0, description="Input/prompt tokens")
    output_tokens: int = Field(default=0, description="Output/completion tokens")
    total_tokens: int = Field(default=0, description="Total tokens")


class CostInfo(BaseModel):
    """Cost information for a completion."""

    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = Field(default=0.0, description="Estimated cost in USD")
    model_name: str = Field(default="", description="Model used")


class ProviderResponse(BaseModel):
    """Response from an LLM provider completion."""

    content: str | None = Field(default="", description="Assistant text content")
    tool_calls: list[ToolCall] = Field(
        default_factory=list, description="Tool calls requested by the model"
    )
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    stop_reason: str | None = Field(
        default=None, description="Why the model stopped (end_turn, tool_call, max_tokens, etc.)"
    )
    raw_response: Any = Field(default=None, description="Provider-specific raw response")

    model_config = {"arbitrary_types_allowed": True}


class AgentConfig(BaseModel):
    """Configuration for an agent."""

    model: ModelConfig = Field(..., description="Model to use")
    system_prompt: str = Field(default="", description="System prompt")
    tools: list[ToolSpec] = Field(default_factory=list, description="Registered tools")
    budget: dict[str, Any] | None = Field(default=None, description="Optional budget config")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata")


__all__ = [
    "ModelConfig",
    "ToolSpec",
    "TaskSpec",
    "ToolCall",
    "Message",
    "TokenUsage",
    "CostInfo",
    "ProviderResponse",
    "AgentConfig",
    # Validation types
    "OutputValidator",
    "ToolOutput",
    "ValidationAction",
    "ValidationAttempt",
    "ValidationContext",
    "ValidationResult",
]
