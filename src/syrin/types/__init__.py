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
    """Configuration passed to LLM providers for completion requests.

    Typically created via ``model.to_config()``. Contains everything the provider
    needs: model_id, api_key, base_url, output schema. Used internally; you rarely
    construct this directly.
    """

    name: str = Field(..., description="Human-readable model name (e.g., gpt-4o-mini)")
    provider: str = Field(
        ...,
        description="Provider identifier: openai, anthropic, ollama, litellm, etc.",
    )
    model_id: str = Field(
        ...,
        description="Full model ID (e.g., openai/gpt-4o, anthropic/claude-sonnet)",
    )
    api_key: str | None = Field(
        default=None,
        description="API key for authentication. Pass explicitly; not auto-read from env.",
    )
    base_url: str | None = Field(
        default=None,
        description="API base URL. Overrides provider default (e.g., for proxies).",
    )
    output: type | None = Field(
        default=None,
        description="Pydantic type for structured output. None = plain text.",
    )


class ToolSpec(BaseModel):
    """Spec for a tool the model can call. Usually built via syrin.tool()."""

    name: str = Field(..., description="Tool name (used in tool_calls.name)")
    description: str = Field(
        default="",
        description="Description for the model. Shown in tool list.",
    )
    parameters_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON schema for parameters. Model uses this to generate args.",
    )
    func: Callable[..., Any] = Field(
        ...,
        description="Python function to run. Receives parsed arguments from model.",
    )

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
    """A single tool/function call requested by the model. In ProviderResponse.tool_calls.

    Execute the named tool with the arguments, then add a tool-role message with the result
    and the same id so the model can continue.
    """

    id: str = Field(..., description="Unique ID for this call. Use when returning tool results.")
    name: str = Field(..., description="Tool/function name to invoke")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON arguments to pass to the tool",
    )


class Message(BaseModel):
    """A single message in a conversation. Used with ``model.complete(messages)``.

    Build a list of Message objects to send to the model. Roles: system (instructions),
    user (human input), assistant (model reply), tool (tool result for function calling).
    """

    role: MessageRole = Field(
        ...,
        description="Message role: system, user, assistant, or tool",
    )
    content: str = Field(default="", description="Message text content")
    tool_call_id: str | None = Field(
        default=None,
        description="ID of the tool call this message responds to (for role=tool)",
    )
    tool_calls: list[ToolCall] = Field(
        default_factory=list,
        description="Tool calls requested by the model (for role=assistant)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata for custom use",
    )


class TokenUsage(BaseModel):
    """Token counts for a completion. Used for cost estimation and budgeting."""

    input_tokens: int = Field(default=0, description="Prompt/input tokens consumed")
    output_tokens: int = Field(default=0, description="Completion/output tokens generated")
    total_tokens: int = Field(default=0, description="Total tokens (input + output)")


class CostInfo(BaseModel):
    """Cost info for a completion. From model.get_pricing(token_usage)."""

    token_usage: TokenUsage = Field(
        default_factory=TokenUsage,
        description="Input/output token counts for this completion",
    )
    cost_usd: float = Field(
        default=0.0,
        description="Estimated cost in USD. Use for budgeting and dashboards.",
    )
    model_name: str = Field(
        default="",
        description="Model used (e.g., gpt-4o-mini). Helps when comparing costs.",
    )


class ProviderResponse(BaseModel):
    """Response from ``model.complete()`` or ``model.acomplete()``.

    The main fields you use: ``content`` (the text), ``token_usage`` (for cost/budget),
    and ``tool_calls`` (if the model requested function calls).
    """

    content: str | None = Field(
        default="",
        description="Assistant text content. Parsed to Pydantic if model has output type.",
    )
    tool_calls: list[ToolCall] = Field(
        default_factory=list,
        description="Tool/function calls requested by the model. Process and add tool results.",
    )
    token_usage: TokenUsage = Field(
        default_factory=TokenUsage,
        description="Input/output token counts. Use for cost and budget tracking.",
    )
    stop_reason: str | None = Field(
        default=None,
        description="Why the model stopped: end_turn, tool_call, max_tokens, etc.",
    )
    raw_response: Any = Field(
        default=None,
        description="Provider-specific raw response (OpenAI, Anthropic, etc.).",
    )

    model_config = {"arbitrary_types_allowed": True}


class AgentConfig(BaseModel):
    """Agent configuration. Created from Agent.to_config() or when building agents."""

    model: ModelConfig = Field(..., description="Model used for completions")
    system_prompt: str = Field(
        default="",
        description="System instructions. Sets agent personality and constraints.",
    )
    tools: list[ToolSpec] = Field(
        default_factory=list,
        description="Tools the agent can call. Empty = no tools.",
    )
    budget: dict[str, Any] | None = Field(
        default=None,
        description="Budget config (max_cost, max_tokens, etc.). None = no limit.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Custom metadata for logging, tracing, or app logic",
    )


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
