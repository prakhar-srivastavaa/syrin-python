"""Tests for core types (types.py)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from syrin.types import (
    AgentConfig,
    CostInfo,
    Message,
    ModelConfig,
    TaskSpec,
    TokenUsage,
)


def test_model_config_serialize_deserialize() -> None:
    cfg = ModelConfig(
        name="test",
        provider="anthropic",
        model_id="anthropic/claude-3-5-sonnet",
        api_key=None,
        base_url=None,
    )
    data = cfg.model_dump()
    assert data["name"] == "test"
    assert data["provider"] == "anthropic"
    assert data["model_id"] == "anthropic/claude-3-5-sonnet"
    restored = ModelConfig.model_validate(data)
    assert restored.model_id == cfg.model_id


def test_message_roles() -> None:
    for role in ("system", "user", "assistant", "tool"):
        msg = Message(role=role, content="hi")
        assert msg.role == role
        assert msg.content == "hi"
        assert msg.tool_call_id is None


def test_token_usage_defaults() -> None:
    u = TokenUsage()
    assert u.input_tokens == 0
    assert u.output_tokens == 0
    assert u.total_tokens == 0


def test_token_usage_custom() -> None:
    u = TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15)
    assert u.total_tokens == 15


def test_cost_info() -> None:
    usage = TokenUsage(input_tokens=100, output_tokens=50)
    cost = CostInfo(token_usage=usage, cost_usd=0.01, model_name="gpt-4")
    assert cost.cost_usd == 0.01
    assert cost.token_usage.input_tokens == 100


def test_task_spec() -> None:
    spec = TaskSpec(name="my_task", parameters={"x": "int"}, return_type=None)
    assert spec.name == "my_task"
    assert spec.parameters == {"x": "int"}


def test_agent_config_requires_model() -> None:
    with pytest.raises(ValidationError):
        AgentConfig.model_validate({})
    cfg = AgentConfig(model=ModelConfig(name="m", provider="p", model_id="mid"))
    assert cfg.system_prompt == ""
    assert cfg.tools == []


# =============================================================================
# TYPES EDGE CASES - TRY TO BREAK FUNCTIONALITY
# =============================================================================


def test_message_with_empty_content() -> None:
    """Message with empty content should be allowed."""
    msg = Message(role="user", content="")
    assert msg.content == ""


def test_message_with_tool_call_id() -> None:
    """Message with tool_call_id."""
    msg = Message(role="tool", content="result", tool_call_id="call_123")
    assert msg.tool_call_id == "call_123"


def test_message_with_very_long_content() -> None:
    """Message with very long content."""
    long_content = "x" * 100000
    msg = Message(role="user", content=long_content)
    assert len(msg.content) == 100000


def test_message_unicode_content() -> None:
    """Message with unicode content."""
    msg = Message(role="user", content="Hello 🌍 你好 🔥")
    assert "🌍" in msg.content


def test_token_usage_with_large_values() -> None:
    """TokenUsage with very large values."""
    u = TokenUsage(
        input_tokens=10_000_000,
        output_tokens=10_000_000,
        total_tokens=20_000_000,
    )
    assert u.total_tokens == 20_000_000


def test_token_usage_requires_total() -> None:
    """TokenUsage total must be explicitly provided."""
    u = TokenUsage(input_tokens=100, output_tokens=50)
    # Note: total is not auto-calculated - this is current behavior
    assert u.total_tokens == 0


def test_cost_info_with_model_name() -> None:
    """CostInfo with model_name."""
    cost = CostInfo(token_usage=TokenUsage(), cost_usd=0.01, model_name="gpt-4")
    assert cost.model_name == "gpt-4"


def test_cost_info_with_zero_cost() -> None:
    """CostInfo with zero cost."""
    cost = CostInfo(token_usage=TokenUsage(), cost_usd=0.0)
    assert cost.cost_usd == 0.0


def test_model_config_with_settings() -> None:
    """ModelConfig with settings."""
    cfg = ModelConfig(
        name="full",
        provider="openai",
        model_id="gpt-4",
        api_key="sk-key",
        base_url="https://api.openai.com/v1",
    )
    assert cfg.name == "full"
    assert cfg.api_key == "sk-key"


def test_task_spec_with_defaults() -> None:
    """TaskSpec with default values."""
    spec = TaskSpec(name="test")
    assert spec.name == "test"
    assert spec.parameters == {}
    assert spec.return_type is None


def test_task_spec_with_complex_parameters() -> None:
    """TaskSpec with complex parameters."""
    spec = TaskSpec(
        name="complex",
        parameters={
            "users": [{"id": "int", "name": "str"}],
            "options": {"flag": "bool", "count": "int"},
        },
    )
    assert "users" in spec.parameters


def test_agent_config_with_all_fields() -> None:
    """AgentConfig with all fields."""
    cfg = AgentConfig(
        model=ModelConfig(name="m", provider="p", model_id="mid"),
        system_prompt="You are a bot.",
        tools=[],
        memory=None,
        budget=None,
    )
    assert cfg.system_prompt == "You are a bot."
