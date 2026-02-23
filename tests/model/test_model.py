"""Tests for Model and ModelRegistry (model.py)."""

from __future__ import annotations

import pytest

from syrin.exceptions import ModelNotFoundError
from syrin.model import Model, ModelRegistry
from syrin.types import ModelConfig


def test_model_creation_and_provider_detection() -> None:
    m = Model("anthropic/claude-3-5-sonnet-20241022")
    assert m.model_id == "anthropic/claude-3-5-sonnet-20241022"
    assert m.provider == "anthropic"


def test_model_openai_prefix() -> None:
    m = Model("openai/gpt-4")
    assert m.provider == "openai"


def test_model_gpt_pattern() -> None:
    m = Model("gpt-4-turbo")
    assert m.provider == "openai"


def test_model_ollama() -> None:
    m = Model("ollama/llama2")
    assert m.provider == "ollama"


def test_model_fallback_litellm() -> None:
    m = Model("custom/unknown-model")
    assert m.provider == "litellm"


def test_model_to_config() -> None:
    m = Model("anthropic/claude-3", api_key="sk-fake", base_url=None)
    cfg = m.to_config()
    assert isinstance(cfg, ModelConfig)
    assert cfg.provider == "anthropic"
    assert cfg.model_id == "anthropic/claude-3"
    assert cfg.api_key == "sk-fake"


def test_model_repr() -> None:
    m = Model("openai/gpt-4")
    r = repr(m)
    assert "openai/gpt-4" in r
    assert "openai" in r


def test_model_env_var_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_MODEL", "anthropic/claude-3-5-sonnet")
    m = Model("$MY_MODEL")
    assert m.model_id == "anthropic/claude-3-5-sonnet"
    assert m.provider == "anthropic"


def test_model_env_var_unset_returns_raw() -> None:
    m = Model("$NONEXISTENT_VAR_12345")
    assert m.model_id == "$NONEXISTENT_VAR_12345"


def test_registry_singleton() -> None:
    r1 = ModelRegistry()
    r2 = ModelRegistry()
    assert r1 is r2


def test_registry_register_and_get() -> None:
    reg = ModelRegistry()
    reg._models.clear()
    m = Model("anthropic/claude-3")
    reg.register("default", m)
    got = reg.get("default")
    assert got is m
    assert reg.list_names() == ["default"]


def test_registry_get_missing_raises() -> None:
    reg = ModelRegistry()
    reg._models.clear()
    with pytest.raises(ModelNotFoundError):
        reg.get("nonexistent")


# =============================================================================
# AGGRESSIVE EDGE CASES - TRY TO BREAK MODEL
# =============================================================================


def test_model_bare_claude_name() -> None:
    """Bare claude model names should detect provider."""
    m = Model("claude-4")
    assert m.provider == "anthropic"


def test_model_bare_gemini_name() -> None:
    """Bare gemini model names should detect provider."""
    m = Model("gemini-2")
    assert m.provider == "google"


def test_model_bare_gpt_name() -> None:
    """Bare gpt model names should detect provider."""
    m = Model("gpt-5")
    assert m.provider == "openai"


def test_model_with_very_long_name() -> None:
    """Very long model names should work."""
    long_name = "a" * 1000
    m = Model(long_name)
    assert m.model_id == long_name


def test_model_special_chars_in_name() -> None:
    """Special characters in model names."""
    m = Model("model-v2.0_alpha-test")
    assert m.provider == "litellm"  # Falls back to litellm


def test_model_case_sensitivity() -> None:
    """Provider detection should be case-sensitive or handle properly."""
    m = Model("GPT-4")
    assert m.provider == "openai"  # Should normalize or handle


def test_model_unicode_in_name() -> None:
    """Unicode characters in model names."""
    m = Model("模型-4")
    assert m.provider == "litellm"  # Falls back


def test_registry_with_many_models() -> None:
    """Registry handles many registered models."""
    reg = ModelRegistry()
    reg._models.clear()
    for i in range(100):
        reg.register(f"model_{i}", Model(f"openai/gpt-{i}"))
    assert len(reg.list_names()) == 100


def test_model_config_with_none_values() -> None:
    """ModelConfig should handle None values gracefully."""
    cfg = ModelConfig(
        name="test",
        provider="openai",
        model_id="gpt-4",
        api_key=None,
    )
    assert cfg.api_key is None


def test_model_with_temperature_extreme_values() -> None:
    """Temperature at extreme bounds."""
    m1 = Model(provider="openai", model_id="gpt-4", temperature=0.0)
    assert m1.settings.temperature == 0.0

    m2 = Model(provider="openai", model_id="gpt-4", temperature=2.0)
    assert m2.settings.temperature == 2.0

    m3 = Model(provider="openai", model_id="gpt-4", temperature=1.0)
    assert m3.settings.temperature == 1.0
