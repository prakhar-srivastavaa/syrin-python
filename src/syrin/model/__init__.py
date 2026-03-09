"""Syrin Model - Extensible Model abstraction for LLM providers.

Usage:
    # Using provider namespaces (recommended)
    model = Model.OpenAI("gpt-4o")
    model = Model.Anthropic("claude-sonnet")
    model = Model.Ollama("llama3")
    model = Model.Google("gemini-2.0-flash")
    model = Model.OpenRouter("openai/gpt-4o")

    # Tweak properties: temperature, max_tokens, context_window, etc.
    model = Model.OpenAI("gpt-4o-mini", temperature=0.7, max_tokens=2048, api_key="...")

    # Direct Model with explicit provider
    model = Model(provider="openai", model_id="gpt-4o")

    # Third-party OpenAI-compatible APIs (DeepSeek, KIMI, Grok, etc.)
    model = Model.Custom("deepseek-chat", api_base="https://api.deepseek.com/v1", api_key="...")

    # OpenRouter — 100+ models via a single API key
    model = Model.OpenRouter("arcee-ai/trinity-large-preview:free", api_key="...")

    # For fully custom LLM providers, inherit from Model:
    class MyModel(Model):
        def complete(self, messages, **kwargs):
            # Your implementation
            pass

Structured output:
    - @structured decorator
    - @output shorthand
    - OutputType wrapper
"""

from syrin.model.core import (
    Middleware,
    Model,
    ModelRegistry,
    ModelSettings,
    ModelVariable,
    ModelVersion,
)

# Setup provider namespaces (import first to avoid circular deps)
from syrin.model.providers import (
    Anthropic,
    Google,
    LiteLLM,
    Ollama,
    OpenAI,
    OpenRouter,
    setup_provider_namespaces,
)

setup_provider_namespaces(Model)

# Structured output
from syrin.model.factory import create_model, make_model
from syrin.model.structured import (  # noqa: E402
    OutputType,
    StructuredOutput,
    output,
    structured,
)

__all__ = [
    # Core
    "Model",
    "ModelRegistry",
    "ModelSettings",
    "ModelVariable",
    "ModelVersion",
    "Middleware",
    # Provider namespaces
    "OpenAI",
    "Anthropic",
    "Ollama",
    "Google",
    "LiteLLM",
    "OpenRouter",
    # Structured output
    "OutputType",
    "output",
    "structured",
    "StructuredOutput",
    "create_model",
    "make_model",
]
