"""Syrin Model - Extensible Model abstraction for LLM providers.

Usage:
    # Using provider namespaces (recommended)
    model = Model.OpenAI("gpt-4o")
    model = Model.Anthropic("claude-sonnet")
    model = Model.Ollama("llama3")
    model = Model.Google("gemini-2.0-flash")

    # Direct Model with explicit provider
    model = Model(provider="openai", model_id="gpt-4o")

    # For custom LLM providers, inherit from Model:
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
    setup_provider_namespaces,
)

setup_provider_namespaces(Model)
Model.OpenAI = OpenAI  # type: ignore[method-assign]
Model.Anthropic = Anthropic  # type: ignore[method-assign]
Model.Ollama = Ollama  # type: ignore[method-assign]
Model.Google = Google  # type: ignore[method-assign]
Model.LiteLLM = LiteLLM  # type: ignore[method-assign]

# Structured output
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
    # Structured output
    "OutputType",
    "output",
    "structured",
    "StructuredOutput",
]
