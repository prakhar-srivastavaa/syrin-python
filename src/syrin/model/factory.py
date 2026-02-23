"""Factory functions for creating models.

Usage:
    from syrin.model import create_model, make_model

    # Quick creation
    model = create_model("openai", "gpt-4o")

    # Create new LLM class
    KimiModel = make_model(
        name="Kimi",
        provider="kimi",
        default_model="kimi2.5",
        base_url="https://api.moonshot.cn/v1",
        api_key_env="KIMI_API_KEY",
    )
    model = KimiModel()
"""

from __future__ import annotations

import os
from typing import Any

from syrin.model.core import Model


def create_model(
    provider: str,
    model_name: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> Model:
    """Factory function to create a model from provider and model name.

    Usage:
        model = create_model("openai", "gpt-4o")
        model = create_model("anthropic", "claude-sonnet-4-5")
        model = create_model("ollama", "llama3")

    Args:
        provider: Provider name (openai, anthropic, ollama, google, etc.)
        model_name: Model name (e.g., "gpt-4o", "claude-sonnet")
        api_key: Optional API key
        base_url: Optional base URL
        **kwargs: Additional model parameters

    Returns:
        Configured Model instance
    """
    if not provider or not provider.strip():
        raise ValueError("Provider name cannot be empty")

    if not model_name or not model_name.strip():
        raise ValueError("Model name cannot be empty")

    provider = provider.strip().lower()

    if provider == "openai":
        from syrin.model.providers import OpenAI

        return OpenAI(model_name, api_key=api_key, base_url=base_url, **kwargs)
    elif provider == "anthropic":
        from syrin.model.providers import Anthropic

        return Anthropic(model_name, api_key=api_key, base_url=base_url, **kwargs)
    elif provider == "ollama":
        from syrin.model.providers import Ollama

        return Ollama(model_name, api_key=api_key, base_url=base_url, **kwargs)
    elif provider == "google":
        from syrin.model.providers import Google

        return Google(model_name, api_key=api_key, base_url=base_url, **kwargs)
    elif provider == "litellm":
        from syrin.model.providers import LiteLLM

        return LiteLLM(model_name, api_key=api_key, base_url=base_url, **kwargs)
    else:
        return Model(
            model_id=f"{provider}/{model_name}",
            name=model_name,
            api_key=api_key,
            api_base=base_url,
            _internal=True,
            **kwargs,
        )


def make_model(
    *,
    name: str,
    provider: str,
    default_model: str,
    base_url: str,
    api_key_env: str | None = None,
    context_window: int | None = None,
    **defaults: Any,
) -> type[Model]:
    """Factory to create a pre-configured Model class for any LLM.

    This is the easiest way to add support for a new LLM - just provide config!

    Usage:
        # Create a Kimi model class
        KimiModel = make_model(
            name="Kimi",
            provider="kimi",
            default_model="kimi2.5",
            base_url="https://api.moonshot.cn/v1",
            api_key_env="KIMI_API_KEY",
            context_window=128000,
        )

        # Use it!
        model = KimiModel()
        model = KimiModel("kimi2.5-flash")

    Args:
        name: Human-readable name (e.g., "Kimi")
        provider: Provider ID (e.g., "kimi")
        default_model: Default model name
        base_url: API base URL
        api_key_env: Environment variable for API key
        context_window: Default context window size
        **defaults: Additional default parameters

    Returns:
        Model subclass with pre-configured defaults
    """
    if not name or not provider or not default_model or not base_url:
        raise ValueError("name, provider, default_model, and base_url are required")

    class CustomModel(Model):
        DEFAULT_PROVIDER = provider
        DEFAULT_BASE_URL = base_url
        DEFAULT_MODEL = default_model

        def __init__(
            self,
            model_name: str = default_model,
            *,
            api_key: str | None = None,
            base_url: str | None = None,
            **model_kwargs: Any,
        ) -> None:
            super().__init__(
                model_id=f"{provider}/{model_name}",
                name=model_name,
                api_base=base_url or os.getenv("OPENAI_BASE_URL") or self.DEFAULT_BASE_URL,
                api_key=api_key or (os.getenv(api_key_env) if api_key_env else None),
                context_window=model_kwargs.pop("context_window", context_window),
                _internal=True,
                **defaults,
                **model_kwargs,
            )

    CustomModel.__name__ = f"{name}Model"
    return CustomModel


__all__ = [
    "create_model",
    "make_model",
]
