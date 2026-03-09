"""Provider registry: resolve provider name to Provider instance.

Moves provider resolution out of Agent so Agent depends on the Provider
abstraction only. New providers register here.
"""

from __future__ import annotations

from syrin.exceptions import ProviderNotFoundError
from syrin.providers.base import Provider

_KNOWN_PROVIDERS = frozenset({"anthropic", "openai", "openrouter", "ollama", "litellm"})


def get_provider(provider_name: str, *, strict: bool = False) -> Provider:
    """Return the Provider instance for the given provider name.

    Use when you have a provider identifier (e.g. from ModelConfig.provider)
    but no Model instance. When using Model, prefer model.get_provider().

    Args:
        provider_name: openai, anthropic, openrouter, ollama, litellm, etc.
        strict: If True, raise ProviderNotFoundError for unknown provider names.
            If False (default), unknown names fall back to LiteLLMProvider.

    Returns:
        Provider instance for the given name.

    Raises:
        ProviderNotFoundError: When strict=True and provider_name is not
            one of openai, anthropic, openrouter, ollama, litellm.
    """
    name = (provider_name or "litellm").strip().lower()
    if strict and name and name not in _KNOWN_PROVIDERS:
        raise ProviderNotFoundError(
            f"Unknown provider: {provider_name!r}. "
            f"Known providers: {', '.join(sorted(_KNOWN_PROVIDERS))}."
        )
    if name == "anthropic":
        from syrin.providers.anthropic import AnthropicProvider

        return AnthropicProvider()
    if name == "openai":
        from syrin.providers.openai import OpenAIProvider

        return OpenAIProvider()
    if name == "openrouter":
        from syrin.providers.openai import OpenAIProvider

        return OpenAIProvider()
    if name in ("ollama", "litellm"):
        from syrin.providers.litellm import LiteLLMProvider

        return LiteLLMProvider()
    from syrin.providers.litellm import LiteLLMProvider

    return LiteLLMProvider()
