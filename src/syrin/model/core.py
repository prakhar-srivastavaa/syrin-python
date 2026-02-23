"""Core Model class and base functionality."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast, overload

from pydantic import BaseModel

from syrin.cost import ModelPricing
from syrin.exceptions import ModelNotFoundError, ProviderError
from syrin.types import (
    Message,
    ModelConfig,
    ProviderResponse,
    ToolSpec,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

T = TypeVar("T", bound=BaseModel)

_PROVIDER_PREFIXES = [
    ("anthropic/", "anthropic"),
    ("openai/", "openai"),
    ("google/", "google"),
    ("ollama/", "ollama"),
    ("azure/", "azure"),
    ("cohere/", "cohere"),
    ("deepseek/", "deepseek"),
    ("kimi/", "kimi"),
    ("sarvam/", "sarvam"),
]
# Patterns for bare model names (without prefix)
_PROVIDER_PATTERNS = [
    (re.compile(r"^gpt-", re.IGNORECASE), "openai"),
    (re.compile(r"^claude-", re.IGNORECASE), "anthropic"),
    (re.compile(r"^gemini-", re.IGNORECASE), "google"),
    (re.compile(r"^llama-", re.IGNORECASE), "ollama"),
]


def _resolve_env_var(value: str) -> str:
    """Resolve $VAR or ${VAR} to environment variable if present."""
    if not value or value[0] != "$":
        return value
    name = value[1:].strip("{}")
    return os.environ.get(name, value)


def _detect_provider(model_id: str) -> str:
    """Detect provider from model_id prefix or pattern."""
    resolved = _resolve_env_var(model_id)
    for prefix, provider in _PROVIDER_PREFIXES:
        if resolved.lower().startswith(prefix):
            return provider
    for pattern, provider in _PROVIDER_PATTERNS:
        if pattern.match(resolved):
            return provider
    return "litellm"


class ModelVersion:
    """Version tracking for models."""

    def __init__(self, major: int = 1, minor: int = 0, patch: int = 0) -> None:
        self.major = major
        self.minor = minor
        self.patch = patch

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def bump_major(self) -> ModelVersion:
        return ModelVersion(self.major + 1, 0, 0)

    def bump_minor(self) -> ModelVersion:
        return ModelVersion(self.major, self.minor + 1, 0)

    def bump_patch(self) -> ModelVersion:
        return ModelVersion(self.major, self.minor, self.patch + 1)


class ModelVariable:
    """Metadata about a model configuration parameter."""

    def __init__(
        self,
        name: str,
        type_hint: type,
        default: Any = None,
        description: str = "",
        required: bool = False,
    ) -> None:
        self.name = name
        self.type_hint = type_hint
        self.default = default
        self.description = description
        self.required = required


class ModelSettings:
    """Model-level settings for budget, context, and memory."""

    def __init__(
        self,
        context_window: int | None = None,
        max_output_tokens: int | None = None,
        max_input_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        **extra: Any,
    ) -> None:
        self.context_window = context_window
        self.max_output_tokens = max_output_tokens
        self.max_input_tokens = max_input_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.stop = stop
        self.extra = extra


class Middleware:
    """Hook for transforming requests/responses.

    Users can subclass this to add custom transformation layers.

    Example:
        class MyMiddleware(Middleware):
            def transform_request(self, messages, **kwargs):
                return messages, kwargs

            def transform_response(self, response):
                return response
    """

    def transform_request(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> tuple[list[Message], dict[str, Any]]:
        return messages, kwargs

    def transform_response(self, response: ProviderResponse) -> ProviderResponse:
        return response


class Model:
    """
    Extensible LLM model with full customization support.

    Usage:
        # Using provider namespaces (recommended)
        model = Model.OpenAI("gpt-4o")
        model = Model.Anthropic("claude-sonnet")
        model = Model.Ollama("llama3")

        # With config
        model = Model.OpenAI("gpt-4o", temperature=0.7)

        # For custom LLM providers, inherit from Model:
        class MyModel(Model):
            def complete(self, messages, **kwargs):
                # Your implementation
                pass
    """

    # Provider namespace - use Model.OpenAI("gpt-4o"), Model.Anthropic("claude"), etc.
    # These are defined as static methods below for IDE support

    @staticmethod
    def OpenAI(
        model_name: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_output_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        context_window: int | None = None,
        output: type | None = None,
        input_price: float | None = None,
        output_price: float | None = None,
        fallback: list[Model] | None = None,
        **kwargs: Any,
    ) -> Model:
        """Create an OpenAI model.

        Usage:
            Model.OpenAI("gpt-4o")
            Model.OpenAI("gpt-4o", temperature=0.7)

        Args:
            model_name: Model name (e.g., "gpt-4o", "gpt-4o-mini")
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Max output tokens
            api_key: API key (or use OPENAI_API_KEY)
            api_base: Custom base URL
            **kwargs: Additional Model parameters
        """
        import os

        return Model(
            model_id=f"openai/{model_name}",
            name=model_name,
            provider="openai",
            api_base=api_base or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1",
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            context_window=context_window or 128000,
            temperature=temperature,
            max_tokens=max_tokens,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            output=output,
            input_price=input_price,
            output_price=output_price,
            fallback=fallback,
            **kwargs,
        )

    @staticmethod
    def Anthropic(
        model_name: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_output_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        context_window: int | None = None,
        output: type | None = None,
        input_price: float | None = None,
        output_price: float | None = None,
        fallback: list[Model] | None = None,
        **kwargs: Any,
    ) -> Model:
        """Create an Anthropic Claude model.

        Usage:
            Model.Anthropic("claude-sonnet-4-5")
            Model.Anthropic("claude-opus-4-5")

        Args:
            model_name: Model name (e.g., "claude-sonnet-4-5-20241022")
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Max output tokens
            api_key: API key (or use ANTHROPIC_API_KEY)
            api_base: Custom base URL
            **kwargs: Additional Model parameters
        """
        import os

        return Model(
            model_id=f"anthropic/{model_name}",
            name=model_name,
            provider="anthropic",
            api_base=api_base or os.getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com",
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            context_window=context_window or 200000,
            temperature=temperature,
            max_tokens=max_tokens,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            output=output,
            input_price=input_price,
            output_price=output_price,
            fallback=fallback,
            **kwargs,
        )

    @staticmethod
    def Ollama(
        model_name: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_output_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        context_window: int | None = None,
        output: type | None = None,
        input_price: float | None = None,
        output_price: float | None = None,
        fallback: list[Model] | None = None,
        **kwargs: Any,
    ) -> Model:
        """Create an Ollama (local) model.

        Usage:
            Model.Ollama("llama3")
            Model.Ollama("mistral")

        Args:
            model_name: Model name (e.g., "llama3", "mistral")
            api_base: Base URL (default: http://localhost:11434)
            **kwargs: Additional Model parameters
        """
        import os

        return Model(
            model_id=f"ollama/{model_name}",
            name=model_name,
            provider="ollama",
            api_base=api_base or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434",
            api_key=api_key,
            context_window=context_window or 8192,
            temperature=temperature,
            max_tokens=max_tokens,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            output=output,
            input_price=input_price,
            output_price=output_price,
            fallback=fallback,
            **kwargs,
        )

    @staticmethod
    def Google(
        model_name: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_output_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        context_window: int | None = None,
        output: type | None = None,
        input_price: float | None = None,
        output_price: float | None = None,
        fallback: list[Model] | None = None,
        **kwargs: Any,
    ) -> Model:
        """Create a Google Gemini model.

        Usage:
            Model.Google("gemini-2.0-flash")
            Model.Google("gemini-1.5-pro")

        Args:
            model_name: Model name (e.g., "gemini-2.0-flash")
            temperature: Sampling temperature
            max_tokens: Max output tokens
            api_key: API key (or use GOOGLE_API_KEY)
            **kwargs: Additional Model parameters
        """
        import os

        return Model(
            model_id=model_name,
            name=model_name,
            provider="google",
            api_base=api_base
            or os.getenv("GOOGLE_BASE_URL")
            or "https://generativelanguage.googleapis.com/v1beta",
            api_key=api_key or os.getenv("GOOGLE_API_KEY"),
            context_window=context_window or 1048576,
            temperature=temperature,
            max_tokens=max_tokens,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            output=output,
            input_price=input_price,
            output_price=output_price,
            fallback=fallback,
            **kwargs,
        )

    @staticmethod
    def LiteLLM(
        model_name: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_output_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        context_window: int | None = None,
        output: type | None = None,
        input_price: float | None = None,
        output_price: float | None = None,
        fallback: list[Model] | None = None,
        **kwargs: Any,
    ) -> Model:
        """Create a LiteLLM model (100+ providers).

        Usage:
            Model.LiteLLM("anthropic/claude-3-5-sonnet")
            Model.LiteLLM("openai/gpt-4o")

        Args:
            model_name: Full model ID (e.g., "anthropic/claude-3-5-sonnet")
            api_key: API key (or use LITELLM_API_KEY)
            api_base: Custom base URL
            **kwargs: Additional Model parameters
        """
        import os

        name = model_name.split("/")[-1] if "/" in model_name else model_name
        return Model(
            model_id=model_name,
            name=name,
            provider="litellm",
            api_base=api_base or os.getenv("LITELLM_BASE_URL") or "https://api.litellm.ai",
            api_key=api_key or os.getenv("LITELLM_API_KEY"),
            context_window=context_window,
            temperature=temperature,
            max_tokens=max_tokens,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            output=output,
            input_price=input_price,
            output_price=output_price,
            fallback=fallback,
            **kwargs,
        )

    def __init__(
        self,
        model_id: str | None = None,
        *,
        provider: str | None = None,
        name: str | None = None,
        description: str = "",
        version: ModelVersion | None = None,
        fallback: list[Model] | None = None,
        output: type | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_output_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        context_window: int | None = None,
        pricing: ModelPricing | None = None,
        input_price: float | None = None,
        output_price: float | None = None,
        transformer: Middleware | None = None,
        **provider_kwargs: Any,
    ) -> None:
        # Check if this is a subclass (for custom LLM providers)
        is_subclass = type(self) is not Model

        # If no model_id provided but this is a subclass, allow it
        # (subclasses may override __init__ differently)
        if model_id is None and not is_subclass:
            raise TypeError(
                "Model requires either model_id or provider. "
                "Usage: Model(provider='openai', model_id='gpt-4o') "
                "or Model.Provider('gpt-4o', provider='openai') "
                "or inherit from Model for custom LLM providers."
            )

        self._model_id_raw = model_id
        self._model_id = _resolve_env_var(model_id) if model_id else ""

        # Use provided provider or detect from model_id
        if provider:
            self._provider = provider.lower()
        else:
            self._provider = _detect_provider(model_id) if model_id else "litellm"

        self._name = (
            name or (self._model_id.split("/")[-1] if "/" in self._model_id else self._model_id)
            if self._model_id
            else ""
        )
        self._description = description
        self._version = version or ModelVersion(1, 0, 0)

        self._api_key = api_key
        self._api_base = api_base

        # Handle pricing
        self._pricing: ModelPricing | None
        if pricing is not None:
            self._pricing = pricing
        elif input_price is not None or output_price is not None:
            self._pricing = ModelPricing(
                input_per_1m=input_price or 0.0,
                output_per_1m=output_price or 0.0,
            )
        else:
            self._pricing = None
        self._output_type = output
        self._transformer = transformer
        self._fallback: list[Model] = list(fallback) if fallback else []

        self._settings = ModelSettings(
            context_window=context_window,
            max_output_tokens=max_output_tokens or max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
        )

        self._provider_kwargs = provider_kwargs
        self._variables = self._extract_variables()

    def _extract_variables(self) -> list[ModelVariable]:
        """Extract configuration parameters."""
        variables = []

        if self._settings.temperature is not None:
            variables.append(
                ModelVariable(
                    name="temperature",
                    type_hint=float,
                    default=self._settings.temperature,
                    description="Sampling temperature (0.0-2.0)",
                    required=False,
                )
            )

        if self._settings.max_output_tokens is not None:
            variables.append(
                ModelVariable(
                    name="max_tokens",
                    type_hint=int,
                    default=self._settings.max_output_tokens,
                    description="Maximum tokens to generate",
                    required=False,
                )
            )

        if self._settings.top_p is not None:
            variables.append(
                ModelVariable(
                    name="top_p",
                    type_hint=float,
                    default=self._settings.top_p,
                    description="Nucleus sampling parameter",
                    required=False,
                )
            )

        if self._settings.context_window is not None:
            variables.append(
                ModelVariable(
                    name="context_window",
                    type_hint=int,
                    default=self._settings.context_window,
                    description="Maximum context window size",
                    required=False,
                )
            )

        return variables

    @property
    def model_id(self) -> str:
        """The model identifier."""
        return self._model_id

    @property
    def name(self) -> str:
        """Human-readable model name."""
        return self._name

    @property
    def provider(self) -> str:
        """Provider identifier."""
        return self._provider

    @property
    def description(self) -> str:
        """Model description."""
        return self._description

    @property
    def version(self) -> ModelVersion:
        """Model version info."""
        return self._version

    @property
    def metadata(self) -> dict[str, Any]:
        """Custom metadata dictionary."""
        return {
            "model_id": self._model_id,
            "provider": self._provider,
            "name": self._name,
            "description": self._description,
            "version": str(self._version),
            "has_fallback": len(self._fallback) > 0,
            "has_output_type": self._output_type is not None,
            "context_window": self._settings.context_window,
            "max_output_tokens": self._settings.max_output_tokens,
        }

    @property
    def variables(self) -> list[ModelVariable]:
        """List of model configuration parameters."""
        return self._variables

    @property
    def fallback(self) -> list[Model]:
        """List of fallback models."""
        return list(self._fallback)

    @property
    def output_type(self) -> type | None:
        """Output type for structured responses."""
        return self._output_type

    @property
    def settings(self) -> ModelSettings:
        """Model-level settings."""
        return self._settings

    @property
    def pricing(self) -> ModelPricing | None:
        """Pricing info per 1M tokens."""
        return self._pricing

    @property
    def api_base(self) -> str | None:
        """Custom base URL for API endpoint."""
        return self._api_base

    @property
    def api_key(self) -> str | None:
        """API key for authentication."""
        return self._api_key

    def to_config(self) -> ModelConfig:
        """Convert to ModelConfig for provider use."""
        return ModelConfig(
            name=self._name,
            provider=self._provider,
            model_id=self._model_id,
            api_key=self._api_key,
            base_url=self._api_base,
            output=self._output_type,
        )

    def with_params(
        self,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_output_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        context_window: int | None = None,
        output: type | None = None,
        **kwargs: Any,
    ) -> Model:
        """Create a copy with overridden parameters."""
        return Model(
            model_id=self._model_id,
            name=self._name,
            description=self._description,
            version=self._version,
            fallback=self._fallback.copy() if self._fallback else None,
            output=output or self._output_type,
            temperature=temperature if temperature is not None else self._settings.temperature,
            max_tokens=max_tokens,
            max_output_tokens=max_output_tokens or max_tokens or self._settings.max_output_tokens,
            top_p=top_p if top_p is not None else self._settings.top_p,
            top_k=top_k if top_k is not None else self._settings.top_k,
            stop=stop if stop is not None else self._settings.stop,
            api_key=self._api_key,
            api_base=self._api_base,
            context_window=context_window
            if context_window is not None
            else self._settings.context_window,
            pricing=self._pricing,
            transformer=self._transformer,
            _internal=True,
            **self._provider_kwargs,
            **kwargs,
        )

    def with_fallback(self, *models: Model) -> Model:
        """Create a copy with fallback models."""
        new_fallback = self._fallback.copy()
        for m in models:
            new_fallback.append(m)

        return Model(
            model_id=self._model_id,
            name=self._name,
            description=self._description,
            version=self._version,
            fallback=new_fallback,
            output=self._output_type,
            temperature=self._settings.temperature,
            max_output_tokens=self._settings.max_output_tokens,
            top_p=self._settings.top_p,
            top_k=self._settings.top_k,
            stop=self._settings.stop,
            api_key=self._api_key,
            api_base=self._api_base,
            context_window=self._settings.context_window,
            pricing=self._pricing,
            transformer=self._transformer,
            _internal=True,
            **self._provider_kwargs,
        )

    def with_output(
        self,
        output: type,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Model:
        """Create a copy configured for structured output."""
        return self.with_params(
            output=output,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def with_middleware(self, middleware: Middleware) -> Model:
        """Create a copy with a middleware."""
        return Model(
            model_id=self._model_id,
            name=self._name,
            description=self._description,
            version=self._version,
            fallback=self._fallback.copy() if self._fallback else None,
            output=self._output_type,
            temperature=self._settings.temperature,
            max_output_tokens=self._settings.max_output_tokens,
            top_p=self._settings.top_p,
            top_k=self._settings.top_k,
            stop=self._settings.stop,
            api_key=self._api_key,
            api_base=self._api_base,
            context_window=self._settings.context_window,
            pricing=self._pricing,
            transformer=middleware,
            _internal=True,
            **self._provider_kwargs,
        )

    def _get_provider_instance(self) -> Any:
        """Get the provider instance for this model."""
        if self._provider == "anthropic":
            from syrin.providers.anthropic import AnthropicProvider

            return AnthropicProvider()
        if self._provider == "openai":
            from syrin.providers.openai import OpenAIProvider

            return OpenAIProvider()
        if self._provider in ("ollama", "litellm"):
            from syrin.providers.litellm import LiteLLMProvider

            return LiteLLMProvider()
        return LiteLLMProvider()

    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> ProviderResponse | Iterator[ProviderResponse]:
        """Send messages to the LLM and get a response."""
        provider = self._get_provider_instance()

        settings = {
            "temperature": temperature if temperature is not None else self._settings.temperature,
            "max_tokens": max_tokens or self._settings.max_output_tokens,
            "top_p": self._settings.top_p,
            "stop": self._settings.stop,
            **self._provider_kwargs,
            **kwargs,
        }

        transformer_result = self._apply_transformer("request", messages, **settings)
        if isinstance(transformer_result, tuple):
            messages, settings = transformer_result
        else:
            # Unexpected single response during request phase
            raise ProviderError(
                "Transformer returned unexpected response type during request phase"
            )

        try:
            if stream:
                return cast(
                    Iterator[ProviderResponse],
                    provider.stream_sync(messages, self.to_config(), tools, **settings),
                )

            response = provider.complete_sync(
                messages=messages,
                model=self.to_config(),
                tools=tools,
                **settings,
            )

            if response is None:
                raise ProviderError(f"Provider {self._provider} returned no response")

            transformer_response = self._apply_transformer("response", response)
            if isinstance(transformer_response, tuple):
                raise ProviderError("Transformer returned tuple instead of response")
            response = transformer_response

            if self._output_type and response.content:
                response = self._parse_structured_output(response)

            return response
        except Exception:
            if self._fallback:
                return self._try_fallback(messages, tools=tools, **kwargs)
            raise

    async def acomplete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> ProviderResponse | AsyncIterator[ProviderResponse]:
        """Async version of complete()."""
        if stream:
            return self._astream_internal(
                messages, tools=tools, temperature=temperature, max_tokens=max_tokens, **kwargs
            )

        provider = self._get_provider_instance()

        settings = {
            "temperature": temperature if temperature is not None else self._settings.temperature,
            "max_tokens": max_tokens or self._settings.max_output_tokens,
            "top_p": self._settings.top_p,
            "stop": self._settings.stop,
            **self._provider_kwargs,
            **kwargs,
        }

        transformer_result = self._apply_transformer("request", messages, **settings)
        if isinstance(transformer_result, tuple):
            messages, settings = transformer_result
        else:
            # Unexpected single response during request phase
            raise ProviderError(
                "Transformer returned unexpected response type during request phase"
            )

        try:
            response = await provider.complete(
                messages=messages,
                model=self.to_config(),
                tools=tools,
                **settings,
            )

            transformer_response = self._apply_transformer("response", response)
            if isinstance(transformer_response, tuple):
                raise ProviderError("Transformer returned tuple instead of response")
            response = transformer_response

            if self._output_type and response.content:
                response = self._parse_structured_output(response)

            return response
        except Exception:
            if self._fallback:
                return await self._atry_fallback(messages, tools=tools, **kwargs)
            raise

    async def _astream_internal(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ProviderResponse]:
        """Internal async streaming implementation."""
        provider = self._get_provider_instance()

        settings = {
            "temperature": temperature if temperature is not None else self._settings.temperature,
            "max_tokens": max_tokens or self._settings.max_output_tokens,
            "top_p": self._settings.top_p,
            "stop": self._settings.stop,
            **self._provider_kwargs,
            **kwargs,
        }

        transformer_result = self._apply_transformer("request", messages, **settings)
        if isinstance(transformer_result, tuple):
            messages, settings = transformer_result
        else:
            # Unexpected single response during request phase
            raise ProviderError(
                "Transformer returned unexpected response type during request phase"
            )

        try:
            async for chunk in provider.stream(messages, self.to_config(), tools, **settings):
                yield chunk
        except Exception:
            if self._fallback:
                async for chunk in self._astream_fallback(messages, tools=tools, **kwargs):
                    yield chunk
            else:
                raise

    async def astream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ProviderResponse]:
        """Stream response chunks asynchronously."""
        async for chunk in self._astream_internal(
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        ):
            yield chunk

    def count_tokens(self, text: str) -> int:
        """Count tokens for the given text."""
        from syrin.cost import count_tokens

        return count_tokens(text, self._model_id)

    def get_pricing(self) -> ModelPricing | None:
        """Return pricing info."""
        if self._pricing is not None:
            return self._pricing

        from syrin.cost import _resolve_pricing

        inp, out = _resolve_pricing(self._model_id)
        if inp > 0 or out > 0:
            return ModelPricing(input_per_1m=inp, output_per_1m=out)
        return None

    @overload
    def _apply_transformer(
        self,
        phase: Literal["request"],
        messages_or_response: list[Message],
        **kwargs: Any,
    ) -> tuple[list[Message], dict[str, Any]]: ...

    @overload
    def _apply_transformer(
        self,
        phase: Literal["response"],
        messages_or_response: ProviderResponse,
        **kwargs: Any,
    ) -> ProviderResponse: ...

    def _apply_transformer(
        self,
        phase: str,
        messages_or_response: list[Message] | ProviderResponse,
        **kwargs: Any,
    ) -> tuple[list[Message], dict[str, Any]] | ProviderResponse:
        """Apply response transformer if set."""
        if self._transformer is None:
            if phase == "request":
                # Cast to list[Message] since phase is "request"
                return cast(list[Message], messages_or_response), kwargs
            # Cast to ProviderResponse since phase is "response"
            return cast(ProviderResponse, messages_or_response)

        if phase == "request":
            return self._transformer.transform_request(
                cast(list[Message], messages_or_response), **kwargs
            )
        else:
            return self._transformer.transform_response(
                cast(ProviderResponse, messages_or_response)
            )

    def _try_fallback(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse | Iterator[ProviderResponse]:
        """Try fallback models in order."""
        for fb in self._fallback:
            try:
                return fb.complete(messages, tools=tools, **kwargs)
            except Exception:
                continue
        raise ProviderError("All fallback models failed")

    async def _atry_fallback(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse | AsyncIterator[ProviderResponse]:
        """Try fallback models in order (async)."""
        for fb in self._fallback:
            try:
                return await fb.acomplete(messages, tools=tools, **kwargs)
            except Exception:
                continue
        raise ProviderError("All fallback models failed")

    async def _astream_fallback(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ProviderResponse]:
        """Try fallback models in order for streaming."""
        for fb in self._fallback:
            try:
                async for chunk in fb.astream(messages, tools=tools, **kwargs):
                    yield chunk
                return
            except Exception:
                continue
        raise ProviderError("All fallback models failed")

    def _parse_structured_output(self, response: ProviderResponse) -> ProviderResponse:
        """Parse response content into structured output type."""
        if not self._output_type or not response.content:
            return response

        try:
            import json

            content = response.content.strip()

            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            data = json.loads(content)

            # Cast to BaseModel type since we've checked _output_type is not None
            output_type = cast(type[BaseModel], self._output_type)
            if isinstance(data, dict):
                parsed = output_type.model_validate(data)
            else:
                parsed = output_type.model_validate_json(response.content)

            response.content = parsed.model_dump_json()
            response.raw_response = parsed

        except Exception as e:
            raise ProviderError(
                f"Failed to parse structured output: {e}. "
                f"Expected {getattr(self._output_type, '__name__', str(self._output_type))}, got: {response.content[:200]}"
            ) from e

        return response

    @classmethod
    def _create(cls, **kwargs: Any) -> Model:
        """Internal method for creating Model instances (for testing).

        Use provider namespaces or inherit from Model instead:
            Model.OpenAI('gpt-4o')
            Model.Anthropic('claude-sonnet')

            class MyModel(Model):
                ...
        """
        return cls(**kwargs, _internal=True)

    def __repr__(self) -> str:
        fallback_info = f", {len(self._fallback)} fallbacks" if self._fallback else ""
        output_info = ""
        if self._output_type:
            if hasattr(self._output_type, "__name__"):
                output_info = f", output={self._output_type.__name__}"
            else:
                output_info = f", output={type(self._output_type).__name__}"
        return f"Model({self._model_id!r}, provider={self._provider!r}{fallback_info}{output_info})"


class ModelRegistry:
    """Singleton registry of named models for lookup."""

    _instance: ModelRegistry | None = None
    _models: dict[str, Model]

    def __new__(cls) -> ModelRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models = {}
        return cls._instance

    def register(self, name: str, model: Model) -> None:
        """Register a model with a name."""
        self._models[name] = model

    def get(self, name: str) -> Model:
        """Get a registered model by name."""
        if name not in self._models:
            raise ModelNotFoundError(f"Model not found: {name}")
        return self._models[name]

    def list_names(self) -> list[str]:
        """List all registered model names."""
        return list(self._models)

    def clear(self) -> None:
        """Clear all registered models."""
        self._models.clear()


__all__ = [
    "Model",
    "ModelVersion",
    "ModelVariable",
    "ModelSettings",
    "ModelRegistry",
    "Middleware",
]
