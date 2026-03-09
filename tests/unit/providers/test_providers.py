"""Tests for providers module (base, litellm, openai, anthropic, openrouter)."""

from __future__ import annotations

import pytest

from syrin.providers.base import Provider
from syrin.types import Message, ModelConfig, ProviderResponse, TokenUsage

# =============================================================================
# PROVIDER CONFIGURATION TESTS
# =============================================================================


class TestProviderConfig:
    """Tests for Provider configuration (via ModelConfig)."""

    def test_basic_config(self):
        """Test basic provider config."""
        config = ModelConfig(
            name="test",
            provider="openai",
            model_id="gpt-4",
            api_key="test-key",
            base_url="https://api.example.com",
        )
        assert config.api_key == "test-key"
        assert config.base_url == "https://api.example.com"

    def test_config_defaults(self):
        """Test config with defaults."""
        config = ModelConfig(name="test", provider="openai", model_id="gpt-4")
        assert config.base_url is None

    def test_config_with_all_fields(self):
        """Test config with all valid fields."""
        config = ModelConfig(
            name="test",
            provider="openai",
            model_id="gpt-4",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
        )
        assert config.api_key == "test-key"
        assert config.base_url == "https://api.openai.com/v1"


# =============================================================================
# BASE PROVIDER TESTS
# =============================================================================


class TestBaseProvider:
    """Tests for Provider abstract class."""

    def test_cannot_instantiate_base(self):
        """Provider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Provider()

    def test_provider_interface(self):
        """Test provider interface methods exist."""

        class TestProvider(Provider):
            async def complete(self, messages, model, tools=None, **kwargs):
                return ProviderResponse(content="test")

        provider = TestProvider()
        assert hasattr(provider, "complete")
        assert hasattr(provider, "complete_sync")
        assert hasattr(provider, "stream")
        assert hasattr(provider, "stream_sync")


# =============================================================================
# PROVIDER CONFIGURATION EDGE CASES
# =============================================================================


class TestProviderConfigEdgeCases:
    """Edge cases for provider configuration."""

    def test_empty_api_key(self):
        """Config with empty API key."""
        config = ModelConfig(name="test", provider="openai", model_id="gpt-4", api_key="")
        assert config.api_key == ""

    def test_very_long_api_key(self):
        """Config with very long API key."""
        long_key = "sk-" + "a" * 1000
        config = ModelConfig(name="test", provider="openai", model_id="gpt-4", api_key=long_key)
        assert len(config.api_key) == 1003

    def test_url_with_trailing_slash(self):
        """URL with trailing slash."""
        config = ModelConfig(
            name="test", provider="openai", model_id="gpt-4", base_url="https://api.example.com/"
        )
        assert config.base_url.endswith("/")

    def test_url_without_scheme(self):
        """URL without scheme."""
        config = ModelConfig(
            name="test", provider="openai", model_id="gpt-4", base_url="api.example.com"
        )
        assert "api.example.com" in config.base_url


# =============================================================================
# PROVIDER RESPONSE TESTS
# =============================================================================


class TestProviderResponse:
    """Tests for ProviderResponse."""

    def test_basic_response(self):
        """Test basic response creation."""
        response = ProviderResponse(
            content="Hello world",
            token_usage=TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30),
        )
        assert response.content == "Hello world"
        assert response.token_usage.total_tokens == 30

    def test_response_with_tool_calls(self):
        """Test response with tool calls."""
        from syrin.types import ToolCall

        response = ProviderResponse(
            content="",
            tool_calls=[ToolCall(id="call_1", name="search", arguments={"q": "test"})],
            token_usage=TokenUsage(),
        )
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "search"

    def test_response_empty_content(self):
        """Test response with empty content."""
        response = ProviderResponse(
            content="",
            token_usage=TokenUsage(),
        )
        assert response.content == ""

    def test_response_none_content(self):
        """Test response with None content."""
        response = ProviderResponse(
            content=None,
            token_usage=TokenUsage(),
        )
        assert response.content is None

    def test_response_very_long_content(self):
        """Test response with very long content."""
        long_content = "x" * 100000
        response = ProviderResponse(
            content=long_content,
            token_usage=TokenUsage(),
        )
        assert len(response.content) == 100000

    def test_response_unicode_content(self):
        """Test response with unicode content."""
        response = ProviderResponse(
            content="Hello 🌍 你好 🔥",
            token_usage=TokenUsage(),
        )
        assert "🌍" in response.content


# =============================================================================
# MOCK PROVIDER TESTS
# =============================================================================


class TestMockProvider:
    """Tests using mock providers."""

    @pytest.mark.asyncio
    async def test_mock_provider_response(self):
        """Test with mock provider response."""

        class MockProvider(Provider):
            async def complete(self, messages, model, tools=None, **kwargs):
                return ProviderResponse(
                    content="Mock response",
                    token_usage=TokenUsage(input_tokens=5, output_tokens=10, total_tokens=15),
                )

        provider = MockProvider()
        model_config = ModelConfig(name="test", provider="test", model_id="test")
        response = await provider.complete(
            messages=[Message(role="user", content="Hello")],
            model=model_config,
        )

        assert response.content == "Mock response"
        assert response.token_usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_mock_provider_with_tools(self):
        """Test mock provider with tools."""

        class MockProvider(Provider):
            async def complete(self, messages, model, tools=None, **kwargs):
                if tools:
                    from syrin.types import ToolCall

                    return ProviderResponse(
                        content="",
                        tool_calls=[ToolCall(id="1", name="test_tool", arguments={})],
                        token_usage=TokenUsage(),
                    )
                return ProviderResponse(content="No tools", token_usage=TokenUsage())

        provider = MockProvider()
        model_config = ModelConfig(name="test", provider="test", model_id="test")

        # Test without tools
        response1 = await provider.complete(
            messages=[],
            model=model_config,
        )
        assert response1.content == "No tools"

        # Test with tools
        response2 = await provider.complete(
            messages=[],
            model=model_config,
            tools=[{"name": "test_tool"}],
        )
        assert len(response2.tool_calls) == 1

    @pytest.mark.asyncio
    async def test_mock_provider_error(self):
        """Test mock provider raising error."""

        class ErrorProvider(Provider):
            async def complete(self, messages, model, tools=None, **kwargs):
                raise ConnectionError("Network error")

        provider = ErrorProvider()
        model_config = ModelConfig(name="test", provider="test", model_id="test")

        with pytest.raises(ConnectionError):
            await provider.complete(messages=[], model=model_config)


# =============================================================================
# PROVIDER EDGE CASES
# =============================================================================


class TestProviderEdgeCases:
    """Edge cases for providers."""

    @pytest.mark.asyncio
    async def test_provider_empty_messages(self):
        """Provider with empty messages list."""

        class TestProvider(Provider):
            async def complete(self, messages, model, tools=None, **kwargs):
                return ProviderResponse(
                    content="Response",
                    token_usage=TokenUsage(),
                )

        provider = TestProvider()
        model_config = ModelConfig(name="test", provider="test", model_id="test")
        response = await provider.complete(messages=[], model=model_config)
        assert response.content == "Response"

    @pytest.mark.asyncio
    async def test_provider_many_messages(self):
        """Provider with many messages."""

        class TestProvider(Provider):
            async def complete(self, messages, model, tools=None, **kwargs):
                return ProviderResponse(
                    content=f"Received {len(messages)} messages",
                    token_usage=TokenUsage(),
                )

        provider = TestProvider()
        model_config = ModelConfig(name="test", provider="test", model_id="test")
        messages = [Message(role="user", content=f"msg{i}") for i in range(100)]
        response = await provider.complete(messages=messages, model=model_config)
        assert "100" in response.content

    @pytest.mark.asyncio
    async def test_provider_special_chars_in_model(self):
        """Provider with special characters in model name."""

        class TestProvider(Provider):
            async def complete(self, messages, model, tools=None, **_kwargs):
                return ProviderResponse(content=model.model_id, token_usage=TokenUsage())

        provider = TestProvider()
        model_config = ModelConfig(name="test", provider="test", model_id="model-v1.2_test")
        response = await provider.complete(messages=[], model=model_config)
        assert response.content == "model-v1.2_test"
