"""
Syrin Model Examples - Complete Guide to Creating and Using Models

This examples file demonstrates:
1. Basic model creation with provider namespaces
2. Structured output for type-safe responses
3. Fallback chains for reliability
4. Response transformers for custom processing
5. Creating custom model classes for new LLMs

Run: python -m examples.models.quickstart
"""

import os

# =============================================================================
# Organized Imports - By Category
# =============================================================================
import syrin

# Core
from syrin import Model

# Structured Output
from syrin import structured


# =============================================================================
# Example 1: Basic Model Creation with Provider Namespaces
# =============================================================================


def example_basic_models():
    """The simplest way to create a model - use provider namespaces."""

    print("=" * 60)
    print("Example 1: Basic Model Creation")
    print("=" * 60)

    # OpenAI models
    model = Model.OpenAI("gpt-4o")
    print(f"OpenAI gpt-4o: {model}")
    print(f"  - Provider: {model.provider}")
    print(f"  - Model ID: {model.model_id}")
    print(f"  - Context Window: {model.settings.context_window}")

    # Anthropic models
    model = Model.Anthropic("claude-sonnet-4-5-20241022")
    print(f"\nAnthropic claude-sonnet: {model}")
    print(f"  - Provider: {model.provider}")
    print(f"  - Model ID: {model.model_id}")

    # Ollama (local)
    model = Model.Ollama("llama3")
    print(f"\nOllama llama3: {model}")
    print(f"  - Provider: {model.provider}")

    # Google
    model = Model.Google("gemini-2.0-flash")
    print(f"\nGoogle gemini-2.0-flash: {model}")
    print(f"  - Provider: {model.provider}")


# =============================================================================
# Example 2: Model with Configuration
# =============================================================================


def example_model_config():
    """Create model with custom configuration."""

    print("\n" + "=" * 60)
    print("Example 2: Model with Configuration")
    print("=" * 60)

    # With temperature
    model = Model.OpenAI("gpt-4o", temperature=0.7)
    print(f"With temperature: {model}")
    print(f"  - Temperature: {model.settings.temperature}")

    # With max_tokens
    model = Model.OpenAI("gpt-4o", max_tokens=1000)
    print(f"\nWith max_tokens: {model}")
    print(f"  - Max Output Tokens: {model.settings.max_output_tokens}")

    # With custom API key and base URL
    model = Model.OpenAI(
        "gpt-4o", api_key="sk-custom-key", api_base="https://custom.endpoint.com/v1"
    )
    print(f"\nWith custom API: {model}")
    print(f"  - API Key: {model.api_key[:10]}...")
    print(f"  - API Base: {model.api_base}")


# =============================================================================
# Example 3: Structured Output
# =============================================================================


def example_structured_output():
    """Use structured output for type-safe responses."""

    print("\n" + "=" * 60)
    print("Example 3: Structured Output")
    print("=" * 60)

    from pydantic import BaseModel

    class SentimentAnalysis(BaseModel):
        sentiment: str
        confidence: float

    # With structured output
    model = Model.OpenAI("gpt-4o", output=SentimentAnalysis)
    print(f"Structured output: {model}")
    print(f"  - Output type: {model.output_type}")


# =============================================================================
# Example 4: Fallback Chains
# =============================================================================


def example_fallback():
    """Create fallback chains for reliability."""

    print("\n" + "=" * 60)
    print("Example 4: Fallback Chains")
    print("=" * 60)

    # Using with_fallback
    model = Model.Anthropic("claude-sonnet").with_fallback(
        Model.OpenAI("gpt-4o"),
        Model.OpenAI("gpt-4o-mini"),  # Cheaper fallback
        Model.Ollama("llama3"),  # Local fallback
    )
    print(f"With fallback: {model}")
    print(f"  - Fallbacks: {len(model.fallback)}")

    # Fallback with structured output
    class SentimentAnalysis(BaseModel):
        sentiment: str

    model = Model.Anthropic("claude-sonnet", output=SentimentAnalysis).with_fallback(
        Model.OpenAI("gpt-4o", output=SentimentAnalysis)
    )
    print(f"\nWith fallback + structured output: {model}")


# =============================================================================
# Example 5: Custom Model via Inheritance
# =============================================================================


def example_custom_model():
    """Create custom model for new LLM providers."""

    print("\n" + "=" * 60)
    print("Example 5: Custom Model via Inheritance")
    print("=" * 60)

    class MyCustomModel(Model):
        """Custom model for any LLM API."""

        def complete(self, messages, **kwargs):
            # Your custom implementation here
            print(f"  - Called with {len(messages)} messages")
            return None

    model = MyCustomModel("my-model")
    print(f"Custom model: {model}")
    print(f"  - Provider: {model.provider}")


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    example_basic_models()
    example_model_config()
    example_structured_output()
    example_fallback()
    example_custom_model()
