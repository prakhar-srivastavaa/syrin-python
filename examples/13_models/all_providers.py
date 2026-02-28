"""All Model Providers Example.

Demonstrates:
- Model.OpenAI, Model.Anthropic, Model.Google, Model.Ollama, Model.LiteLLM
- Model.Custom for third-party OpenAI-compatible APIs
- Model.Almock for testing without API keys
- Model settings: temperature, max_tokens, context_window
- api_key and api_base configuration

Run: python -m examples.13_models.all_providers
Visit: http://localhost:8000/playground
Requires: uv pip install syrin[serve]
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from syrin import Model

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# 1. Provider namespaces
model = Model.OpenAI("gpt-4o", api_key=os.getenv("OPENAI_API_KEY", "sk-test"))
print(f"OpenAI: {model.provider}, {model.model_id}")
model = Model.Anthropic("claude-sonnet-4-5", api_key=os.getenv("ANTHROPIC_API_KEY", "sk-test"))
model = Model.Google("gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY", "test"))
model = Model.Ollama("llama3")
model = Model.LiteLLM("openai/gpt-4o", api_key=os.getenv("OPENAI_API_KEY", "sk-test"))
model = Model.Almock(latency_seconds=0.01, lorem_length=50)

# 2. Model configuration
model = Model.OpenAI(
    "gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY", "sk-test"),
    temperature=0.7,
    max_tokens=2048,
    context_window=128000,
)

# 3. Model.Custom (third-party APIs)
model = Model.Custom(
    "deepseek-chat",
    api_base="https://api.deepseek.com/v1",
    api_key=os.getenv("DEEPSEEK_API_KEY", "sk-test"),
    context_window=128_000,
)

# 4. Structured output
from pydantic import BaseModel


class SentimentAnalysis(BaseModel):
    sentiment: str
    confidence: float


model = Model.OpenAI(
    "gpt-4o",
    output=SentimentAnalysis,
    api_key=os.getenv("OPENAI_API_KEY", "sk-test"),
)

# 5. Fallback chains
key = os.getenv("OPENAI_API_KEY", "sk-test")
model = Model.Anthropic(
    "claude-sonnet-4-5",
    api_key=os.getenv("ANTHROPIC_API_KEY", "sk-test"),
).with_fallback(
    Model.OpenAI("gpt-4o", api_key=key),
    Model.OpenAI("gpt-4o-mini", api_key=key),
    Model.Ollama("llama3"),
)
print(f"Fallbacks: {len(model.fallback)}")

if __name__ == "__main__":
    from syrin import Agent

    class ProvidersDemoAgent(Agent):
        name = "providers-demo"
        description = "Agent with Almock (model providers demo)"
        model = Model.Almock(latency_seconds=0.01, lorem_length=50)
        system_prompt = "You are a helpful assistant."

    agent = ProvidersDemoAgent()
    print("Serving at http://localhost:8000/playground")
    agent.serve(port=8000, enable_playground=True, debug=True)
