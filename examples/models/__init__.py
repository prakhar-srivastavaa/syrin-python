"""Model examples - Use pre-built models or create custom ones.

Usage:
    # Import centralized models (used across examples)
    from examples.models.models import gpt4, claude, deepseek_chat, kimi, grok, openrouter_free

    agent = Agent(model=gpt4, system_prompt="You are helpful.")

    # Or create inline - always pass api_key explicitly
    from syrin import Model
    model = Model.OpenAI("gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
    model = Model.OpenRouter("openai/gpt-4o", api_key=os.getenv("OPENROUTER_API_KEY"))
"""

# Pre-built models in models.py - import and use anywhere
