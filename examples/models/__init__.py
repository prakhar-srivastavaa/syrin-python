"""Custom model examples - Use pre-built models from Syrin!

Usage:
    from syrin import Model, Provider

    # Basic - auto-detect provider
    model = Model("gpt-4o")

    # Explicit provider
    model = Model("claude-sonnet", provider="anthropic")

    # Provider namespace
    model = Model.Provider("gpt-4o", provider="openai")

    # Custom model via inheritance
    class MyModel(Model):
        def complete(self, messages, **kwargs):
            # your implementation
            pass
"""

# Import pre-built models directly from Syrin
# See above for usage examples
