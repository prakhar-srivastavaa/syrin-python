"""Configuration Example.

Demonstrates:
- Using configure() to set global config
- Using get_config() to access config
- Setting default model
- Enabling/disabling tracing
- Environment variable configuration

Run: python -m examples.advanced.config
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from syrin import configure, get_config

logging.basicConfig(level=logging.ERROR)
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("httpcore").setLevel(logging.CRITICAL)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def example_get_config() -> None:
    """Getting global configuration."""
    print("\n" + "=" * 50)
    print("Get Config Example")
    print("=" * 50)

    config = get_config()

    print(f"Trace enabled: {config.trace}")
    print(f"Default model: {config.default_model}")
    print(f"Default API key: {config.default_api_key}")


def example_configure() -> None:
    """Configuring global settings."""
    print("\n" + "=" * 50)
    print("Configure Example")
    print("=" * 50)

    # Configure tracing
    configure(trace=True)

    config = get_config()
    print(f"Trace after configure: {config.trace}")

    # Reset
    configure(trace=False)


def example_configure_model() -> None:
    """Setting default model."""
    print("\n" + "=" * 50)
    print("Configure Model Example")
    print("=" * 50)

    from syrin.types import ModelConfig

    # Set default model (name is required by ModelConfig)
    model_config = ModelConfig(
        name="gpt-4o-mini",
        provider="openai",
        model_id="openai/gpt-4o-mini",
    )
    configure(default_model=model_config)

    config = get_config()
    print(f"Default model: {config.default_model}")


def example_config_api_key() -> None:
    """Setting default API key."""
    print("\n" + "=" * 50)
    print("Configure API Key Example")
    print("=" * 50)

    # Set API key
    configure(default_api_key="sk-test-key123")

    config = get_config()
    print(f"API key set: {config.default_api_key is not None}")
    print(f"API key value: {config.default_api_key[:10]}...")


def example_config_multiple() -> None:
    """Setting multiple config values."""
    print("\n" + "=" * 50)
    print("Multiple Config Values")
    print("=" * 50)

    configure(
        trace=True,
        default_api_key="sk-test-key",
    )

    config = get_config()
    print(f"Trace: {config.trace}")
    print(f"API key: {config.default_api_key is not None}")


def example_env_variables() -> None:
    """Configuration via environment variables."""
    print("\n" + "=" * 50)
    print("Environment Variables")
    print("=" * 50)

    print("Supported environment variables:")
    print("  SYRIN_TRACE - Enable tracing (1, true, yes)")
    print("  SYRIN_API_KEY - Default API key")

    # Check current values
    print(f"\nCurrent SYRIN_TRACE: {os.environ.get('SYRIN_TRACE', 'not set')}")
    print(f"Current SYRIN_API_KEY: {'set' if os.environ.get('SYRIN_API_KEY') else 'not set'}")


def example_config_get_set() -> None:
    """Using get/set methods."""
    print("\n" + "=" * 50)
    print("Get/Set Methods")
    print("=" * 50)

    config = get_config()

    # Get value
    trace = config.get("trace")
    print(f"Get trace: {trace}")

    # Set value
    config.set(trace=True)
    print(f"After set: {config.trace}")

    # Get with default
    value = config.get("nonexistent", "default_value")
    print(f"Get with default: {value}")


def example_thread_safety() -> None:
    """Thread-safe configuration."""
    print("\n" + "=" * 50)
    print("Thread Safety Example")
    print("=" * 50)

    import threading

    def update_config(value: bool, thread_id: int):
        configure(trace=value)
        config = get_config()
        print(f"Thread {thread_id}: trace = {config.trace}")

    threads = []
    for i in range(3):
        t = threading.Thread(target=update_config, args=(i % 2 == 0, i))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("Thread-safe updates completed")


if __name__ == "__main__":
    example_get_config()
    example_configure()
    example_configure_model()
    example_config_api_key()
    example_config_multiple()
    example_env_variables()
    example_config_get_set()
    example_thread_safety()
