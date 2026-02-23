"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import os

os.environ["SYRIN_TESTING"] = "1"

import pytest


@pytest.fixture(autouse=True)
def _clear_model_registry() -> None:
    """Reset ModelRegistry singleton state between tests that mutate it."""
    from syrin.model import ModelRegistry

    reg = ModelRegistry()
    reg._models.clear()
    yield
    reg._models.clear()


@pytest.fixture
def sample_model_id() -> str:
    """A sample model ID for tests."""
    return "claude-sonnet-4-5"


@pytest.fixture
def sample_model():
    """Fixture to create Model instances for testing.

    Usage:
        def test_something(sample_model):
            model = sample_model(provider="openai", model_id="gpt-4o")
    """
    from syrin.model import Model

    def _create(**kwargs):
        return Model(**kwargs)

    return _create
