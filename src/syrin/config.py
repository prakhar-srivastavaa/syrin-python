"""Global configuration for Syrin."""

from __future__ import annotations

import os
import threading
from typing import Any

from syrin.types import ModelConfig


class GlobalConfig:
    """Global Syrin configuration."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._trace: bool = False
        self._default_model: ModelConfig | None = None
        self._default_api_key: str | None = None
        self._env_prefix = "SYRIN_"
        self._load_from_env()

    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        if os.environ.get(f"{self._env_prefix}TRACE", "").lower() in ("1", "true", "yes"):
            self._trace = True
        if os.environ.get(f"{self._env_prefix}TRACE", ""):
            pass
        self._default_api_key = os.environ.get(f"{self._env_prefix}API_KEY", None)

    @property
    def trace(self) -> bool:
        """Whether tracing is enabled."""
        return self._trace

    @trace.setter
    def trace(self, value: bool) -> None:
        with self._lock:
            self._trace = value

    @property
    def default_model(self) -> ModelConfig | None:
        """Default model to use when none is specified."""
        return self._default_model

    @default_model.setter
    def default_model(self, value: ModelConfig | None) -> None:
        with self._lock:
            self._default_model = value

    @property
    def default_api_key(self) -> str | None:
        """Default API key to use when none is specified."""
        return self._default_api_key

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return getattr(self, key, default)

    def set(self, **kwargs: Any) -> None:
        """Set multiple configuration values."""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)


_config = GlobalConfig()


def get_config() -> GlobalConfig:
    """Get the global configuration instance."""
    return _config


def configure(**kwargs: Any) -> None:
    """Configure global Syrin settings.

    Args:
        trace: Enable tracing (default: False)
        default_model: Default model to use (Model or ModelConfig)
        default_api_key: Default API key to use

    Example:
        >>> import syrin
        >>> syrin.configure(trace=True)
        >>> syrin.configure(default_model="openai/gpt-4o")
    """
    _config.set(**kwargs)


__all__ = ["GlobalConfig", "get_config", "configure", "GlobalConfig"]
