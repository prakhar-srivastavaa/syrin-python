"""Global configuration for Syrin."""

from __future__ import annotations

import os
import threading

from syrin.types import ModelConfig


class GlobalConfig:
    """Global Syrin configuration. Accessed via get_config().

    Attributes:
        trace: Whether tracing is enabled. Set via configure(trace=True).
        debug: Whether debug mode is enabled. Set via configure(debug=True).
        default_model: Default ModelConfig when none specified.
        default_api_key: Default API key (rarely used; pass per Model).
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._trace: bool = False
        self._debug: bool = False
        self._default_model: ModelConfig | None = None
        self._default_api_key: str | None = None
        self._env_prefix = "SYRIN_"
        self._load_from_env()

    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        if os.environ.get(f"{self._env_prefix}TRACE", "").lower() in ("1", "true", "yes"):
            self._trace = True
        # API key is never auto-loaded from env; user must pass via configure() or Model(api_key=...)

    @property
    def trace(self) -> bool:
        """Whether tracing is enabled."""
        return self._trace

    @trace.setter
    def trace(self, value: bool) -> None:
        with self._lock:
            self._trace = value

    @property
    def debug(self) -> bool:
        """Whether debug mode is enabled."""
        return self._debug

    @debug.setter
    def debug(self, value: bool) -> None:
        with self._lock:
            self._debug = value

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

    @default_api_key.setter
    def default_api_key(self, value: str | None) -> None:
        with self._lock:
            self._default_api_key = value

    def get(self, key: str, default: object = None) -> object:
        """Get a configuration value."""
        return getattr(self, key, default)

    def set(self, **kwargs: object) -> None:
        """Set multiple configuration values."""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)


_config = GlobalConfig()


def get_config() -> GlobalConfig:
    """Get the global Syrin configuration instance.

    Returns:
        GlobalConfig singleton.
    """
    return _config


def configure(**kwargs: object) -> None:
    """Configure global Syrin settings.

    Args:
        trace: Enable tracing (default: False)
        debug: Enable debug mode (default: False)
        default_model: Default model to use (Model or ModelConfig)
        default_api_key: Default API key to use

    Example:
        >>> import syrin
        >>> syrin.configure(trace=True)
        >>> syrin.configure(default_model="openai/gpt-4o")
    """
    _config.set(**kwargs)


__all__ = ["GlobalConfig", "get_config", "configure"]
