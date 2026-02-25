"""Ensure every symbol in syrin.__all__ is importable and exported.

Valid: all names in __all__ exist and can be retrieved via getattr(syrin, name).
Invalid: duplicate names, names that don't exist, or wrong types.
"""

from __future__ import annotations

import syrin


def test_all_exported_names_are_importable() -> None:
    """Every name in syrin.__all__ must be getattr(syrin, name)."""
    for name in syrin.__all__:
        assert hasattr(syrin, name), (
            f"syrin.__all__ contains '{name}' but syrin has no such attribute"
        )
        obj = getattr(syrin, name)
        assert obj is not None or name == "__version__", (
            f"syrin.{name} is None (except __version__)"
        )


def test_no_duplicate_exports() -> None:
    """__all__ must not list the same name twice."""
    seen: set[str] = set()
    for name in syrin.__all__:
        assert name not in seen, f"Duplicate in __all__: '{name}'"
        seen.add(name)


def test_version_is_string() -> None:
    """__version__ must be a non-empty string."""
    assert isinstance(syrin.__version__, str)
    assert len(syrin.__version__.strip()) > 0


def test_run_is_callable() -> None:
    """run must be callable (convenience API)."""
    assert callable(syrin.run)


def test_configure_and_get_config_are_callable() -> None:
    """configure and get_config must be callable."""
    assert callable(syrin.configure)
    assert callable(syrin.get_config)


def test_get_config_returns_global_config_type() -> None:
    """get_config() must return an object with expected config attributes."""
    from syrin.config import GlobalConfig

    config = syrin.get_config()
    assert isinstance(config, GlobalConfig)
    assert hasattr(config, "default_model")
    assert hasattr(config, "trace")
    assert hasattr(config, "get")
    assert hasattr(config, "set")


def test_run_signature_has_typed_return() -> None:
    """run() must be annotated to return Response[str] (public API typing)."""
    import typing

    hints = typing.get_type_hints(syrin.run)
    assert "return" in hints
    ret = hints["return"]
    # Response[str] or Response; accept both
    assert "Response" in str(ret)
