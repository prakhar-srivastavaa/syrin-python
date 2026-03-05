"""Config resolver: validate and apply overrides to live agents.

Applies OverridePayload via RemoteConfigurable per-section apply. Rejects
remote_excluded and unknown paths; flags hot-swap blocklist paths as pending_restart.
"""

from __future__ import annotations

import dataclasses
import logging

from pydantic import ValidationError

from syrin.agent import Agent
from syrin.remote._protocol import RemoteConfigurable
from syrin.remote._resolver_helpers import _coerce_enum, _normalize_enum_value
from syrin.remote._schema import _get_configurable, extract_agent_schema
from syrin.remote._types import (
    AgentSchema,
    FieldSchema,
    OverridePayload,
)

_log = logging.getLogger(__name__)

# Paths that require agent restart to take effect (backend re-init). Applied in memory but flagged.
HOT_SWAP_BLOCKLIST: frozenset[str] = frozenset(
    {
        "memory.backend",
        "memory.path",
        "checkpoint.storage",
        "checkpoint.path",
    }
)


@dataclasses.dataclass
class ResolveResult:
    """Result of applying overrides: accepted, rejected, and pending-restart paths.

    Attributes:
        accepted: Paths that were applied successfully.
        rejected: (path, reason) for overrides that were not applied.
        pending_restart: Paths that were applied but require agent restart to take full effect.
    """

    accepted: list[str]
    rejected: list[tuple[str, str]]
    pending_restart: list[str]


def _field_for_path(schema: AgentSchema, path: str) -> FieldSchema | None:
    """Return FieldSchema for dotted path, or None if not found. Checks fields and children."""
    parts = path.split(".")
    if len(parts) < 2:
        return None
    section_key = parts[0]
    config_schema = schema.sections.get(section_key)
    if not config_schema:
        return None
    # Remaining path after section: e.g. "run" or "per.hour"
    rest = ".".join(parts[1:])
    for f in config_schema.fields:
        if f.path == path:
            return f
        if f.children and rest.startswith(f.name + "."):
            for c in f.children:
                if c.path == path:
                    return c
            # Deeper nested: we don't store full path in schema for 3+ levels; match by prefix
            for c in f.children:
                if path == c.path:
                    return c
    return None


def _all_schema_paths(schema: AgentSchema) -> set[str]:
    """Return set of all dotted paths present in schema (for unknown path check)."""
    out: set[str] = set()
    for config_schema in schema.sections.values():
        for f in config_schema.fields:
            out.add(f.path)
            if f.children:
                for c in f.children:
                    out.add(c.path)
    return out


class ConfigResolver:
    """Validates and applies remote config overrides to a live agent.

    Uses agent schema for path validation and enum coercion. Applies overrides
    by section; on validation failure for a section, that section is rejected
    and the agent is left unchanged for that section.
    """

    def apply_overrides(
        self,
        agent: Agent,
        payload: OverridePayload,
        schema: AgentSchema | None = None,
    ) -> ResolveResult:
        """Apply payload overrides to the agent. Returns accepted, rejected, and pending_restart.

        Schema is optional: if None, obtained from registry by payload.agent_id, then
        from extract_agent_schema(agent) if not registered.
        """
        from syrin.remote._registry import get_registry

        schema = (
            schema or get_registry().get_schema(payload.agent_id) or extract_agent_schema(agent)
        )
        known_paths = _all_schema_paths(schema)

        accepted: list[str] = []
        rejected: list[tuple[str, str]] = []
        pending_restart: list[str] = []

        # Group overrides by section
        by_section: dict[str, list[tuple[str, object]]] = {}
        for ov in payload.overrides:
            path = ov.path
            parts = path.split(".", 1)
            if len(parts) < 2:
                rejected.append((path, "invalid_path"))
                continue
            section = parts[0]
            if path not in known_paths:
                rejected.append((path, "unknown_path"))
                continue
            field = _field_for_path(schema, path)
            if field and field.remote_excluded:
                rejected.append((path, "remote_excluded"))
                continue
            val = ov.value
            if (
                field
                and field.enum_values is not None
                and isinstance(val, str)
                and val not in field.enum_values
            ):
                normalized = _normalize_enum_value(path, val, field)
                if normalized is not None:
                    val = normalized
                else:
                    rejected.append((path, "invalid_enum_value"))
                    continue
            by_section.setdefault(section, []).append((path, val))

        # Coerce enums for each override
        for section, pairs in list(by_section.items()):
            new_pairs: list[tuple[str, object]] = []
            for path, val in pairs:
                field = _field_for_path(schema, path)
                if field:
                    val = _coerce_enum(val, field)
                new_pairs.append((path, val))
            by_section[section] = new_pairs

        # Apply per section via RemoteConfigurable (atomic: one section fails -> whole section rejected)
        remote_sections = getattr(type(agent), "REMOTE_CONFIG_SECTIONS", None) or {}
        for section, pairs in by_section.items():
            if not pairs:
                continue
            spec = remote_sections.get(section)
            configurable = _get_configurable(agent, spec)
            if configurable is None or not isinstance(configurable, RemoteConfigurable):
                for path, _ in pairs:
                    rejected.append((path, "unknown_section"))
                continue
            section_schema = schema.sections.get(section)
            if not section_schema:
                for path, _ in pairs:
                    rejected.append((path, "missing_schema"))
                continue
            try:
                configurable.apply_remote_overrides(agent, pairs, section_schema)
                for path, _ in pairs:
                    accepted.append(path)
                    if path in HOT_SWAP_BLOCKLIST:
                        pending_restart.append(path)
            except (ValidationError, ValueError, TypeError) as e:
                msg = str(e)
                _log.warning("Remote config section %s rejected: %s", section, msg)
                for path, _ in pairs:
                    rejected.append((path, msg))
            except Exception as e:
                _log.warning("Remote config section %s failed: %s", section, e)
                for path, _ in pairs:
                    rejected.append((path, str(e)))

        return ResolveResult(accepted=accepted, rejected=rejected, pending_restart=pending_restart)
