# Remote Config

Remote config lets a backend (Syrin Cloud or your own server) push configuration overrides to running agents—temperature, budget limits, memory settings, etc.—without code deploys. The library provides types and infrastructure for schema extraction, registration, and override application.

## Overview

- **Agent → backend:** The agent connects outbound to the config backend. Works behind NAT, firewalls, and load balancers.
- **No feature flags in OSS:** All gating is server-side. On the free tier, agents run with local config only.
- **Pluggable transport:** Overrides can be delivered via SSE (SaaS), HTTP routes on `agent.serve()`, or polling.

## What's in remote config today

| Section | Agent attribute | In dashboard? | Notes |
|--------|------------------|---------------|--------|
| **agent** | (top-level) | ✅ Yes | `max_tool_iterations`, `debug`, `loop_strategy`, `system_prompt`, `hitl_timeout` |
| **budget** | `_budget` | ✅ Yes | Run limit, reserve, per (rate limits), thresholds, shared |
| **memory** | `_persistent_memory` | ✅ Yes | Backend, path, types, top_k, decay, scope, etc. |
| **context** | `_context.context` | ✅ Yes | max_tokens, reserve, thresholds, budget (token caps), encoding |
| **checkpoint** | `_checkpoint_config` | ✅ Yes | enabled, storage, path, trigger, max_checkpoints, compress |
| **rate_limit** | `_rate_limit_manager.config` | ✅ Yes | rpm, tpm, rpd, thresholds, wait_backoff, auto_switch |
| **circuit_breaker** | `_circuit_breaker` | ✅ Yes | failure_threshold, recovery_timeout, half_open_max |
| **output** | `_output` | ✅ Yes | validation_retries, context, strict (output validation config) |
| **guardrails** | (agent) | ✅ Yes | Enable/disable by name: `guardrails.{name}.enabled` (bool). |
| **prompt_vars** | (agent) | ✅ Yes | Realtime template variables: `prompt_vars.{key}` (str). |
| **tools** | (agent) | ✅ Yes | Enable/disable by tool name: `tools.{name}.enabled` (bool). Agent tools and MCP tools. |
| **mcp** | (agent) | ✅ Yes | Enable/disable MCP server by index: `mcp.{index}.enabled` (bool). |
| **observability** | `_tracer`, `_audit` | ❌ No | Tracer/audit are set at init; no structured “observability config” section yet. |
| **model** | `_model` / `_model_config` | ❌ No | Not exposed. Model identity/API is security-sensitive; override via code or env. |

So: **Budget, Rate limit, Memory, Context, Circuit breaker, Checkpoint, Output, agent top-level (including system prompt), guardrails (enable/disable), prompt_vars (realtime), tools (enable/disable by name), and mcp (enable/disable by index)** are all covered.

## Override store, baseline, and revert (scale-friendly)

The API uses an **override store** so dashboards can revert to code values and stay consistent at scale:

- **Baseline** — Values from code (frozen on first GET /config). Stored per agent as `_remote_baseline_values`.
- **Overrides** — User-applied path → value (only overridden paths). Stored as `_remote_overrides`. PATCH adds or updates; **`value: null` removes that path** (revert to baseline).
- **Current** — Effective values = baseline + overrides (overrides win). Returned as `current_values` and used to sync agent state after each PATCH.

GET /config returns `baseline_values`, `overrides`, and `current_values`, and enriches each **field** in `sections` with `baseline_value`, `current_value`, and `overridden` so the dashboard can render one row per field without merging. Revert one path: PATCH with `{"path": "budget.run", "value": null}`.

## Dashboard: what to expose, what to skip

**Expose in the dashboard (already in schema):**

- **agent** — Edit system prompt, max tool iterations, debug, loop strategy, hitl_timeout. High value, safe.
- **budget** — Run limit, reserve, per-period limits, thresholds. Core for cost control.
- **memory** — Backend, top_k, decay, scope. Useful for tuning retrieval and retention.
- **context** — max_tokens, reserve, thresholds. Controls context window and compaction.
- **checkpoint** — When/where to checkpoint. Useful for ops (e.g. enable/disable, change trigger).
- **rate_limit** — rpm/tpm/rpd and thresholds. Prevents provider throttling.
- **circuit_breaker** — failure_threshold, recovery_timeout. Resilience tuning.
- **output** — Validation retries, strict mode. Niche but valid.
- **guardrails** — Toggle each guardrail by name (`guardrails.{name}.enabled`). Enable/disable checks without code deploy.
- **prompt_vars** — Edit template variables in realtime (`prompt_vars.{key}`). Use for A/B prompts, tenant-specific copy.
- **tools** — Toggle each tool by name (`tools.{name}.enabled`). Covers agent tools and tools from MCP servers.
- **mcp** — Toggle each MCP server by index (`mcp.{index}.enabled`). Disabling a server disables all tools from it.

**Do not expose (or treat as read-only / advanced):**

- **Model** — Don’t allow switching model/API from the dashboard by default (credentials, safety). If you add a “model” section later, limit to model *name* only and keep keys in env/code.
- **Observability** — No section yet. If you add one (e.g. enable/disable tracing, sampling), expose only simple toggles or sampling rates, not exporters/credentials.

**Optional future sections:** `model` (model name only, keys in env), or observability toggles—add when product needs them.

### Not configurable remotely

The following are intentionally not exposed in the schema and cannot be changed via remote overrides:

- **Model** — Identity and API credentials; override via code or env.
- **Observability** — Tracer/audit are set at init; no structured observability section yet.

## Testing remote config

Remote config is tested in two ways:

1. **Automated E2E** — `tests/unit/serve/test_http_config_routes.py` (class `TestRemoteConfigE2EFullFeatures`) runs an agent with guardrails, prompt_vars, and tools behind the config routes; GET /config, PATCH with overrides (guardrails.*.enabled, prompt_vars.*, tools.*.enabled), then asserts agent state (`_guardrails_disabled`, `_prompt_vars`, `_tools_disabled`, `agent.tools`) and GET /config `current_values`. Run:
   ```bash
   uv run pytest tests/unit/serve/test_http_config_routes.py -v -k TestRemoteConfigE2EFullFeatures
   ```

2. **Manual** — Use the full-featured example server, then override via curl and verify:
   - Start: `PYTHONPATH=. python examples/12_remote_config/serve_full_features.py`
   - GET /config → note `agent_id`, sections, current_values
   - PATCH /config with overrides (e.g. disable a guardrail, set prompt_vars, disable a tool)
   - GET /config again → confirm current_values reflect overrides
   - Optional: POST /chat to confirm disabled tool is not offered and prompt_vars are used in the prompt

Step-by-step manual flow and curl examples: **`examples/12_remote_config/TESTING_REMOTE_CONFIG.md`**.

## Data Types

All types live in `syrin.remote` and are used by schema extraction, registry, resolver, and transports.

### FieldSchema

One configurable field: name, dotted path, type, default, constraints, enum values, and nested children. In GET /config responses, each field is enriched with `baseline_value`, `current_value`, and `overridden` for dashboard UX.

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Field name (e.g. `run`, `top_k`) |
| `path` | str | Dotted path for overrides (e.g. `budget.run`, `memory.decay.rate`) |
| `type` | str | Type name: `float`, `str`, `int`, `bool`, `object`, etc. |
| `default` | object \| None | Default value; JSON-serializable |
| `description` | str \| None | Human-readable description |
| `constraints` | dict | Validation: `ge`, `le`, `gt`, `lt`, `pattern`, `min_length`, `max_length` |
| `enum_values` | list[str] \| None | For StrEnum fields, allowed string values |
| `children` | list[FieldSchema] \| None | Nested fields when this field is an object |
| `remote_excluded` | bool | If True, not writable via remote overrides (e.g. callables) |
| `baseline_value` | object \| None | (Response only.) Value from code; used for revert. |
| `current_value` | object \| None | (Response only.) Effective value (baseline + overrides). |
| `overridden` | bool | (Response only.) True if this path has a remote override. |

### ConfigSchema

All fields for one config object (e.g. Budget, Memory).

| Field | Type | Description |
|-------|------|-------------|
| `section` | str | Section key (e.g. `budget`, `memory`) |
| `class_name` | str | Python type name (e.g. `Budget`, `Memory`) |
| `fields` | list[FieldSchema] | Field schemas for this section |

### AgentSchema

Full schema for a registered agent: sections, baseline (code) values, overrides, and current values. GET /config returns this with baseline frozen at first read; overrides = user-applied path→value; current = baseline + overrides. Revert = remove path from overrides.

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | str | Unique agent identifier |
| `agent_name` | str | Human-readable name |
| `class_name` | str | Python class name |
| `sections` | dict[str, ConfigSchema] | Map of section key to config schema (fields enriched with baseline_value, current_value, overridden in response) |
| `baseline_values` | dict[str, object] | Values from code (frozen at first GET); used for revert. |
| `overrides` | dict[str, object] | User-applied overrides (path → value). Only overridden paths. Revert = remove path. |
| `current_values` | dict[str, object] | Effective values (baseline + overrides). |

### ConfigOverride

Single override: path and value. Applied by the resolver. **`value: null` in PATCH = revert that path to baseline** (remove from override store).

| Field | Type | Description |
|-------|------|-------------|
| `path` | str | Dotted path (e.g. `budget.run`) |
| `value` | object \| null | New value; type must match schema. Use `null` to revert path to baseline. |

### OverridePayload

List of overrides from the backend, with a monotonic version. Used by SSE, polling, and PATCH responses.

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | str | Target agent identifier |
| `version` | int | Monotonic version number (≥ 0) |
| `overrides` | list[ConfigOverride] | Overrides to apply |

### SyncRequest / SyncResponse

Registration handshake. The agent sends `SyncRequest` (agent_id, schema, library_version). The backend responds with `SyncResponse` (ok, optional initial_overrides, optional error). When `ok` is False (e.g. backend down), the agent continues with local config.

- **SyncRequest:** `agent_id`, `agent_schema` (serialized as `"schema"` on the wire), `library_version`.
- **SyncResponse:** `ok`, `initial_overrides` (list of ConfigOverride or None), `error` (str or None).

## Schema extraction

The library can extract field schemas from your config classes so the backend knows what can be overridden.

- **`extract_schema(cls, prefix)`** — Auto-detects Pydantic, dataclass, or plain class and returns a list of `FieldSchema`. Use a dotted `prefix` (e.g. `"budget"`, `"memory.decay"`). Nested models (e.g. `Budget.per` → `RateLimit`) are recursed into; callables, Protocols, and `type[...]` are marked `remote_excluded`.
- **`extract_agent_schema(agent)`** — Builds a full `AgentSchema` from a live agent: agent section (max_tool_iterations, debug, loop_strategy, system_prompt, hitl_timeout), plus budget, memory, context, checkpoint, rate_limit, circuit_breaker, and output sections with current values.

Supported sources: Pydantic (`Budget`, `Memory`, `Decay`, `CheckpointConfig`), dataclasses (`Context`, `Output`, `APIRateLimit`), and plain classes with `__init__` (e.g. `CircuitBreaker`). StrEnum fields get `enum_values`; Pydantic constraints (ge, le, gt, lt, pattern, min_length, max_length) are extracted from `Field` metadata.

### Per-class schema (RemoteConfigurable)

Schema and apply logic are owned per config class via the **`RemoteConfigurable`** protocol (`syrin.remote.RemoteConfigurable`). Each section (agent, budget, memory, context, checkpoint, rate_limit, circuit_breaker, output) is implemented by the corresponding class:

- **`get_remote_config_schema(self, section_key)`** — Returns `(ConfigSchema, current_values)` for this section. Implementations use `build_section_schema_from_obj` from `syrin.remote._schema` and the existing `extract_schema` reflection.
- **`apply_remote_overrides(self, agent, pairs, section_schema)`** — Applies a list of `(path, value)` pairs for this section to the agent.

The agent’s **`REMOTE_CONFIG_SECTIONS`** class attribute maps section keys to attribute names (or `None` for the agent section). Adding a new section means implementing `RemoteConfigurable` on the config class and adding one entry to `REMOTE_CONFIG_SECTIONS`. See `docs/remote-config-scalability.md` for the design rationale.

## Registry

The **config registry** tracks live agents and their schemas. It is a singleton accessed via `get_registry()`. Used by the integration layer when `syrin.init()` is called: agents register on construction and the backend receives their schema.

- **`get_registry()`** — Returns the global `ConfigRegistry` singleton. Thread-safe.
- **`ConfigRegistry.register(agent)`** — Extracts schema from the agent, stores the agent (weak ref) and schema. Returns the `AgentSchema` with canonical `agent_id`. Re-registering the same agent overwrites the stored schema.
- **`ConfigRegistry.unregister(agent_id)`** — Removes the agent and schema for that id. Idempotent if the id is unknown.
- **`ConfigRegistry.get_agent(agent_id)`** — Returns the live agent or `None` if unknown or the agent was garbage-collected.
- **`ConfigRegistry.get_schema(agent_id)`** — Returns the stored `AgentSchema` or `None`.
- **`ConfigRegistry.all_schemas()`** — Returns a copy of all stored schemas (`dict[str, AgentSchema]`).
- **`ConfigRegistry.make_agent_id(agent)`** — Deterministic id: `name:ClassName` when the agent has an explicit name (and it is not the class name lowercased), otherwise `ClassName:uuid8` (8 hex chars) so multiple default-named agents get unique ids.

Agents are held by weak reference so they can be garbage-collected; after GC, `get_agent(agent_id)` returns `None`. Schema entries remain until `unregister(agent_id)` is called.

## Resolver

The **config resolver** applies override payloads to a live agent with full validation. It rejects unknown paths, `remote_excluded` fields, invalid enum values, and values that fail Pydantic/dataclass validation; failed sections leave the agent unchanged for that section.

- **`ConfigResolver().apply_overrides(agent, payload, schema=None)`** — Applies `payload.overrides` to the agent. Returns a `ResolveResult`. Schema is optional: if omitted, taken from the registry by `payload.agent_id`, then from `extract_agent_schema(agent)` if not registered.
- **`ResolveResult`** — Dataclass with:
  - **`accepted`** — Paths that were applied successfully.
  - **`rejected`** — List of `(path, reason)` for overrides not applied (e.g. `unknown_path`, `remote_excluded`, `invalid_enum_value`, or validation error message).
  - **`pending_restart`** — Paths that were applied but require an agent restart to take full effect (e.g. `memory.backend`, `checkpoint.storage`).
- **Hot-swap blocklist** — Paths in `syrin.remote._resolver.HOT_SWAP_BLOCKLIST` (`memory.backend`, `memory.path`, `checkpoint.storage`, `checkpoint.path`) are applied in memory and included in `pending_restart`; the agent should be restarted for backend re-init.
- **Section atomicity** — If any override in a section fails validation, the whole section is rejected and the agent is unchanged for that section; other sections can still be applied.
- **StrEnum** — String values for enum fields (e.g. `memory.decay.strategy="linear"`) are coerced to the corresponding enum; invalid enum strings are rejected before apply.

## Transports

Overrides are delivered through a **ConfigTransport** protocol. The resolver is transport-agnostic; each transport implements `register(schema)`, `on_override(agent_id, callback)`, and `stop()`. Callbacks may be invoked from a background thread (SSE and polling).

| Transport | When to use | Behavior |
|-----------|-------------|----------|
| **SSETransport** | SaaS mode (`syrin.init(api_key=...)`) | POST schema to backend, open GET stream for overrides. Auto-reconnect with exponential backoff (1s–60s). Backend down at register returns `SyncResponse(ok=False)`. |
| **ServeTransport** | Self-hosted: config routes on `agent.serve()` | No network. `register()` returns ok. `on_override(agent_id, callback)` stores callback; route handlers use `get_callback(agent_id)(payload)`. |
| **PollingTransport** | Environments where SSE is blocked | Periodic `GET /agents/{id}/overrides?since_version=v`. Configurable interval (default 30s). |

- **ConfigTransport** — Protocol: `register(schema) -> SyncResponse`, `on_override(agent_id, callback)`, `stop()`.
- **SSETransport(base_url, api_key, client=None)** — Uses httpx. Register: `POST {base_url}/agents/{id}/register`. Stream: `GET {base_url}/agents/{id}/stream`; parses SSE `event: override`, `data: <OverridePayload JSON>`.
- **PollingTransport(base_url, api_key, poll_interval=30, client=None)** — Daemon thread GETs overrides and invokes callback.
- **ServeTransport()** — In-memory. `get_callback(agent_id)` for PATCH handler. `stop()` clears callbacks.

## syrin.init() and integration

Call **`syrin.init()`** once at startup to enable remote config. When enabled, every new `Agent` registers with the registry and (when a transport is set) with the backend; overrides are applied automatically.

### API

```python
import syrin

# SaaS mode — agent phones home via SSETransport (default)
syrin.init(
    api_key="sk-syrin-...",       # or set SYRIN_API_KEY env
    base_url="https://api.syrin.ai/v1",  # optional; default above
)

# With env only (e.g. in production)
import os
# SYRIN_API_KEY=sk-... is set in deployment
if os.getenv("SYRIN_API_KEY"):
    syrin.init()

# Custom transport
from syrin.remote import PollingTransport
syrin.init(
    transport=PollingTransport(
        base_url="https://my-config-server.internal/v1",
        api_key="sk-...",
        poll_interval=15,
    ),
)
```

- **No `syrin.init()`:** Agents run with local config only. Zero overhead; no registry or network.
- **`syrin.init(api_key=...)` or `SYRIN_API_KEY`:** Creates default `SSETransport`; agents register and receive overrides via SSE.
- **`syrin.init(transport=...)`:** Uses your transport (e.g. `ServeTransport`, `PollingTransport`). `api_key`/`base_url` are still stored on `get_config()` for reference.

### Agent hook and hooks

At the end of `Agent.__init__`, the library calls the remote init hook. When `get_config().cloud_enabled` is True it: registers the agent, calls `transport.register(schema)`, applies any `initial_overrides` from the registration response, and starts listening for overrides via `transport.on_override(agent_id, callback)`. When overrides are applied (or rejected), the agent emits:

- **`Hook.REMOTE_CONFIG_UPDATE`** — When at least one override was applied (context: `accepted`, `pending_restart`).
- **`Hook.REMOTE_CONFIG_ERROR`** — When any override was rejected (context: `rejected`).

### Config routes on agent.serve()

When you serve an agent (e.g. `agent.serve()` or `AgentRouter`), the following routes are added automatically (same prefix as `/chat`, `/health`, etc.):

| Method | Path | Description |
|--------|------|-------------|
| GET | `/config` | Schema and current values (for dashboards). |
| PATCH | `/config` | Apply overrides. Body: `OverridePayload` (agent_id, version, overrides). Returns `{accepted, rejected, pending_restart}`. Rejected paths are rolled back from the override store. |
| GET | `/config/stream` | SSE stream: `event: heartbeat` (data: `{version}`), then `event: override` when PATCH is applied. |

No `syrin.init()` is required for these routes; they work with the in-process resolver. When you use a self-hosted dashboard, you can `curl` or `PATCH` to apply overrides without Syrin Cloud.

## Example: Building types by hand

```python
from syrin.remote import (
    AgentSchema,
    ConfigOverride,
    ConfigSchema,
    FieldSchema,
    OverridePayload,
    SyncRequest,
    SyncResponse,
)

# Single override
override = ConfigOverride(path="budget.run", value=2.0)

# Payload for one agent
payload = OverridePayload(
    agent_id="my_agent:MyAgent",
    version=1,
    overrides=[override],
)

# Registration response (e.g. from backend)
response = SyncResponse(
    ok=True,
    initial_overrides=[ConfigOverride(path="budget.run", value=1.5)],
    error=None,
)
```

Schema types are normally produced by the extractor. Example: extract from a class or from a live agent:

```python
from syrin import Agent, Budget, Model
from syrin.remote import extract_schema, extract_agent_schema

# From a config class (e.g. for a custom dashboard)
fields = extract_schema(Budget, "budget")
# fields[0].path == "budget.run", .type == "float", .constraints == {"ge": 0}, etc.

# Full agent schema (all sections + current values)
agent = Agent(model=Model.Almock(), budget=Budget(run=1.0))
schema = extract_agent_schema(agent)
# schema.sections["budget"], schema.current_values["budget.run"] == 1.0

# Registry (e.g. for integration or tests)
from syrin.remote import get_registry

reg = get_registry()
reg.register(agent)
agent_id = reg.make_agent_id(agent)  # e.g. "my_agent:Agent" or "Agent:a1b2c3d4"
schema = reg.get_schema(agent_id)
reg.unregister(agent_id)

# Resolver: apply overrides to a live agent
from syrin.remote import ConfigResolver, ResolveResult

resolver = ConfigResolver()
result = resolver.apply_overrides(agent, payload, schema=schema)
# result.accepted, result.rejected, result.pending_restart

# Transports (for integration or custom backends)
from syrin.remote import ConfigTransport, PollingTransport, ServeTransport, SSETransport

# Serve: in-memory; route handler calls get_callback(agent_id)(payload)
serve = ServeTransport()
serve.register(schema)
serve.on_override(agent_id, lambda p: resolver.apply_overrides(agent, p))
# serve.get_callback(agent_id)  # used by PATCH /config handler

# SSE: agent phones home (use with syrin.init(api_key=...) in integration)
# sse = SSETransport(base_url="https://api.syrin.ai/v1", api_key="sk-...")

# Polling: fallback when SSE is blocked
# poll = PollingTransport(base_url="...", api_key="...", poll_interval=15)
```
