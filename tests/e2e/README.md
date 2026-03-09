# E2E tests

End-to-end tests run against the full stack (no internal mocks). Current contents:

- **`test_e2e_smoke.py`** — Smoke test: Agent → provider (Almock) → response over the real call path. No patching; always runs in CI.

To add e2e tests that hit real APIs (OpenAI, Anthropic, OpenRouter, etc.), add new `test_*.py` files here and use `pytest.importorskip` or `pytest.mark.skipif` when required env vars (e.g. API keys) are missing, so CI stays green without secrets.
