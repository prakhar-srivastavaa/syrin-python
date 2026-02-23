"""Abstract provider interface for LLM completions."""

from __future__ import annotations

import asyncio
import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import Any

from syrin.types import Message, ModelConfig, ProviderResponse, ToolSpec

_log = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message=".*Event loop is closed.*")

_default_loop: asyncio.AbstractEventLoop | None = None


def _get_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create a persistent event loop to avoid 'Event loop is closed' errors."""
    global _default_loop
    if _default_loop is None or _default_loop.is_closed():
        _default_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_default_loop)
    return _default_loop


class Provider(ABC):
    """Abstract base for LLM providers. Implement complete() and optionally complete_sync()."""

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        model: ModelConfig,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Run a completion. Returns content, tool_calls, and token_usage."""
        ...

    def _run_async(self, coro: Any) -> ProviderResponse | None:
        """Run async coroutine using a persistent event loop."""
        loop = _get_event_loop()
        try:
            result: ProviderResponse | None = loop.run_until_complete(coro)
            return result
        except RuntimeError as e:
            if "Event loop is closed" in str(e):
                _log.debug("Async cleanup warning (non-fatal): %s", e)
                return None
            raise
        except asyncio.CancelledError:
            return None

    @staticmethod
    def _handle_task_exception(task: asyncio.Task[Any]) -> None:
        """Suppress event loop closed errors in background tasks."""
        try:
            task.result()
        except RuntimeError as e:
            if "Event loop is closed" not in str(e):
                _log.debug("Background task error (non-fatal): %s", e)
        except asyncio.CancelledError:
            pass

    def complete_sync(
        self,
        messages: list[Message],
        model: ModelConfig,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse | None:
        """Synchronous wrapper using asyncio.run()."""
        return self._run_async(self.complete(messages=messages, model=model, tools=tools, **kwargs))

    async def stream(
        self,
        messages: list[Message],
        model: ModelConfig,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ProviderResponse]:
        """Stream response chunks. Yields partial responses."""
        response = await self.complete(messages, model, tools, **kwargs)
        yield response

    def stream_sync(
        self,
        messages: list[Message],
        model: ModelConfig,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> Iterator[ProviderResponse]:
        """Synchronous streaming - yields responses."""
        response = self._run_async(self.complete(messages, model, tools, **kwargs))
        if response:
            yield response
