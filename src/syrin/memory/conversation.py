"""Conversation memory implementations (buffer, windowed)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from syrin.types import Message


def _to_message(m: Message | dict[str, Any]) -> Message:
    """Convert Message or dict to Message."""
    if isinstance(m, Message):
        return m
    return Message.model_validate(m)


class ConversationMemory(ABC):
    """Base class for conversation memory. Subclasses implement storage policy.

    Use with Agent(memory=BufferMemory()) for session history. Implements
    add/get_messages/clear. Memory config uses Memory; ConversationMemory
    is for session-level buffers.
    """

    @abstractmethod
    def add(self, message: Message) -> None:
        """Append a message to memory."""
        ...

    @abstractmethod
    def get_messages(self) -> list[Message]:
        """Return messages to include in the next request (order preserved)."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all messages from memory."""
        ...

    def load_messages(self, messages: list[Message] | list[dict[str, Any]]) -> None:
        """Replace all messages with the given list. For restore from checkpoint.

        Accepts Message objects or dicts (from serialized checkpoint). Default
        implementation: clear then add each message. Subclasses may override
        for efficiency.
        """
        self.clear()
        for m in messages:
            self.add(_to_message(m))


class BufferMemory(ConversationMemory):
    """Stores all messages with no limit. Default for session memory."""

    def __init__(self) -> None:
        self._messages: list[Message] = []

    def add(self, message: Message) -> None:
        self._messages.append(message)

    def get_messages(self) -> list[Message]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()


class WindowMemory(ConversationMemory):
    """Keeps only the last k message pairs (user + assistant).

    Use when context is limited; older turns are dropped.
    """

    def __init__(self, k: int = 10) -> None:
        if k < 1:
            raise ValueError("WindowMemory k must be >= 1")
        self._k = k
        self._messages: list[Message] = []

    def add(self, message: Message) -> None:
        self._messages.append(message)

    def get_messages(self) -> list[Message]:
        n = self._k * 2
        if len(self._messages) <= n:
            return list(self._messages)
        return list(self._messages[-n:])

    def clear(self) -> None:
        self._messages.clear()


__all__ = [
    "ConversationMemory",
    "BufferMemory",
    "WindowMemory",
]
