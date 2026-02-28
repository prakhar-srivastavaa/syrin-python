"""Conversation memory implementations (buffer, windowed)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from syrin.types import Message


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
