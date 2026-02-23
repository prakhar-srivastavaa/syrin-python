"""Conversation memory for agents (buffer, windowed)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from syrin.types import Message


class Memory(ABC):
    """Base class for conversation memory. Subclasses implement storage policy."""

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


class BufferMemory(Memory):
    """Stores all messages (no limit). Default memory type."""

    def __init__(self) -> None:
        self._messages: list[Message] = []

    def add(self, message: Message) -> None:
        self._messages.append(message)

    def get_messages(self) -> list[Message]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()


class WindowMemory(Memory):
    """Keeps only the last k message pairs (user + assistant)."""

    def __init__(self, k: int = 10) -> None:
        if k < 1:
            raise ValueError("WindowMemory k must be >= 1")
        self._k = k
        self._messages: list[Message] = []

    def add(self, message: Message) -> None:
        self._messages.append(message)

    def get_messages(self) -> list[Message]:
        # Return last k*2 messages (pairs); if odd, we have one extra (e.g. user at end)
        n = self._k * 2
        if len(self._messages) <= n:
            return list(self._messages)
        return list(self._messages[-n:])

    def clear(self) -> None:
        self._messages.clear()
