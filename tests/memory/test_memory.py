"""Tests for conversation memory (memory.py)."""

from __future__ import annotations

import pytest

from syrin.memory import BufferMemory, WindowMemory
from syrin.types import Message


def test_buffer_memory_add_and_get() -> None:
    mem = BufferMemory()
    assert mem.get_messages() == []
    mem.add(Message(role="user", content="Hi"))
    mem.add(Message(role="assistant", content="Hello"))
    msgs = mem.get_messages()
    assert len(msgs) == 2
    assert msgs[0].content == "Hi"
    assert msgs[1].content == "Hello"


def test_buffer_memory_clear() -> None:
    mem = BufferMemory()
    mem.add(Message(role="user", content="Hi"))
    mem.clear()
    assert mem.get_messages() == []


def test_window_memory_keeps_last_k_pairs() -> None:
    mem = WindowMemory(k=2)
    for i in range(5):
        mem.add(Message(role="user", content=f"u{i}"))
        mem.add(Message(role="assistant", content=f"a{i}"))
    msgs = mem.get_messages()
    assert len(msgs) == 4
    assert msgs[0].content == "u3"
    assert msgs[1].content == "a3"
    assert msgs[2].content == "u4"
    assert msgs[3].content == "a4"


def test_window_memory_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        WindowMemory(k=0)


# =============================================================================
# MEMORY EDGE CASES - TRY TO BREAK FUNCTIONALITY
# =============================================================================


def test_buffer_memory_with_very_long_content() -> None:
    """BufferMemory with very long message content."""
    mem = BufferMemory()
    long_content = "x" * 100000
    mem.add(Message(role="user", content=long_content))
    msgs = mem.get_messages()
    assert len(msgs) == 1
    assert len(msgs[0].content) == 100000


def test_buffer_memory_with_unicode() -> None:
    """BufferMemory with unicode content."""
    mem = BufferMemory()
    mem.add(Message(role="user", content="Hello 🌍 你好 🔥"))
    msgs = mem.get_messages()
    assert "🌍" in msgs[0].content


def test_buffer_memory_clear_twice() -> None:
    """BufferMemory clear twice should be safe."""
    mem = BufferMemory()
    mem.add(Message(role="user", content="Hi"))
    mem.clear()
    mem.clear()  # Should not raise
    assert mem.get_messages() == []


def test_buffer_memory_many_messages() -> None:
    """BufferMemory with many messages."""
    mem = BufferMemory()
    for i in range(1000):
        mem.add(Message(role="user", content=f"message {i}"))
    assert len(mem.get_messages()) == 1000


def test_window_memory_with_odd_count() -> None:
    """WindowMemory with odd number of messages."""
    mem = WindowMemory(k=2)
    mem.add(Message(role="user", content="u1"))
    mem.add(Message(role="assistant", content="a1"))
    mem.add(Message(role="user", content="u2"))
    # Only user message, no assistant
    msgs = mem.get_messages()
    assert len(msgs) == 3


def test_window_memory_with_single_message() -> None:
    """WindowMemory with single message."""
    mem = WindowMemory(k=1)
    mem.add(Message(role="user", content="Hi"))
    msgs = mem.get_messages()
    assert len(msgs) == 1


def test_window_memory_k_equals_one() -> None:
    """WindowMemory with k=1."""
    mem = WindowMemory(k=1)
    for i in range(4):
        mem.add(Message(role="user", content=f"u{i}"))
        mem.add(Message(role="assistant", content=f"a{i}"))
    msgs = mem.get_messages()
    assert len(msgs) == 2


def test_window_memory_clear() -> None:
    """WindowMemory clear should work."""
    mem = WindowMemory(k=2)
    mem.add(Message(role="user", content="Hi"))
    mem.clear()
    assert mem.get_messages() == []


def test_message_with_all_roles() -> None:
    """Message with all possible roles."""
    roles = ["system", "user", "assistant", "tool"]
    for role in roles:
        msg = Message(role=role, content="test")
        assert msg.role == role


def test_message_with_tool_call_id() -> None:
    """Message with tool_call_id."""
    msg = Message(role="tool", content="result", tool_call_id="call_123")
    assert msg.tool_call_id == "call_123"


def test_buffer_memory_preserves_order() -> None:
    """BufferMemory should preserve message order."""
    mem = BufferMemory()
    for i in range(10):
        mem.add(Message(role="user", content=f"user {i}"))
        mem.add(Message(role="assistant", content=f"assistant {i}"))
    msgs = mem.get_messages()
    for i in range(10):
        assert msgs[i * 2].content == f"user {i}"
        assert msgs[i * 2 + 1].content == f"assistant {i}"
