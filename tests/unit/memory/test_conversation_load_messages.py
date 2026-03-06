"""Tests for ConversationMemory.load_messages - long-running session restore."""

from __future__ import annotations

import pytest

from syrin.enums import MessageRole
from syrin.memory.conversation import BufferMemory, WindowMemory
from syrin.types import Message


class TestBufferMemoryLoadMessages:
    """Tests for BufferMemory.load_messages()."""

    def test_load_messages_empty_replaces_all(self) -> None:
        """Load empty list clears buffer."""
        mem = BufferMemory()
        mem.add(Message(role=MessageRole.USER, content="Hello"))
        mem.add(Message(role=MessageRole.ASSISTANT, content="Hi"))
        mem.load_messages([])
        assert mem.get_messages() == []

    def test_load_messages_replaces_existing(self) -> None:
        """Load replaces all existing messages."""
        mem = BufferMemory()
        mem.add(Message(role=MessageRole.USER, content="A"))
        mem.add(Message(role=MessageRole.ASSISTANT, content="B"))
        new_msgs = [
            Message(role=MessageRole.USER, content="X"),
            Message(role=MessageRole.ASSISTANT, content="Y"),
        ]
        mem.load_messages(new_msgs)
        got = mem.get_messages()
        assert len(got) == 2
        assert got[0].content == "X"
        assert got[1].content == "Y"

    def test_load_messages_preserves_order(self) -> None:
        """Load preserves message order."""
        mem = BufferMemory()
        msgs = [
            Message(role=MessageRole.USER, content="1"),
            Message(role=MessageRole.ASSISTANT, content="2"),
            Message(role=MessageRole.USER, content="3"),
        ]
        mem.load_messages(msgs)
        got = mem.get_messages()
        assert [m.content for m in got] == ["1", "2", "3"]

    def test_load_messages_with_tool_calls(self) -> None:
        """Load handles messages with tool_calls."""
        from syrin.types import ToolCall

        mem = BufferMemory()
        msgs = [
            Message(
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[ToolCall(id="tc1", name="search", arguments={"q": "test"})],
            ),
        ]
        mem.load_messages(msgs)
        got = mem.get_messages()
        assert len(got) == 1
        assert len(got[0].tool_calls) == 1
        assert got[0].tool_calls[0].name == "search"

    def test_load_messages_does_not_mutate_input(self) -> None:
        """Load should not mutate the input list."""
        mem = BufferMemory()
        msgs = [Message(role=MessageRole.USER, content="hi")]
        original_len = len(msgs)
        mem.load_messages(msgs)
        assert len(msgs) == original_len
        assert msgs[0].content == "hi"


class TestWindowMemoryLoadMessages:
    """Tests for WindowMemory.load_messages()."""

    def test_load_messages_empty_replaces_all(self) -> None:
        """Load empty list clears window."""
        mem = WindowMemory(k=5)
        mem.add(Message(role=MessageRole.USER, content="A"))
        mem.add(Message(role=MessageRole.ASSISTANT, content="B"))
        mem.load_messages([])
        assert mem.get_messages() == []

    def test_load_messages_replaces_existing(self) -> None:
        """Load replaces existing messages (within window policy)."""
        mem = WindowMemory(k=2)
        mem.add(Message(role=MessageRole.USER, content="A"))
        mem.add(Message(role=MessageRole.ASSISTANT, content="B"))
        new_msgs = [
            Message(role=MessageRole.USER, content="X"),
            Message(role=MessageRole.ASSISTANT, content="Y"),
        ]
        mem.load_messages(new_msgs)
        got = mem.get_messages()
        assert len(got) == 2
        assert got[0].content == "X" and got[1].content == "Y"

    def test_load_messages_exceeds_window_stores_all_internally(self) -> None:
        """Load more than window size stores all; get_messages returns last k*2."""
        mem = WindowMemory(k=2)
        msgs = [
            Message(role=MessageRole.USER, content="u0"),
            Message(role=MessageRole.ASSISTANT, content="a0"),
            Message(role=MessageRole.USER, content="u1"),
            Message(role=MessageRole.ASSISTANT, content="a1"),
            Message(role=MessageRole.USER, content="u2"),
            Message(role=MessageRole.ASSISTANT, content="a2"),
        ]
        mem.load_messages(msgs)
        got = mem.get_messages()
        assert len(got) <= 4  # k*2
        assert got[-1].content == "a2"


class TestLoadMessagesEdgeCases:
    """Edge cases and invalid inputs."""

    def test_load_messages_empty_memory(self) -> None:
        """Load into empty memory."""
        mem = BufferMemory()
        mem.load_messages([Message(role=MessageRole.USER, content="hi")])
        assert len(mem.get_messages()) == 1
        assert mem.get_messages()[0].content == "hi"

    def test_load_messages_from_dict_checkpoint_serialization(self) -> None:
        """Load accepts dicts (from checkpoint serialization)."""
        mem = BufferMemory()
        msgs_dict = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        mem.load_messages(msgs_dict)
        got = mem.get_messages()
        assert len(got) == 2
        assert got[0].content == "Hello"
        assert got[1].content == "Hi"

    def test_load_messages_invalid_element_raises(self) -> None:
        """Passing non-Message and non-dict element raises."""
        from pydantic import ValidationError

        mem = BufferMemory()
        with pytest.raises(ValidationError):
            mem.load_messages([123])  # type: ignore[list-item]
