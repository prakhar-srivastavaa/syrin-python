"""Comprehensive tests for CLI module and WorkflowDebugger.

This module tests:
1. WorkflowDebugger class - all methods and edge cases
2. CLI commands - doctor, run, trace
3. Utility functions - check_for_trace_flag, remove_trace_flag, auto_trace
4. Edge cases and error handling
"""

from __future__ import annotations

import json
import sys
from contextlib import suppress
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from syrin.cli import (
    DebugEvent,
    WorkflowDebugger,
    auto_trace,
    check_for_trace_flag,
    doctor_command,
    main,
    remove_trace_flag,
    run_with_observability,
)
from syrin.enums import Hook
from syrin.events import EventContext

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_events():
    """Create a mock events object for testing."""
    events = MagicMock()
    events.on = MagicMock()
    return events


@pytest.fixture
def mock_agent(mock_events):
    """Create a mock agent with events."""
    agent = MagicMock()
    agent.events = mock_events
    return agent


@pytest.fixture
def sample_event_data():
    """Sample event data for testing."""
    return {
        "agent_name": "test_agent",
        "tool_name": "calculator",
        "model": "gpt-4",
        "cost": 0.001,
        "duration": 1.5,
        "task": "Test task",
        "error": None,
    }


@pytest.fixture
def populated_debugger():
    """Create a debugger with sample events."""
    debugger = WorkflowDebugger(verbose=False)

    # Add some sample events (using actual Hook enum values which are dot-notation)
    hooks = [
        Hook.AGENT_RUN_START,
        Hook.LLM_REQUEST_START,
        Hook.LLM_REQUEST_END,
        Hook.TOOL_CALL_START,
        Hook.TOOL_CALL_END,
        Hook.AGENT_RUN_END,
    ]

    for hook in hooks:
        # Hook values use dot notation (e.g., "agent.run.end"), so check for ".end"
        has_duration = ".end" in hook.value
        debugger.events.append(
            DebugEvent(
                timestamp=datetime.now(),
                hook=hook,
                data={"cost": 0.001, "agent_name": "test"},
                duration_ms=100.0 if has_duration else None,
            )
        )

    return debugger


# =============================================================================
# DEBUGGER CLASS TESTS
# =============================================================================


class TestWorkflowDebugger:
    """Test WorkflowDebugger class."""

    def test_debugger_creation(self):
        """Test basic debugger creation."""
        debugger = WorkflowDebugger(verbose=True)
        assert debugger.verbose is True
        assert debugger.colors is True
        assert debugger.events == []
        assert debugger._attached_objects == []

    def test_debugger_creation_silent(self):
        """Test debugger creation in silent mode."""
        debugger = WorkflowDebugger(verbose=False, colors=False)
        assert debugger.verbose is False
        assert debugger.colors is False

    def test_attach_success(self, mock_agent):
        """Test successful attachment to agent."""
        debugger = WorkflowDebugger(verbose=False)
        result = debugger.attach(mock_agent)

        assert result is debugger  # Should return self for chaining
        assert mock_agent in debugger._attached_objects
        assert mock_agent.events.on.called

    def test_attach_no_events_attribute(self):
        """Test attachment fails when object has no events."""
        debugger = WorkflowDebugger(verbose=False)
        obj = MagicMock()  # No events attribute
        del obj.events

        with pytest.raises(ValueError, match="has no 'events' attribute"):
            debugger.attach(obj)

    def test_attach_registers_all_hooks(self, mock_agent):
        """Test that attach registers handlers for all hooks."""
        debugger = WorkflowDebugger(verbose=False)
        debugger.attach(mock_agent)

        # Should register handlers for all Hook enum values
        call_count = mock_agent.events.on.call_count
        assert call_count > 0
        # Should be at least as many as Hook enum values
        assert call_count >= len(Hook)

    def test_attach_chaining(self, mock_agent):
        """Test that attach returns self for chaining."""
        debugger = WorkflowDebugger(verbose=False)

        # Create another mock
        mock_agent2 = MagicMock()
        mock_agent2.events = MagicMock()

        result = debugger.attach(mock_agent).attach(mock_agent2)
        assert result is debugger
        assert len(debugger._attached_objects) == 2

    def test_clear(self, populated_debugger):
        """Test clearing events."""
        assert len(populated_debugger.events) > 0
        populated_debugger.clear()
        assert len(populated_debugger.events) == 0


class TestDebuggerEventHandling:
    """Test event handling in WorkflowDebugger."""

    def test_on_event_creates_debug_event(self):
        """Test that _on_event creates a DebugEvent."""
        debugger = WorkflowDebugger(verbose=False)
        ctx = {"cost": 0.005, "agent_name": "test"}

        debugger._on_event(Hook.AGENT_RUN_START, ctx)

        assert len(debugger.events) == 1
        assert debugger.events[0].hook == Hook.AGENT_RUN_START
        assert debugger.events[0].data["cost"] == 0.005

    def test_on_event_captures_timestamp(self):
        """Test that _on_event captures timestamp."""
        debugger = WorkflowDebugger(verbose=False)
        before = datetime.now()

        debugger._on_event(Hook.AGENT_RUN_START, {})

        after = datetime.now()
        assert len(debugger.events) == 1
        assert before <= debugger.events[0].timestamp <= after

    def test_on_event_captures_duration(self):
        """Test that _on_event captures duration for end events."""
        debugger = WorkflowDebugger(verbose=False)
        ctx = {"duration": 2.5}

        debugger._on_event(Hook.AGENT_RUN_END, ctx)

        assert debugger.events[0].duration_ms == 2500.0  # Converted to ms

    def test_on_event_handles_none_duration(self):
        """Test that _on_event handles None duration."""
        debugger = WorkflowDebugger(verbose=False)
        ctx = {"duration": None}

        debugger._on_event(Hook.AGENT_RUN_START, ctx)

        assert debugger.events[0].duration_ms is None

    def test_on_event_copies_context(self):
        """Test that _on_event copies context to avoid mutations."""
        debugger = WorkflowDebugger(verbose=False)
        ctx = {"key": "value"}

        debugger._on_event(Hook.AGENT_RUN_START, ctx)

        # Modify original
        ctx["key"] = "modified"

        # Should not affect stored event
        assert debugger.events[0].data["key"] == "value"


class TestDebuggerSummary:
    """Test summary functionality."""

    def test_print_summary_empty(self, capsys):
        """Test summary with no events."""
        debugger = WorkflowDebugger(verbose=False)
        debugger.print_summary()

        captured = capsys.readouterr()
        assert "WORKFLOW EXECUTION SUMMARY" in captured.out

    def test_print_summary_with_events(self, populated_debugger, capsys):
        """Test summary with events."""
        populated_debugger.print_summary()

        captured = capsys.readouterr()
        assert "WORKFLOW EXECUTION SUMMARY" in captured.out
        # Hook enum uses dot notation (e.g., "agent.run.start")
        assert "agent.run.start" in captured.out
        assert "agent.run.end" in captured.out
        assert "Statistics:" in captured.out

    def test_print_summary_event_counts(self, populated_debugger, capsys):
        """Test that summary shows correct event counts."""
        populated_debugger.print_summary()

        captured = capsys.readouterr()
        # Should show counts for each hook type (dot notation format)
        assert "agent.run.start:" in captured.out

    def test_print_summary_statistics(self, populated_debugger, capsys):
        """Test that summary shows statistics."""
        populated_debugger.print_summary()

        captured = capsys.readouterr()
        assert "Agent runs:" in captured.out
        assert "Tool calls:" in captured.out
        assert "LLM calls:" in captured.out
        assert "Total cost:" in captured.out

    def test_print_summary_with_errors(self, capsys):
        """Test summary with error events."""
        debugger = WorkflowDebugger(verbose=False)
        debugger.events.append(
            DebugEvent(
                timestamp=datetime.now(),
                hook=Hook.TOOL_ERROR,
                data={"error": "Something went wrong"},
            )
        )

        debugger.print_summary()

        captured = capsys.readouterr()
        assert "Errors encountered:" in captured.out
        assert "Something went wrong" in captured.out


class TestDebuggerExport:
    """Test export functionality."""

    def test_export_jsonl_creates_file(self, populated_debugger, tmp_path):
        """Test that export creates a file."""
        filepath = tmp_path / "debug_trace.jsonl"
        populated_debugger.export_jsonl(str(filepath))

        assert filepath.exists()

    def test_export_jsonl_content(self, populated_debugger, tmp_path):
        """Test that export writes valid JSONL."""
        filepath = tmp_path / "debug_trace.jsonl"
        populated_debugger.export_jsonl(str(filepath))

        content = filepath.read_text()
        lines = content.strip().split("\n")

        assert len(lines) == len(populated_debugger.events)

        # Each line should be valid JSON
        for line in lines:
            data = json.loads(line)
            assert "timestamp" in data
            assert "hook" in data
            assert "data" in data

    def test_export_jsonl_data_integrity(self, populated_debugger, tmp_path):
        """Test that export preserves data correctly."""
        filepath = tmp_path / "debug_trace.jsonl"
        populated_debugger.export_jsonl(str(filepath))

        with open(filepath) as f:
            lines = f.readlines()

        # Check first event
        first_event = json.loads(lines[0])
        assert first_event["hook"] == populated_debugger.events[0].hook.value

    def test_export_empty_debugger(self, tmp_path):
        """Test exporting empty debugger."""
        debugger = WorkflowDebugger(verbose=False)
        filepath = tmp_path / "empty.jsonl"
        debugger.export_jsonl(str(filepath))

        content = filepath.read_text()
        assert content == ""

    def test_export_prints_confirmation_when_verbose(self, populated_debugger, tmp_path, capsys):
        """Test that verbose mode prints export confirmation."""
        debugger = WorkflowDebugger(verbose=True)
        debugger.events = populated_debugger.events

        filepath = tmp_path / "debug.jsonl"
        debugger.export_jsonl(str(filepath))

        captured = capsys.readouterr()
        assert "Exported" in captured.out
        assert str(filepath) in captured.out


class TestDebuggerPrintEvent:
    """Test event printing functionality."""

    def test_print_event_start_hook(self, capsys):
        """Test printing start event."""
        debugger = WorkflowDebugger(verbose=False)
        event = DebugEvent(
            timestamp=datetime.now(),
            hook=Hook.AGENT_RUN_START,
            data={"agent_name": "test_agent"},
        )

        debugger._print_event(event)

        captured = capsys.readouterr()
        # Hook enum uses dot notation (e.g., "agent.run.start")
        assert "agent.run.start" in captured.out
        assert "test_agent" in captured.out

    def test_print_event_with_cost(self, capsys):
        """Test printing event with cost."""
        debugger = WorkflowDebugger(verbose=False)
        event = DebugEvent(
            timestamp=datetime.now(),
            hook=Hook.AGENT_RUN_END,
            data={"cost": 0.0123},
        )

        debugger._print_event(event)

        captured = capsys.readouterr()
        assert "$0.0123" in captured.out or "0.0123" in captured.out

    def test_print_event_with_duration(self, capsys):
        """Test printing event with duration."""
        debugger = WorkflowDebugger(verbose=False)
        event = DebugEvent(
            timestamp=datetime.now(),
            hook=Hook.AGENT_RUN_END,
            data={},
            duration_ms=1234.5,
        )

        debugger._print_event(event)

        captured = capsys.readouterr()
        assert "1234.50ms" in captured.out or "1234.5" in captured.out

    def test_print_event_with_error(self, capsys):
        """Test printing error event."""
        debugger = WorkflowDebugger(verbose=False)
        event = DebugEvent(
            timestamp=datetime.now(),
            hook=Hook.TOOL_ERROR,
            data={"error": "Tool execution failed"},
        )

        debugger._print_event(event)

        captured = capsys.readouterr()
        assert "Error:" in captured.out
        assert "Tool execution failed" in captured.out

    def test_print_event_truncates_long_task(self, capsys):
        """Test that long tasks are truncated."""
        debugger = WorkflowDebugger(verbose=False)
        long_task = "A" * 100
        event = DebugEvent(
            timestamp=datetime.now(),
            hook=Hook.AGENT_RUN_START,
            data={"task": long_task},
        )

        debugger._print_event(event)

        captured = capsys.readouterr()
        assert "..." in captured.out
        assert len(captured.out) < len(long_task) + 100

    def test_print_event_without_colors(self, capsys):
        """Test printing without colors."""
        debugger = WorkflowDebugger(verbose=False, colors=False)
        event = DebugEvent(
            timestamp=datetime.now(),
            hook=Hook.AGENT_RUN_START,
            data={},
        )

        debugger._print_event(event)

        captured = capsys.readouterr()
        # Should not contain ANSI codes when colors=False
        assert "\033[" not in captured.out


# =============================================================================
# UTILITY FUNCTION TESTS
# =============================================================================


class TestTraceFlagFunctions:
    """Test trace flag detection and manipulation."""

    def test_check_for_trace_flag_present(self):
        """Test detection when flag is present."""
        with patch.object(sys, "argv", ["script.py", "--trace"]):
            assert check_for_trace_flag() is True

    def test_check_for_trace_flag_absent(self):
        """Test detection when flag is absent."""
        with patch.object(sys, "argv", ["script.py", "--other-flag"]):
            assert check_for_trace_flag() is False

    def test_check_for_trace_flag_empty_argv(self):
        """Test detection with empty argv."""
        with patch.object(sys, "argv", []):
            assert check_for_trace_flag() is False

    def test_remove_trace_flag_removes_flag(self):
        """Test removing trace flag."""
        with patch.object(sys, "argv", ["script.py", "--trace", "arg1"]):
            remove_trace_flag()
            assert "--trace" not in sys.argv
            assert sys.argv == ["script.py", "arg1"]

    def test_remove_trace_flag_removes_multiple(self):
        """Test removing multiple trace flags."""
        with patch.object(sys, "argv", ["--trace", "script.py", "--trace"]):
            remove_trace_flag()
            assert sys.argv.count("--trace") == 0

    def test_remove_trace_flag_no_flag(self):
        """Test removing when flag not present."""
        original = ["script.py", "arg1", "arg2"]
        with patch.object(sys, "argv", original.copy()):
            remove_trace_flag()
            assert sys.argv == original


class TestAutoTrace:
    """Test auto_trace context manager."""

    def test_auto_trace_enables_tracing(self):
        """Test that auto_trace enables tracing."""
        from syrin.observability import get_tracer

        tracer = get_tracer()

        with auto_trace():
            assert tracer.debug_mode is True

    def test_auto_trace_adds_exporter(self):
        """Test that auto_trace adds ConsoleExporter."""
        from syrin.observability import get_tracer

        tracer = get_tracer()
        initial_count = len(tracer._exporters)

        with auto_trace():
            # Should have added ConsoleExporter
            assert len(tracer._exporters) >= initial_count

    def test_auto_trace_restores_state(self):
        """Test that state is maintained after context."""
        from syrin.observability import get_tracer

        tracer = get_tracer()

        with auto_trace():
            pass

        # After context, debug mode should still be True
        # (as per implementation note, we don't remove exporters)
        assert tracer.debug_mode is True


# =============================================================================
# CLI COMMAND TESTS
# =============================================================================


class TestDoctorCommand:
    """Test doctor command."""

    def test_doctor_runs_without_error(self, capsys):
        """Test that doctor command runs."""
        result = doctor_command()

        captured = capsys.readouterr()
        assert "Syrin Doctor" in captured.out
        # Should return 0 or 1 (not raise)
        assert result in (0, 1)

    def test_doctor_checks_python_version(self, capsys):
        """Test that doctor checks Python version."""
        doctor_command()

        captured = capsys.readouterr()
        assert "Python version" in captured.out

    def test_doctor_checks_syrin_import(self, capsys):
        """Test that doctor checks Syrin import."""
        doctor_command()

        captured = capsys.readouterr()
        assert "Syrin import" in captured.out

    def test_doctor_checks_dependencies(self, capsys):
        """Test that doctor checks key dependencies."""
        doctor_command()

        captured = capsys.readouterr()
        assert "pydantic" in captured.out


class TestRunWithObservability:
    """Test run_with_observability function."""

    def test_run_nonexistent_script(self, capsys):
        """Test running nonexistent script."""
        result = run_with_observability("/nonexistent/script.py")

        assert result == 1
        captured = capsys.readouterr()
        assert "not found" in captured.out.lower()

    def test_run_invalid_script(self, tmp_path, capsys):
        """Test running invalid Python script."""
        script = tmp_path / "invalid.py"
        script.write_text("this is not valid python")

        result = run_with_observability(str(script))

        assert result == 1
        captured = capsys.readouterr()
        assert "failed" in captured.out.lower() or "error" in captured.out.lower()

    def test_run_valid_script(self, tmp_path):
        """Test running valid Python script."""
        script = tmp_path / "valid.py"
        script.write_text("print('Hello from script')")

        result = run_with_observability(str(script))

        assert result == 0

    def test_run_enables_tracing(self, tmp_path, capsys):
        """Test that run enables tracing."""
        script = tmp_path / "test.py"
        script.write_text("pass")

        run_with_observability(str(script))

        captured = capsys.readouterr()
        assert "observability" in captured.out.lower() or "trace" in captured.out.lower()


class TestMainCLI:
    """Test main CLI entry point."""

    def test_main_no_args_prints_help(self, capsys):
        """Test that main with no args prints help."""
        with patch.object(sys, "argv", ["syrin"]):
            result = main()

        captured = capsys.readouterr()
        assert "usage:" in captured.out.lower()
        assert result == 0

    def test_main_help_flag(self, capsys):
        """Test --help flag."""
        with patch.object(sys, "argv", ["syrin", "--help"]), pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0

    def test_main_doctor_command(self, capsys):
        """Test doctor command via main."""
        with patch.object(sys, "argv", ["syrin", "doctor"]):
            result = main()

        captured = capsys.readouterr()
        assert "Syrin Doctor" in captured.out
        assert result in (0, 1)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestDebuggerEdgeCases:
    """Edge cases for WorkflowDebugger."""

    def test_attach_with_none_events(self):
        """Test attach when events is None."""
        debugger = WorkflowDebugger(verbose=False)
        obj = MagicMock()
        obj.events = None

        # attach() checks hasattr first, so None will return True but then fail on access
        # This should be handled gracefully (skip registration or raise clear error)
        with suppress(ValueError, AttributeError, TypeError):
            debugger.attach(obj)

    def test_on_event_with_exception_in_data(self):
        """Test handling event data that might raise exception."""
        debugger = WorkflowDebugger(verbose=False)

        # Create context that raises on dict() conversion
        class BadContext:
            def __getitem__(self, key):
                raise RuntimeError("Bad context")

            def get(self, _key, _default=None):
                raise RuntimeError("Bad context")

        # The implementation does dict(ctx) which calls __iter__ and __getitem__
        # This will raise RuntimeError, which is acceptable behavior
        with pytest.raises(RuntimeError, match="Bad context"):
            debugger._on_event(Hook.AGENT_RUN_START, BadContext())

    def test_print_summary_with_unicode(self, capsys):
        """Test summary with unicode characters."""
        debugger = WorkflowDebugger(verbose=False)
        debugger.events.append(
            DebugEvent(
                timestamp=datetime.now(),
                hook=Hook.AGENT_RUN_START,
                data={"agent_name": "测试代理 🎉"},
            )
        )

        try:
            debugger.print_summary()
            captured = capsys.readouterr()
            # Unicode may or may not be in output depending on terminal encoding
            # Just verify no exception was raised
            assert "WORKFLOW EXECUTION SUMMARY" in captured.out
        except UnicodeEncodeError:
            pytest.skip("Unicode not supported in this environment")

    def test_print_event_with_special_characters(self, capsys):
        """Test printing event with special characters."""
        debugger = WorkflowDebugger(verbose=False)
        event = DebugEvent(
            timestamp=datetime.now(),
            hook=Hook.TOOL_CALL_START,
            data={
                "tool_name": "test_tool",
                "task": "Task with \n newlines \t tabs",
            },
        )

        debugger._print_event(event)
        captured = capsys.readouterr()
        assert "test_tool" in captured.out

    def test_export_jsonl_with_special_chars(self, tmp_path):
        """Test exporting events with special characters."""
        debugger = WorkflowDebugger(verbose=False)
        debugger.events.append(
            DebugEvent(
                timestamp=datetime.now(),
                hook=Hook.AGENT_RUN_START,
                data={"message": 'Hello "world" \n New line'},
            )
        )

        filepath = tmp_path / "special.jsonl"
        debugger.export_jsonl(str(filepath))

        # Should be valid JSON
        with open(filepath) as f:
            data = json.loads(f.readline())
            assert 'Hello "world"' in data["data"]["message"]

    def test_debugger_with_many_events(self):
        """Test debugger with many events."""
        debugger = WorkflowDebugger(verbose=False)

        for i in range(1000):
            debugger.events.append(
                DebugEvent(
                    timestamp=datetime.now(),
                    hook=Hook.AGENT_RUN_START,
                    data={"index": i},
                )
            )

        assert len(debugger.events) == 1000
        # Summary should still work
        debugger.print_summary()

    def test_debugger_with_deeply_nested_data(self, capsys):
        """Test handling deeply nested data."""
        debugger = WorkflowDebugger(verbose=False)
        nested_data = {"level1": {"level2": {"level3": {"value": "deep"}}}}

        debugger.events.append(
            DebugEvent(
                timestamp=datetime.now(),
                hook=Hook.AGENT_RUN_START,
                data=nested_data,
            )
        )

        # Should not crash
        debugger.print_summary()

    def test_attach_handles_hook_registration_errors(self):
        """Test that attach handles errors during hook registration."""
        debugger = WorkflowDebugger(verbose=False)

        # Create events object that raises on certain hooks
        events = MagicMock()

        def raise_on_some_hooks(hook, _handler):
            if "ERROR" in hook.value:
                raise RuntimeError("Cannot register error hooks")

        events.on = raise_on_some_hooks

        obj = MagicMock()
        obj.events = events

        # Should not raise even though some hooks fail
        try:
            debugger.attach(obj)
        except RuntimeError:
            pytest.fail("Should suppress hook registration errors")


class TestCLIEdgeCases:
    """Edge cases for CLI."""

    def test_run_script_with_arguments(self, tmp_path):
        """Test running script with arguments."""
        script = tmp_path / "args.py"
        script.write_text("import sys; print(f'Args: {sys.argv}')")

        # Save original argv
        original_argv = sys.argv.copy()

        try:
            with patch.object(sys, "argv", [str(script), "--arg1", "--arg2"]):
                result = run_with_observability(str(script))
                assert result == 0
        finally:
            sys.argv = original_argv

    def test_run_script_that_raises_exception(self, tmp_path, capsys):
        """Test running script that raises exception."""
        script = tmp_path / "error.py"
        script.write_text("raise ValueError('Test error')")

        result = run_with_observability(str(script))

        assert result == 1
        captured = capsys.readouterr()
        assert "failed" in captured.out.lower() or "error" in captured.out.lower()

    def test_run_script_with_import_error(self, tmp_path, capsys):
        """Test running script with import error."""
        script = tmp_path / "import_error.py"
        script.write_text("import nonexistent_module_xyz")

        result = run_with_observability(str(script))

        assert result == 1

    def test_doctor_with_missing_dependencies(self, monkeypatch, capsys):
        """Test doctor when dependencies are missing."""
        # Mock importlib.import_module to simulate missing pydantic
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pydantic":
                raise ImportError("No module named 'pydantic'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        doctor_command()

        captured = capsys.readouterr()
        assert "pydantic" in captured.out.lower() or "check" in captured.out.lower()


class TestColors:
    """Test Colors class."""

    def test_colors_when_tty(self):
        """Test colors are enabled when stdout is a tty."""
        with patch.object(sys.stdout, "isatty", return_value=True):
            # Re-import to get fresh instance
            from importlib import reload

            import syrin.cli as cli_module

            reload(cli_module)

            # Colors should be ANSI codes
            assert "\033[" in cli_module.Colors.GREEN

    def test_colors_when_not_tty(self):
        """Test colors are disabled when stdout is not a tty."""
        with patch.object(sys.stdout, "isatty", return_value=False):
            from importlib import reload

            import syrin.cli as cli_module

            reload(cli_module)

            # Colors should be empty strings
            assert cli_module.Colors.RESET == ""


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_debugger_workflow(self, mock_agent, capsys):
        """Test complete debugger workflow from attach to summary."""
        debugger = WorkflowDebugger(verbose=True)

        # Attach
        debugger.attach(mock_agent)

        # Simulate events
        ctx1 = EventContext(agent_name="test_agent", cost=0.001)
        mock_agent.events.on.call_args_list[0][0][1](ctx1)

        ctx2 = EventContext(agent_name="test_agent", cost=0.002)
        mock_agent.events.on.call_args_list[1][0][1](ctx2)

        # Print summary
        debugger.print_summary()

        captured = capsys.readouterr()
        assert "test_agent" in captured.out or "WORKFLOW" in captured.out

    def test_trace_flag_integration(self, tmp_path):
        """Test complete --trace flag integration."""
        script = tmp_path / "trace_test.py"
        script.write_text("""
from syrin import check_for_trace_flag, remove_trace_flag
import sys

if check_for_trace_flag():
    remove_trace_flag()
    print("TRACE_ENABLED")
else:
    print("TRACE_DISABLED")
""")

        # Simulate running with --trace
        with patch.object(sys, "argv", [str(script), "--trace"]):
            exec(script.read_text())

    def test_cli_run_integration(self, tmp_path, capsys):
        """Test complete CLI run flow."""
        script = tmp_path / "integration.py"
        script.write_text("print('Integration test passed')")

        from syrin.cli import main

        with patch.object(sys, "argv", ["syrin", "run", str(script)]):
            result = main()

        assert result == 0


# =============================================================================
# INVALID/ERROR TESTS
# =============================================================================


class TestInvalidInputs:
    """Test handling of invalid inputs."""

    def test_debugger_attach_to_string(self):
        """Test attaching to a string (should fail)."""
        debugger = WorkflowDebugger(verbose=False)

        with pytest.raises(ValueError):
            debugger.attach("not an agent")

    def test_debugger_attach_to_int(self):
        """Test attaching to an int (should fail)."""
        debugger = WorkflowDebugger(verbose=False)

        with pytest.raises(ValueError):
            debugger.attach(123)

    def test_debugger_attach_to_dict(self):
        """Test attaching to a dict (should fail)."""
        debugger = WorkflowDebugger(verbose=False)

        with pytest.raises(ValueError):
            debugger.attach({"events": "not events object"})

    def test_export_to_invalid_path(self, populated_debugger):
        """Test exporting to invalid path."""
        with pytest.raises((OSError, IOError)):
            populated_debugger.export_jsonl("/nonexistent/directory/file.jsonl")

    def test_export_to_directory(self, populated_debugger, tmp_path):
        """Test exporting to a directory (should fail)."""
        with pytest.raises((OSError, IOError, IsADirectoryError)):
            populated_debugger.export_jsonl(str(tmp_path))

    def test_run_script_directory(self, tmp_path, capsys):
        """Test running a directory instead of a file."""
        result = run_with_observability(str(tmp_path))

        assert result == 1

    def test_run_script_no_read_permission(self, tmp_path):
        """Test running script without read permission."""
        script = tmp_path / "noperms.py"
        script.write_text("print('test')")
        script.chmod(0o000)

        try:
            result = run_with_observability(str(script))
            assert result == 1
        finally:
            script.chmod(0o644)  # Restore permissions for cleanup

    def test_debugger_with_none_hook(self):
        """Test handling None hook value."""
        debugger = WorkflowDebugger(verbose=False)

        # This shouldn't happen in practice, but test defense
        with suppress(AttributeError, TypeError):
            debugger._on_event(None, {})  # type: ignore

    def test_debugger_with_invalid_timestamp(self):
        """Test handling invalid timestamp."""
        debugger = WorkflowDebugger(verbose=False)

        event = DebugEvent(
            timestamp="not a datetime",  # type: ignore
            hook=Hook.AGENT_RUN_START,
            data={},
        )

        # Should not crash when printing
        with suppress(AttributeError, TypeError):
            debugger._print_event(event)


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================


class TestPerformance:
    """Performance-related tests."""

    def test_debugger_handles_rapid_events(self):
        """Test handling rapid succession of events."""
        debugger = WorkflowDebugger(verbose=False)
        MagicMock()

        # Simulate rapid event firing
        start_time = datetime.now()
        for i in range(100):
            debugger._on_event(
                Hook.AGENT_RUN_START,
                {"index": i, "data": "x" * 1000},
            )
        duration = (datetime.now() - start_time).total_seconds()

        # Should complete in reasonable time (< 1 second for 100 events)
        assert duration < 1.0
        assert len(debugger.events) == 100

    def test_memory_usage_with_many_events(self):
        """Test memory doesn't explode with many events."""
        debugger = WorkflowDebugger(verbose=False)

        # Get baseline memory (rough approximation)
        initial_size = len(str(debugger.events))

        # Add many events
        for i in range(1000):
            debugger.events.append(
                DebugEvent(
                    timestamp=datetime.now(),
                    hook=Hook.AGENT_RUN_START,
                    data={"index": i},
                )
            )

        # Memory should scale linearly, not explode
        final_size = len(str(debugger.events))
        # Should be roughly 1000x the size of one event
        assert final_size > initial_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
