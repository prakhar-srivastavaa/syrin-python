"""Command-line interface for Syrin with built-in observability.

This module provides CLI features for running and debugging Syrin agents:

1. **Auto-tracing with --trace flag:**
   When users run their scripts with `--trace`, observability is automatically enabled.

   Example:
       $ python my_agent.py --trace
       $ python -m syrin.run my_agent.py --trace

2. **WorkflowDebugger:**
   Built-in debugger class that captures all workflow events with pretty output.
   Users don't need to write their own hook handlers.

   Example:
       >>> from Syrin import Agent
       >>> from syrin.cli import WorkflowDebugger

       >>> debugger = WorkflowDebugger()
       >>> agent = Agent(...)
       >>> debugger.attach(agent)
       >>> result = agent.response("Hello")
       >>> debugger.print_summary()

 3. **CLI Commands:**
   - `syrin run <script.py>` - Run a script with optional tracing
   - `syrin trace <script.py>` - Run with tracing enabled
   - `syrin doctor` - Check installation and configuration
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import traceback
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from syrin.enums import Hook
from syrin.events import Events
from syrin.observability import (
    ConsoleExporter,
    SpanKind,
    SpanStatus,
    get_tracer,
    trace,
)


def _color(code: str) -> str:
    """Get ANSI color code if terminal supports it."""
    if sys.stdout.isatty():
        return code
    return ""


class Colors:
    """ANSI color codes for terminal output."""

    RESET = _color("\033[0m")
    BOLD = _color("\033[1m")
    DIM = _color("\033[2m")
    GREEN = _color("\033[92m")
    BLUE = _color("\033[94m")
    YELLOW = _color("\033[93m")
    MAGENTA = _color("\033[95m")
    CYAN = _color("\033[96m")
    RED = _color("\033[91m")
    WHITE = _color("\033[97m")


@dataclass
class DebugEvent:
    """A single debug event captured from the system."""

    timestamp: datetime
    hook: Hook
    data: dict[str, Any]
    duration_ms: float | None = None


class WorkflowDebugger:
    """Built-in workflow debugger for Syrin agents and pipelines.

    This class provides easy-to-use debugging capabilities without requiring
    users to write their own hook handlers. Simply attach it to an agent
    or pipeline and it will capture all events.

    Features:
    - Color-coded terminal output
    - Event timing and duration tracking
    - Cost aggregation
    - Summary statistics
    - Export to JSONL for further analysis

    Example - Basic usage with Agent:
        >>> from Syrin import Agent
        >>> from syrin.cli import WorkflowDebugger

        >>> debugger = WorkflowDebugger(verbose=True)
        >>> agent = Agent(...)
        >>> debugger.attach(agent)

        >>> result = agent.response("Research quantum computing")
        >>> debugger.print_summary()

    Example - With DynamicPipeline:
        >>> from syrin.agent.multi_agent import DynamicPipeline
        >>>
        >>> debugger = WorkflowDebugger(verbose=True)
        >>> pipeline = DynamicPipeline(agents=[...])
        >>> debugger.attach(pipeline)
        >>>
        >>> result = pipeline.run("Complex task")
        >>> debugger.print_summary()

    Example - Silent mode (capture only, no output):
        >>> debugger = WorkflowDebugger(verbose=False)
        >>> debugger.attach(agent)
        >>> result = agent.response("Hello")
        >>>
        >>> # Analyze events programmatically
        >>> for event in debugger.events:
        ...     if event.hook == Hook.TOOL_CALL_START:
        ...         print(f"Tool called: {event.data.get('tool_name')}")

    Example - Export to file:
        >>> debugger = WorkflowDebugger()
        >>> debugger.attach(agent)
        >>> result = agent.response("Hello")
        >>> debugger.export_jsonl("/tmp/debug_trace.jsonl")
    """

    def __init__(self, verbose: bool = True, colors: bool = True):
        """Initialize the workflow debugger.

        Args:
            verbose: Whether to print events to terminal as they occur
            colors: Whether to use ANSI colors in terminal output
        """
        self.verbose = verbose
        self.colors = colors
        self.events: list[DebugEvent] = []
        self._attached_objects: list[Any] = []
        self._start_time: datetime | None = None

    def attach(self, obj: Any) -> WorkflowDebugger:
        """Attach debugger to an agent, pipeline, or any object with events.

        Args:
            obj: Object with an `events` attribute (Agent, DynamicPipeline, etc.)

        Returns:
            Self for method chaining

        Example:
            >>> debugger.attach(agent).attach(pipeline)
        """
        if not hasattr(obj, "events"):
            raise ValueError(f"Object {type(obj).__name__} has no 'events' attribute")

        events: Events = obj.events

        # Attach to all available hooks dynamically
        def make_handler(h: Hook) -> Any:
            return lambda ctx: self._on_event(h, ctx)

        for hook in Hook:
            with suppress(Exception):
                events.on(hook, make_handler(hook))

        self._attached_objects.append(obj)

        if self.verbose:
            print(f"{Colors.CYAN}▶ Debugger attached to {type(obj).__name__}{Colors.RESET}")

        return self

    def _on_event(self, hook: Hook, ctx: dict[str, Any]) -> None:
        """Handle an event from the system."""
        now = datetime.now()

        # Calculate duration if this is an end event
        duration_ms = None
        if hasattr(ctx, "get"):
            duration_ms = ctx.get("duration")
            if duration_ms is not None:
                duration_ms = float(duration_ms) * 1000  # Convert to ms

        event = DebugEvent(
            timestamp=now,
            hook=hook,
            data=dict(ctx),
            duration_ms=duration_ms,
        )
        self.events.append(event)

        if self.verbose:
            self._print_event(event)

    def _print_event(self, event: DebugEvent) -> None:
        """Pretty print a single event."""
        if not self.colors:
            c = Colors
            for attr in dir(c):
                if not attr.startswith("_"):
                    setattr(c, attr, "")
        else:
            c = Colors

        timestamp = event.timestamp.strftime("%H:%M:%S.%f")[:-3]
        hook_name = event.hook.value

        # Choose color and symbol based on hook type
        if "start" in hook_name or "init" in hook_name:
            color = c.GREEN
            symbol = "▶"
        elif "end" in hook_name or "complete" in hook_name:
            color = c.BLUE
            symbol = "✓"
        elif "spawn" in hook_name or "handoff" in hook_name:
            color = c.CYAN
            symbol = "→"
        elif "error" in hook_name or "exceeded" in hook_name:
            color = c.RED
            symbol = "✗"
        elif "plan" in hook_name or "check" in hook_name:
            color = c.MAGENTA
            symbol = "◉"
        elif "tool" in hook_name:
            color = c.YELLOW
            symbol = "🔧"
        elif "llm" in hook_name:
            color = c.WHITE
            symbol = "💬"
        else:
            color = c.DIM
            symbol = "•"

        print(f"{color}{symbol} {timestamp} {hook_name}{c.RESET}")

        # Print relevant context data
        data = event.data
        indent = "     "

        if "task" in data:
            task = data["task"]
            if isinstance(task, str) and len(task) > 60:
                task = task[:57] + "..."
            print(f"{indent}Task: {task}")

        if "agent_type" in data or "agent_name" in data:
            name = data.get("agent_type") or data.get("agent_name")
            print(f"{indent}Agent: {name}")

        if "tool_name" in data:
            print(f"{indent}Tool: {data['tool_name']}")

        if "model" in data:
            print(f"{indent}Model: {data['model']}")

        if "plan_count" in data:
            print(f"{indent}Plan: {data['plan_count']} agents")

        if "mode" in data:
            print(f"{indent}Mode: {data['mode']}")

        if "cost" in data and data["cost"] is not None:
            cost_val = float(data["cost"])
            # Format as decimal, never scientific notation
            if cost_val > 0 and cost_val < 0.0001:
                print(f"{indent}Cost: ${cost_val:.8f}")
            elif cost_val > 0:
                print(f"{indent}Cost: ${cost_val:.6f}")
            else:
                print(f"{indent}Cost: $0.000000")

        if "total_cost" in data and data["total_cost"] is not None:
            total_val = float(data["total_cost"])
            # Format as decimal, never scientific notation
            if total_val > 0 and total_val < 0.0001:
                print(f"{indent}Total: ${total_val:.8f}")
            elif total_val > 0:
                print(f"{indent}Total: ${total_val:.6f}")
            else:
                print(f"{indent}Total: $0.000000")

        if event.duration_ms is not None:
            print(f"{indent}Duration: {event.duration_ms:.2f}ms")
        elif "duration" in data and data["duration"] is not None:
            print(f"{indent}Duration: {float(data['duration']) * 1000:.2f}ms")

        if "error" in data:
            print(f"{indent}{c.RED}Error: {data['error']}{c.RESET}")

        print()

    def print_summary(self) -> None:
        """Print a summary of all captured events."""
        if not self.colors:
            c = Colors
            for attr in dir(c):
                if not attr.startswith("_"):
                    setattr(c, attr, "")
        else:
            c = Colors

        print(f"\n{c.BOLD}{'=' * 70}{c.RESET}")
        print(f"{c.BOLD}WORKFLOW EXECUTION SUMMARY{c.RESET}")
        print(f"{c.BOLD}{'=' * 70}{c.RESET}")

        if not self.events:
            print(f"\n{c.YELLOW}No events captured{c.RESET}")
            return

        # Count events by hook
        event_counts: dict[str, int] = {}
        for event in self.events:
            hook = event.hook.value
            event_counts[hook] = event_counts.get(hook, 0) + 1

        print(f"\n{c.CYAN}Events captured:{c.RESET}")
        for hook, count in sorted(event_counts.items()):
            print(f"  {hook}: {count}")
        print()  # Add blank line after event list

        # Calculate totals
        total_cost = sum(
            float(e.data.get("cost", 0) or 0)
            for e in self.events
            if "cost" in e.data and e.data.get("cost") is not None
        )

        agent_starts = [e for e in self.events if e.hook == Hook.AGENT_RUN_START]
        agent_ends = [e for e in self.events if e.hook == Hook.AGENT_RUN_END]
        tool_calls = [e for e in self.events if "tool.call" in e.hook.value]
        llm_calls = [e for e in self.events if "llm.request" in e.hook.value]
        errors = [e for e in self.events if ".error" in e.hook.value]

        print(f"\n{c.CYAN}Statistics:{c.RESET}")
        print(f"  Agent runs: {len(agent_starts)} started, {len(agent_ends)} completed")
        print(f"  Tool calls: {len(tool_calls)}")
        print(f"  LLM calls: {len(llm_calls)}")
        print(f"  Errors: {len(errors)}")
        # Format as decimal, never scientific notation
        if total_cost > 0 and total_cost < 0.0001:
            print(f"  Total cost: ${total_cost:.8f}")
        elif total_cost > 0:
            print(f"  Total cost: ${total_cost:.6f}")
        else:
            print("  Total cost: $0.000000")
        print(f"  Total events: {len(self.events)}")

        if errors:
            print(f"\n{c.RED}Errors encountered:{c.RESET}")
            for event in errors:
                print(f"  - {event.hook.value}: {event.data.get('error', 'Unknown')}")

        print()

    def export_jsonl(self, filepath: str) -> None:
        """Export all events to a JSONL file for analysis.

        Args:
            filepath: Path to output JSONL file
        """
        import json

        with open(filepath, "w") as f:
            for event in self.events:
                record = {
                    "timestamp": event.timestamp.isoformat(),
                    "hook": event.hook.value,
                    "data": event.data,
                    "duration_ms": event.duration_ms,
                }
                f.write(json.dumps(record, default=str) + "\n")

        if self.verbose:
            print(f"{Colors.GREEN}✓ Exported {len(self.events)} events to {filepath}{Colors.RESET}")

    def clear(self) -> None:
        """Clear all captured events."""
        self.events.clear()


@contextmanager
def auto_trace() -> Iterator[None]:
    """Context manager that automatically enables tracing.

    This is used internally when --trace flag is detected.
    It adds a ConsoleExporter to the global tracer.

    Example:
        >>> with auto_trace():
        ...     result = agent.response("Hello")
        ...     # All operations are traced to console
    """
    tracer = get_tracer()
    exporter = ConsoleExporter(verbose=True)
    tracer.add_exporter(exporter)
    tracer.set_debug_mode(True)

    try:
        yield
    finally:
        # Note: We don't remove the exporter to keep traces from being lost
        pass


def check_for_trace_flag() -> bool:
    """Check if --trace flag is in sys.argv.

    Returns:
        True if --trace flag is present
    """
    return "--trace" in sys.argv


def remove_trace_flag() -> None:
    """Remove --trace flag from sys.argv so it doesn't break user scripts."""
    while "--trace" in sys.argv:
        sys.argv.remove("--trace")


def run_with_observability(script_path: str) -> int:
    """Run a Python script with observability enabled.

    Args:
        script_path: Path to Python script to run

    Returns:
        Exit code from script execution
    """
    print(f"{Colors.CYAN}▶ Running {script_path} with observability enabled{Colors.RESET}\n")

    # Enable tracing
    tracer = get_tracer()
    tracer.add_exporter(ConsoleExporter(colors=True, verbose=True))
    tracer.set_debug_mode(True)

    # Load and run the script
    script_file = Path(script_path).resolve()
    if not script_file.exists():
        print(f"{Colors.RED}✗ Script not found: {script_file}{Colors.RESET}")
        return 1

    # Add script directory to path
    sys.path.insert(0, str(script_file.parent))

    # Load the module
    spec = importlib.util.spec_from_file_location("__main__", script_file)
    if spec is None or spec.loader is None:
        print(f"{Colors.RED}✗ Failed to load script: {script_file}{Colors.RESET}")
        return 1

    module = importlib.util.module_from_spec(spec)

    try:
        with trace.span(
            "script.run",
            kind=SpanKind.INTERNAL,
            attributes={
                "script.path": str(script_file),
                "script.name": script_file.name,
            },
        ) as span:
            spec.loader.exec_module(module)
            span.set_status(SpanStatus.OK)
    except Exception:
        print(f"\n{Colors.RED}✗ Script failed with error:{Colors.RESET}")
        traceback.print_exc()
        return 1

    print(f"\n{Colors.GREEN}✓ Script completed successfully{Colors.RESET}")
    return 0


def doctor_command() -> int:
    """Run diagnostics on Syrin installation.

    Returns:
        Exit code (0 for healthy, 1 for issues)
    """
    print(f"{Colors.BOLD}Syrin Doctor{Colors.RESET}\n")

    checks = []

    # Check Python version
    import sys as _sys

    py_version = _sys.version_info
    checks.append(
        (
            "Python version",
            py_version >= (3, 10),
            f"{py_version.major}.{py_version.minor}.{py_version.micro}",
            "Python 3.10+ required",
        )
    )

    # Check Syrin installation
    try:
        import syrin

        checks.append(("Syrin import", True, f"v{getattr(syrin, '__version__', 'unknown')}", ""))
    except ImportError as e:
        checks.append(("Syrin import", False, str(e), "Run: pip install syrin"))

    # Check key dependencies
    deps = ["pydantic", "typing_extensions"]
    for dep in deps:
        try:
            __import__(dep)
            checks.append((f"{dep} installed", True, "✓", ""))
        except ImportError:
            checks.append((f"{dep} installed", False, "✗", f"pip install {dep}"))

    # Check optional providers
    providers = ["openai", "anthropic", "groq"]
    for provider in providers:
        try:
            __import__(provider)
            checks.append((f"{provider} SDK", True, "✓", ""))
        except ImportError:
            checks.append((f"{provider} SDK", False, "-", f"pip install {provider}"))

    # Print results
    all_ok = True
    for name, ok, value, fix in checks:
        status = f"{Colors.GREEN}✓{Colors.RESET}" if ok else f"{Colors.RED}✗{Colors.RESET}"
        print(f"{status} {name:25} {value}")
        if not ok and fix:
            print(f"  {Colors.YELLOW}→ {fix}{Colors.RESET}")
            all_ok = False

    print()
    if all_ok:
        print(f"{Colors.GREEN}All checks passed!{Colors.RESET}")
        return 0
    else:
        print(f"{Colors.YELLOW}Some checks failed. See above for fixes.{Colors.RESET}")
        return 1


def main() -> int:
    """Main CLI entry point.

    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        prog="syrin",
        description="Syrin CLI - Run and debug AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  syrin doctor                    Check installation health
  syrin run my_agent.py           Run a script
  syrin run my_agent.py --trace   Run with observability enabled
  syrin trace my_agent.py         Shortcut for run --trace

Auto-tracing in user scripts:
  When users run their scripts with --trace flag, observability
  is automatically enabled:
    $ python my_agent.py --trace
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Doctor command
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check installation and configuration",
    )
    doctor_parser.set_defaults(func=doctor_command)

    # Run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run a Python script with optional tracing",
    )
    run_parser.add_argument("script", help="Python script to run")
    run_parser.add_argument(
        "--trace",
        action="store_true",
        help="Enable observability tracing",
    )
    run_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Additional arguments to pass to script",
    )

    # Trace command (shortcut for run --trace)
    trace_parser = subparsers.add_parser(
        "trace",
        help="Run a script with observability enabled",
    )
    trace_parser.add_argument("script", help="Python script to run")
    trace_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Additional arguments to pass to script",
    )
    trace_parser.set_defaults(trace=True)

    # Parse args
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "doctor":
        return doctor_command()

    if args.command in ("run", "trace"):
        # Pass remaining args to script
        if args.args:
            # Remove the '--' separator if present
            if args.args[0] == "--":
                args.args = args.args[1:]
            sys.argv = [args.script] + args.args
        else:
            sys.argv = [args.script]

        if getattr(args, "trace", False):
            return run_with_observability(args.script)
        else:
            # Run without observability
            script_path = Path(args.script).resolve()
            if not script_path.exists():
                print(f"{Colors.RED}✗ Script not found: {script_path}{Colors.RESET}")
                return 1

            sys.path.insert(0, str(script_path.parent))
            spec = importlib.util.spec_from_file_location("__main__", script_path)
            if spec is None or spec.loader is None:
                print(f"{Colors.RED}✗ Failed to load script{Colors.RESET}")
                return 1

            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
                return 0
            except Exception:
                traceback.print_exc()
                return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
