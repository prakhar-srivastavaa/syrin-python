"""OpenTelemetry Protocol (OTLP) exporter for Syrin.

Exports spans to any OTLP-compatible backend (Jaeger, Tempo, Datadog, etc.)

Example:
    >>> from syrin.observability.otlp import OTLPExporter
    >>>
    >>> exporter = OTLPExporter(
    ...     endpoint="http://localhost:4318/v1/traces",
    ...     headers={"Authorization": "Bearer token"}
    ... )
    >>> trace.add_exporter(exporter)
"""

from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)

# Module-level declarations for optional dependencies
otel_trace: Any = None
OTelOTLPExporter: Any = None
TracerProvider: Any = None
BatchSpanProcessor: Any = None


def _try_import_opentelemetry() -> bool:
    """Try to import opentelemetry modules."""
    global otel_trace, OTelOTLPExporter, TracerProvider, BatchSpanProcessor
    try:
        from opentelemetry import trace as _otel_trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as _OTelOTLPExporter,
        )
        from opentelemetry.sdk.trace import TracerProvider as _TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor as _BatchSpanProcessor

        otel_trace = _otel_trace
        OTelOTLPExporter = _OTelOTLPExporter
        TracerProvider = _TracerProvider
        BatchSpanProcessor = _BatchSpanProcessor
        return True
    except ImportError:
        return False


OPENTELEMETRY_AVAILABLE = _try_import_opentelemetry()
