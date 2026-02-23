"""Rate limit configuration and stats."""

from dataclasses import dataclass, field
from typing import Any

from syrin.enums import ThresholdMetric
from syrin.threshold import RateLimitThreshold, ThresholdContext


@dataclass
class RateLimitStats:
    """Statistics about rate limit usage."""

    rpm_used: int = 0
    rpm_limit: int = 0
    tpm_used: int = 0
    tpm_limit: int = 0
    rpd_used: int = 0
    rpd_limit: int = 0
    thresholds_triggered: list[str] = field(default_factory=list)


@dataclass
class APIRateLimit:
    """API rate limit configuration for provider calls.

    Provides proactive rate limit management with threshold actions.

    Args:
        rpm: Requests per minute limit
        tpm: Tokens per minute limit
        rpd: Requests per day limit
        thresholds: List of RateLimitThreshold
        wait_backoff: Seconds to wait when WAIT action triggers (default 1.0)
        auto_switch: Auto-switch model on exceeded (default True)

    Example:
        >>> from syrin import Agent, Model
        >>> from syrin.ratelimit import APIRateLimit
        >>> from syrin.threshold import RateLimitThreshold
        >>> from syrin.enums import ThresholdMetric
        >>>
        >>> agent = Agent(
        ...     model=Model("openai/gpt-4o"),
        ...     rate_limit=APIRateLimit(
        ...         rpm=500,
        ...         thresholds=[
        ...             RateLimitThreshold(at=80, action=lambda ctx: print(f"RPM at {ctx.percentage}%"), metric=ThresholdMetric.RPM),
        ...         ]
        ...     )
        ... )
    """

    rpm: int | None = None
    tpm: int | None = None
    rpd: int | None = None
    thresholds: list[RateLimitThreshold] = field(default_factory=list)
    wait_backoff: float = 1.0
    auto_switch: bool = True
    auto_detect: bool = False
    retry_on_429: bool = True
    max_retries: int = 3
    _model_id: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.rpm is None and self.tpm is None and self.rpd is None and not self.auto_detect:
            raise ValueError(
                "At least one of rpm, tpm, or rpd must be set, or set auto_detect=True"
            )
        self._validate_thresholds()

    def _validate_thresholds(self) -> None:
        """Validate that only RateLimitThreshold is used."""
        for th in self.thresholds:
            if not isinstance(th, RateLimitThreshold):
                raise ValueError(
                    f"APIRateLimit thresholds only accept RateLimitThreshold, got {type(th).__name__}"
                )

    @classmethod
    def auto_detect_for_model(cls, model_id: str) -> "APIRateLimit":
        """Create an APIRateLimit with auto-detected limits for the given model."""
        from syrin.ratelimit.providers import auto_detect_limits

        limits = auto_detect_limits(model_id)
        return cls(
            rpm=limits.get("rpm"),
            tpm=limits.get("tpm"),
            rpd=limits.get("rpd"),
            auto_detect=True,
            _model_id=model_id,
        )

    def get_thresholds_for_metric(self, metric: ThresholdMetric | str) -> list[RateLimitThreshold]:
        """Get thresholds for a specific metric."""
        metric_str = metric.value if hasattr(metric, "value") else str(metric)
        return [t for t in self.thresholds if str(t.metric.value) == metric_str]

    def check_thresholds(
        self, metric: ThresholdMetric, current: int, parent: Any = None
    ) -> list[RateLimitThreshold]:
        """Check thresholds for a specific metric."""
        limit = getattr(self, metric.value, None)
        if limit is None or limit <= 0:
            return []

        percentage = int((current / limit) * 100)
        triggered = []

        for threshold in self.thresholds:
            if threshold.should_trigger(percentage, metric):
                ctx = ThresholdContext(
                    percentage=percentage,
                    metric=metric,
                    current_value=float(current),
                    limit_value=float(limit),
                    parent=parent,
                )
                threshold.execute(ctx)
                triggered.append(threshold)

        return triggered


__all__ = [
    "APIRateLimit",
    "RateLimitStats",
    "RateLimitThreshold",
]
