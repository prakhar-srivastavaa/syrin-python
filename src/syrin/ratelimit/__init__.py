"""Rate limiting with proactive threshold actions.

Example:
    >>> from Syrin import Agent, Model
    >>> from syrin.ratelimit import APIRateLimit
    >>> from syrin.threshold import RateLimitThreshold
    >>> from syrin.enums import ThresholdMetric
    >>>
    >>> # Simple usage - just set limits
    >>> agent = Agent(
    ...     model=Model("openai/gpt-4o"),
    ...     rate_limit=APIRateLimit(rpm=500, tpm=150000)
    ... )
    >>>
    >>> # With thresholds
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

from syrin.ratelimit._manager import (
    DefaultRateLimitManager,
    RateLimitManager,
    create_rate_limit_manager,
)
from syrin.ratelimit.backends import (
    MemoryRateLimitBackend,
    RedisRateLimitBackend,
    SQLiteRateLimitBackend,
    get_rate_limit_backend,
)
from syrin.ratelimit.config import APIRateLimit, RateLimitStats
from syrin.threshold import RateLimitThreshold, ThresholdContext

__all__ = [
    "APIRateLimit",
    "RateLimitStats",
    "RateLimitThreshold",
    "ThresholdContext",
    # Manager
    "RateLimitManager",
    "DefaultRateLimitManager",
    "create_rate_limit_manager",
    # Backends
    "get_rate_limit_backend",
    "MemoryRateLimitBackend",
    "SQLiteRateLimitBackend",
    "RedisRateLimitBackend",
]
