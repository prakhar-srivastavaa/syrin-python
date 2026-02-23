"""Budget models, threshold actions, and budget tracker."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from syrin.cost import ModelPricing, Pricing
from syrin.enums import OnExceeded, ThresholdMetric
from syrin.threshold import BudgetThreshold, Threshold, ThresholdContext
from syrin.types import CostInfo

__all__ = [
    "Budget",
    "BudgetStatus",
    "BudgetSummary",
    "CostEntry",
    "ModelPricing",
    "OnExceeded",
    "Pricing",
    "RateLimit",
    "Threshold",
    "BudgetThreshold",
]


class BudgetStatus(str, Enum):
    """Result of a budget check."""

    OK = "ok"
    THRESHOLD = "threshold"
    EXCEEDED = "exceeded"


class RateLimit(BaseModel):
    """Rate-based cost limits in USD."""

    hour: float | None = Field(default=None, ge=0, description="Max USD per hour")
    day: float | None = Field(default=None, ge=0, description="Max USD per day")
    week: float | None = Field(default=None, ge=0, description="Max USD per week")
    month: float | None = Field(default=None, ge=0, description="Max USD per month")


class Budget(BaseModel):
    """Budget configuration: run limit, rate limits, on_exceeded behavior, thresholds, sharing.

    Args:
        run: Max cost per run in USD
        per: Rate limits (hourly, daily, weekly, monthly)
        on_exceeded: What to do when budget exceeded (ERROR, WARN, etc.)
        thresholds: List of BudgetThreshold (only BudgetThreshold allowed)
        shared: Whether budget is shared with child agents

    Example:
        >>> from syrin import Budget
        >>> from syrin.threshold import BudgetThreshold
        >>>
        >>> budget = Budget(
        ...     run=10.0,
        ...     thresholds=[
        ...         BudgetThreshold(at=80, action=lambda ctx: print(f"At {ctx.percentage}%"))
        ...     ]
        ... )
    """

    model_config = {"str_strip_whitespace": True, "arbitrary_types_allowed": True}

    run: float | None = Field(default=None, ge=0, description="Max cost per run (USD)")
    per: RateLimit | None = Field(default=None, description="Rate limits")
    on_exceeded: OnExceeded = Field(
        default=OnExceeded.ERROR, description="Behavior when budget exceeded"
    )
    thresholds: list[Any] = Field(
        default_factory=list,
        description="Ordered list of threshold actions (e.g. at 80% switch model)",
    )
    shared: bool = Field(
        default=False,
        description="If True, this budget is shared with child agents (borrow mechanism)",
    )
    _parent_budget: Budget | None = PrivateAttr(default=None)
    _spent: float = PrivateAttr(default=0.0)

    @model_validator(mode="after")
    def _validate_thresholds(self) -> Budget:
        """Validate that only BudgetThreshold is used and metrics are valid."""
        for th in self.thresholds:
            if not isinstance(th, BudgetThreshold):
                raise TypeError(f"Budget only accepts BudgetThreshold, got {type(th).__name__}")
        valid_metrics = {ThresholdMetric.COST, ThresholdMetric.TOKENS}
        for th in self.thresholds:
            metric_val = th.metric.value if hasattr(th.metric, "value") else th.metric
            if metric_val not in valid_metrics:
                raise ValueError(
                    f"Budget thresholds only support {valid_metrics}, got {metric_val}"
                )
        return self

    @property
    def remaining(self) -> float | None:
        """Get remaining budget. Returns None if run limit not set."""
        if self.run is None:
            return None
        return self.run - self._spent

    def _set_spent(self, amount: float) -> None:
        """Internal method to track spent amount. Called by BudgetTracker."""
        self._spent = amount

    def __str__(self) -> str:
        if self.run is not None:
            remaining = self.remaining if self.remaining is not None else self.run
            return f"Budget(${self.run:.2f}, remaining=${remaining:.2f}, shared={self.shared})"
        return f"Budget(unlimited, shared={self.shared})"


@dataclass
class CostEntry:
    """Timestamped cost entry for rolling windows (wall-clock for persistence)."""

    cost_usd: float
    timestamp: float = field(default_factory=time.time)
    model_name: str = ""


@dataclass
class BudgetSummary:
    """Current budget state summary."""

    current_run_cost: float
    hourly_cost: float
    daily_cost: float
    weekly_cost: float
    monthly_cost: float
    entries_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_run_cost": self.current_run_cost,
            "hourly_cost": self.hourly_cost,
            "daily_cost": self.daily_cost,
            "weekly_cost": self.weekly_cost,
            "monthly_cost": self.monthly_cost,
            "entries_count": self.entries_count,
        }


# Rolling window lengths in seconds (wall-clock for persistence)
_SEC_PER_HOUR = 3600.0
_SEC_PER_DAY = 86400.0
_SEC_PER_WEEK = 604800.0
_SEC_PER_MONTH = 2592000.0  # 30 days


class BudgetTracker:
    """
    Tracks cost per run and over rolling windows (hour, day, week, month).
    Uses wall-clock time so state can be persisted across restarts.
    """

    def __init__(self) -> None:
        self._cost_history: list[CostEntry] = []
        self._run_start: float = time.time()

    def record(self, cost: CostInfo) -> None:
        """Add a cost entry and update all windows. Prunes entries older than 30 days."""
        self._cost_history.append(
            CostEntry(
                cost_usd=cost.cost_usd,
                timestamp=time.time(),
                model_name=cost.model_name or "",
            )
        )
        self._prune_old()

    def _prune_old(self) -> None:
        """Remove entries older than 30 days."""
        now = time.time()
        cutoff = now - _SEC_PER_MONTH
        self._cost_history = [e for e in self._cost_history if e.timestamp >= cutoff]

    def get_state(self) -> dict[str, Any]:
        """Serialize state for persistence (cost_history, run_start)."""
        return {
            "cost_history": [
                {"cost_usd": e.cost_usd, "timestamp": e.timestamp, "model_name": e.model_name}
                for e in self._cost_history
            ],
            "run_start": self._run_start,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore from persisted state."""
        self._cost_history = [
            CostEntry(
                cost_usd=item["cost_usd"],
                timestamp=item["timestamp"],
                model_name=item.get("model_name", ""),
            )
            for item in state.get("cost_history", [])
        ]
        self._run_start = state.get("run_start", time.time())

    @property
    def current_run_cost(self) -> float:
        """Cost since last reset_run()."""
        run_start = self._run_start
        return sum(e.cost_usd for e in self._cost_history if e.timestamp >= run_start)

    def _window_cost(self, window_sec: float) -> float:
        now = time.time()
        cutoff = now - window_sec
        return sum(e.cost_usd for e in self._cost_history if e.timestamp >= cutoff)

    @property
    def hourly_cost(self) -> float:
        return self._window_cost(_SEC_PER_HOUR)

    @property
    def daily_cost(self) -> float:
        return self._window_cost(_SEC_PER_DAY)

    @property
    def weekly_cost(self) -> float:
        return self._window_cost(_SEC_PER_WEEK)

    @property
    def monthly_cost(self) -> float:
        return self._window_cost(_SEC_PER_MONTH)

    def reset_run(self) -> None:
        """Reset per-run counter (next record starts a new run)."""
        self._run_start = time.time()

    def check_budget(self, budget: Budget) -> BudgetStatus:
        """
        Returns OK, THRESHOLD, or EXCEEDED. EXCEEDED if any limit is over.
        THRESHOLD if a threshold is crossed but limits not exceeded.
        """
        if budget.run is not None and self.current_run_cost >= budget.run:
            return BudgetStatus.EXCEEDED
        per = budget.per
        if per is not None:
            if per.hour is not None and self.hourly_cost >= per.hour:
                return BudgetStatus.EXCEEDED
            if per.day is not None and self.daily_cost >= per.day:
                return BudgetStatus.EXCEEDED
            if per.week is not None and self.weekly_cost >= per.week:
                return BudgetStatus.EXCEEDED
            if per.month is not None and self.monthly_cost >= per.month:
                return BudgetStatus.EXCEEDED
        triggered = self.check_thresholds(budget)
        if triggered:
            return BudgetStatus.THRESHOLD
        return BudgetStatus.OK

    def check_thresholds(self, budget: Budget, parent: Any = None) -> list[BudgetThreshold]:
        """Check and execute thresholds that are currently crossed.

        Args:
            budget: The budget configuration
            parent: Reference to parent object (e.g., Agent) for context

        Returns:
            List of thresholds that were triggered
        """
        if not budget.run or budget.run <= 0:
            return []
        run_cost = self.current_run_cost
        limit = budget.run
        pct = int((run_cost / limit) * 100) if limit else 0
        triggered = []
        for th in budget.thresholds:
            if pct >= th.at:
                ctx = ThresholdContext(
                    percentage=pct,
                    metric=ThresholdMetric.COST,
                    current_value=run_cost,
                    limit_value=limit,
                    budget_run=limit,
                    parent=parent,
                )
                th.execute(ctx)
                triggered.append(th)
        return triggered

    def get_summary(self) -> BudgetSummary:
        return BudgetSummary(
            current_run_cost=self.current_run_cost,
            hourly_cost=self.hourly_cost,
            daily_cost=self.daily_cost,
            weekly_cost=self.weekly_cost,
            monthly_cost=self.monthly_cost,
            entries_count=len(self._cost_history),
        )
