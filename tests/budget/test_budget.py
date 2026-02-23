"""Tests for budget models and BudgetTracker (budget.py)."""

from __future__ import annotations

import pytest

from syrin.budget import (
    Budget,
    BudgetStatus,
    BudgetThreshold,
    BudgetTracker,
    RateLimit,
    Threshold,
)
from syrin.enums import OnExceeded, ThresholdMetric
from syrin.types import CostInfo, TokenUsage


def test_rate_limit_model() -> None:
    r = RateLimit(hour=10.0, day=100.0)
    assert r.hour == 10.0
    assert r.day == 100.0
    assert r.week is None


def test_budget_model() -> None:
    b = Budget(run=5.0, on_exceeded="warn", thresholds=[])
    assert b.run == 5.0
    assert b.on_exceeded == "warn"
    assert b.thresholds == []


def test_threshold_with_action() -> None:
    """Threshold now takes a function as action."""
    t = Threshold(
        at=80, action=lambda ctx: print(f"Budget at {ctx.percentage}%"), metric=ThresholdMetric.COST
    )
    assert t.at == 80
    assert callable(t.action)


def test_threshold_with_lambda() -> None:
    """Test threshold with lambda action."""
    t = Threshold(
        at=80, action=lambda ctx: print(f"At {ctx.percentage}%"), metric=ThresholdMetric.COST
    )
    assert t.at == 80


def test_threshold_with_function() -> None:
    """Test threshold with function action."""

    def my_action(ctx):
        pass

    t = Threshold(at=90, action=my_action, metric=ThresholdMetric.COST)
    assert t.at == 90


def test_budget_tracker_record_and_run_cost() -> None:
    tracker = BudgetTracker()
    tracker.record(CostInfo(cost_usd=0.5, token_usage=TokenUsage()))
    tracker.record(CostInfo(cost_usd=0.3, token_usage=TokenUsage()))
    assert tracker.current_run_cost == 0.8
    tracker.reset_run()
    assert tracker.current_run_cost == 0.0


def test_budget_tracker_check_budget_ok() -> None:
    tracker = BudgetTracker()
    tracker.record(CostInfo(cost_usd=0.1, token_usage=TokenUsage()))
    budget = Budget(run=5.0)
    assert tracker.check_budget(budget) == BudgetStatus.OK


def test_budget_tracker_check_budget_exceeded() -> None:
    tracker = BudgetTracker()
    tracker.record(CostInfo(cost_usd=10.0, token_usage=TokenUsage()))
    budget = Budget(run=5.0)
    assert tracker.check_budget(budget) == BudgetStatus.EXCEEDED


def test_budget_tracker_check_thresholds() -> None:
    triggered_actions = []

    def capture_action(ctx):
        triggered_actions.append(ctx.percentage)

    tracker = BudgetTracker()
    tracker.record(CostInfo(cost_usd=4.0, token_usage=TokenUsage()))
    budget = Budget(
        run=5.0, thresholds=[Threshold(at=80, action=capture_action, metric=ThresholdMetric.COST)]
    )
    triggered = tracker.check_thresholds(budget)
    assert len(triggered) == 1
    assert triggered[0].at == 80
    assert 80 in triggered_actions  # Action was executed


def test_budget_tracker_get_summary() -> None:
    tracker = BudgetTracker()
    tracker.record(CostInfo(cost_usd=1.0, token_usage=TokenUsage()))
    s = tracker.get_summary()
    assert s.current_run_cost == 1.0
    assert s.entries_count == 1
    d = s.to_dict()
    assert "current_run_cost" in d
    assert "hourly_cost" in d


def test_budget_tracker_rolling_window() -> None:
    tracker = BudgetTracker()
    # All entries are "now" in monotonic time, so hourly/daily include them
    tracker.record(CostInfo(cost_usd=1.0, token_usage=TokenUsage()))
    assert tracker.hourly_cost == 1.0
    assert tracker.daily_cost == 1.0


# =============================================================================
# AGGRESSIVE EDGE CASES - TRY TO BREAK BUDGET
# =============================================================================


def test_rate_limit_negative_values_should_fail() -> None:
    """Negative rate limits should raise validation error."""
    with pytest.raises((ValueError, TypeError)):
        RateLimit(hour=-1.0)


def test_rate_limit_zero_values_allowed() -> None:
    """Zero rate limits should be valid (allow no spending)."""
    r = RateLimit(hour=0.0, day=0.0)
    assert r.hour == 0.0
    assert r.day == 0.0


def test_rate_limit_very_high_values() -> None:
    """Very high rate limits should be allowed."""
    r = RateLimit(hour=1_000_000.0, day=10_000_000.0)
    assert r.hour == 1_000_000.0


def test_budget_with_very_many_thresholds() -> None:
    """Many threshold actions should work."""
    thresholds = [
        Threshold(at=i, action=lambda _: None, metric=ThresholdMetric.COST)
        for i in range(0, 100, 5)
    ]
    b = Budget(run=10.0, thresholds=thresholds)
    assert len(b.thresholds) == 20


def test_budget_threshold_edge_cases() -> None:
    """Threshold at exact boundaries."""
    t0 = Threshold(at=0, action=lambda _: None, metric=ThresholdMetric.COST)
    assert t0.at == 0

    t100 = Threshold(at=100, action=lambda _: None, metric=ThresholdMetric.COST)
    assert t100.at == 100

    with pytest.raises(ValueError):
        Threshold(at=-1, action=lambda _: None, metric=ThresholdMetric.COST)

    with pytest.raises(ValueError):
        Threshold(at=101, action=lambda _: None, metric=ThresholdMetric.COST)


def test_budget_tracker_very_many_entries() -> None:
    """Many budget entries should be tracked efficiently."""
    tracker = BudgetTracker()
    for _i in range(10000):
        tracker.record(CostInfo(cost_usd=0.001, token_usage=TokenUsage()))

    assert tracker.current_run_cost == 10.0
    assert tracker.get_summary().entries_count == 10000


def test_budget_tracker_check_budget_at_exact_limit() -> None:
    """Test budget check at exact limit boundary - uses >= so exact = EXCEEDED."""
    tracker = BudgetTracker()
    tracker.record(CostInfo(cost_usd=5.0, token_usage=TokenUsage()))
    budget = Budget(run=5.0)
    # At exact limit triggers EXCEEDED (uses >= not >)
    assert tracker.check_budget(budget) == BudgetStatus.EXCEEDED


def test_budget_tracker_check_budget_slightly_over() -> None:
    """Test budget check slightly over limit."""
    tracker = BudgetTracker()
    tracker.record(CostInfo(cost_usd=5.01, token_usage=TokenUsage()))
    budget = Budget(run=5.0)
    assert tracker.check_budget(budget) == BudgetStatus.EXCEEDED


def test_budget_with_no_thresholds() -> None:
    """Budget with empty thresholds list."""
    b = Budget(run=10.0, thresholds=[])
    assert b.thresholds == []


def test_budget_on_exceeded_various_valid_actions() -> None:
    """Test all valid on_exceeded actions."""
    b1 = Budget(run=1.0, on_exceeded=OnExceeded.ERROR)
    assert b1.on_exceeded == OnExceeded.ERROR

    b2 = Budget(run=1.0, on_exceeded=OnExceeded.WARN)
    assert b2.on_exceeded == OnExceeded.WARN

    b3 = Budget(run=1.0, on_exceeded=OnExceeded.STOP)
    assert b3.on_exceeded == OnExceeded.STOP


def test_budget_tracker_reset_run_multiple_times() -> None:
    """Multiple reset runs should work correctly."""
    tracker = BudgetTracker()
    tracker.record(CostInfo(cost_usd=5.0, token_usage=TokenUsage()))
    assert tracker.current_run_cost == 5.0

    tracker.reset_run()
    assert tracker.current_run_cost == 0.0

    tracker.record(CostInfo(cost_usd=3.0, token_usage=TokenUsage()))
    assert tracker.current_run_cost == 3.0

    tracker.reset_run()
    assert tracker.current_run_cost == 0.0

    tracker.record(CostInfo(cost_usd=2.0, token_usage=TokenUsage()))
    assert tracker.current_run_cost == 2.0


def test_budget_rejects_wrong_threshold_class() -> None:
    """Budget should reject ContextThreshold."""
    from syrin.threshold import ContextThreshold

    with pytest.raises(TypeError, match="Budget only accepts BudgetThreshold"):
        Budget(
            run=10.0,
            thresholds=[ContextThreshold(at=80, action=lambda _: None)],
        )


def test_budget_rejects_wrong_threshold_class_ratelimit() -> None:
    """Budget should reject RateLimitThreshold."""
    from syrin.threshold import RateLimitThreshold

    with pytest.raises(TypeError, match="Budget only accepts BudgetThreshold"):
        Budget(
            run=10.0,
            thresholds=[
                RateLimitThreshold(at=80, action=lambda _: None, metric=ThresholdMetric.RPM)
            ],
        )


def test_budget_rejects_invalid_metric() -> None:
    """Budget should reject thresholds with invalid metrics like RPM."""
    with pytest.raises(ValueError, match="Budget thresholds only support"):
        Budget(
            run=10.0,
            thresholds=[BudgetThreshold(at=80, action=lambda _: None, metric=ThresholdMetric.RPM)],
        )


def test_budget_threshold_accepts_cost_metric() -> None:
    """BudgetThreshold should accept COST metric."""
    threshold = BudgetThreshold(at=50, action=lambda _: None, metric=ThresholdMetric.COST)
    assert threshold.at == 50
    assert threshold.metric == ThresholdMetric.COST


def test_budget_threshold_accepts_tokens_metric() -> None:
    """BudgetThreshold should accept TOKENS metric."""
    threshold = BudgetThreshold(at=50, action=lambda _: None, metric=ThresholdMetric.TOKENS)
    assert threshold.at == 50
    assert threshold.metric == ThresholdMetric.TOKENS


def test_budget_threshold_default_metric_is_cost() -> None:
    """BudgetThreshold should default to COST metric."""
    threshold = BudgetThreshold(at=50, action=lambda _: None)
    assert threshold.metric == ThresholdMetric.COST


def test_budget_threshold_at_zero() -> None:
    """BudgetThreshold at 0%."""
    threshold = BudgetThreshold(at=0, action=lambda _: None)
    assert threshold.at == 0
    assert threshold.should_trigger(0) is True


def test_budget_threshold_at_100() -> None:
    """BudgetThreshold at 100%."""
    threshold = BudgetThreshold(at=100, action=lambda _: None)
    assert threshold.at == 100
    assert threshold.should_trigger(100) is True


def test_budget_threshold_invalid_at_negative() -> None:
    """BudgetThreshold should reject negative at value."""
    with pytest.raises(ValueError, match="Threshold 'at' must be between 0 and 100"):
        BudgetThreshold(at=-1, action=lambda _: None)


def test_budget_threshold_invalid_at_over_100() -> None:
    """BudgetThreshold should reject at > 100."""
    with pytest.raises(ValueError, match="Threshold 'at' must be between 0 and 100"):
        BudgetThreshold(at=101, action=lambda _: None)


def test_budget_threshold_requires_action() -> None:
    """BudgetThreshold should require action."""
    with pytest.raises(ValueError, match="Threshold 'action' is required"):
        BudgetThreshold(at=80, action=None)  # type: ignore
