"""Tests for rate limit module with unified RateLimitThreshold."""

from unittest.mock import MagicMock

import pytest

from syrin.enums import ThresholdMetric
from syrin.ratelimit import (
    APIRateLimit,
    DefaultRateLimitManager,
    RateLimitStats,
    RateLimitThreshold,
    create_rate_limit_manager,
)
from syrin.threshold import ThresholdContext


class TestAPIRateLimit:
    """Tests for APIRateLimit configuration."""

    def test_creation_with_rpm_only(self):
        """Test creating APIRateLimit with just RPM."""
        limit = APIRateLimit(rpm=500)
        assert limit.rpm == 500
        assert limit.tpm is None
        assert limit.rpd is None

    def test_creation_with_all_limits(self):
        """Test creating APIRateLimit with all limits."""
        limit = APIRateLimit(rpm=500, tpm=150000, rpd=10000)
        assert limit.rpm == 500
        assert limit.tpm == 150000
        assert limit.rpd == 10000

    def test_creation_raises_if_no_limits(self):
        """Test that creating without limits raises ValueError."""
        with pytest.raises(ValueError, match="At least one of rpm, tpm, or rpd"):
            APIRateLimit()

    def test_get_thresholds_for_metric(self):
        """Test filtering thresholds by metric."""
        thresholds = [
            RateLimitThreshold(at=80, action=lambda _: None, metric=ThresholdMetric.RPM),
            RateLimitThreshold(at=90, action=lambda _: None, metric=ThresholdMetric.TPM),
            RateLimitThreshold(at=100, action=lambda _: None, metric=ThresholdMetric.RPM),
        ]
        limit = APIRateLimit(rpm=500, tpm=150000, thresholds=thresholds)

        rpm_thresholds = limit.get_thresholds_for_metric("rpm")
        assert len(rpm_thresholds) == 2

        tpm_thresholds = limit.get_thresholds_for_metric("tpm")
        assert len(tpm_thresholds) == 1


class TestRateLimitThreshold:
    """Tests for unified RateLimitThreshold with rate limits."""

    def test_creation_basic(self):
        """Test basic threshold creation."""
        threshold = RateLimitThreshold(
            at=80,
            action=lambda _: None,
            metric=ThresholdMetric.RPM,
        )
        assert threshold.at == 80
        assert threshold.metric == ThresholdMetric.RPM
        assert callable(threshold.action)

    def test_validation_at_out_of_range(self):
        """Test that at must be 0-100."""
        with pytest.raises(ValueError, match="between 0 and 100"):
            RateLimitThreshold(at=101, action=lambda _: None, metric=ThresholdMetric.RPM)
        with pytest.raises(ValueError, match="between 0 and 100"):
            RateLimitThreshold(at=-1, action=lambda _: None, metric=ThresholdMetric.RPM)

    def test_should_trigger(self):
        """Test should_trigger method."""
        threshold = RateLimitThreshold(at=80, action=lambda _: None, metric=ThresholdMetric.RPM)
        assert threshold.should_trigger(85, ThresholdMetric.RPM) is True
        assert threshold.should_trigger(75, ThresholdMetric.RPM) is False
        assert threshold.should_trigger(85, ThresholdMetric.TPM) is False  # Wrong metric


class TestThresholdContext:
    """Tests for ThresholdContext."""

    def test_creation(self):
        """Test context creation."""
        ctx = ThresholdContext(
            percentage=80,
            metric=ThresholdMetric.RPM,
            current_value=400.0,
            limit_value=500.0,
        )
        assert ctx.percentage == 80
        assert ctx.metric == ThresholdMetric.RPM
        assert ctx.current_value == 400.0
        assert ctx.limit_value == 500.0


class TestDefaultRateLimitManager:
    """Tests for DefaultRateLimitManager."""

    def test_creation(self):
        """Test manager creation."""
        config = APIRateLimit(rpm=500, tpm=150000)
        manager = DefaultRateLimitManager(config=config)
        assert manager.config == config

    def test_record_increments_counters(self):
        """Test that record increments request and token counters."""
        config = APIRateLimit(rpm=500, tpm=150000)
        manager = DefaultRateLimitManager(config=config)

        manager.record(tokens_used=1000)

        assert manager.current_rpm == 1
        assert manager.current_tpm == 1000

    def test_check_allows_when_under_limit(self):
        """Test check allows request when under limit."""
        config = APIRateLimit(rpm=500, tpm=150000)
        manager = DefaultRateLimitManager(config=config)

        allowed, reason = manager.check()

        assert allowed is True
        assert reason == "OK"

    def test_check_blocks_when_rpm_exceeded(self):
        """Test check blocks when RPM exceeded."""
        config = APIRateLimit(rpm=1, tpm=150000)
        manager = DefaultRateLimitManager(config=config)
        manager.record()

        allowed, reason = manager.check()

        assert allowed is False
        assert "RPM exceeded" in reason

    def test_check_blocks_when_tpm_exceeded(self):
        """Test check blocks when TPM exceeded."""
        config = APIRateLimit(rpm=500, tpm=100)
        manager = DefaultRateLimitManager(config=config)
        manager.record(tokens_used=101)

        allowed, reason = manager.check()

        assert allowed is False
        assert "TPM exceeded" in reason

    def test_stats_update(self):
        """Test that stats are updated correctly."""
        config = APIRateLimit(rpm=500, tpm=150000, rpd=10000)
        manager = DefaultRateLimitManager(config=config)
        manager.record(tokens_used=1000)

        stats = manager.stats

        assert stats.rpm_used == 1
        assert stats.tpm_used == 1000
        assert stats.rpm_limit == 500
        assert stats.tpm_limit == 150000
        assert stats.rpd_limit == 10000

    def test_threshold_triggered(self):
        """Test threshold is triggered and action executed."""
        triggered = []

        def track_action(ctx):
            triggered.append(ctx.percentage)

        config = APIRateLimit(
            rpm=100,
            thresholds=[
                RateLimitThreshold(at=50, action=track_action, metric=ThresholdMetric.RPM),
            ],
        )
        manager = DefaultRateLimitManager(config=config)

        for _ in range(51):
            manager.record()

        allowed, reason = manager.check()

        assert allowed is True
        assert len(triggered) > 0

    def test_threshold_custom_handler(self):
        """Test custom threshold handler is called."""
        handler = MagicMock()
        config = APIRateLimit(
            rpm=100,
            thresholds=[
                RateLimitThreshold(at=50, action=handler, metric=ThresholdMetric.RPM),
            ],
        )
        manager = DefaultRateLimitManager(config=config)

        for _ in range(51):
            manager.record()

        manager.check()

        assert handler.called

    def test_get_triggered_threshold(self):
        """Test get_triggered_threshold returns correct threshold."""
        config = APIRateLimit(
            rpm=100,
            thresholds=[
                RateLimitThreshold(at=100, action=lambda _: None, metric=ThresholdMetric.RPM),
            ],
        )
        manager = DefaultRateLimitManager(config=config)

        # Make 100 requests
        for _ in range(100):
            manager.record()

        triggered = manager.get_triggered_threshold()

        assert triggered is not None
        assert triggered.at == 100
        assert triggered.metric == ThresholdMetric.RPM

    def test_get_triggered_action_returns_none_when_no_threshold(self):
        """Test get_triggered_threshold returns None when no threshold hit."""
        config = APIRateLimit(rpm=100)
        manager = DefaultRateLimitManager(config=config)

        triggered = manager.get_triggered_threshold()

        assert triggered is None

    def test_events_emitted_on_threshold(self):
        """Test events are emitted when threshold triggered."""
        emit_fn = MagicMock()
        config = APIRateLimit(
            rpm=100,
            thresholds=[
                RateLimitThreshold(
                    at=50,
                    action=lambda _: None,
                    metric=ThresholdMetric.RPM,
                ),
            ],
        )
        manager = DefaultRateLimitManager(config=config)
        manager.set_emit_fn(emit_fn)

        # Make 51 requests to trigger 50% threshold
        for _ in range(51):
            manager.record()

        manager.check()

        # Should have emitted ratelimit.threshold
        emit_fn.assert_called()
        calls = emit_fn.call_args_list
        event_names = [call[0][0] for call in calls]
        assert "ratelimit.threshold" in event_names

    def test_events_emitted_on_exceeded(self):
        """Test events are emitted when limit exceeded."""
        emit_fn = MagicMock()
        config = APIRateLimit(rpm=1)
        manager = DefaultRateLimitManager(config=config)
        manager.set_emit_fn(emit_fn)
        manager.record()

        manager.check()

        # Should have emitted ratelimit.exceeded
        calls = emit_fn.call_args_list
        event_names = [call[0][0] for call in calls]
        assert "ratelimit.exceeded" in event_names


class TestCreateRateLimitManager:
    """Tests for create_rate_limit_manager factory."""

    def test_creates_with_config(self):
        """Test creating manager with custom config."""
        config = APIRateLimit(rpm=500)
        manager = create_rate_limit_manager(config)
        assert manager.config == config

    def test_sets_emit_fn(self):
        """Test emit function is set."""
        emit_fn = MagicMock()
        config = APIRateLimit(rpm=500)
        manager = create_rate_limit_manager(config, emit_fn=emit_fn)
        # Function should be set
        assert manager._emit_fn is emit_fn


class TestRollingWindows:
    """Tests for rolling window behavior."""

    def test_rpm_window_expires(self):
        """Test RPM counter resets after window expires."""
        config = APIRateLimit(rpm=10)
        manager = DefaultRateLimitManager(config=config)
        manager.record()

        # Within window
        assert manager.current_rpm == 1

    def test_tokens_accumulated_in_window(self):
        """Test tokens are accumulated in rolling window."""
        config = APIRateLimit(tpm=10000)
        manager = DefaultRateLimitManager(config=config)

        manager.record(tokens_used=1000)
        manager.record(tokens_used=2000)

        assert manager.current_tpm == 3000


class TestRateLimitStats:
    """Tests for RateLimitStats."""

    def test_defaults(self):
        """Test default values."""
        stats = RateLimitStats()
        assert stats.rpm_used == 0
        assert stats.rpm_limit == 0
        assert stats.tpm_used == 0
        assert stats.tpm_limit == 0
        assert stats.rpd_used == 0
        assert stats.rpd_limit == 0
        assert stats.thresholds_triggered == []

    def test_with_values(self):
        """Test with custom values."""
        stats = RateLimitStats(
            rpm_used=100,
            rpm_limit=500,
            tpm_used=10000,
            tpm_limit=150000,
            rpd_used=500,
            rpd_limit=10000,
            thresholds_triggered=["rpm"],
        )
        assert stats.rpm_used == 100
        assert stats.thresholds_triggered == ["rpm"]


# =============================================================================
# RATELIMIT EDGE CASES - TRY TO BREAK FUNCTIONALITY
# =============================================================================


class TestRateLimitEdgeCases:
    """Edge cases for rate limiting."""

    def test_api_rate_limit_with_rpd_only(self):
        """APIRateLimit with only rpd."""
        limit = APIRateLimit(rpd=1000)
        assert limit.rpd == 1000
        assert limit.rpm is None

    def test_api_rate_limit_with_very_high_values(self):
        """Very high rate limit values."""
        limit = APIRateLimit(rpm=1_000_000, tpm=100_000_000)
        assert limit.rpm == 1_000_000
        assert limit.tpm == 100_000_000

    def test_rate_limit_threshold_at_zero(self):
        """RateLimitThreshold at 0%."""
        threshold = RateLimitThreshold(
            at=0,
            action=lambda _: None,
            metric=ThresholdMetric.RPM,
        )
        assert threshold.at == 0

    def test_rate_limit_threshold_at_100(self):
        """RateLimitThreshold at 100%."""
        threshold = RateLimitThreshold(
            at=100,
            action=lambda _: None,
            metric=ThresholdMetric.RPM,
        )
        assert threshold.at == 100

    def test_manager_with_zero_tokens(self):
        """Record with zero tokens."""
        config = APIRateLimit(rpm=500, tpm=150000)
        manager = DefaultRateLimitManager(config=config)
        manager.record(tokens_used=0)
        assert manager.current_tpm == 0

    def test_manager_record_multiple_times(self):
        """Multiple records accumulate correctly."""
        config = APIRateLimit(rpm=500, tpm=150000)
        manager = DefaultRateLimitManager(config=config)

        for _ in range(10):
            manager.record(tokens_used=100)

        assert manager.current_rpm == 10
        assert manager.current_tpm == 1000

    def test_stats_with_no_thresholds(self):
        """Stats with no thresholds triggered."""
        stats = RateLimitStats(rpm_used=50, rpm_limit=100)
        assert stats.thresholds_triggered == []

    def test_check_with_no_limits(self):
        """Check when no limits configured."""
        config = APIRateLimit(rpm=1)
        manager = DefaultRateLimitManager(config=config)
        manager.record()  # Use 1 request

        allowed, reason = manager.check()
        # After 1 request of 1, should be OK since we check AFTER using
        # But actually with RPM=1, after 1 request we're at 100%

    def test_threshold_with_multiple_metrics(self):
        """Multiple threshold metrics."""
        thresholds = [
            RateLimitThreshold(at=50, action=lambda _: None, metric=ThresholdMetric.RPM),
            RateLimitThreshold(at=50, action=lambda _: None, metric=ThresholdMetric.TPM),
        ]
        limit = APIRateLimit(rpm=100, tpm=1000, thresholds=thresholds)
        assert len(limit.thresholds) == 2

    def test_threshold_with_lambda(self):
        """Test threshold with lambda action."""
        threshold = RateLimitThreshold(at=80, action=lambda _: None, metric=ThresholdMetric.RPM)
        assert callable(threshold.action)

    def test_threshold_with_function(self):
        """Test threshold with function action."""

        def my_action(ctx):
            pass

        threshold = RateLimitThreshold(at=90, action=my_action, metric=ThresholdMetric.TPM)
        assert callable(threshold.action)


class TestRateLimitThresholdValidation:
    """Edge case tests for threshold validation in APIRateLimit."""

    def test_rate_limit_rejects_budget_threshold(self):
        """APIRateLimit should reject BudgetThreshold."""
        from syrin.threshold import BudgetThreshold

        with pytest.raises(
            ValueError, match="APIRateLimit thresholds only accept RateLimitThreshold"
        ):
            APIRateLimit(
                rpm=100,
                thresholds=[BudgetThreshold(at=80, action=lambda _: None)],
            )

    def test_rate_limit_rejects_context_threshold(self):
        """APIRateLimit should reject ContextThreshold."""
        from syrin.threshold import ContextThreshold

        with pytest.raises(
            ValueError, match="APIRateLimit thresholds only accept RateLimitThreshold"
        ):
            APIRateLimit(
                rpm=100,
                thresholds=[ContextThreshold(at=80, action=lambda _: None)],
            )

    def test_rate_limit_threshold_requires_metric(self):
        """RateLimitThreshold should require metric."""
        from syrin.threshold import RateLimitThreshold

        with pytest.raises(ValueError, match="RateLimitThreshold 'metric' is required"):
            RateLimitThreshold(at=80, action=lambda _: None)

    def test_rate_limit_threshold_invalid_metric(self):
        """RateLimitThreshold should reject invalid metrics like COST."""
        from syrin.threshold import RateLimitThreshold

        with pytest.raises(ValueError, match="RateLimitThreshold only supports"):
            RateLimitThreshold(at=80, action=lambda _: None, metric=ThresholdMetric.COST)

    def test_rate_limit_threshold_accepts_rpm(self):
        """RateLimitThreshold should accept RPM."""
        threshold = RateLimitThreshold(at=50, action=lambda _: None, metric=ThresholdMetric.RPM)
        assert threshold.at == 50
        assert threshold.metric == ThresholdMetric.RPM

    def test_rate_limit_threshold_accepts_tpm(self):
        """RateLimitThreshold should accept TPM."""
        threshold = RateLimitThreshold(at=50, action=lambda _: None, metric=ThresholdMetric.TPM)
        assert threshold.at == 50
        assert threshold.metric == ThresholdMetric.TPM

    def test_rate_limit_threshold_accepts_rpd(self):
        """RateLimitThreshold should accept RPD."""
        threshold = RateLimitThreshold(at=50, action=lambda _: None, metric=ThresholdMetric.RPD)
        assert threshold.at == 50
        assert threshold.metric == ThresholdMetric.RPD

    def test_rate_limit_threshold_rejects_tokens_metric(self):
        """RateLimitThreshold should reject TOKENS metric."""
        with pytest.raises(ValueError, match="RateLimitThreshold only supports"):
            RateLimitThreshold(at=80, action=lambda _: None, metric=ThresholdMetric.TOKENS)

    def test_rate_limit_threshold_invalid_at_negative(self):
        """RateLimitThreshold should reject negative at value."""
        with pytest.raises(ValueError, match="Threshold 'at' must be between 0 and 100"):
            RateLimitThreshold(at=-1, action=lambda _: None, metric=ThresholdMetric.RPM)

    def test_rate_limit_threshold_invalid_at_over_100(self):
        """RateLimitThreshold should reject at > 100."""
        with pytest.raises(ValueError, match="Threshold 'at' must be between 0 and 100"):
            RateLimitThreshold(at=101, action=lambda _: None, metric=ThresholdMetric.RPM)

    def test_rate_limit_threshold_requires_action(self):
        """RateLimitThreshold should require action."""
        with pytest.raises(ValueError, match="Threshold 'action' is required"):
            RateLimitThreshold(at=80, action=None, metric=ThresholdMetric.RPM)  # type: ignore

    def test_threshold_context_fields(self):
        """ThresholdContext should have all required fields."""
        ctx = ThresholdContext(
            percentage=80,
            metric=ThresholdMetric.RPM,
            current_value=80,
            limit_value=100,
            budget_run=10.0,
            parent=None,
            metadata={"key": "value"},
        )
        assert ctx.percentage == 80
        assert ctx.metric == ThresholdMetric.RPM
        assert ctx.current_value == 80
        assert ctx.limit_value == 100
        assert ctx.budget_run == 10.0
        assert ctx.metadata == {"key": "value"}

    def test_threshold_should_trigger_with_matching_metric(self):
        """Threshold.should_trigger should work with matching metric."""
        threshold = RateLimitThreshold(at=80, action=lambda _: None, metric=ThresholdMetric.RPM)
        assert threshold.should_trigger(80, ThresholdMetric.RPM) is True
        assert threshold.should_trigger(90, ThresholdMetric.RPM) is True

    def test_threshold_should_trigger_with_non_matching_metric(self):
        """Threshold.should_trigger should return False with non-matching metric."""
        threshold = RateLimitThreshold(at=80, action=lambda _: None, metric=ThresholdMetric.RPM)
        assert threshold.should_trigger(80, ThresholdMetric.TPM) is False
        assert threshold.should_trigger(90, ThresholdMetric.TPM) is False
