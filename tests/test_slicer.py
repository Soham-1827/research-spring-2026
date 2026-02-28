"""Tests for time window computation and nearest price finding."""

from datetime import datetime, timedelta, timezone

from ensemble.models import MarketCandle, TimeWindowLabel
from ensemble.slicer import compute_window_timestamps, find_nearest_price


class TestComputeWindowTimestamps:
    def test_compute_window_timestamps(self) -> None:
        """T-7d is 7 days before close, T-1d is 1 day, T-1h is 1 hour."""
        close = datetime(2024, 12, 18, 19, 0, 0, tzinfo=timezone.utc)
        windows = compute_window_timestamps(close)

        assert windows[TimeWindowLabel.T_7D] == close - timedelta(days=7)
        assert windows[TimeWindowLabel.T_1D] == close - timedelta(days=1)
        assert windows[TimeWindowLabel.T_1H] == close - timedelta(hours=1)
        assert len(windows) == 3


def _make_candle(ts: datetime, close_cents: int) -> MarketCandle:
    """Helper to create a candle at a given timestamp."""
    return MarketCandle(
        end_period_ts=ts,
        close_cents=close_cents,
        open_cents=close_cents,
        high_cents=close_cents,
        low_cents=close_cents,
        volume=100,
    )


class TestFindNearestPrice:
    def test_find_nearest_price_exact_match(self) -> None:
        """Candle at exact target timestamp returns its close_cents."""
        target = datetime(2024, 12, 11, 19, 0, 0, tzinfo=timezone.utc)
        candles = [_make_candle(target, 72)]
        assert find_nearest_price(candles, target) == 72

    def test_find_nearest_price_within_tolerance(self) -> None:
        """Candle 30 minutes from target (within 2h default) returns price."""
        target = datetime(2024, 12, 11, 19, 0, 0, tzinfo=timezone.utc)
        candle_ts = target + timedelta(minutes=30)
        candles = [_make_candle(candle_ts, 65)]
        assert find_nearest_price(candles, target) == 65

    def test_find_nearest_price_outside_tolerance(self) -> None:
        """Candle 3 hours from target (outside 2h default) returns None."""
        target = datetime(2024, 12, 11, 19, 0, 0, tzinfo=timezone.utc)
        candle_ts = target + timedelta(hours=3)
        candles = [_make_candle(candle_ts, 65)]
        assert find_nearest_price(candles, target) is None

    def test_find_nearest_price_empty_list(self) -> None:
        """Empty candle list returns None."""
        target = datetime(2024, 12, 11, 19, 0, 0, tzinfo=timezone.utc)
        assert find_nearest_price([], target) is None

    def test_find_nearest_price_picks_closest(self) -> None:
        """When multiple candles exist, picks the one closest to target."""
        target = datetime(2024, 12, 11, 19, 0, 0, tzinfo=timezone.utc)
        candles = [
            _make_candle(target - timedelta(hours=1), 60),
            _make_candle(target - timedelta(minutes=10), 72),
            _make_candle(target + timedelta(hours=1), 80),
        ]
        assert find_nearest_price(candles, target) == 72
