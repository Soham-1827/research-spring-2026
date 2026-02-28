"""Time window price extraction from candlestick data."""

from __future__ import annotations

from datetime import datetime, timedelta

from ensemble.models import MarketCandle, TimeWindowLabel


def compute_window_timestamps(close_time: datetime) -> dict[TimeWindowLabel, datetime]:
    """Compute snapshot timestamps for each time window relative to close_time.

    Returns timestamps for T-7d, T-1d, T-1h by subtracting offsets from close_time.
    All datetimes must be timezone-aware (UTC).
    """
    return {
        TimeWindowLabel.T_7D: close_time - timedelta(days=7),
        TimeWindowLabel.T_1D: close_time - timedelta(days=1),
        TimeWindowLabel.T_1H: close_time - timedelta(hours=1),
    }


def find_nearest_price(
    candles: list[MarketCandle],
    target_ts: datetime,
    max_gap_hours: int = 2,
) -> int | None:
    """Find the candlestick closest to target_ts and return its close_cents.

    Returns None if no candle is within max_gap_hours of the target timestamp.
    """
    if not candles:
        return None

    best_candle = None
    best_delta = None

    for candle in candles:
        delta = abs((candle.end_period_ts - target_ts).total_seconds())
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_candle = candle

    if best_delta is None or best_delta > max_gap_hours * 3600:
        return None

    assert best_candle is not None
    return best_candle.close_cents
