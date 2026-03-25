"""Benchmark dataset validation tests.

Validates that data/benchmark/events.json meets all research requirements:
- 15 events with category diversity
- All window prices in valid range
- Schema validation passes
- No outcome data leaks into snapshots
"""

import json
from pathlib import Path

import pytest

from ensemble.loader import load_events
from ensemble.models import TimeWindowLabel

BENCHMARK_PATH = Path("data/benchmark/events.json")


@pytest.fixture
def benchmark_events():
    """Load the benchmark dataset."""
    if not BENCHMARK_PATH.exists():
        pytest.skip("Benchmark dataset not yet created")
    return load_events(BENCHMARK_PATH)


@pytest.fixture
def benchmark_raw():
    """Load raw benchmark JSON for metadata checks."""
    if not BENCHMARK_PATH.exists():
        pytest.skip("Benchmark dataset not yet created")
    with open(BENCHMARK_PATH) as f:
        return json.load(f)


class TestBenchmarkDataset:
    def test_benchmark_loads_successfully(self, benchmark_events) -> None:
        """load_events() returns exactly 15 Event objects."""
        assert len(benchmark_events) == 15, f"Expected 15 events, got {len(benchmark_events)}"

    def test_benchmark_has_category_diversity(self, benchmark_events) -> None:
        """At least 4 unique categories across the 15 events."""
        categories = {e.category for e in benchmark_events}
        assert len(categories) >= 4, (
            f"Need at least 4 categories, got {len(categories)}: {categories}"
        )

    def test_benchmark_all_windows_have_prices(self, benchmark_events) -> None:
        """Every event has window_prices for T-7d, T-1d, T-1h."""
        for event in benchmark_events:
            for window in TimeWindowLabel:
                assert window in event.window_prices, (
                    f"Event {event.event_ticker} missing price for {window.value}"
                )

    def test_benchmark_prices_in_valid_range(self, benchmark_events) -> None:
        """All window prices between 1-99 cents."""
        for event in benchmark_events:
            for window, price in event.window_prices.items():
                assert 1 <= price <= 99, (
                    f"Event {event.event_ticker} window {window.value} has "
                    f"price {price}¢ outside valid range 1-99"
                )

    def test_benchmark_outcomes_are_valid(self, benchmark_events) -> None:
        """All outcomes are 'yes' or 'no'."""
        for event in benchmark_events:
            assert event.outcome.value in ("yes", "no"), (
                f"Event {event.event_ticker} has invalid outcome: {event.outcome}"
            )

    def test_benchmark_snapshots_exclude_outcome(self, benchmark_events) -> None:
        """For each event, all 3 snapshots contain no outcome data."""
        forbidden = {"outcome", "result", "settlement", "close_time", "resolution"}
        for event in benchmark_events:
            for window in TimeWindowLabel:
                if window in event.window_prices:
                    snap = event.snapshot(window)
                    snap_fields = set(type(snap).model_fields.keys())
                    leaked = snap_fields & forbidden
                    assert leaked == set(), (
                        f"Snapshot for {event.event_ticker}/{window.value} "
                        f"contains forbidden fields: {leaked}"
                    )

    def test_benchmark_has_metadata(self, benchmark_raw) -> None:
        """JSON file has metadata block with required fields."""
        assert "metadata" in benchmark_raw
        meta = benchmark_raw["metadata"]
        assert "created" in meta
        assert "total_events" in meta
        assert meta["total_events"] == 15
