"""Tests for Pydantic data models: Event, EventSnapshot, and type boundary."""

from datetime import datetime, timezone

import pytest

from ensemble.models import Event, EventSnapshot, Outcome, TimeWindowLabel


FORBIDDEN_FIELDS = {"outcome", "result", "settlement", "close_time", "resolution"}


class TestEventSnapshotTypeBoundary:
    """Verify that EventSnapshot structurally cannot contain outcome data."""

    def test_event_snapshot_has_no_outcome_field(self) -> None:
        """EventSnapshot model fields must NOT include outcome-related names."""
        snapshot_fields = set(EventSnapshot.model_fields.keys())
        leaked = snapshot_fields & FORBIDDEN_FIELDS
        assert leaked == set(), (
            f"EventSnapshot contains forbidden fields: {leaked}. "
            "These fields must NEVER be in the LLM-safe model."
        )

    def test_event_snapshot_prices_are_int(self, sample_event: Event) -> None:
        """Prices must be stored as integer cents, never floats."""
        snap = sample_event.snapshot(TimeWindowLabel.T_7D)
        assert isinstance(snap.yes_price_cents, int)
        assert isinstance(snap.no_price_cents, int)


class TestEventSnapshotCreation:
    """Verify snapshot creation from Event via composition."""

    def test_event_snapshot_creation_from_event(self, sample_event: Event) -> None:
        """Event.snapshot() should produce a valid EventSnapshot with correct fields."""
        snap = sample_event.snapshot(TimeWindowLabel.T_7D)

        assert isinstance(snap, EventSnapshot)
        assert snap.event_ticker == sample_event.event_ticker
        assert snap.market_ticker == sample_event.market_ticker
        assert snap.title == sample_event.title
        assert snap.question == sample_event.question
        assert snap.description == sample_event.description
        assert snap.category == sample_event.category
        assert snap.yes_price_cents == 72
        assert snap.no_price_cents == 28
        assert snap.window == TimeWindowLabel.T_7D

    def test_no_price_cents_computed(self) -> None:
        """no_price_cents should always equal 100 - yes_price_cents."""
        test_cases = [
            (72, 28),
            (50, 50),
            (0, 100),
            (100, 0),
            (1, 99),
            (99, 1),
        ]
        for yes_price, expected_no_price in test_cases:
            snap = EventSnapshot(
                event_ticker="TEST",
                market_ticker="TEST-MKT",
                title="Test event",
                question="Test?",
                description="Test description",
                category="test",
                yes_price_cents=yes_price,
                no_price_cents=100 - yes_price,
                window=TimeWindowLabel.T_7D,
                snapshot_timestamp=datetime(2024, 12, 11, 19, 0, 0, tzinfo=timezone.utc),
            )
            assert snap.no_price_cents == expected_no_price, (
                f"For yes_price={yes_price}, expected no_price={expected_no_price}, "
                f"got {snap.no_price_cents}"
            )

    def test_all_three_windows_produce_snapshots(self, sample_event: Event) -> None:
        """All three time windows should produce valid snapshots with different prices."""
        snapshots = {
            w: sample_event.snapshot(w)
            for w in TimeWindowLabel
        }

        assert len(snapshots) == 3
        assert snapshots[TimeWindowLabel.T_7D].yes_price_cents == 72
        assert snapshots[TimeWindowLabel.T_1D].yes_price_cents == 85
        assert snapshots[TimeWindowLabel.T_1H].yes_price_cents == 93

        # Each snapshot should have the correct window label
        for window, snap in snapshots.items():
            assert snap.window == window

        # Timestamps should differ
        timestamps = [snap.snapshot_timestamp for snap in snapshots.values()]
        assert len(set(timestamps)) == 3, "Each window should have a unique timestamp"


class TestEventValidation:
    """Verify Event model validation rules."""

    def test_event_validation_rejects_invalid_outcome(self) -> None:
        """Event should reject outcome values other than 'yes' or 'no'."""
        with pytest.raises(ValueError):
            Event(
                event_ticker="TEST",
                market_ticker="TEST-MKT",
                series_ticker="TEST-SERIES",
                title="Test",
                question="Test?",
                description="Test",
                category="test",
                outcome="maybe",  # type: ignore[arg-type]
                close_time=datetime(2024, 12, 18, 19, 0, 0, tzinfo=timezone.utc),
                open_time=datetime(2024, 10, 1, 0, 0, 0, tzinfo=timezone.utc),
                window_prices={},
            )
