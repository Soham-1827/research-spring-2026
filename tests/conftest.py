"""Shared test fixtures for the ensemble test suite."""

from datetime import datetime, timezone

import pytest

from ensemble.models import Event, Outcome, TimeWindowLabel


@pytest.fixture
def sample_event() -> Event:
    """A realistic sample event for testing: Fed rate decision Dec 2024."""
    return Event(
        event_ticker="FED-24DEC-T",
        market_ticker="FED-24DEC-T5050",
        series_ticker="FED-24DEC",
        title="Fed interest rate decision December 2024",
        question="Will the Fed cut rates by 25bp at the December 2024 meeting?",
        description="Federal Reserve monetary policy decision for December 2024 FOMC meeting.",
        category="economics",
        outcome=Outcome.YES,
        close_time=datetime(2024, 12, 18, 19, 0, 0, tzinfo=timezone.utc),
        open_time=datetime(2024, 10, 1, 0, 0, 0, tzinfo=timezone.utc),
        window_prices={
            TimeWindowLabel.T_7D: 72,
            TimeWindowLabel.T_1D: 85,
            TimeWindowLabel.T_1H: 93,
        },
    )
