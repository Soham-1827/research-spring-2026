"""
Pydantic data models for the LLM Prediction Market Ensemble.

Core pattern: Composition-based type boundary between Event and EventSnapshot.
EventSnapshot is an independent model that structurally cannot contain outcome data.
Event holds outcome data and can produce EventSnapshot via .snapshot() method.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum

from typing import Literal

from pydantic import BaseModel, Field


class TimeWindowLabel(str, Enum):
    """Time window labels for price snapshots."""

    T_7D = "T-7d"
    T_1D = "T-1d"
    T_1H = "T-1h"


class Outcome(str, Enum):
    """Possible outcomes for a prediction market event."""

    YES = "yes"
    NO = "no"


class EventSnapshot(BaseModel):
    """LLM-safe model. This is what goes into the prompt.

    INVARIANT: This model NEVER contains outcome, result, settlement,
    or resolution data. If you need those fields, use Event instead --
    but NEVER pass Event to an LLM.
    """

    event_ticker: str
    market_ticker: str
    title: str
    question: str
    description: str
    category: str
    yes_price_cents: int = Field(description="Market YES price in cents (0-100)")
    no_price_cents: int = Field(description="Market NO price in cents (0-100), always 100 - yes_price_cents")
    window: TimeWindowLabel
    snapshot_timestamp: datetime


class Action(str, Enum):
    """Trading actions a persona can take."""

    BUY_YES = "BUY_YES"
    BUY_NO = "BUY_NO"
    SKIP = "SKIP"


class PersonaDecision(BaseModel):
    """Structured decision from a single persona -- used as OpenAI response_format.

    This is the schema that gpt-5-nano returns via structured outputs.
    """

    action: Action = Field(description="BUY_YES, BUY_NO, or SKIP")
    stake_dollars: float = Field(
        ge=0, description="Dollar amount to stake (0 if SKIP)"
    )
    reasoning: str = Field(description="Why this decision reflects the persona's bias")


class DecisionRecord(BaseModel):
    """A serialized decision with full context for decisions.jsonl."""

    event_ticker: str
    market_ticker: str
    window: TimeWindowLabel
    persona_id: str
    persona_name: str
    action: Action
    stake_dollars: float
    reasoning: str
    yes_price_cents: int
    no_price_cents: int
    timestamp: datetime


class RevealRecord(BaseModel):
    """Reveal phase log: all persona positions for one event+window."""

    event_ticker: str
    window: TimeWindowLabel
    decisions: list[DecisionRecord]
    timestamp: datetime


class CouncilResponse(BaseModel):
    """Structured response from a persona during the council debate phase.

    Contains both the debate argument and a revised decision.
    """

    debate_argument: str = Field(
        description="Natural rebuttal addressing other personas by name"
    )
    revised_action: Action = Field(description="BUY_YES, BUY_NO, or SKIP after debate")
    revised_stake_dollars: float = Field(
        ge=0, description="Revised stake amount (0 if SKIP)"
    )
    changed_mind: bool = Field(
        description="True if the revised decision differs from the blind decision"
    )
    reasoning: str = Field(
        description="Why the persona kept or changed their position after debate"
    )


class MarketCandle(BaseModel):
    """A single candlestick data point from Kalshi market history."""

    end_period_ts: datetime
    close_cents: int
    open_cents: int
    high_cents: int
    low_cents: int
    volume: int


class Event(BaseModel):
    """Full event including outcome -- used for settlement ONLY.

    WARNING: Never pass this model or its outcome field to an LLM.
    Use .snapshot(window) to create an LLM-safe EventSnapshot.
    """

    event_ticker: str
    market_ticker: str
    series_ticker: str
    title: str
    question: str
    description: str
    category: str
    outcome: Outcome
    close_time: datetime
    open_time: datetime
    window_prices: dict[TimeWindowLabel, int] = Field(
        description="YES price in cents at each time window snapshot"
    )

    def _window_timestamp(self, window: TimeWindowLabel) -> datetime:
        """Compute the timestamp for a given time window relative to close_time."""
        offsets = {
            TimeWindowLabel.T_7D: timedelta(days=7),
            TimeWindowLabel.T_1D: timedelta(days=1),
            TimeWindowLabel.T_1H: timedelta(hours=1),
        }
        return self.close_time - offsets[window]

    def snapshot(self, window: TimeWindowLabel) -> EventSnapshot:
        """Create an LLM-safe EventSnapshot for the given time window.

        This is the ONLY way to produce data safe for LLM consumption.
        The type signature enforces that outcome data cannot leak.
        """
        yes_price = self.window_prices[window]
        return EventSnapshot(
            event_ticker=self.event_ticker,
            market_ticker=self.market_ticker,
            title=self.title,
            question=self.question,
            description=self.description,
            category=self.category,
            yes_price_cents=yes_price,
            no_price_cents=100 - yes_price,
            window=window,
            snapshot_timestamp=self._window_timestamp(window),
        )
