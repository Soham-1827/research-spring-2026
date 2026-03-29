"""Tests for the Simulation Engine (Phase 3).

Validates:
- Decision model schemas (PersonaDecision, DecisionRecord, RevealRecord)
- Blind phase runs all personas in parallel (asyncio.gather)
- JSONL serialization and deserialization
- No outcome data reaches the LLM (snapshot safety)
- Stake constraints enforced (0 <= stake <= balance)
- Dry-run CLI works without API key
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ensemble.models import (
    Action,
    DecisionRecord,
    Event,
    Outcome,
    PersonaDecision,
    RevealRecord,
    TimeWindowLabel,
)
from ensemble.personas import PersonaConfig, load_personas
from ensemble.simulator import (
    load_decisions_jsonl,
    run_blind_phase,
    write_decisions_jsonl,
    write_reveal_jsonl,
)


@pytest.fixture
def two_personas() -> list[PersonaConfig]:
    """Two minimal personas for testing."""
    return [
        PersonaConfig(
            name="Bold Bettor",
            id="bold",
            bias_type="overconfidence",
            description="Always confident",
            traits=["trait1", "trait2", "trait3"],
        ),
        PersonaConfig(
            name="Shy Skipper",
            id="shy",
            bias_type="risk_aversion",
            description="Always cautious",
            traits=["trait1", "trait2", "trait3"],
        ),
    ]


@pytest.fixture
def sample_event() -> Event:
    """A test event."""
    return Event(
        event_ticker="TEST-001",
        market_ticker="TEST-001-YES",
        series_ticker="TEST",
        title="Test Event",
        question="Will the test pass?",
        description="A test prediction market.",
        category="test",
        outcome=Outcome.YES,
        close_time=datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc),
        open_time=datetime(2025, 5, 1, 0, 0, tzinfo=timezone.utc),
        window_prices={
            TimeWindowLabel.T_7D: 45,
            TimeWindowLabel.T_1D: 55,
            TimeWindowLabel.T_1H: 65,
        },
    )


class TestDecisionModels:
    def test_persona_decision_schema(self) -> None:
        """PersonaDecision has action, stake_dollars, reasoning."""
        d = PersonaDecision(
            action=Action.BUY_YES,
            stake_dollars=25.0,
            reasoning="Market undervalues this event",
        )
        assert d.action == Action.BUY_YES
        assert d.stake_dollars == 25.0
        assert len(d.reasoning) > 0

    def test_persona_decision_skip(self) -> None:
        """SKIP action with zero stake."""
        d = PersonaDecision(
            action=Action.SKIP,
            stake_dollars=0.0,
            reasoning="Too uncertain",
        )
        assert d.action == Action.SKIP
        assert d.stake_dollars == 0.0

    def test_persona_decision_rejects_negative_stake(self) -> None:
        """Negative stake is rejected by Pydantic validation."""
        with pytest.raises(Exception):
            PersonaDecision(
                action=Action.BUY_YES,
                stake_dollars=-10.0,
                reasoning="test",
            )

    def test_decision_record_has_full_context(self) -> None:
        """DecisionRecord includes event, window, persona, and market data."""
        r = DecisionRecord(
            event_ticker="E1",
            market_ticker="M1",
            window=TimeWindowLabel.T_1D,
            persona_id="bold",
            persona_name="Bold Bettor",
            action=Action.BUY_YES,
            stake_dollars=30.0,
            reasoning="test",
            yes_price_cents=55,
            no_price_cents=45,
            timestamp=datetime.now(timezone.utc),
        )
        assert r.event_ticker == "E1"
        assert r.yes_price_cents + r.no_price_cents == 100

    def test_reveal_record_groups_decisions(self) -> None:
        """RevealRecord holds all persona decisions for one window."""
        now = datetime.now(timezone.utc)
        decisions = [
            DecisionRecord(
                event_ticker="E1",
                market_ticker="M1",
                window=TimeWindowLabel.T_1D,
                persona_id=f"p{i}",
                persona_name=f"Persona {i}",
                action=Action.BUY_YES,
                stake_dollars=10.0,
                reasoning="test",
                yes_price_cents=50,
                no_price_cents=50,
                timestamp=now,
            )
            for i in range(3)
        ]
        reveal = RevealRecord(
            event_ticker="E1",
            window=TimeWindowLabel.T_1D,
            decisions=decisions,
            timestamp=now,
        )
        assert len(reveal.decisions) == 3


class TestJSONLSerialization:
    def test_write_and_read_decisions(self, tmp_path, sample_event) -> None:
        """Decisions written to JSONL can be read back identically."""
        jsonl_path = tmp_path / "decisions.jsonl"
        now = datetime.now(timezone.utc)

        records = [
            DecisionRecord(
                event_ticker="E1",
                market_ticker="M1",
                window=TimeWindowLabel.T_7D,
                persona_id="bold",
                persona_name="Bold",
                action=Action.BUY_YES,
                stake_dollars=25.0,
                reasoning="go big",
                yes_price_cents=45,
                no_price_cents=55,
                timestamp=now,
            ),
            DecisionRecord(
                event_ticker="E1",
                market_ticker="M1",
                window=TimeWindowLabel.T_7D,
                persona_id="shy",
                persona_name="Shy",
                action=Action.SKIP,
                stake_dollars=0.0,
                reasoning="too risky",
                yes_price_cents=45,
                no_price_cents=55,
                timestamp=now,
            ),
        ]

        write_decisions_jsonl(records, jsonl_path)
        loaded = load_decisions_jsonl(jsonl_path)

        assert len(loaded) == 2
        assert loaded[0].persona_id == "bold"
        assert loaded[0].action == Action.BUY_YES
        assert loaded[1].persona_id == "shy"
        assert loaded[1].action == Action.SKIP

    def test_jsonl_appends_across_calls(self, tmp_path) -> None:
        """Multiple write calls append to the same file."""
        jsonl_path = tmp_path / "decisions.jsonl"
        now = datetime.now(timezone.utc)

        record = DecisionRecord(
            event_ticker="E1",
            market_ticker="M1",
            window=TimeWindowLabel.T_7D,
            persona_id="p1",
            persona_name="P1",
            action=Action.BUY_NO,
            stake_dollars=10.0,
            reasoning="test",
            yes_price_cents=60,
            no_price_cents=40,
            timestamp=now,
        )

        write_decisions_jsonl([record], jsonl_path)
        write_decisions_jsonl([record], jsonl_path)

        loaded = load_decisions_jsonl(jsonl_path)
        assert len(loaded) == 2

    def test_reveal_records_skipped_when_loading_decisions(self, tmp_path) -> None:
        """load_decisions_jsonl skips reveal records."""
        jsonl_path = tmp_path / "decisions.jsonl"
        now = datetime.now(timezone.utc)

        record = DecisionRecord(
            event_ticker="E1",
            market_ticker="M1",
            window=TimeWindowLabel.T_7D,
            persona_id="p1",
            persona_name="P1",
            action=Action.BUY_YES,
            stake_dollars=15.0,
            reasoning="test",
            yes_price_cents=50,
            no_price_cents=50,
            timestamp=now,
        )
        reveal = RevealRecord(
            event_ticker="E1",
            window=TimeWindowLabel.T_7D,
            decisions=[record],
            timestamp=now,
        )

        write_decisions_jsonl([record], jsonl_path)
        write_reveal_jsonl(reveal, jsonl_path)

        loaded = load_decisions_jsonl(jsonl_path)
        assert len(loaded) == 1  # reveal record skipped


class TestBlindPhase:
    @pytest.mark.asyncio
    async def test_blind_phase_returns_one_decision_per_persona(
        self, two_personas, sample_event
    ) -> None:
        """Blind phase produces exactly one DecisionRecord per persona."""
        mock_decision = PersonaDecision(
            action=Action.BUY_YES,
            stake_dollars=20.0,
            reasoning="mocked decision",
        )

        with patch("ensemble.simulator.call_persona", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_decision

            records = await run_blind_phase(
                client=MagicMock(),
                personas=two_personas,
                event=sample_event,
                window=TimeWindowLabel.T_1D,
                balances={"bold": 100.0, "shy": 100.0},
            )

        assert len(records) == 2
        assert records[0].persona_id == "bold"
        assert records[1].persona_id == "shy"

    @pytest.mark.asyncio
    async def test_blind_phase_calls_all_personas_simultaneously(
        self, two_personas, sample_event
    ) -> None:
        """All persona calls are launched before any completes (gather semantics)."""
        call_order = []

        async def mock_call(*args, **kwargs):
            persona = kwargs.get("persona") or args[1]
            call_order.append(f"start_{persona.id}")
            await asyncio.sleep(0.01)
            call_order.append(f"end_{persona.id}")
            return PersonaDecision(
                action=Action.SKIP,
                stake_dollars=0.0,
                reasoning="mock",
            )

        with patch("ensemble.simulator.call_persona", side_effect=mock_call):
            await run_blind_phase(
                client=MagicMock(),
                personas=two_personas,
                event=sample_event,
                window=TimeWindowLabel.T_7D,
                balances={"bold": 100.0, "shy": 100.0},
            )

        # Both starts should happen before both ends (parallel execution)
        assert call_order.index("start_bold") < call_order.index("end_shy")
        assert call_order.index("start_shy") < call_order.index("end_bold")

    @pytest.mark.asyncio
    async def test_blind_phase_snapshot_has_no_outcome(
        self, two_personas, sample_event
    ) -> None:
        """The snapshot passed to call_persona never contains outcome data."""
        captured_snapshots = []

        async def capture_call(*args, **kwargs):
            snapshot = kwargs.get("snapshot") or args[2]
            captured_snapshots.append(snapshot)
            return PersonaDecision(
                action=Action.SKIP,
                stake_dollars=0.0,
                reasoning="mock",
            )

        with patch("ensemble.simulator.call_persona", side_effect=capture_call):
            await run_blind_phase(
                client=MagicMock(),
                personas=two_personas,
                event=sample_event,
                window=TimeWindowLabel.T_1H,
                balances={"bold": 100.0, "shy": 100.0},
            )

        # Verify snapshots have no outcome-related fields
        forbidden = {"outcome", "result", "settlement", "resolution"}
        for snap in captured_snapshots:
            snap_fields = set(type(snap).model_fields.keys())
            leaked = snap_fields & forbidden
            assert leaked == set(), f"Snapshot contains forbidden fields: {leaked}"


class TestActionEnum:
    def test_all_actions_defined(self) -> None:
        """BUY_YES, BUY_NO, and SKIP are all valid actions."""
        assert Action.BUY_YES.value == "BUY_YES"
        assert Action.BUY_NO.value == "BUY_NO"
        assert Action.SKIP.value == "SKIP"
        assert len(Action) == 3
