"""Tests for Portfolio & Evaluation (Phase 4).

Validates:
- Kalshi pricing math (PORT-01)
- Portfolio balance tracking across events
- CSV output (PORT-02)
- Calibration metrics (PORT-03)
- Skip rate reporting (PORT-04)
"""

from datetime import datetime, timezone

import pytest

from ensemble.models import Action, DecisionRecord, Outcome, TimeWindowLabel
from ensemble.portfolio import BetResult, PortfolioTracker, settle_bet
from ensemble.metrics import compute_brier_score, compute_persona_metrics


def _make_decision(
    action: Action,
    stake: float,
    yes_price: int = 60,
    persona_id: str = "bold",
    event_ticker: str = "E1",
    window: TimeWindowLabel = TimeWindowLabel.T_1D,
) -> DecisionRecord:
    return DecisionRecord(
        event_ticker=event_ticker,
        market_ticker=f"{event_ticker}-YES",
        window=window,
        persona_id=persona_id,
        persona_name="Bold Bettor",
        action=action,
        stake_dollars=stake,
        reasoning="test",
        yes_price_cents=yes_price,
        no_price_cents=100 - yes_price,
        timestamp=datetime.now(timezone.utc),
    )


class TestKalshiPricingMath:
    def test_buy_yes_wins(self) -> None:
        """BUY YES at 60c, outcome YES → payout = stake * 100/60."""
        d = _make_decision(Action.BUY_YES, 30.0, yes_price=60)
        result = settle_bet(d, Outcome.YES)
        assert result.won is True
        assert result.payout == 50.0  # 30 * 100/60
        assert result.profit == 20.0  # 50 - 30

    def test_buy_yes_loses(self) -> None:
        """BUY YES at 60c, outcome NO → lose stake."""
        d = _make_decision(Action.BUY_YES, 30.0, yes_price=60)
        result = settle_bet(d, Outcome.NO)
        assert result.won is False
        assert result.payout == 0.0
        assert result.profit == -30.0

    def test_buy_no_wins(self) -> None:
        """BUY NO at 40c (yes=60c), outcome NO → payout = stake * 100/40."""
        d = _make_decision(Action.BUY_NO, 20.0, yes_price=60)
        result = settle_bet(d, Outcome.NO)
        assert result.won is True
        assert result.payout == 50.0  # 20 * 100/40
        assert result.profit == 30.0  # 50 - 20

    def test_buy_no_loses(self) -> None:
        """BUY NO at 40c, outcome YES → lose stake."""
        d = _make_decision(Action.BUY_NO, 20.0, yes_price=60)
        result = settle_bet(d, Outcome.YES)
        assert result.won is False
        assert result.payout == 0.0
        assert result.profit == -20.0

    def test_skip_no_effect(self) -> None:
        """SKIP → zero stake, zero payout, zero profit."""
        d = _make_decision(Action.SKIP, 0.0)
        result = settle_bet(d, Outcome.YES)
        assert result.profit == 0.0
        assert result.payout == 0.0
        assert result.won is False

    def test_buy_yes_at_50_cents(self) -> None:
        """BUY YES at 50c, outcome YES → payout = 2x stake (even odds)."""
        d = _make_decision(Action.BUY_YES, 10.0, yes_price=50)
        result = settle_bet(d, Outcome.YES)
        assert result.payout == 20.0  # 10 * 100/50
        assert result.profit == 10.0

    def test_buy_yes_at_90_cents(self) -> None:
        """BUY YES at 90c, outcome YES → small profit (low odds)."""
        d = _make_decision(Action.BUY_YES, 90.0, yes_price=90)
        result = settle_bet(d, Outcome.YES)
        assert result.payout == 100.0  # 90 * 100/90
        assert result.profit == 10.0

    def test_buy_yes_at_10_cents(self) -> None:
        """BUY YES at 10c, outcome YES → 10x payout (long shot)."""
        d = _make_decision(Action.BUY_YES, 10.0, yes_price=10)
        result = settle_bet(d, Outcome.YES)
        assert result.payout == 100.0  # 10 * 100/10
        assert result.profit == 90.0


class TestPortfolioTracker:
    def test_starting_balance(self) -> None:
        """Each persona starts with the configured balance."""
        tracker = PortfolioTracker(starting_balance=100.0)
        tracker.init_persona("bold")
        assert tracker.balances["bold"] == 100.0

    def test_balance_after_win(self) -> None:
        """Balance increases after a winning bet."""
        tracker = PortfolioTracker(starting_balance=100.0)
        d = _make_decision(Action.BUY_YES, 20.0, yes_price=50)
        result = tracker.apply_bet(d, Outcome.YES)
        # profit = 20 * (100/50 - 1) = 20
        assert tracker.balances["bold"] == 120.0

    def test_balance_after_loss(self) -> None:
        """Balance decreases after a losing bet."""
        tracker = PortfolioTracker(starting_balance=100.0)
        d = _make_decision(Action.BUY_YES, 20.0, yes_price=50)
        tracker.apply_bet(d, Outcome.NO)
        assert tracker.balances["bold"] == 80.0

    def test_balance_unchanged_on_skip(self) -> None:
        """SKIP does not change balance."""
        tracker = PortfolioTracker(starting_balance=100.0)
        d = _make_decision(Action.SKIP, 0.0)
        tracker.apply_bet(d, Outcome.YES)
        assert tracker.balances["bold"] == 100.0

    def test_multiple_bets_accumulate(self) -> None:
        """Multiple bets affect balance cumulatively."""
        tracker = PortfolioTracker(starting_balance=100.0)
        # Win: +20 (profit on 20 stake at 50c)
        d1 = _make_decision(Action.BUY_YES, 20.0, yes_price=50, event_ticker="E1")
        tracker.apply_bet(d1, Outcome.YES)
        assert tracker.balances["bold"] == 120.0

        # Lose: -15
        d2 = _make_decision(Action.BUY_YES, 15.0, yes_price=60, event_ticker="E2")
        tracker.apply_bet(d2, Outcome.NO)
        assert tracker.balances["bold"] == 105.0

    def test_process_event_records_snapshot(self) -> None:
        """process_event adds a balance snapshot to history."""
        tracker = PortfolioTracker(starting_balance=100.0)
        decisions = [
            _make_decision(Action.BUY_YES, 10.0, yes_price=50, persona_id="bold"),
            _make_decision(Action.SKIP, 0.0, persona_id="shy"),
        ]
        tracker.init_persona("shy")
        tracker.process_event(decisions, Outcome.YES, "E1")

        assert len(tracker.history) == 1
        assert tracker.history[0]["event_ticker"] == "E1"
        assert tracker.history[0]["bold"] == 110.0
        assert tracker.history[0]["shy"] == 100.0

    def test_write_portfolio_csv(self, tmp_path) -> None:
        """Portfolio CSV has header and one row per event (PORT-02)."""
        tracker = PortfolioTracker(starting_balance=100.0)
        tracker.init_persona("bold")
        tracker.init_persona("shy")

        d1 = _make_decision(Action.BUY_YES, 10.0, yes_price=50, persona_id="bold", event_ticker="E1")
        d2 = _make_decision(Action.SKIP, 0.0, persona_id="shy", event_ticker="E1")
        tracker.process_event([d1, d2], Outcome.YES, "E1")

        csv_path = tmp_path / "portfolio.csv"
        tracker.write_portfolio_csv(csv_path)

        lines = csv_path.read_text().strip().split("\n")
        assert len(lines) == 2  # header + 1 event
        assert "event_ticker" in lines[0]
        assert "bold" in lines[0]
        assert "shy" in lines[0]


class TestBrierScore:
    def test_perfect_brier_score(self) -> None:
        """All correct predictions → Brier = 0.0."""
        decisions = [
            _make_decision(Action.BUY_YES, 10.0, event_ticker="E1"),
            _make_decision(Action.BUY_NO, 10.0, event_ticker="E2"),
        ]
        outcomes = {"E1": Outcome.YES, "E2": Outcome.NO}
        brier = compute_brier_score(decisions, outcomes)
        assert brier == 0.0

    def test_worst_brier_score(self) -> None:
        """All wrong predictions → Brier = 1.0."""
        decisions = [
            _make_decision(Action.BUY_YES, 10.0, event_ticker="E1"),
            _make_decision(Action.BUY_NO, 10.0, event_ticker="E2"),
        ]
        outcomes = {"E1": Outcome.NO, "E2": Outcome.YES}
        brier = compute_brier_score(decisions, outcomes)
        assert brier == 1.0

    def test_brier_skips_excluded(self) -> None:
        """SKIP decisions are excluded from Brier calculation."""
        decisions = [
            _make_decision(Action.BUY_YES, 10.0, event_ticker="E1"),
            _make_decision(Action.SKIP, 0.0, event_ticker="E2"),
        ]
        outcomes = {"E1": Outcome.YES, "E2": Outcome.NO}
        brier = compute_brier_score(decisions, outcomes)
        assert brier == 0.0  # only E1 counted, which is correct

    def test_brier_empty_returns_zero(self) -> None:
        """No bets → Brier = 0.0."""
        brier = compute_brier_score([], {})
        assert brier == 0.0


class TestPersonaMetrics:
    def test_full_metrics_computation(self) -> None:
        """compute_persona_metrics returns all fields correctly."""
        decisions = [
            _make_decision(Action.BUY_YES, 10.0, event_ticker="E1"),  # win
            _make_decision(Action.BUY_NO, 10.0, event_ticker="E2"),   # win
            _make_decision(Action.BUY_YES, 10.0, event_ticker="E3"),  # lose
            _make_decision(Action.SKIP, 0.0, event_ticker="E4"),      # skip
        ]
        outcomes = {"E1": Outcome.YES, "E2": Outcome.NO, "E3": Outcome.NO, "E4": Outcome.YES}

        metrics = compute_persona_metrics(
            persona_id="bold",
            persona_name="Bold Bettor",
            decisions=decisions,
            outcomes=outcomes,
            starting_balance=100.0,
            final_balance=115.0,
        )

        assert metrics.total_decisions == 4
        assert metrics.total_bets == 3
        assert metrics.total_skips == 1
        assert metrics.wins == 2
        assert metrics.losses == 1
        assert metrics.accuracy_pct == pytest.approx(66.7, abs=0.1)
        assert metrics.skip_rate_pct == 25.0
        assert metrics.roi_pct == 15.0
        assert metrics.brier_score < 1.0

    def test_skip_rate_all_skips(self) -> None:
        """PORT-04: 100% skip rate when all decisions are SKIP."""
        decisions = [
            _make_decision(Action.SKIP, 0.0, event_ticker="E1"),
            _make_decision(Action.SKIP, 0.0, event_ticker="E2"),
        ]
        metrics = compute_persona_metrics(
            "shy", "Shy", decisions, {}, 100.0, 100.0
        )
        assert metrics.skip_rate_pct == 100.0
        assert metrics.accuracy_pct == 0.0
