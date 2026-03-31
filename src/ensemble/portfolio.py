"""Portfolio tracker with Kalshi pricing math.

Kalshi contract pricing:
- BUY YES at X¢: pay $stake. If YES wins → payout = stake * 100/X.
  Profit = stake * (100-X)/X. Loss = -stake.
- BUY NO at (100-X)¢: pay $stake. If NO wins → payout = stake * 100/(100-X).
  Profit = stake * X/(100-X). Loss = -stake.
- SKIP: no effect on balance.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from ensemble.models import Action, DecisionRecord, Outcome


@dataclass
class BetResult:
    """Result of settling a single bet."""

    persona_id: str
    event_ticker: str
    window: str
    action: str
    stake: float
    yes_price_cents: int
    payout: float
    profit: float
    won: bool


def settle_bet(decision: DecisionRecord, outcome: Outcome) -> BetResult:
    """Settle a single bet against the actual outcome.

    Returns a BetResult with payout, profit, and win/loss.
    """
    if decision.action == Action.SKIP:
        return BetResult(
            persona_id=decision.persona_id,
            event_ticker=decision.event_ticker,
            window=decision.window.value,
            action="SKIP",
            stake=0.0,
            yes_price_cents=decision.yes_price_cents,
            payout=0.0,
            profit=0.0,
            won=False,
        )

    stake = decision.stake_dollars
    yes_price = decision.yes_price_cents
    no_price = decision.no_price_cents

    if decision.action == Action.BUY_YES:
        if outcome == Outcome.YES:
            payout = stake * 100 / yes_price
            return BetResult(
                persona_id=decision.persona_id,
                event_ticker=decision.event_ticker,
                window=decision.window.value,
                action="BUY_YES",
                stake=stake,
                yes_price_cents=yes_price,
                payout=round(payout, 2),
                profit=round(payout - stake, 2),
                won=True,
            )
        else:
            return BetResult(
                persona_id=decision.persona_id,
                event_ticker=decision.event_ticker,
                window=decision.window.value,
                action="BUY_YES",
                stake=stake,
                yes_price_cents=yes_price,
                payout=0.0,
                profit=round(-stake, 2),
                won=False,
            )

    # BUY_NO
    if outcome == Outcome.NO:
        payout = stake * 100 / no_price
        return BetResult(
            persona_id=decision.persona_id,
            event_ticker=decision.event_ticker,
            window=decision.window.value,
            action="BUY_NO",
            stake=stake,
            yes_price_cents=yes_price,
            payout=round(payout, 2),
            profit=round(payout - stake, 2),
            won=True,
        )
    else:
        return BetResult(
            persona_id=decision.persona_id,
            event_ticker=decision.event_ticker,
            window=decision.window.value,
            action="BUY_NO",
            stake=stake,
            yes_price_cents=yes_price,
            payout=0.0,
            profit=round(-stake, 2),
            won=False,
        )


@dataclass
class PortfolioTracker:
    """Tracks running balance per persona across all events."""

    starting_balance: float = 100.0
    balances: dict[str, float] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)

    def init_persona(self, persona_id: str) -> None:
        """Initialize a persona with the starting balance."""
        if persona_id not in self.balances:
            self.balances[persona_id] = self.starting_balance

    def apply_bet(self, decision: DecisionRecord, outcome: Outcome) -> BetResult:
        """Apply a single bet: deduct stake, then settle.

        Balance change: -stake + payout (payout=0 if lost, >stake if won).
        """
        self.init_persona(decision.persona_id)
        result = settle_bet(decision, outcome)

        # Balance = old - stake + payout
        self.balances[decision.persona_id] += result.profit

        return result

    def process_event(
        self,
        decisions: list[DecisionRecord],
        outcome: Outcome,
        event_ticker: str,
    ) -> list[BetResult]:
        """Process all decisions for one event and record balance snapshot."""
        results = []
        for d in decisions:
            result = self.apply_bet(d, outcome)
            results.append(result)

        # Record balance snapshot after this event
        snapshot = {"event_ticker": event_ticker}
        for pid, bal in self.balances.items():
            snapshot[pid] = round(bal, 2)
        self.history.append(snapshot)

        return results

    def write_portfolio_csv(self, path: Path) -> None:
        """Write running balance per persona after each event to CSV."""
        if not self.history:
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        persona_ids = [k for k in self.history[0] if k != "event_ticker"]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["event_ticker"] + persona_ids)
            writer.writeheader()
            for row in self.history:
                writer.writerow(row)
