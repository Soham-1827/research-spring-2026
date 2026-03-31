"""Calibration metrics: Brier score, accuracy, ROI, skip rate.

All metrics computed per persona from decision records + outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass

from ensemble.models import Action, DecisionRecord, Outcome


@dataclass
class PersonaMetrics:
    """Aggregated metrics for one persona."""

    persona_id: str
    persona_name: str
    total_decisions: int
    total_bets: int
    total_skips: int
    wins: int
    losses: int
    accuracy_pct: float
    skip_rate_pct: float
    brier_score: float
    starting_balance: float
    final_balance: float
    roi_pct: float


def compute_brier_score(
    decisions: list[DecisionRecord],
    outcomes: dict[str, Outcome],
) -> float:
    """Compute Brier score for a set of decisions.

    For each non-SKIP decision:
    - BUY_YES → forecast = 1.0 (persona predicts YES)
    - BUY_NO → forecast = 0.0 (persona predicts NO)
    - Actual: 1.0 if YES, 0.0 if NO

    Brier = mean((forecast - actual)^2)
    Lower is better. Perfect = 0.0, worst = 1.0.
    """
    scores = []
    for d in decisions:
        if d.action == Action.SKIP:
            continue
        if d.event_ticker not in outcomes:
            continue

        forecast = 1.0 if d.action == Action.BUY_YES else 0.0
        actual = 1.0 if outcomes[d.event_ticker] == Outcome.YES else 0.0
        scores.append((forecast - actual) ** 2)

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def compute_persona_metrics(
    persona_id: str,
    persona_name: str,
    decisions: list[DecisionRecord],
    outcomes: dict[str, Outcome],
    starting_balance: float,
    final_balance: float,
) -> PersonaMetrics:
    """Compute all metrics for one persona."""
    total = len(decisions)
    skips = sum(1 for d in decisions if d.action == Action.SKIP)
    bets = total - skips

    wins = 0
    losses = 0
    for d in decisions:
        if d.action == Action.SKIP:
            continue
        if d.event_ticker not in outcomes:
            continue
        outcome = outcomes[d.event_ticker]
        if (d.action == Action.BUY_YES and outcome == Outcome.YES) or \
           (d.action == Action.BUY_NO and outcome == Outcome.NO):
            wins += 1
        else:
            losses += 1

    accuracy = (wins / bets * 100) if bets > 0 else 0.0
    skip_rate = (skips / total * 100) if total > 0 else 0.0
    brier = compute_brier_score(decisions, outcomes)
    roi = ((final_balance - starting_balance) / starting_balance * 100) if starting_balance > 0 else 0.0

    return PersonaMetrics(
        persona_id=persona_id,
        persona_name=persona_name,
        total_decisions=total,
        total_bets=bets,
        total_skips=skips,
        wins=wins,
        losses=losses,
        accuracy_pct=round(accuracy, 1),
        skip_rate_pct=round(skip_rate, 1),
        brier_score=round(brier, 4),
        starting_balance=starting_balance,
        final_balance=round(final_balance, 2),
        roi_pct=round(roi, 1),
    )
