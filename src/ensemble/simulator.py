"""Simulation engine: blind-phase orchestration, reveal logging, JSONL serialization.

Core flow per event per window:
1. Create EventSnapshot (LLM-safe, no outcome)
2. Blind phase: asyncio.gather() all persona calls simultaneously
3. Serialize decisions to decisions.jsonl immediately
4. Log reveal record (all positions visible together)

After all windows for an event: log per-market bet summary.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI

from ensemble.llm import call_persona
from ensemble.models import (
    Action,
    DecisionRecord,
    Event,
    PersonaDecision,
    RevealRecord,
    TimeWindowLabel,
)
from ensemble.personas import PersonaConfig


async def run_blind_phase(
    client: AsyncOpenAI,
    personas: list[PersonaConfig],
    event: Event,
    window: TimeWindowLabel,
    balances: dict[str, float],
    model: str = "gpt-5-nano",
) -> list[DecisionRecord]:
    """Run the blind phase: all personas decide simultaneously on one snapshot.

    Args:
        client: Shared async OpenAI client
        personas: All personas to query
        event: Full event (snapshot will be extracted)
        window: Which time window to use
        balances: Current balance per persona ID
        model: LLM model ID

    Returns:
        List of DecisionRecords, one per persona
    """
    snapshot = event.snapshot(window)
    now = datetime.now(timezone.utc)

    # Launch all persona calls in parallel -- no persona sees another's decision
    tasks = [
        call_persona(
            client=client,
            persona=p,
            snapshot=snapshot,
            balance=balances.get(p.id, 100.0),
            model=model,
        )
        for p in personas
    ]
    decisions: list[PersonaDecision] = await asyncio.gather(*tasks)

    records = []
    for persona, decision in zip(personas, decisions):
        record = DecisionRecord(
            event_ticker=event.event_ticker,
            market_ticker=event.market_ticker,
            window=window,
            persona_id=persona.id,
            persona_name=persona.name,
            action=decision.action,
            stake_dollars=decision.stake_dollars,
            reasoning=decision.reasoning,
            yes_price_cents=snapshot.yes_price_cents,
            no_price_cents=snapshot.no_price_cents,
            timestamp=now,
        )
        records.append(record)

    return records


def write_decisions_jsonl(records: list[DecisionRecord], path: Path) -> None:
    """Append decision records to a JSONL file.

    Each line is a JSON object representing one persona's decision.
    File is opened in append mode so multiple blind phases accumulate.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")


def write_reveal_jsonl(reveal: RevealRecord, path: Path) -> None:
    """Append a reveal record to the JSONL file.

    Reveal records are tagged with type='reveal' to distinguish from decisions.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        data = reveal.model_dump(mode="json")
        data["record_type"] = "reveal"
        f.write(json.dumps(data, default=str) + "\n")


def load_decisions_jsonl(path: Path) -> list[DecisionRecord]:
    """Load all decision records from a JSONL file (skips reveal records)."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if data.get("record_type") == "reveal":
                continue
            records.append(DecisionRecord.model_validate(data))
    return records


async def simulate_event(
    client: AsyncOpenAI,
    personas: list[PersonaConfig],
    event: Event,
    balances: dict[str, float],
    output_path: Path,
    model: str = "gpt-5-nano",
    on_decisions: callable = None,
) -> list[DecisionRecord]:
    """Run full simulation for one event across all time windows.

    For each window (T-7d, T-1d, T-1h):
    1. Blind phase: all personas decide simultaneously
    2. Decisions serialized to JSONL
    3. Reveal record logged

    Args:
        client: Shared async OpenAI client
        personas: All personas
        event: The event to simulate
        balances: Current balance per persona ID (mutated in place)
        output_path: Path to decisions.jsonl
        model: LLM model ID
        on_decisions: Optional callback(window, records) for progress reporting

    Returns:
        All decision records for this event
    """
    all_records = []

    for window in TimeWindowLabel:
        if window not in event.window_prices:
            continue

        # Blind phase
        records = await run_blind_phase(
            client=client,
            personas=personas,
            event=event,
            window=window,
            balances=balances,
            model=model,
        )

        # Serialize immediately (SIML-03)
        write_decisions_jsonl(records, output_path)

        # Reveal phase log (SIML-04)
        reveal = RevealRecord(
            event_ticker=event.event_ticker,
            window=window,
            decisions=records,
            timestamp=datetime.now(timezone.utc),
        )
        write_reveal_jsonl(reveal, output_path)

        if on_decisions:
            on_decisions(window, records)

        all_records.extend(records)

    return all_records


async def simulate_all(
    events: list[Event],
    personas: list[PersonaConfig],
    output_path: Path,
    model: str = "gpt-5-nano",
    starting_balance: float = 100.0,
    on_event_start: callable = None,
    on_decisions: callable = None,
    on_event_complete: callable = None,
) -> list[DecisionRecord]:
    """Run the full simulation across all events and personas.

    Args:
        events: All benchmark events
        personas: All personas to simulate
        output_path: Path to decisions.jsonl (will be created/overwritten)
        model: LLM model ID
        starting_balance: Initial balance per persona
        on_event_start: Optional callback(event_index, event)
        on_decisions: Optional callback(window, records)
        on_event_complete: Optional callback(event, all_event_records)

    Returns:
        All decision records across all events
    """
    # Clear output file for fresh run
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    # Initialize balances
    balances = {p.id: starting_balance for p in personas}

    client = AsyncOpenAI()
    all_records = []

    for i, event in enumerate(events):
        if on_event_start:
            on_event_start(i, event)

        event_records = await simulate_event(
            client=client,
            personas=personas,
            event=event,
            balances=balances,
            output_path=output_path,
            model=model,
            on_decisions=on_decisions,
        )

        if on_event_complete:
            on_event_complete(event, event_records)

        all_records.extend(event_records)

    return all_records
