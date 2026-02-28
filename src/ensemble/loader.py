"""Load events from JSON files into validated Event models."""

from __future__ import annotations

import json
from pathlib import Path

from ensemble.models import Event, TimeWindowLabel


def load_events(source: Path) -> list[Event]:
    """Load a JSON file containing events and validate into Event models.

    Accepts two formats:
    - Bare list: [{...}, ...]
    - Wrapped: {"metadata": {...}, "events": [{...}, ...]}

    Raises ValueError if validation fails, including which event_ticker failed.
    """
    path = Path(source)
    with open(path) as f:
        raw = json.load(f)

    if isinstance(raw, list):
        event_dicts = raw
    elif isinstance(raw, dict) and "events" in raw:
        event_dicts = raw["events"]
    else:
        raise ValueError(
            f"Invalid format in {path}: expected a list or dict with 'events' key"
        )

    events: list[Event] = []
    for i, d in enumerate(event_dicts):
        ticker = d.get("event_ticker", f"event[{i}]")
        try:
            # Parse window_prices string keys to TimeWindowLabel enum
            if "window_prices" in d and isinstance(d["window_prices"], dict):
                parsed_prices = {}
                for key, val in d["window_prices"].items():
                    parsed_prices[TimeWindowLabel(key)] = val
                d["window_prices"] = parsed_prices

            events.append(Event.model_validate(d))
        except Exception as e:
            raise ValueError(f"Validation failed for event '{ticker}': {e}") from e

    return events
