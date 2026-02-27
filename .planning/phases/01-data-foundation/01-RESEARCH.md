# Phase 1: Data Foundation - Research

**Researched:** 2026-02-27
**Domain:** Kalshi market data ingestion, pydantic v2 modeling, contamination detection, benchmark curation
**Confidence:** HIGH

## Summary

Phase 1 establishes the data layer that all subsequent phases depend on. The core challenge is threefold: (1) acquiring and structuring Kalshi historical market data with correct time-windowed price snapshots, (2) enforcing at the type level that outcome data never leaks into LLM-facing models, and (3) curating a 15-event benchmark dataset with documented contamination checks.

Kalshi provides a public REST API (no auth required for read-only market data) with dedicated historical market endpoints and candlestick (OHLC) time series data. The API returns rich market objects with 70+ fields including `result` (settlement outcome), pricing data, and event metadata. The key architectural pattern is **composition over inheritance**: `EventSnapshot` is a separate pydantic model that simply does not contain outcome fields -- it is not derived from `Event` by exclusion, it is a distinct type that only contains what the LLM should see. The `Event` model holds `EventSnapshot` plus the outcome and resolution metadata needed for settlement.

For contamination detection, the recommended approach is a direct prompt-based probe: ask the model to state what it knows about an event's outcome without revealing it, then score confidence. Events where the model demonstrates knowledge of the outcome are excluded from the benchmark. The 15-event dataset should span multiple categories (economics/Fed, weather, sports, politics, pop culture) with systematic selection criteria (resolved between mid-2024 and early 2025, moderate volume, binary YES/NO markets).

**Primary recommendation:** Build a data pipeline that fetches from Kalshi API (historical markets + candlesticks), validates into pydantic models, slices time windows from candlestick data, and exports a curated JSON benchmark file. Use composition-based type separation (EventSnapshot vs Event) as the contamination firewall.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DATA-01 | Load historical Kalshi market data from JSON/CSV into validated EventSnapshot models (outcome excluded) | Kalshi API provides historical market data (GET /historical/markets/{ticker}) and candlestick time series. Pydantic v2 composition pattern ensures EventSnapshot never contains outcome fields. Data can be fetched via API and cached as JSON. |
| DATA-02 | Slice each event into time window snapshots (T-7d, T-1d, T-1h) with correct market price per window | Kalshi candlestick API (GET /series/{series_ticker}/markets/{ticker}/candlesticks) supports period_interval=60 (hourly) and 1440 (daily). Time windows computed by subtracting timedelta from close_time. Price extracted from candlestick OHLC close values. |
| DATA-03 | Contamination check prompt queries model to determine if it knows outcome | Prompt-based detection: ask model directly about event outcome without revealing it. Score response confidence. Flag events where model demonstrates knowledge. Multiple detection techniques available (direct query, completion-based, knowledge probing). |
| DATA-04 | 15-event benchmark dataset curated and validated with documented methodology | Kalshi API supports filtering by category, volume, date range. Select resolved binary markets across 4-5 categories, resolution dates near GPT-5-nano knowledge cutoff. Document selection criteria. All events must pass schema validation and contamination check. |
</phase_requirements>

## Standard Stack

### Core (Phase 1 specific)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pydantic | v2 (>=2.6) | Data models (Event, EventSnapshot, MarketCandle) | Required by OpenAI SDK for structured outputs; strict validation |
| pandas | 2.x | Data manipulation, CSV/JSON I/O, time series operations | Standard for tabular research data at this scale |
| httpx | >=0.27 | HTTP client for Kalshi API calls | Async-capable, used internally by openai SDK already |
| typer | >=0.12 | CLI entry point (`python -m ensemble load`) | Type-annotated CLI with --help |
| rich | >=13.0 | Pretty-print validated records to terminal | Readable inspection output |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| python-dateutil | >=2.9 | Timezone-aware datetime parsing from Kalshi API | Parsing ISO 8601 timestamps with timezone info |
| openai | >=1.60 | Contamination check prompt (gpt-5-nano call) | DATA-03 only; minimal usage in Phase 1 |

### Not Needed Yet (Phase 2+)

| Library | Deferred To | Why Wait |
|---------|-------------|----------|
| jinja2 | Phase 2 | Persona prompt templates not needed for data loading |
| tenacity | Phase 2 | Retry logic matters for bulk LLM calls, not single contamination checks |
| jsonlines | Phase 2 | decisions.jsonl is a Phase 2/3 concern |
| asyncio (heavy use) | Phase 2 | Phase 1 has only sequential API calls for data fetching |

**Installation (Phase 1):**
```bash
uv add pydantic pandas httpx typer rich python-dateutil openai python-dotenv
uv add --dev pytest ruff
```

## Architecture Patterns

### Recommended Project Structure (Phase 1 deliverables)

```
src/
  ensemble/
    __init__.py
    cli.py              # typer CLI: `load`, `validate`, `contamination-check`
    models.py           # Event, EventSnapshot, MarketCandle, TimeWindow
    loader.py           # Kalshi API fetcher + JSON/CSV file loader
    slicer.py           # Time window slicer (T-7d, T-1d, T-1h)
    contamination.py    # Contamination check prompt + scoring
data/
  benchmark/
    events.json         # Curated 15-event benchmark dataset
    methodology.md      # Selection methodology documentation
tests/
  test_models.py        # Schema validation, outcome exclusion
  test_loader.py        # Data loading from file and API
  test_slicer.py        # Time window computation
  test_contamination.py # Contamination check prompt
```

### Pattern 1: Composition-Based Type Boundary (CRITICAL)

**What:** EventSnapshot and Event are separate pydantic models. EventSnapshot does NOT inherit from Event or use field exclusion. It is an independent type that structurally cannot contain outcome data.

**Why this matters:** This is the core research validity guarantee. If outcome data can leak into LLM context, the entire experiment is invalid. Type-level enforcement means a bug cannot accidentally include outcome fields.

**Example:**
```python
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class TimeWindowLabel(str, Enum):
    T_7D = "T-7d"
    T_1D = "T-1d"
    T_1H = "T-1h"

class EventSnapshot(BaseModel):
    """LLM-safe model. This is what goes into prompts.

    INVARIANT: This model NEVER contains outcome, result,
    settlement, or resolution data. If you need those fields,
    use Event instead -- but NEVER pass Event to an LLM.
    """
    event_ticker: str
    market_ticker: str
    title: str
    question: str                    # YES/NO question text
    description: str                 # event context/background
    category: str                    # e.g. "economics", "weather"
    yes_price_cents: int             # market price at this time window
    no_price_cents: int              # 100 - yes_price_cents
    window: TimeWindowLabel          # which time window this snapshot is from
    snapshot_timestamp: datetime     # when this price was observed

class Outcome(str, Enum):
    YES = "yes"
    NO = "no"

class Event(BaseModel):
    """Full event with outcome -- used for settlement ONLY.

    WARNING: Never pass this model or its outcome field to an LLM.
    Use event.snapshot(window) to get the LLM-safe version.
    """
    event_ticker: str
    market_ticker: str
    series_ticker: str
    title: str
    question: str
    description: str
    category: str
    outcome: Outcome                 # settlement result
    close_time: datetime             # market close / resolution time
    open_time: datetime              # market open time
    # Price data per time window -- populated by slicer
    window_prices: dict[TimeWindowLabel, int]  # window -> yes_price_cents

    def snapshot(self, window: TimeWindowLabel) -> EventSnapshot:
        """Create an LLM-safe snapshot for a specific time window.

        This is the ONLY way to create EventSnapshot from Event.
        Outcome is structurally excluded.
        """
        return EventSnapshot(
            event_ticker=self.event_ticker,
            market_ticker=self.market_ticker,
            title=self.title,
            question=self.question,
            description=self.description,
            category=self.category,
            yes_price_cents=self.window_prices[window],
            no_price_cents=100 - self.window_prices[window],
            window=window,
            snapshot_timestamp=self._window_timestamp(window),
        )

    def _window_timestamp(self, window: TimeWindowLabel) -> datetime:
        from datetime import timedelta
        offsets = {
            TimeWindowLabel.T_7D: timedelta(days=7),
            TimeWindowLabel.T_1D: timedelta(days=1),
            TimeWindowLabel.T_1H: timedelta(hours=1),
        }
        return self.close_time - offsets[window]
```

### Pattern 2: Time Window Slicer Using Candlestick API

**What:** Compute T-7d, T-1d, T-1h prices by querying Kalshi candlestick data around the resolution date and finding the closest available price point.

**Example:**
```python
from datetime import datetime, timedelta

def compute_window_timestamps(close_time: datetime) -> dict[TimeWindowLabel, datetime]:
    """Compute exact timestamps for each time window."""
    return {
        TimeWindowLabel.T_7D: close_time - timedelta(days=7),
        TimeWindowLabel.T_1D: close_time - timedelta(days=1),
        TimeWindowLabel.T_1H: close_time - timedelta(hours=1),
    }

def find_nearest_price(
    candlesticks: list[dict],
    target_ts: datetime,
    max_gap_hours: int = 2,
) -> int | None:
    """Find the candlestick closest to target_ts and return yes price in cents.

    Returns None if no candlestick within max_gap_hours of target.
    Uses the 'close' price from the candlestick period.
    """
    best = None
    best_delta = timedelta(hours=max_gap_hours + 1)
    for candle in candlesticks:
        candle_ts = datetime.fromisoformat(candle["end_period_ts"])
        delta = abs(candle_ts - target_ts)
        if delta < best_delta:
            best_delta = delta
            # price.close is in cents
            best = candle["price"]["close_cents"]
    if best_delta <= timedelta(hours=max_gap_hours):
        return best
    return None
```

### Pattern 3: Contamination Check Prompt

**What:** A structured prompt that probes whether the LLM knows an event's outcome without revealing it.

**Example:**
```python
CONTAMINATION_CHECK_SYSTEM = """You are a research assistant helping detect
training data contamination. You will be given a description of a prediction
market event. Your job is to honestly assess whether you have knowledge of
this event's actual outcome from your training data.

Respond with EXACTLY this JSON format:
{
  "knows_outcome": true/false,
  "confidence": "high"/"medium"/"low"/"none",
  "stated_outcome": "string or null",
  "reasoning": "why you think you know or don't know"
}

Be honest. If you know or strongly suspect the outcome, say so.
The scientific integrity of this research depends on your honesty."""

CONTAMINATION_CHECK_USER = """Event: {title}
Question: {question}
Category: {category}
Market close date: {close_date}

Do you know or can you infer the actual outcome of this event?
What was the result?"""
```

**Scoring logic:**
- `knows_outcome=true` AND `confidence="high"` --> EXCLUDE event
- `knows_outcome=true` AND `confidence="medium"` --> FLAG for review
- `knows_outcome=false` OR `confidence="low"/"none"` --> INCLUDE event

### Anti-Patterns to Avoid

- **Inheriting EventSnapshot from Event and excluding fields:** Fragile. Adding a new field to Event could accidentally expose it. Use composition instead.
- **Storing prices as floats:** Kalshi prices are in cents (integers). Use `int` for cents, convert to dollars only for display. Avoids floating-point errors.
- **Hardcoding the 15 events:** Store in a JSON file (`data/benchmark/events.json`). Document selection methodology separately.
- **Fetching live data at runtime:** Phase 1 should fetch once, cache as JSON, and load from file. No live API dependency during simulation runs.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| HTTP client for Kalshi API | Raw urllib/requests | httpx | Async-capable, timeout handling, connection pooling |
| Datetime parsing from ISO 8601 | Manual string splitting | `datetime.fromisoformat()` + python-dateutil | Timezone handling is tricky; dateutil handles all edge cases |
| CLI argument parsing | argparse boilerplate | typer | Type-annotated, auto --help, less code |
| Data validation | Manual dict checks | pydantic v2 BaseModel | Declarative validation, clear error messages, serialization built in |
| Pretty terminal output | print() + manual formatting | rich | Tables, colors, progress bars with zero effort |

## Common Pitfalls

### Pitfall 1: Kalshi API Price Format Confusion
**What goes wrong:** Kalshi API returns prices in TWO formats: integer cents AND fixed-point dollar strings (e.g., `"0.65"`). Using the wrong one leads to 100x price errors.
**Why it happens:** The API response includes both `yes_bid` (cents) and `yes_bid_dollars` (string). Historical market endpoint uses `last_price_dollars` (FixedPointDollars string).
**How to avoid:** Standardize on cents (integers) internally. Parse dollar strings by multiplying by 100 and rounding. Document the convention in a code comment.
**Warning signs:** Prices showing as 0.65 when they should be 65, or 6500 when they should be 65.

### Pitfall 2: Time Window Price Convergence (Context Contamination)
**What goes wrong:** T-1h snapshot shows price at 97 cents or 3 cents, effectively revealing the outcome through the market price.
**Why it happens:** Kalshi markets converge to 0 or 100 cents as resolution approaches. The T-1h window is the highest risk.
**How to avoid:** Validate all window prices are between 5 and 95 cents. Flag events where T-1h price is outside this range. Consider replacing T-1h with T-4h or T-6h if convergence is a systemic problem.
**Warning signs:** T-1h prediction accuracy dramatically exceeds T-7d accuracy across all personas.

### Pitfall 3: Missing Candlestick Data at Window Timestamps
**What goes wrong:** No trading activity at exact T-7d moment means no candlestick exists. The slicer returns None and the event is silently dropped or crashes.
**Why it happens:** Low-volume markets may have gaps of hours or days without trades.
**How to avoid:** Use `find_nearest_price()` with a configurable `max_gap_hours` tolerance. For daily candlesticks (period_interval=1440), a 24-hour gap tolerance is appropriate. Log warnings when using a price more than 1 hour from the target timestamp.
**Warning signs:** Many events have None prices for T-7d (market wasn't active that early).

### Pitfall 4: Timezone Mismatches in Window Computation
**What goes wrong:** close_time is in UTC, but timedelta subtraction is done on naive datetimes. The 1-hour window could be off by hours due to DST or timezone confusion.
**Why it happens:** Kalshi API returns UTC timestamps. Python datetime subtraction works correctly on timezone-aware datetimes but breaks on mixed aware/naive.
**How to avoid:** Always use timezone-aware datetimes (UTC). Parse with `datetime.fromisoformat()` which preserves timezone. Never strip timezone info.
**Warning signs:** T-1h prices don't match what Kalshi website shows at that time.

### Pitfall 5: Contamination Check Prompt Gaming
**What goes wrong:** The LLM "plays dumb" about events it actually knows, or falsely claims knowledge to appear helpful.
**Why it happens:** LLMs are trained to be helpful and may not honestly report their knowledge boundaries. The prompt must encourage honesty.
**How to avoid:** Run contamination check multiple times with varied phrasing. Cross-reference: ask the model to predict the outcome (without telling it you know), then compare with actual outcome. If prediction accuracy is suspiciously high, the model likely has training data contamination.
**Warning signs:** Model claims it doesn't know outcome of major world events (dishonest). Model claims knowledge of events that happened after its training cutoff (hallucination).

### Pitfall 6: Event Selection Bias in Benchmark
**What goes wrong:** Researcher picks 15 "interesting" events, all from one domain, all with clear outcomes. Results don't generalize.
**Why it happens:** Manual selection favors memorable events, which are also the most likely to be in training data.
**How to avoid:** Use systematic criteria: (1) resolved between Aug 2024 - Feb 2025, (2) binary YES/NO only, (3) volume > 1000 contracts, (4) sample 3 events from each of 5 categories, (5) pass contamination check.
**Warning signs:** All events are political or all are sports. All outcomes are YES. All events resolved decisively (no close calls).

## Code Examples

### Kalshi API Data Fetching

```python
import httpx
from datetime import datetime

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"

async def fetch_historical_market(ticker: str) -> dict:
    """Fetch a single historical market by ticker. No auth required."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{KALSHI_BASE}/historical/markets/{ticker}")
        resp.raise_for_status()
        return resp.json()["market"]

async def fetch_candlesticks(
    series_ticker: str,
    market_ticker: str,
    start_ts: int,  # unix timestamp
    end_ts: int,
    period_interval: int = 60,  # 1=minute, 60=hour, 1440=day
) -> list[dict]:
    """Fetch OHLC candlestick data for a market."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{KALSHI_BASE}/series/{series_ticker}/markets/{market_ticker}/candlesticks",
            params={
                "start_ts": start_ts,
                "end_ts": end_ts,
                "period_interval": period_interval,
            },
        )
        resp.raise_for_status()
        return resp.json()["candlesticks"]

async def fetch_event(event_ticker: str) -> dict:
    """Fetch event details including title, category, and markets."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{KALSHI_BASE}/events/{event_ticker}",
            params={"with_nested_markets": "true"},
        )
        resp.raise_for_status()
        return resp.json()["event"]
```

### Benchmark Dataset JSON Structure

```json
{
  "metadata": {
    "created": "2026-02-27",
    "methodology": "Systematic selection: 3 events per category, resolved Aug 2024-Feb 2025, volume > 1000, binary YES/NO, passed contamination check",
    "categories": ["economics", "weather", "sports", "politics", "technology"],
    "total_events": 15,
    "contamination_check_model": "gpt-5-nano",
    "contamination_check_date": "2026-02-27"
  },
  "events": [
    {
      "event_ticker": "FED-24DEC-T",
      "market_ticker": "FED-24DEC-T5050",
      "series_ticker": "FED-24DEC",
      "title": "Fed interest rate decision December 2024",
      "question": "Will the Fed cut rates by 25bp at the December 2024 meeting?",
      "description": "Federal Reserve monetary policy decision...",
      "category": "economics",
      "outcome": "yes",
      "close_time": "2024-12-18T19:00:00Z",
      "open_time": "2024-10-01T00:00:00Z",
      "window_prices": {
        "T-7d": 72,
        "T-1d": 85,
        "T-1h": 93
      },
      "contamination_check": {
        "knows_outcome": true,
        "confidence": "low",
        "passed": true
      }
    }
  ]
}
```

### CLI Entry Point Structure

```python
import typer
from pathlib import Path

app = typer.Typer(name="ensemble", help="LLM Prediction Market Ensemble")

@app.command()
def load(
    source: Path = typer.Argument(..., help="Path to events JSON/CSV file"),
    validate_only: bool = typer.Option(False, help="Only validate, don't print"),
):
    """Load and validate historical Kalshi market data."""
    ...

@app.command()
def contamination_check(
    events_file: Path = typer.Argument(..., help="Path to benchmark events.json"),
    model: str = typer.Option("gpt-5-nano", help="Model for contamination check"),
):
    """Run contamination check on all events in the benchmark dataset."""
    ...

if __name__ == "__main__":
    app()
```

## Kalshi API Reference

### Key Endpoints for Phase 1

| Endpoint | Method | Auth | Purpose |
|----------|--------|------|---------|
| `/historical/markets` | GET | None | List resolved/archived markets with filters |
| `/historical/markets/{ticker}` | GET | None | Get single historical market (70+ fields incl. `result`) |
| `/events/{event_ticker}` | GET | None | Get event metadata (title, category, markets) |
| `/series/{s}/markets/{t}/candlesticks` | GET | None | OHLC time series with configurable interval |

### Kalshi Market Object Key Fields

| Field | Type | Usage in Phase 1 |
|-------|------|------------------|
| `ticker` | string | Market identifier (e.g., "FED-24DEC-T5050") |
| `event_ticker` | string | Parent event grouping |
| `series_ticker` | string | Required for candlestick API path |
| `title` / `yes_sub_title` / `no_sub_title` | string | Event description for LLM context |
| `close_time` | datetime | Resolution date -- anchor for time windows |
| `result` | string ("yes"/"no") | Settlement outcome -- NEVER in EventSnapshot |
| `last_price_dollars` | FixedPointDollars | Last traded price (string, e.g. "0.65") |
| `rules_primary` | string | Market rules description |
| `category` | string | Event category for benchmark diversity |
| `volume_fp` | FixedPointCount | Trading volume for selection criteria |

### Candlestick Response Fields

| Field | Type | Usage |
|-------|------|-------|
| `end_period_ts` | datetime | Period end timestamp |
| `price.close_cents` | int | Closing price for the period (what we use for window prices) |
| `price.open_cents` | int | Opening price |
| `price.high_cents` / `price.low_cents` | int | Range within period |
| `volume` | int | Contracts traded in period |

### Period Intervals

| Value | Meaning | Use Case |
|-------|---------|----------|
| 1 | 1-minute candles | Not needed -- too granular |
| 60 | 1-hour candles | T-1h window: get hourly price near resolution |
| 1440 | 1-day candles | T-7d and T-1d windows: get daily close prices |

## Benchmark Dataset Curation Strategy

### Selection Criteria

1. **Resolution window:** Aug 2024 - Feb 2025 (near GPT-5-nano training cutoff of ~May 2025, so model may have partial knowledge)
2. **Market type:** Binary YES/NO only (no scalar markets)
3. **Volume threshold:** > 1,000 contracts traded (ensures meaningful price discovery)
4. **Price range:** At least one time window has price between 20-80 cents (interesting uncertainty)
5. **Category diversity:** 3 events from each of 5 categories = 15 total

### Recommended Categories

| Category | Example Markets | Contamination Risk |
|----------|----------------|-------------------|
| Economics/Fed | Rate decisions, inflation prints, GDP | MEDIUM -- well-covered in training data |
| Weather | Hurricane landfall, temperature records | LOW -- niche, poorly covered |
| Sports | Game outcomes, player stats | HIGH -- heavily covered |
| Politics | Policy decisions, approval ratings | MEDIUM-HIGH -- well-covered |
| Technology/Culture | Product launches, award shows | MEDIUM -- varies by event profile |

### Contamination Mitigation

- Run contamination check on all candidate events BEFORE finalizing the 15
- Prefer lower-profile events within each category
- If a category has too many contaminated events, replace with a different subcategory
- Document which events were excluded and why
- Target: at least 12 of 15 events pass contamination check with confidence="low" or "none"

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual JSON parsing of LLM output | Pydantic v2 structured outputs via OpenAI SDK | OpenAI SDK 1.x (2024) | Zero parsing code needed |
| Field exclusion via model_dump(exclude=...) | Separate composition-based models | Pydantic v2 best practice | Type-level safety, not runtime exclusion |
| requests for HTTP | httpx (async-native) | httpx 0.24+ (2023) | Async support, better timeout handling |
| pip + requirements.txt | uv + pyproject.toml | uv 0.1+ (2024) | 10-100x faster installs, lockfile support |

## Open Questions

1. **Kalshi API rate limits for unauthenticated requests**
   - What we know: Historical and candlestick endpoints are public (no auth)
   - What's unclear: Rate limits for unauthenticated access; could be as low as 10 req/min
   - Recommendation: Fetch all data once, cache as JSON. Add 1-second delay between API calls during initial fetch. This is a one-time data collection, not a runtime concern.

2. **GPT-5-nano exact training data cutoff**
   - What we know: Claude's knowledge says May 2025, but GPT-5-nano's actual cutoff may differ
   - What's unclear: Exact date, and whether prediction market outcomes are well-represented in training data
   - Recommendation: Use contamination check as empirical validation regardless of stated cutoff. Select events from a range of dates.

3. **Candlestick data availability for older markets**
   - What we know: Candlestick API exists and returns OHLC data
   - What's unclear: Whether candlestick data is retained for markets that resolved 6+ months ago
   - Recommendation: Test with a known resolved market ticker first. If candlesticks are unavailable, fall back to `last_price_dollars` from the historical market endpoint (less granular but sufficient).

4. **T-1h window viability**
   - What we know: Market prices converge near resolution, potentially revealing outcomes
   - What's unclear: How many benchmark events will have T-1h prices in the 5-95 cent range
   - Recommendation: Collect T-1h data but validate price bounds. If >50% of events have convergent T-1h prices, consider T-4h or T-6h as alternative.

## Sources

### Primary (HIGH confidence)
- [Kalshi Historical Market API](https://docs.kalshi.com/api-reference/historical/get-historical-market) - Response fields, no-auth access confirmed
- [Kalshi Historical Markets List](https://docs.kalshi.com/api-reference/historical/get-historical-markets) - Filtering, pagination, 70+ fields
- [Kalshi Market Candlesticks API](https://docs.kalshi.com/api-reference/market/get-market-candlesticks) - OHLC data, period intervals (1/60/1440)
- [Kalshi Get Event API](https://docs.kalshi.com/api-reference/events/get-event) - Event metadata, categories, nested markets
- [Pydantic v2 Models Documentation](https://docs.pydantic.dev/latest/concepts/models/) - Composition patterns, field exclusion

### Secondary (MEDIUM confidence)
- [Pydantic field exclusion patterns](https://github.com/pydantic/pydantic/issues/11099) - Community patterns for nested exclusion
- [LLM Data Contamination Survey](https://arxiv.org/html/2502.17521v2) - Contamination detection techniques
- [Kalshi market categories](https://kalshi.com) - Category diversity confirmed (economics, weather, sports, politics, tech)

### Tertiary (LOW confidence)
- Kalshi API rate limits for unauthenticated access - not explicitly documented, needs empirical testing
- GPT-5-nano training data overlap with prediction market outcomes - needs empirical contamination checks

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - pydantic v2, pandas, httpx are well-established; Kalshi API docs are public
- Architecture (EventSnapshot/Event boundary): HIGH - composition pattern is standard pydantic v2 best practice
- Kalshi API data format: HIGH - verified from official docs, multiple endpoints confirmed
- Time window slicer: MEDIUM - candlestick API confirmed but data availability for old markets needs testing
- Contamination check: MEDIUM - prompt-based detection is established technique but effectiveness varies
- Benchmark curation: MEDIUM - strategy is sound but depends on Kalshi data availability and contamination results

**Research date:** 2026-02-27
**Valid until:** 2026-03-27 (Kalshi API is stable; pydantic v2 is stable)
