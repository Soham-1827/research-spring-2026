# Architecture Research

**Domain:** LLM multi-agent ensemble / prediction market simulation (Python CLI)
**Researched:** 2026-02-22
**Confidence:** HIGH

## Standard Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI Layer                             │
│  ┌──────────────────────────────────────────────────────┐    │
│  │   run.py  (typer CLI)                                │    │
│  │   --events <file>  --personas <file>  --output <dir> │    │
│  └────────────────────────┬─────────────────────────────┘    │
└───────────────────────────┼─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                    Simulation Engine                          │
│  ┌─────────────────┐   ┌──────────────────────────────────┐  │
│  │  Event Loop     │   │  Time Window Coordinator         │  │
│  │  (15 events)    │──>│  (T-7d → T-1d → T-1h per event) │  │
│  └─────────────────┘   └──────────────┬───────────────────┘  │
└──────────────────────────────────────┼──────────────────────┘
                                        │
         ┌──────────────────────────────▼──────────────────────┐
         │               Blind Phase Orchestrator               │
         │                                                      │
         │  asyncio.gather(                                     │
         │    call_persona(overconfident, market_snapshot),     │
         │    call_persona(risk_averse, market_snapshot),       │
         │    call_persona(recency_biased, market_snapshot),    │
         │    call_persona(base_rate, market_snapshot),         │
         │  )                                                   │
         └──────────────────────────────────────────────────────┘
                │              │              │              │
    ┌───────────▼──┐  ┌────────▼──┐  ┌───────▼──┐  ┌───────▼──┐
    │  Persona A   │  │ Persona B │  │ Persona C│  │ Persona D│
    │ LLM Call     │  │ LLM Call  │  │ LLM Call │  │ LLM Call │
    │ (gpt-5-nano) │  │(gpt-5-nano│  │(gpt-5-nano│  │(gpt-5-nano│
    └──────┬───────┘  └─────┬─────┘  └────┬─────┘  └────┬─────┘
           │                │              │              │
           └────────────────┴──────────────┴──────────────┘
                                    │
                    ┌───────────────▼────────────────┐
                    │     Decision Collector          │
                    │  {persona, action, stake,       │
                    │   reasoning, window, event}     │
                    └───────────────┬────────────────┘
                                    │
         ┌──────────────────────────▼──────────────────────────┐
         │               Portfolio Simulator                    │
         │  Per-persona: balance, trade log, P&L per event     │
         │  Kalshi math: payout = stake / price * $1           │
         └──────────────────────────┬──────────────────────────┘
                                    │
         ┌──────────────────────────▼──────────────────────────┐
         │               Results Reporter                       │
         │  • results/portfolio.csv   (balance curve)          │
         │  • results/decisions.jsonl (full trace)             │
         │  • Console: rich table summary                      │
         └─────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| CLI (run.py) | Entry point, argument parsing, config loading | typer app with `--events`, `--personas`, `--output`, `--dry-run` |
| Event Loader | Read market data files, validate schema | pandas DataFrame or pydantic models from JSON/CSV |
| Time Window Coordinator | For each event, emit market snapshots at each configured window | Filter rows by timestamp offset from resolution date |
| Persona Registry | Load persona definitions (name, system prompt, bias type) | JSON/YAML file → list of Persona pydantic models |
| Prompt Builder | Combine persona system prompt + market snapshot → full API payload | Jinja2 template; keeps prompts editable without code changes |
| Blind Phase Orchestrator | Fire all persona LLM calls in parallel for a given window | `asyncio.gather()` over list of coroutines |
| LLM Caller | Single async call to gpt-5-nano with structured output | `client.beta.chat.completions.parse()` with `PersonaDecision` schema |
| Decision Collector | Aggregate all persona decisions for a window into a list | Simple list append; feeds reveal log and portfolio sim |
| Portfolio Simulator | Track $100 per persona, apply Kalshi payout math, update on resolution | Pure Python; no external dependency needed |
| Results Reporter | Write CSV + JSONL output, render rich CLI table | pandas `.to_csv()`, `jsonlines.open()`, `rich.table.Table` |

## Recommended Project Structure

```
llm-prediction-ensemble/
├── pyproject.toml              # deps, scripts, tool config
├── .env                        # OPENAI_API_KEY (gitignored)
├── data/
│   ├── events/                 # historical event datasets
│   │   └── kalshi_15_events.json
│   └── personas/
│       └── personas.json       # persona definitions
├── prompts/
│   └── persona_prompt.j2       # jinja2 prompt template
├── src/
│   └── ensemble/
│       ├── __init__.py
│       ├── cli.py              # typer CLI entry point
│       ├── models.py           # pydantic schemas (Event, Decision, Persona, Portfolio)
│       ├── loader.py           # data ingestion (events, personas)
│       ├── prompt_builder.py   # jinja2 prompt assembly
│       ├── llm_caller.py       # async OpenAI API call + retry
│       ├── orchestrator.py     # blind phase coordinator, event loop
│       ├── portfolio.py        # portfolio simulator + Kalshi math
│       └── reporter.py         # CSV/JSONL output + rich display
├── results/                    # gitignored; output of runs
│   └── run_20260222_143012/
│       ├── config.json         # full run config snapshot
│       ├── portfolio.csv       # balance curve per persona
│       └── decisions.jsonl     # reasoning trace per decision
└── tests/
    ├── test_portfolio.py       # unit tests for Kalshi math
    ├── test_loader.py
    └── test_orchestrator.py
```

### Structure Rationale

- **data/**: Separated from code; swap event datasets without touching logic
- **prompts/**: Persona prompt templates editable without Python changes; enables rapid persona iteration
- **src/ensemble/**: Importable package; enables unit testing of individual components
- **results/**: Run-stamped directories; each experiment is self-contained and reproducible

## Architectural Patterns

### Pattern 1: Structured Outputs for Decision Parsing

**What:** Use OpenAI's structured outputs feature to constrain persona responses to a typed schema
**When to use:** Always — prevents unparseable free-text responses
**Trade-offs:** Requires pydantic v2; slightly more setup; but eliminates parsing failures entirely

**Example:**
```python
from pydantic import BaseModel
from enum import Enum

class Action(str, Enum):
    YES = "YES"
    NO = "NO"
    SKIP = "SKIP"

class PersonaDecision(BaseModel):
    action: Action
    stake_dollars: float  # 0.0 if SKIP
    reasoning: str        # persona's argument

response = await client.beta.chat.completions.parse(
    model="gpt-5-nano",
    messages=[...],
    response_format=PersonaDecision,
    temperature=0,
)
decision = response.choices[0].message.parsed
```

### Pattern 2: Blind Phase via asyncio.gather

**What:** All persona LLM calls for a given time window fire simultaneously
**When to use:** Always — this is the blind phase; sequential calls would be slower and personas still can't see each other
**Trade-offs:** N concurrent API calls; OpenAI rate limits apply; use tenacity for retries

**Example:**
```python
async def run_blind_phase(personas, market_snapshot):
    tasks = [call_persona(p, market_snapshot) for p in personas]
    decisions = await asyncio.gather(*tasks, return_exceptions=True)
    return [d for d in decisions if not isinstance(d, Exception)]
```

### Pattern 3: EventSnapshot vs Event Type Boundary (Critical for Research Validity)

**What:** Enforce at the type level that LLMs only ever receive `EventSnapshot` (outcome-free), never `Event` (which contains the outcome for settlement).
**When to use:** Always — this is the architectural guarantee that prevents contamination
**Trade-offs:** Slightly more types to maintain; eliminates an entire class of contamination bugs

**Example:**
```python
class EventSnapshot(BaseModel):
    """LLM-safe: no outcome field. This is what goes into prompts."""
    event_id: str
    question: str
    description: str
    yes_price_cents: float  # market price at this time window
    context: str            # pre-event stats/narrative

class Event(BaseModel):
    """Full event including outcome — used for settlement ONLY, never passed to LLM."""
    snapshot: EventSnapshot
    outcome: str            # "YES" or "NO"
    resolution_date: str

def build_prompt(persona: Persona, snapshot: EventSnapshot) -> list[dict]:
    """Type signature enforces: only EventSnapshot goes in, never Event."""
    ...
```

### Pattern 4: Serializable Pipeline Stages

**What:** Save `decisions.jsonl` after the LLM call phase so portfolio simulation and reporting can re-run without re-calling the API
**When to use:** Always — 4 personas × 3 windows × 15 events = 180 API calls; debugging the portfolio math shouldn't cost API tokens
**Trade-offs:** Adds a file write step; trivially worth it

### Pattern 5: Kalshi Pricing Math

**What:** Correctly model YES/NO contract economics
**When to use:** Every trade resolution
**Trade-offs:** Simple arithmetic but easy to get wrong; unit test exhaustively

**Logic:**
```python
def resolve_trade(action: str, stake: float, yes_price_cents: float, outcome: str) -> float:
    """Returns P&L for a resolved trade."""
    if action == "SKIP":
        return 0.0
    yes_price = yes_price_cents / 100  # e.g. 0.65
    no_price = 1 - yes_price           # e.g. 0.35

    if action == "YES":
        contracts = stake / yes_price  # contracts purchased
        return (contracts * 1.0 - stake) if outcome == "YES" else -stake
    elif action == "NO":
        contracts = stake / no_price
        return (contracts * 1.0 - stake) if outcome == "NO" else -stake
```

## Data Flow

### Simulation Flow

```
CLI args (events file, personas file, output dir)
    ↓
Load events → List[Event] with time windows + outcome
Load personas → List[Persona] with system prompts
    ↓
For each Event (15 total):
    For each TimeWindow (T-7d, T-1d, T-1h):
        Build market_snapshot (event context + price at this window)
        ↓
        Blind Phase:
            asyncio.gather → [PersonaDecision × N personas]
        ↓
        Reveal Phase:
            Log all decisions together (persona, action, stake, reasoning)
    ↓
    Event Resolves:
        For each persona's last decision on this event:
            portfolio.resolve(decision, outcome) → update balance
        Log event P&L per persona
↓
All 15 events done:
    Write portfolio.csv (persona × event balance curve)
    Write decisions.jsonl (full reasoning trace)
    Print rich summary table
```

### Key Data Flows

1. **Market snapshot assembly:** Event description + YES/NO question + current market price (at window) + pre-event context stats → fed to Jinja2 template → LLM system+user messages. Outcome is NEVER included.
2. **Portfolio resolution:** On event close, look up each persona's last committed position for that event, apply Kalshi math, update running balance.
3. **Reveal log:** After blind phase, all decisions for a window are collected into a `RevealLog` object — stored in JSONL for future moderator use.

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| 4-6 personas, 15 events | Single process asyncio — current design is sufficient |
| 10+ personas | Monitor OpenAI rate limits; add semaphore to cap concurrent calls |
| 100+ events | Add progress persistence; resume from checkpoint if interrupted |
| Multiple model providers | Add LLMProvider abstraction layer over caller; inject at runtime |

## Anti-Patterns

### Anti-Pattern 1: Sequential Persona Calls

**What people do:** Call personas one by one in a loop: `for p in personas: call(p)`
**Why it's wrong:** 4 personas × 3 windows × 15 events = 180 sequential API calls; slow and unnecessary
**Do this instead:** `asyncio.gather()` for all personas within a window; still sequential across windows

### Anti-Pattern 2: Free-Text Decision Parsing

**What people do:** Ask the LLM to say "I choose YES, betting $30" and parse with regex
**Why it's wrong:** LLMs produce inconsistent formatting; parsing breaks on edge cases
**Do this instead:** OpenAI structured outputs with Pydantic schema; zero parsing code needed

### Anti-Pattern 3: Hardcoded Persona Prompts

**What people do:** Embed system prompts directly in Python source code
**Why it's wrong:** Impossible to iterate on personas without touching code; no separation of concerns
**Do this instead:** Persona definitions in JSON/YAML + Jinja2 prompt templates

### Anti-Pattern 4: Including Outcome in Context

**What people do:** Accidentally include resolution date, final price, or outcome in market snapshot
**Why it's wrong:** Completely invalidates the experiment
**Do this instead:** Explicitly strip outcome fields before building market snapshot; validate in unit test

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| OpenAI API | openai Python SDK, async client | Use `AsyncOpenAI`; set `timeout=60` for long reasoning responses |
| Kalshi historical data | Static file (CSV/JSON) downloaded once | Kalshi provides market history export; no live API needed |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| Loader → Orchestrator | List[Event] pydantic models | Validated at load time; orchestrator trusts data is clean |
| Orchestrator → LLM Caller | MarketSnapshot + Persona pydantic objects | Prompt builder is called inside caller |
| LLM Caller → Portfolio | PersonaDecision pydantic model | Structured output guarantees type safety |
| Portfolio → Reporter | DataFrame / list of TradeRecord | Portfolio exports; reporter doesn't modify portfolio state |

## Sources

- OpenAI async SDK patterns and structured outputs documentation
- asyncio concurrency patterns for IO-bound tasks
- Kalshi prediction market mechanics
- Multi-agent LLM system design patterns (2024-2025)

---
*Architecture research for: LLM ensemble prediction market simulation*
*Researched: 2026-02-22*
