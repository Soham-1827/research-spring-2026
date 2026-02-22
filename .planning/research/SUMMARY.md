# Project Research Summary

**Project:** LLM Prediction Market Ensemble
**Domain:** LLM multi-agent simulation / prediction market research tool (Python CLI)
**Researched:** 2026-02-22
**Confidence:** HIGH

## Executive Summary

This is a Python research tool, not a web application. The architecture is a sequential pipeline with one critical concurrency point — parallel LLM calls per time window (blind phase). The recommended stack is lean: OpenAI Python SDK with async support, Pydantic v2 for structured outputs, pandas for data analysis, and rich/typer for CLI. The most important architectural decision is the `EventSnapshot` vs `Event` type boundary — enforcing at the type level that LLMs only ever see outcome-free snapshots is the core guarantee of research validity.

The biggest risk in this project is not technical — it's scientific: GPT-5-nano's training data almost certainly includes outcomes for well-known historical events, making "prediction" on popular events meaningless. Event selection must be systematic and include contamination checks. The second major risk is persona collapse — without strong, behaviorally-specific system prompts, all personas converge to identical decisions, defeating the purpose of the ensemble. Both risks must be addressed before any results can be trusted.

Build order follows data dependencies: data models first (no API keys needed), then orchestration and LLM calls, then portfolio math (can parallelize with orchestration using fixture data), then reporting and evaluation metrics. Serializing decisions after the LLM phase is essential — it allows portfolio math and reporting to iterate without re-spending API tokens.

## Key Findings

### Recommended Stack

The right stack is minimal and research-focused. The core is the OpenAI Python SDK (>=1.60) with `AsyncOpenAI` for parallel persona calls and structured outputs via Pydantic v2 — this eliminates all decision parsing code. Pandas handles market data ingestion and results analysis. Typer provides a clean CLI, rich provides readable terminal output, and tenacity handles API retry logic.

**Core technologies:**
- `openai` >=1.60 + `AsyncOpenAI`: LLM calls to gpt-5-nano; structured outputs require async client
- `pydantic` v2: `PersonaDecision` schema — YES/NO/SKIP + stake; guaranteed parseable response
- `pandas` 2.x: Market data loading, results CSV output, portfolio curve analysis
- `asyncio` (stdlib): `gather()` for parallel persona calls within each time window
- `tenacity` >=8.0: Retry on rate limit / transient API errors — essential for 180-call runs

**Avoid:** LangChain (unnecessary abstraction), requests directly (SDK handles auth/retry), any web_search tool in API calls (contaminates experiment).

### Expected Features

The v1 must validate the core research mechanic: blind-then-reveal decisions across multiple time windows, tracked as a portfolio simulation.

**Must have (table stakes for research validity):**
- Persona system prompt library (4-6 behavioral bias personas) — without diversity, no ensemble
- Market data ingestion (Kalshi CSV/JSON → structured EventSnapshot) — no outcomes in LLM context
- Blind-phase async LLM calls — all personas decide simultaneously per window
- Structured decision output (YES/NO/SKIP + stake) — Pydantic schema, no free-text parsing
- Portfolio tracker with correct Kalshi pricing math — unit-tested exhaustively
- Multiple time windows per event (T-7d, T-1d, T-1h) — core research variable
- Results output: `portfolio.csv` + `decisions.jsonl` — analysis requires persistent data
- CLI runner with config-driven event/persona selection

**Should have (research quality):**
- Contamination check prompts per event — validates scientific integrity
- Calibration metrics (Brier score, accuracy %, ROI per persona)
- Inter-persona variance metric — low variance = persona prompts need work
- Experiment config snapshot saved with each run — reproducibility

**Defer (v2+):**
- Moderator/synthesis agent — after baseline persona behavior is understood
- Multi-model ensemble (Claude, GPT, Gemini) — after single-model baseline established
- Web UI dashboard — after research direction validated

### Architecture Approach

The system is a 6-component pipeline with one async fan-out. The CLI loads events and personas, the simulation engine loops over 15 events and 3 time windows each, the blind phase orchestrator fires all persona LLM calls in parallel via `asyncio.gather()`, decisions are collected and serialized to `decisions.jsonl`, the portfolio simulator resolves each event and updates balances, and the reporter writes `portfolio.csv` and renders a rich summary table.

**Major components:**
1. **Data Loader** — reads Kalshi JSON/CSV into `EventSnapshot` models (outcome-free); validates schema
2. **Persona Registry** — loads persona definitions (name, system prompt, bias description) from YAML/JSON
3. **Blind Phase Orchestrator** — `asyncio.gather()` over all persona coroutines per time window
4. **LLM Caller** — single async call with `response_format=PersonaDecision`; tenacity retry wrapper
5. **Portfolio Simulator** — Kalshi pricing math; $100 starting balance per persona; unit-tested
6. **Results Reporter** — `portfolio.csv`, `decisions.jsonl`, rich CLI table summary

### Critical Pitfalls

1. **Training data contamination** — GPT-5-nano knows outcomes of major events. Use a contamination check prompt before including any event; prefer niche/low-profile markets; treat suspiciously high accuracy as a warning
2. **Persona collapse** — vague bias instructions produce identical decisions. Make personas behaviorally specific ("never bet more than $10, always skip if favorite is <70¢"); measure inter-persona variance as a metric
3. **Kalshi portfolio math errors** — YES at 65¢ means payout = stake/0.65 × $1; unit test all 5 cases (WIN/LOSS × YES/NO + SKIP) before trusting any results
4. **Context contamination via late prices** — market prices converge to 0¢ or 100¢ near resolution; validate all window prices are between 5¢ and 95¢
5. **Skip rate gaming** — a persona that always skips never loses; report skip rate separately, track participation rate

## Implications for Roadmap

### Phase 1: Foundation — Data Models, Ingestion & CLI
**Rationale:** Everything depends on clean data and the `EventSnapshot`/`Event` type boundary. No API keys or external dependencies needed yet — pure Python data modeling and file I/O.
**Delivers:** `Event`, `EventSnapshot`, `Persona`, `PersonaDecision` models; Kalshi data loader; CLI skeleton; validated 15-event dataset with contamination checks
**Addresses:** Market data ingestion, contamination guard, CLI runner, event selection bias
**Avoids:** Context contamination (enforced at type level), reproducibility failures (config snapshot)

### Phase 2: Persona Engine & LLM Orchestration
**Rationale:** Persona design is the experimental variable — get it right before running at scale. Serialize decisions so Phase 3 can iterate without re-spending tokens.
**Delivers:** Persona library (4-6 bias personas), prompt builder (Jinja2), async LLM caller with structured outputs, blind-phase orchestrator, `decisions.jsonl` serialization
**Uses:** `openai`, `pydantic`, `asyncio`, `tenacity`, `jinja2`
**Avoids:** Persona collapse (behavioral specificity in prompts), parsing failures (structured outputs)

### Phase 3: Portfolio Simulator & Evaluation
**Rationale:** Can be developed in parallel with Phase 2 using fixture `decisions.jsonl` data. Kalshi math must be unit-tested before any results are trusted.
**Delivers:** Portfolio tracker, Kalshi pricing math, P&L per event, portfolio curve, calibration metrics (Brier score, accuracy %, ROI), skip rate reporting, results reporter
**Implements:** Portfolio Simulator + Results Reporter components
**Avoids:** Portfolio math errors, skip rate gaming

### Phase 4: Full Simulation Run & Analysis
**Rationale:** Integration of all components; run full 15-event simulation end-to-end; validate scientific integrity.
**Delivers:** Complete CLI simulation run, results analysis across all personas, inter-persona variance metrics, research findings report
**Avoids:** Event selection bias (systematic dataset), reproducibility failures (config snapshots verified)

### Phase Ordering Rationale

- Phase 1 before everything: the `EventSnapshot` type boundary must be established before any LLM call code is written — it's load-bearing for research validity
- Phase 3 can parallelize with Phase 2: portfolio math depends only on the `PersonaDecision` schema (defined in Phase 1), not on actual LLM calls
- Phase 4 is integration: no new components, just wiring and validation

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 2:** Prompt engineering for behavioral bias personas — no established literature on optimal Kalshi-specific persona prompts; will require iteration
- **Phase 1:** Kalshi historical data availability and format — verify data export format before building ingestion layer

Phases with standard patterns (skip research-phase):
- **Phase 3:** Kalshi pricing math is well-documented; portfolio simulation is pure arithmetic
- **Phase 4:** Integration and CLI — standard Python patterns

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | openai SDK, pydantic, asyncio — all well-established; gpt-5-nano confirmed to support structured outputs |
| Features | HIGH | Core mechanic is clear; feature list derived from research design requirements |
| Architecture | HIGH | Pipeline architecture is standard; EventSnapshot boundary is the key insight |
| Pitfalls | HIGH | Contamination and persona collapse are well-documented failure modes in LLM research |

**Overall confidence:** HIGH

### Gaps to Address

- **Kalshi historical data format:** Verify actual export schema before Phase 1 data loader implementation
- **gpt-5-nano async structured output API:** Confirm exact `client.beta.chat.completions.parse()` syntax for this model during Phase 2 planning
- **Optimal time window selection:** T-7d/T-1d/T-1h is a starting hypothesis; validate that Kalshi markets have sufficient price data at these offsets

## Sources

### Primary (HIGH confidence)
- OpenAI API docs (gpt-5-nano model page) — confirmed structured outputs, 400K context, no web search restriction needed at SDK level
- Kalshi prediction market mechanics — pricing structure, YES/NO contract math

### Secondary (MEDIUM confidence)
- LLM benchmark contamination research (2023-2025) — training data overlap failure modes
- Multi-agent LLM ensemble literature — persona collapse patterns

### Tertiary (LOW confidence)
- Kalshi historical data export availability — verify directly with Kalshi API/export tools

---
*Research completed: 2026-02-22*
*Ready for roadmap: yes*
