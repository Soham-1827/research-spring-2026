# Roadmap: LLM Prediction Market Ensemble

## Overview

This roadmap delivers a Python CLI research tool that simulates prediction market trading by an ensemble of bias-persona LLMs. The build follows data dependencies: validated market data and models first, then persona prompt engineering, then async simulation orchestration, and finally portfolio math and evaluation metrics. Each phase delivers a coherent, independently testable capability.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Data Foundation** - Market data models, ingestion pipeline, 15-event benchmark dataset with contamination checks
- [ ] **Phase 2: Persona Engine** - Behavioral bias persona definitions, Jinja2 prompt templates, config-driven persona registry
- [ ] **Phase 3: Simulation Engine** - Blind-phase async LLM orchestration, structured decision output, decision serialization, reveal logging
- [ ] **Phase 4: Portfolio & Evaluation** - Kalshi pricing math, portfolio tracking, calibration metrics, results reporting

## Phase Details

### Phase 1: Data Foundation
**Goal**: Researcher can load, validate, and inspect historical Kalshi market data with confidence that no outcome data leaks into LLM context
**Depends on**: Nothing (first phase)
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04
**Success Criteria** (what must be TRUE):
  1. Running `python -m ensemble load` ingests a Kalshi JSON/CSV file and prints validated EventSnapshot records with no outcome fields present
  2. Each event produces 3 time window snapshots (T-7d, T-1d, T-1h) with correct market prices per window
  3. A contamination check prompt can be run against the LLM for any event, and the result indicates whether the model likely knows the outcome
  4. A curated 15-event benchmark dataset exists with documented selection methodology, and all events pass schema validation
**Plans**: TBD

Plans:
- [ ] 01-01: TBD
- [ ] 01-02: TBD

### Phase 2: Persona Engine
**Goal**: Researcher can define, inspect, and modify behavioral bias personas without touching Python code
**Depends on**: Phase 1
**Requirements**: PERS-01, PERS-02, PERS-03
**Success Criteria** (what must be TRUE):
  1. Six distinct personas (overconfident, risk-averse, recency-biased, base-rate-focused, contrarian, anchoring-biased) are defined and loadable from config
  2. Each persona's system prompt is rendered from a Jinja2 template that accepts market context variables
  3. Adding a new persona requires only a YAML/JSON entry and a prompt template file -- no Python changes needed
**Plans**: TBD

Plans:
- [ ] 02-01: TBD

### Phase 3: Simulation Engine
**Goal**: Researcher can run blind-then-reveal simulation rounds where all personas independently analyze market snapshots and produce structured, serialized decisions
**Depends on**: Phase 1, Phase 2
**Requirements**: SIML-01, SIML-02, SIML-03, SIML-04, SIML-05
**Success Criteria** (what must be TRUE):
  1. All 6 personas receive the same market snapshot simultaneously via async parallel calls -- no persona sees another's decision before submitting
  2. Each persona returns a parseable structured decision (YES/NO/SKIP + stake + reasoning) via Pydantic schema
  3. All decisions are serialized to `decisions.jsonl` immediately after each blind phase -- portfolio simulation can re-run from this file without API calls
  4. After each time window, a reveal record logs all persona positions together (persona, action, stake, reasoning, window, event)
  5. At the end of each market, a per-market bet summary displays each persona's stake, entry window, and win/loss outcome
**Plans**: TBD

Plans:
- [ ] 03-01: TBD
- [ ] 03-02: TBD

### Phase 4: Portfolio & Evaluation
**Goal**: Researcher can evaluate which behavioral biases lead to profitable prediction market behavior through correct portfolio math and calibration metrics
**Depends on**: Phase 3
**Requirements**: PORT-01, PORT-02, PORT-03, PORT-04
**Success Criteria** (what must be TRUE):
  1. Portfolio tracker applies correct Kalshi pricing: YES at X cents means stake/X * $1 payout if YES wins; NO at (100-X) cents means stake/(100-X) * $1 payout if NO wins; SKIP is a no-op
  2. `portfolio.csv` shows running balance per persona after each of the 15 events (balance curve from $100 starting balance)
  3. Calibration metrics (Brier score, prediction accuracy %, ROI) are computed and displayed per persona
  4. Skip rate is reported separately per persona -- total markets skipped and as a percentage of all markets
**Plans**: TBD

Plans:
- [ ] 04-01: TBD
- [ ] 04-02: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Data Foundation | 0/TBD | Not started | - |
| 2. Persona Engine | 0/TBD | Not started | - |
| 3. Simulation Engine | 0/TBD | Not started | - |
| 4. Portfolio & Evaluation | 0/TBD | Not started | - |
