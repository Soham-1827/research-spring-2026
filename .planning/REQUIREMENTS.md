# Requirements: LLM Prediction Market Ensemble

**Defined:** 2026-02-22
**Core Value:** Determine whether LLMs with diverse behavioral biases, reasoning independently then revealing positions, can model effective prediction market decision-making — and surface which biases lead to better returns.

## v1 Requirements

### Data

- [ ] **DATA-01**: System can load historical Kalshi market data from JSON/CSV into validated `EventSnapshot` models (event description, YES/NO question, market prices at each time window — outcome field excluded)
- [ ] **DATA-02**: System slices each event into multiple time window snapshots (T-7d, T-1d, T-1h before resolution) and serves the correct market price per window
- [ ] **DATA-03**: System supports a contamination check prompt — queries the model to determine if it already knows the outcome of an event before including it in the benchmark dataset
- [ ] **DATA-04**: A 15-event benchmark dataset of historical Kalshi markets is curated and validated, with systematic selection methodology documented

### Personas

- [ ] **PERS-01**: System includes 6 behavioral bias personas: overconfident, risk-averse, recency-biased, base-rate-focused, contrarian, and anchoring-biased — each implemented as a distinct system prompt
- [ ] **PERS-02**: Persona system prompts are defined in Jinja2 templates — editable without modifying Python code
- [ ] **PERS-03**: Persona definitions (name, bias type, system prompt template) are stored in YAML/JSON config — new personas can be added without Python changes

### Simulation

- [ ] **SIML-01**: Blind phase: all 6 personas receive the same market snapshot simultaneously per time window via `asyncio.gather()` — no persona sees another's decision before submitting their own
- [ ] **SIML-02**: Each persona's LLM call returns a structured decision via Pydantic schema: `action` (YES / NO / SKIP), `stake_dollars` (0.0 if SKIP), and `reasoning` (persona's argument) — guaranteed parseable via OpenAI structured outputs
- [ ] **SIML-03**: All persona decisions per time window are serialized to `decisions.jsonl` immediately after the blind phase — portfolio simulation and reporting can re-run from this file without additional API calls
- [ ] **SIML-04**: Reveal phase log: after each time window's blind phase, all persona positions are logged together in a structured reveal record (persona, action, stake, reasoning, window, event) — stored in `decisions.jsonl`
- [ ] **SIML-05**: Per-market bet summary: at the end of each market, the system displays each persona's stake amount, the time window at which they placed their bet, and whether they won or lost

### Portfolio

- [ ] **PORT-01**: Portfolio tracker maintains a $100 starting balance per persona and applies correct Kalshi pricing math: buying YES at X¢ means stake/X × $1 payout if YES wins; buying NO at (100-X)¢ means stake/(100-X) × $1 payout if NO wins; SKIP is a no-op
- [ ] **PORT-02**: Results are written to `portfolio.csv` showing the running balance per persona after each of the 15 events (balance curve)
- [ ] **PORT-03**: Calibration metrics computed per persona: Brier score, prediction accuracy %, and ROI over the 15-event benchmark
- [ ] **PORT-04**: Skip rate reported separately per persona — total markets skipped and as a percentage of all markets entered

## v2 Requirements

### Ensemble Enhancement

- **ENS-01**: Moderator/synthesis agent — after blind-then-reveal, a neutral LLM synthesizes all positions into a consensus prediction with rationale
- **ENS-02**: Multi-model ensemble — run personas across multiple underlying models (e.g., Claude, GPT, Gemini) to study model capability vs. persona bias

### Interface

- **UI-01**: Web UI dashboard — visualize portfolio curves, persona decisions, and market outcomes in a browser interface

### Expansion

- **EXP-01**: Live Kalshi integration — connect to Kalshi API for real-time prediction market data (after historical validation is complete)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Web search in LLM API calls | Core research constraint — model must not look up outcomes; invalidates experiment |
| Fine-tuning personas | gpt-5-nano does not support fine-tuning; system prompt engineering only |
| Real-time market data ingestion | Research uses static historical data; live API adds complexity without research value in v1 |
| Live trading execution | Research only — no actual money; Kalshi API integration deferred to post-validation |
| Multi-model ensemble (v1) | Confounds model capability with persona bias; establish single-model baseline first |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Pending |
| DATA-02 | Phase 1 | Pending |
| DATA-03 | Phase 1 | Pending |
| DATA-04 | Phase 1 | Pending |
| PERS-01 | Phase 2 | Pending |
| PERS-02 | Phase 2 | Pending |
| PERS-03 | Phase 2 | Pending |
| SIML-01 | Phase 3 | Pending |
| SIML-02 | Phase 3 | Pending |
| SIML-03 | Phase 3 | Pending |
| SIML-04 | Phase 3 | Pending |
| SIML-05 | Phase 3 | Pending |
| PORT-01 | Phase 4 | Pending |
| PORT-02 | Phase 4 | Pending |
| PORT-03 | Phase 4 | Pending |
| PORT-04 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 16 total
- Mapped to phases: 16
- Unmapped: 0 ✓

---
*Requirements defined: 2026-02-22*
*Last updated: 2026-02-22 after initial definition*
