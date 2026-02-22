# LLM Prediction Market Ensemble

## What This Is

A research system that simulates prediction market trading by an ensemble of LLMs, each embodying a distinct behavioral bias persona. Given historical Kalshi market data across multiple time windows (no outcome knowledge), each persona independently decides whether to BUY YES, BUY NO, or SKIP — and chooses their stake from a $100 portfolio. Results are evaluated against actual outcomes across 15 historical events to determine which cognitive styles lead to profitable prediction market behavior.

## Core Value

Determine whether LLMs with diverse behavioral biases, reasoning independently then revealing positions, can model effective prediction market decision-making — and surface which biases lead to better returns.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Ingest historical Kalshi market data (event description, question, market prices at multiple time windows)
- [ ] Define behavioral bias personas as system prompts (e.g., overconfident, risk-averse, recency-biased, base-rate-focused)
- [ ] Blind-then-reveal mechanism: each persona independently analyzes each time window before seeing others' reasoning
- [ ] Persona decision output: BUY YES / BUY NO / SKIP + stake amount from $100 starting portfolio
- [ ] Multiple time windows per event (e.g., T-7d, T-1d, T-1h before resolution)
- [ ] Portfolio tracker: running balance per persona across 15 events, accounting for Kalshi-style YES/NO pricing
- [ ] Reasoning trace capture: full LLM output (argument + decision + stake) stored per persona per window
- [ ] CLI runner: execute a full simulation run from command line, output results to console and file
- [ ] Results report: per-persona portfolio curve, decision log, aggregate accuracy

### Out of Scope

- Web search in LLM calls — prevents outcome contamination, core to research integrity
- Moderator/synthesis agent — v2 after baseline personas are validated
- Web UI dashboard — v2
- Live Kalshi trading integration — research only, historical data
- Multi-model ensemble (GPT-4, Gemini) — v2; v1 uses one model with different system prompts

## Context

- Research project: evaluating LLM collective intelligence in prediction markets
- Kalshi is a regulated US prediction market; historical event data includes market prices over time with known outcomes
- Behavioral bias personas are the primary variable — same model, same data, different cognitive framing
- Python CLI is the target interface for running experiments
- "Blind then reveal" mirrors how real traders form independent views before seeing market flow
- 15 historical events chosen as the benchmark dataset; mix of sports, politics, finance
- No ground truth should be present in LLM context — simulate pre-event information environment only

## Constraints

- **Data integrity**: LLM calls must never include outcome data — strict pre-event context boundary
- **No web search**: LLM tool use must be disabled or absent; all context provided via prompt
- **Model**: OpenAI `gpt-5-nano` (snapshot: `gpt-5-nano-2025-08-07`) — 400K context window, $0.05/1M input tokens, differentiated only by system prompt persona
- **Portfolio math**: Must reflect real Kalshi pricing mechanics (YES at X¢ means $1 payout if correct)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| One model + system prompts (not multi-model) | Isolates persona bias as the variable; removes model capability confound | — Pending |
| Behavioral biases as persona type | More interesting for research than domain experts; maps to real trader psychology | — Pending |
| Multiple time windows per event | Lets us study whether personas enter at different times and what that means for returns | — Pending |
| Historical Kalshi data (not synthetic) | Real market data = real pricing, real complexity, grounded evaluation | — Pending |
| Skip option for personas | Reflects real trading discipline; prevents forced low-confidence bets | — Pending |

---
*Last updated: 2026-02-22 after initialization*
