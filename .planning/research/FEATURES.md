# Feature Research

**Domain:** LLM multi-agent ensemble / prediction market simulation (research tool)
**Researched:** 2026-02-22
**Confidence:** HIGH

## Feature Landscape

### Table Stakes (Research Validity Requires These)

Features without which the research results are meaningless or the tool is unusable.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Persona system prompt library | Without distinct personas, there's no ensemble — just one LLM run many times | MEDIUM | Define 4-6 behavioral bias personas (overconfident, risk-averse, recency-biased, base-rate-focused, contrarian, anchoring-biased) |
| Market data ingestion | Need historical Kalshi event data as structured input to LLMs | MEDIUM | Event description, YES/NO question, market prices at each time window, known outcome |
| Blind-phase LLM call per persona | Core research mechanic — each persona must decide independently | LOW | Async parallel calls; no persona sees others' positions until reveal |
| Structured decision output | YES / NO / SKIP + stake amount must be parseable | LOW | Use OpenAI structured outputs with Pydantic schema |
| Portfolio tracker | $100 starting balance per persona, updated after each event resolves | MEDIUM | Must correctly implement Kalshi pricing math |
| Multiple time windows per event | T-7d, T-1d, T-1h — persona can bet at any window | MEDIUM | Each window is a separate decision opportunity; persona can change or hold position |
| Results output to file | Research data must be saved for analysis | LOW | CSV for portfolio curves, JSON for reasoning traces |
| Reasoning trace capture | Understanding WHY each persona bet is core to research value | LOW | Save full LLM response (reasoning + decision) per persona per window |
| Skip option | Persona can choose not to bet on a given market/window | LOW | Tracks discipline; prevents forced low-confidence bets |
| No web search in LLM calls | Experiment validity — model must not look up outcomes | LOW | Never pass web_search tool to API; strictly prompt-only |

### Differentiators (Research Quality and Novelty)

Features that make this research more insightful than a naive ensemble.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Reveal phase log | After blind decisions, all positions are logged together — enables future moderator | LOW | Prepares data structure for v2 moderator agent |
| Kalshi-accurate pricing model | YES at X¢ means $1 payout if correct — models real market economics | MEDIUM | Incorrect math invalidates portfolio comparisons |
| Per-event portfolio curve | Track balance after each of 15 events, not just final | LOW | Shows which events each persona wins/loses on |
| Calibration metrics | Brier score, accuracy %, ROI per persona | MEDIUM | Standard prediction market evaluation metrics |
| Configurable event dataset | Swap in different 15-event sets without code changes | LOW | Config-driven event selection |
| Inter-persona variance metric | Measure disagreement between personas per event | LOW | High variance = more interesting/contested markets |
| Temperature=0 reproducibility | Same input → same output; essential for research | LOW | Set in API call config |
| Experiment config snapshot | Save full run config alongside results | LOW | Enables exact reproduction of any experiment |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Live Kalshi trading integration | "Let's make it real" | Legal risk, API complexity, defeats historical evaluation purpose | Historical data only for v1 |
| Web search for LLMs | "Give them more context" | Contaminates experiment — model can find outcomes | Provide pre-event data explicitly in prompt |
| Real-time market data API | "Make it current" | Adds latency, API costs, defeats purpose of controlled historical study | Static CSV/JSON historical dataset |
| Moderator/synthesis agent in v1 | "Get a consensus answer" | Adds complexity before we understand baseline persona behavior | Blind-then-reveal log; add moderator in v2 |
| Multi-model ensemble in v1 | "Use Claude + GPT + Gemini" | Confounds model capability with persona bias — can't isolate the variable | One model, different system prompts only |
| UI dashboard in v1 | "Make it pretty" | Builds interface before validating core logic | CLI output + CSV; dashboard in v2 |
| Fine-tuning personas | "Train on Kalshi data" | gpt-5-nano doesn't support fine-tuning; also changes the research question | System prompt engineering only |

## Feature Dependencies

```
[Market Data Ingestion]
    └──required by──> [Time Window Slicer]
                          └──required by──> [Blind-Phase LLM Call]
                                                └──required by──> [Portfolio Tracker]
                                                                      └──required by──> [Results Output]

[Persona System Prompt Library]
    └──required by──> [Blind-Phase LLM Call]

[Structured Decision Output]
    └──required by──> [Portfolio Tracker]

[Skip Option]
    └──part of──> [Structured Decision Output]

[Reveal Phase Log]
    └──enhances──> [Blind-Phase LLM Call] (post-processing)
    └──enables──> [Moderator Agent (v2)]
```

### Dependency Notes

- **Market data ingestion required before anything:** All LLM calls depend on structured event data being available
- **Structured output required before portfolio tracker:** Portfolio math can only run on parsed decisions, not free-text
- **Time windows require ingestion:** The slicer needs to know what data is available at each snapshot point
- **Calibration metrics require portfolio tracker:** Derived from resolved trade outcomes

## MVP Definition

### Launch With (v1)

Minimum viable research tool — validates the core concept.

- [ ] Persona library (4 behavioral bias personas) — without this there's no ensemble
- [ ] Market data loader (read CSV/JSON with event + time windows + outcome) — need data to run
- [ ] Blind-phase async LLM calls for all personas per time window — core mechanic
- [ ] Structured decision output (YES/NO/SKIP + stake) — parseable results
- [ ] Portfolio tracker with Kalshi pricing math — measure performance
- [ ] CLI runner: `python run.py --events data/events.json --output results/` — usable
- [ ] Results CSV + reasoning trace JSON — analyzable output

### Add After Validation (v1.x)

- [ ] Calibration metrics (Brier score, accuracy %, ROI) — once data is flowing
- [ ] Inter-persona variance metric — after first run reveals patterns
- [ ] Reveal phase structured log — foundation for v2 moderator

### Future Consideration (v2+)

- [ ] Moderator/synthesis agent — after baseline persona behavior is understood
- [ ] Multi-model ensemble — after single-model baseline is established
- [ ] Web UI dashboard — after research direction is validated
- [ ] Live Kalshi integration — if research transitions to real-time prediction

## Feature Prioritization Matrix

| Feature | Research Value | Implementation Cost | Priority |
|---------|----------------|---------------------|----------|
| Persona prompt library | HIGH | LOW | P1 |
| Market data ingestion | HIGH | MEDIUM | P1 |
| Blind-phase LLM calls | HIGH | LOW | P1 |
| Structured decision output | HIGH | LOW | P1 |
| Portfolio tracker | HIGH | MEDIUM | P1 |
| Multiple time windows | HIGH | MEDIUM | P1 |
| CLI runner | HIGH | LOW | P1 |
| Results output (CSV/JSON) | HIGH | LOW | P1 |
| Calibration metrics | MEDIUM | LOW | P2 |
| Reveal phase log | MEDIUM | LOW | P2 |
| Inter-persona variance | MEDIUM | LOW | P2 |
| Experiment config snapshot | HIGH | LOW | P2 |
| Moderator agent | HIGH | HIGH | P3 |
| Web UI dashboard | LOW | HIGH | P3 |

## Sources

- Kalshi prediction market mechanics and pricing structure
- LLM multi-agent ensemble research literature (2024-2025)
- Prediction market evaluation methodology (Brier score, calibration)
- Research tool design principles

---
*Feature research for: LLM ensemble prediction market simulation*
*Researched: 2026-02-22*
