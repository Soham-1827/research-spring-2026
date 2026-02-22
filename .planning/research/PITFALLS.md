# Pitfalls Research

**Domain:** LLM multi-agent ensemble / prediction market simulation (research tool)
**Researched:** 2026-02-22
**Confidence:** HIGH

## Critical Pitfalls

### Pitfall 1: LLM Training Data Contamination

**What goes wrong:**
GPT-5-nano's training data includes outcomes of major historical events — sports championships, elections, major financial events. When asked to predict a past event, the model may recall the actual outcome from training rather than reasoning from the provided data. Results look impressive but are scientifically invalid.

**Why it happens:**
Researchers select well-known events for convenience (easy to find data). The model "predicts" correctly not through reasoning but through memorization.

**How to avoid:**
- Prefer Kalshi markets on niche, lower-profile events that are less likely to be in training data
- Include a "contamination check" prompt before running: ask the model directly "Do you know the outcome of [event]?" — if it answers confidently, exclude that event
- Use events with resolution dates close to the model's knowledge cutoff (May 2024) where training coverage is sparse
- Treat high accuracy as a warning sign, not just a success signal

**Warning signs:**
- Model states outcome with very high confidence and no hedging
- Model's reasoning references the outcome indirectly ("given what we now know...")
- All personas agree immediately with no disagreement

**Phase to address:** Phase 1 (Data & Event Selection)

---

### Pitfall 2: Persona Collapse (All Personas Converge)

**What goes wrong:**
Despite different system prompts, all personas produce nearly identical decisions. The ensemble provides no diversity of thought — it's just running the same LLM 4 times.

**Why it happens:**
LLMs have strong priors about "correct" answers. A short bias instruction ("You are overconfident") is often insufficient to override the model's default tendency toward balanced, hedged responses.

**How to avoid:**
- Make persona instructions highly specific and behavioral: instead of "You are risk-averse," write "You have lost money on upsets before. You always bet conservatively — never more than $10, and you skip any market where the favorite isn't at 70¢+ YES"
- Include explicit contrast in prompts: "Other analysts may be too aggressive. Your edge is caution."
- Measure inter-persona variance as a first-class metric — if variance is low, iterate on persona prompts before analyzing results
- Test personas on a single event before running the full 15-event suite

**Warning signs:**
- All personas bet YES/NO on the same side for every event
- Stake amounts are similar across personas
- Skip rates are identical across personas

**Phase to address:** Phase 2 (Persona Design & Prompt Engineering)

---

### Pitfall 3: Kalshi Portfolio Math Errors

**What goes wrong:**
Incorrect implementation of Kalshi pricing leads to wrong P&L calculations. Common errors: treating stake as number of contracts instead of dollars invested, ignoring the price at time of entry, or computing payout incorrectly.

**Why it happens:**
Kalshi YES/NO contracts are priced like binary options (YES at 65¢ = $0.65 per contract, pays $1 if correct). Developers unfamiliar with this model treat it like a simple 1:1 bet.

**How to avoid:**
- Implement portfolio math with explicit unit tests covering: WIN on YES, LOSS on YES, WIN on NO, LOSS on NO, SKIP
- Document the math with a worked example in a docstring:
  ```
  Event: YES at 0.65 (65¢), stake $20
  Contracts bought: 20 / 0.65 = 30.77
  If YES wins: 30.77 * $1 - $20 = +$10.77
  If NO wins: -$20 (lose stake)
  ```
- Run a manual sanity check: a persona that always bets $10 on the favorite should have a specific expected return given the 15-event dataset

**Warning signs:**
- Portfolio balances go negative (impossible — can only lose stake)
- Persona that always skips shows non-zero P&L
- P&L is symmetric across YES/NO bets on the same event

**Phase to address:** Phase 2 (Portfolio Simulator)

---

### Pitfall 4: Context Contamination via Time Window Data

**What goes wrong:**
A time window snapshot accidentally includes data from after the event resolves (e.g., the T-1h snapshot shows the final price, which reveals the outcome in prediction markets because final prices cluster at 0¢ or 100¢ as resolution approaches).

**Why it happens:**
Kalshi market prices converge to 0 or 100 cents in the hours before resolution as traders price in known information. A T-1h snapshot might effectively reveal the outcome.

**How to avoid:**
- Validate that market price at each time window is strictly bounded (e.g., between 5¢ and 95¢ — anything outside this range is suspicious)
- Prefer T-7d and T-1d windows; T-1h is highest risk for late-resolving price convergence
- Add an assertion: if YES price at any window is >90¢ or <10¢, flag the event for manual review

**Warning signs:**
- Market price at T-1h is 95¢+ or 5¢- for most events
- Model's T-1h prediction accuracy dramatically exceeds T-7d accuracy

**Phase to address:** Phase 1 (Data Ingestion)

---

### Pitfall 5: Reproducibility Failures

**What goes wrong:**
Two runs with identical inputs produce different results because temperature > 0, random seeds aren't set, or the run config isn't saved alongside results.

**Why it happens:**
Default LLM temperature introduces randomness. Without logging the exact config, it's impossible to reproduce or verify any finding.

**How to avoid:**
- Always use `temperature=0` for reproducibility in research runs
- Save a `config.json` snapshot with every results directory: model name, temperature, personas file hash, events file hash, timestamp
- Version the personas file and events file (don't modify in place)

**Warning signs:**
- Re-running the same experiment yields different portfolio results
- Can't reconstruct which persona file was used for a result
- Results directory has no config snapshot

**Phase to address:** Phase 1 (CLI & Config)

---

### Pitfall 6: Skip Rate Gaming

**What goes wrong:**
A persona that always SKIPs "wins" by never losing money. Portfolio results show it as top performer, but it provided no predictive signal.

**Why it happens:**
If the system rewards final portfolio balance without penalizing abstention, a skip-everything strategy dominates by default.

**How to avoid:**
- Track and report skip rate per persona separately from portfolio performance
- Consider a "participation rate" metric: minimum % of markets a persona must enter to be compared in the main leaderboard
- Alternatively, apply a small opportunity cost for skipping (e.g., lose $0.50 per skip) — document the choice clearly
- Report both "including skips" and "excluding skips" accuracy/ROI

**Warning signs:**
- One persona has highest balance but >80% skip rate
- Persona with most conservative stake amounts consistently outperforms

**Phase to address:** Phase 3 (Evaluation & Reporting)

---

### Pitfall 7: Event Selection Bias

**What goes wrong:**
Manually selecting 15 "interesting" events introduces survivorship or recency bias — the dataset reflects what the researcher thought was notable, not a representative sample of Kalshi markets.

**Why it happens:**
Researchers pick events they know about or that had clear outcomes. This biases results toward the researcher's prior knowledge and domain.

**How to avoid:**
- Use a systematic selection criterion: e.g., top 15 markets by volume in a given date range, or random sample from a Kalshi export
- Document the selection methodology clearly
- Run experiments on 2-3 different event sets to check result stability

**Warning signs:**
- All 15 events are from the same domain (all sports, or all political)
- All events resolve clearly (no close calls) — real markets have many near-50% outcomes

**Phase to address:** Phase 1 (Data & Event Selection)

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Hardcode personas in Python | Faster to write | Can't iterate without code changes | Never — use JSON config |
| No unit tests for portfolio math | Ship faster | Silent math errors corrupt all results | Never — test this first |
| Single monolithic script | Simple at start | Impossible to test components in isolation | Prototype only, refactor before analysis |
| Skip retry logic on API calls | Less code | Experiment fails mid-run on transient error, wasting time | Never |
| No results directory versioning | Simpler output | Old results overwritten; can't compare runs | Never |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| OpenAI structured outputs | Using `response_format={"type": "json_object"}` (old API) | Use `client.beta.chat.completions.parse(response_format=PersonaDecision)` with Pydantic model |
| OpenAI async client | Using sync client in async code | Instantiate `AsyncOpenAI()` not `OpenAI()` |
| Kalshi data export | Assuming consistent column names across exports | Validate schema at load time with pydantic |
| tenacity retry | Retrying on ALL exceptions | Only retry on `openai.RateLimitError` and `openai.APIConnectionError`; fail fast on auth errors |

## "Looks Done But Isn't" Checklist

- [ ] **Portfolio math:** Verify with a manually-computed expected outcome for a known event before trusting results
- [ ] **Contamination guard:** Test each event with a "do you know the outcome?" prompt before including in dataset
- [ ] **Persona diversity:** Check inter-persona agreement rate — if >80% agreement, personas need stronger differentiation
- [ ] **Outcome exclusion:** Confirm outcome field is never present in market_snapshot passed to LLM (grep for "outcome" in prompt builder)
- [ ] **Time window prices:** Verify no window shows price <5¢ or >95¢ — flag for review if so
- [ ] **Reproducibility:** Run same experiment twice with temperature=0, verify identical results

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Training data contamination | Phase 1: Data & Event Selection | Contamination check prompt for each event |
| Persona collapse | Phase 2: Persona Design | Measure inter-persona variance on test event |
| Portfolio math errors | Phase 2: Portfolio Simulator | Unit tests cover all 5 cases (WIN/LOSS × YES/NO + SKIP) |
| Context contamination via late prices | Phase 1: Data Ingestion | Validate price bounds in loader |
| Reproducibility failures | Phase 1: CLI & Config | Re-run test, compare results |
| Skip rate gaming | Phase 3: Evaluation | Report skip rate alongside portfolio balance |
| Event selection bias | Phase 1: Data & Event Selection | Document selection methodology |

## Sources

- LLM benchmark contamination research (2023-2025) — documented in NLP literature
- Prediction market mechanics: Kalshi market rules and pricing structure
- Multi-agent LLM system failure modes (2024 research)
- Research reproducibility standards in ML

---
*Pitfalls research for: LLM ensemble prediction market simulation*
*Researched: 2026-02-22*
