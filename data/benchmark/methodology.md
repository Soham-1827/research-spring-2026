# Benchmark Dataset Selection Methodology

## Overview

This document describes how the 15-event benchmark dataset was curated for the LLM Prediction Market Ensemble research project.

## Selection Criteria

1. **Resolution window**: Events resolved between July 2024 and February 2026
2. **Market type**: Binary YES/NO prediction markets only
3. **Volume**: Prefer markets with significant trading volume (>500 contracts where verifiable)
4. **Price range**: Window prices between 5-95 cents (genuine uncertainty, not near-certain outcomes)
5. **Category diversity**: Minimum 4 categories, target 5, with 3 events per category
6. **Time horizon**: Events must be open long enough for T-7d window to be meaningful (at least 2 weeks from open to close)

## Categories (5 categories, 3 events each)

| Category | Events | Source |
|----------|--------|--------|
| Economics | Q3 GDP >2.9%, Q3 GDP >2.4%, Q2 GDP >2.4% | Kalshi API (confirmed tickers) |
| Politics | 2024 Presidential, Senate control, FOMC Sep 50bp cut | Kalshi markets (well-documented) |
| World | Next Pope, G7 leader departure, Klarna vs Stripe IPO | Kalshi API (confirmed tickers) |
| Sports | Super Bowl LX, Giannis trade deadline, Chiefs SB appearance | Kalshi markets (reported in media) |
| Entertainment | Oscar Best Picture, Grammy AOTY, Golden Globe Best Drama | Kalshi markets |

## Data Sources

- **Primary**: Kalshi public API (`api.elections.kalshi.com/trade-api/v2`)
  - GDP market tickers confirmed directly via API
  - Pope, G7, Klarna tickers confirmed via API
- **Secondary**: Kalshi market reporting and financial media
  - Super Bowl volume ($871M reported by Fortune)
  - Giannis trade market ($23M reported)
  - Awards markets documented on kalshi.com
- **Window prices**: Approximated from last traded prices and market history
  - GDP markets: prices from Kalshi API response
  - Other markets: estimated from reported odds at different time points
  - All prices represent YES contract price in cents

## Window Price Methodology

- **T-7d**: Market price approximately 7 days before close/resolution
- **T-1d**: Market price approximately 1 day before close/resolution
- **T-1h**: Market price approximately 1 hour before close/resolution
- Prices are integer cents (e.g., 55 = $0.55 YES contract price)
- NO price = 100 - YES price (always)

**Important**: Window prices for non-GDP events are approximate. For production research, these should be refined using the Kalshi candlestick API (`/series/{s}/markets/{t}/candlesticks`). The approximate prices are reasonable for validating the research pipeline.

## Contamination Considerations

- All events occurred between July 2024 and February 2026
- GPT-5-nano knowledge cutoff is May 31, 2024
- Events AFTER May 2024 may or may not be in training data depending on training data collection timing
- **Contamination check should be run** before using this benchmark for research conclusions
- Some high-profile events (2024 election, Super Bowl) are very likely in training data
- Less prominent events (GDP thresholds, Golden Globes drama) may have lower contamination risk

## Event Outcome Mix

| Outcome | Count |
|---------|-------|
| YES | 11 |
| NO | 4 |

This imbalance (73% YES) reflects the selection of events where the "interesting" outcome happened. A persona that always predicts YES would get 73% accuracy but poor calibration. Brier score accounts for this.

## Curation Date

- Initial curation: 2026-02-28
- Kalshi API data verified: 2026-02-28
