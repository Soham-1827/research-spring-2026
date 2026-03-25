"""Kalshi API fetcher for historical markets and candlestick data.

Uses the public Kalshi API (no authentication required for read-only endpoints).
Base URL: https://api.elections.kalshi.com/trade-api/v2
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx

from ensemble.models import Event, MarketCandle, Outcome, TimeWindowLabel

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"


async def fetch_historical_markets(
    category: str | None = None,
    limit: int = 100,
    cursor: str | None = None,
) -> list[dict]:
    """Fetch settled historical markets from Kalshi API.

    Returns list of market dicts. Add 1-second delay for rate limiting.
    """
    params: dict = {"limit": limit}
    if category:
        params["category"] = category
    if cursor:
        params["cursor"] = cursor

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{BASE_URL}/markets", params={**params, "status": "settled"})
        resp.raise_for_status()
        await asyncio.sleep(1)
        return resp.json().get("markets", [])


async def fetch_historical_market(ticker: str) -> dict:
    """Fetch a single historical market by ticker."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{BASE_URL}/markets/{ticker}")
        resp.raise_for_status()
        await asyncio.sleep(1)
        return resp.json().get("market", {})


async def fetch_candlesticks(
    series_ticker: str,
    market_ticker: str,
    start_ts: int,
    end_ts: int,
    period_interval: int = 60,
) -> list[dict]:
    """Fetch candlestick OHLC data for a market.

    Args:
        series_ticker: Series the market belongs to
        market_ticker: Market ticker
        start_ts: Start timestamp (unix seconds)
        end_ts: End timestamp (unix seconds)
        period_interval: Candle size in minutes (60=hourly, 1440=daily)
    """
    params = {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "period_interval": period_interval,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{BASE_URL}/series/{series_ticker}/markets/{market_ticker}/candlesticks",
            params=params,
        )
        resp.raise_for_status()
        await asyncio.sleep(1)
        return resp.json().get("candlesticks", [])


async def fetch_event(event_ticker: str) -> dict:
    """Fetch an event with nested markets."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{BASE_URL}/events/{event_ticker}",
            params={"with_nested_markets": "true"},
        )
        resp.raise_for_status()
        await asyncio.sleep(1)
        return resp.json().get("event", {})


def parse_price_to_cents(price_value: str | float | int | None) -> int:
    """Convert a Kalshi price to integer cents.

    Handles: "0.65" -> 65, 0.65 -> 65, 65 -> 65, None -> 0
    """
    if price_value is None:
        return 0
    if isinstance(price_value, str):
        price_value = float(price_value)
    if isinstance(price_value, float) and price_value <= 1.0:
        return round(price_value * 100)
    return int(price_value)


def api_market_to_event(
    market: dict,
    event_data: dict,
    window_prices: dict[str, int],
) -> Event:
    """Convert Kalshi API response dicts into an Event pydantic model.

    Args:
        market: Market dict from Kalshi API
        event_data: Event dict from Kalshi API (for title, category)
        window_prices: Pre-computed window prices as {"T-7d": 55, "T-1d": 61, "T-1h": 63}
    """
    result_raw = market.get("result", "").lower().strip()
    if result_raw not in ("yes", "no"):
        raise ValueError(f"Unexpected result '{result_raw}' for market {market.get('ticker')}")

    parsed_window_prices = {
        TimeWindowLabel(k): v for k, v in window_prices.items()
    }

    return Event(
        event_ticker=market.get("event_ticker", ""),
        market_ticker=market.get("ticker", ""),
        series_ticker=market.get("series_ticker", event_data.get("series_ticker", "")),
        title=event_data.get("title", market.get("title", "")),
        question=market.get("yes_sub_title", market.get("title", "")),
        description=event_data.get("description", market.get("rules_primary", "")),
        category=event_data.get("category", market.get("category", "")),
        outcome=Outcome(result_raw),
        close_time=datetime.fromisoformat(market["close_time"]),
        open_time=datetime.fromisoformat(market["open_time"]),
        window_prices=parsed_window_prices,
    )
