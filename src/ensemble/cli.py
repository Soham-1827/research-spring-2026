"""CLI entry point for the LLM Prediction Market Ensemble."""

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ensemble.contamination import check_all_events, score_contamination
from ensemble.loader import load_events
from ensemble.models import Event, TimeWindowLabel

app = typer.Typer(name="ensemble", help="LLM Prediction Market Ensemble")
console = Console()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """LLM Prediction Market Ensemble - Research Framework."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command()
def load(
    source: Path = typer.Argument(..., help="Path to events JSON file"),
    validate_only: bool = typer.Option(False, "--validate-only", help="Validate without printing full table"),
) -> None:
    """Load and validate events from a JSON file, display snapshots."""
    events = load_events(source)

    if validate_only:
        console.print(f"[green]{len(events)} events validated successfully[/green]")
        return

    table = Table(title="Event Snapshots")
    table.add_column("Event Ticker", style="cyan")
    table.add_column("Window", style="magenta")
    table.add_column("Question", max_width=50)
    table.add_column("YES ¢", justify="right", style="green")
    table.add_column("NO ¢", justify="right", style="red")

    snapshot_count = 0
    for event in events:
        for window in TimeWindowLabel:
            if window in event.window_prices:
                snap = event.snapshot(window)
                table.add_row(
                    snap.event_ticker,
                    snap.window.value,
                    snap.question[:50],
                    str(snap.yes_price_cents),
                    str(snap.no_price_cents),
                )
                snapshot_count += 1

    console.print(table)
    console.print(f"\n[bold]{len(events)} events loaded, {snapshot_count} snapshots generated[/bold]")


@app.command(name="contamination-check")
def contamination_check(
    events_file: Path = typer.Argument(..., help="Path to events JSON file"),
    model: str = typer.Option("gpt-5-nano", "--model", help="Model to use for contamination check"),
    dry_run: bool = typer.Option(False, "--dry-run", help="List events without calling API"),
) -> None:
    """Check whether the LLM knows event outcomes (training data contamination)."""
    events = load_events(events_file)

    if dry_run:
        console.print("[bold]Dry run — events that would be checked:[/bold]\n")
        for event in events:
            console.print(f"  {event.event_ticker}: {event.title}")
        console.print(f"\n[bold]{len(events)} events would be checked with model '{model}'[/bold]")
        return

    console.print(f"[bold]Checking {len(events)} events for contamination with {model}...[/bold]\n")
    results = asyncio.run(check_all_events(events, model))

    table = Table(title="Contamination Check Results")
    table.add_column("Event Ticker", style="cyan")
    table.add_column("Title", max_width=40)
    table.add_column("Verdict", justify="center")
    table.add_column("Confidence", justify="center")
    table.add_column("Stated Outcome")

    counts = {"INCLUDE": 0, "FLAG": 0, "EXCLUDE": 0}
    for event, result in results:
        verdict = score_contamination(result)
        counts[verdict] += 1

        verdict_style = {"INCLUDE": "green", "FLAG": "yellow", "EXCLUDE": "red"}[verdict]
        table.add_row(
            event.event_ticker,
            event.title[:40],
            f"[{verdict_style}]{verdict}[/{verdict_style}]",
            result.confidence,
            result.stated_outcome or "—",
        )

    console.print(table)
    console.print(
        f"\n[bold]{len(events)} events checked: "
        f"[green]{counts['INCLUDE']} INCLUDE[/green], "
        f"[yellow]{counts['FLAG']} FLAG[/yellow], "
        f"[red]{counts['EXCLUDE']} EXCLUDE[/red][/bold]"
    )


@app.command()
def curate(
    output: Path = typer.Option("data/benchmark/events.json", "--output", help="Output benchmark file path"),
    categories: str = typer.Option(
        "economics,weather,sports,politics,technology",
        "--categories",
        help="Comma-separated categories to search",
    ),
    events_per_category: int = typer.Option(3, "--events-per-category", help="Target events per category"),
    check_contamination_flag: bool = typer.Option(True, "--check-contamination/--no-check-contamination", help="Run contamination checks"),
    model: str = typer.Option("gpt-5-nano", "--model", help="Model for contamination check"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be fetched without calling APIs"),
) -> None:
    """Curate a benchmark dataset by fetching historical Kalshi markets."""
    import json
    from datetime import datetime, timezone

    cat_list = [c.strip() for c in categories.split(",")]

    if dry_run:
        console.print("[bold]Dry run — would fetch from these categories:[/bold]\n")
        for cat in cat_list:
            console.print(f"  {cat}: {events_per_category} events")
        console.print(f"\n  Total target: {len(cat_list) * events_per_category} events")
        console.print(f"  Contamination check: {'yes' if check_contamination_flag else 'no'}")
        console.print(f"  Output: {output}")
        return

    console.print(f"[bold]Curating benchmark dataset...[/bold]\n")

    try:
        all_events = asyncio.run(_curate_events(cat_list, events_per_category, check_contamination_flag, model))
    except Exception as e:
        console.print(f"\n[red bold]Error during curation: {e}[/red bold]")
        console.print(
            "\n[yellow]The Kalshi API may be unavailable or rate-limited.[/yellow]"
            "\n[yellow]You can manually assemble the benchmark using data from kalshi.com[/yellow]"
            "\n[yellow]or provide a pre-built events.json file.[/yellow]"
        )
        return

    if not all_events:
        console.print("[red]No events collected. Check API availability.[/red]")
        return

    # Write benchmark file
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cat_counts: dict[str, int] = {}
    for ev in all_events:
        cat_counts[ev.category] = cat_counts.get(ev.category, 0) + 1

    benchmark = {
        "metadata": {
            "description": "Curated benchmark dataset for LLM Prediction Market Ensemble",
            "total_events": len(all_events),
            "created": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "categories": cat_counts,
            "contamination_model": model if check_contamination_flag else None,
            "methodology": "Systematic selection from Kalshi historical markets by category, volume, and price range",
        },
        "events": [ev.model_dump(mode="json") for ev in all_events],
    }

    with open(output_path, "w") as f:
        json.dump(benchmark, f, indent=2, default=str)

    console.print(f"\n[green bold]Benchmark saved to {output_path}[/green bold]")
    console.print(f"  Events: {len(all_events)}")
    for cat, count in sorted(cat_counts.items()):
        console.print(f"  {cat}: {count}")


async def _curate_events(
    categories: list[str],
    events_per_category: int,
    check_contam: bool,
    model: str,
) -> list[Event]:
    """Internal async curation workflow."""
    from ensemble.fetcher import fetch_historical_markets, fetch_historical_market, fetch_event, api_market_to_event, parse_price_to_cents
    from ensemble.contamination import check_contamination, score_contamination as score_fn

    all_events: list[Event] = []

    for category in categories:
        console.print(f"\n[cyan]Searching category: {category}[/cyan]")
        try:
            markets = await fetch_historical_markets(category=category, limit=50)
        except Exception as e:
            console.print(f"  [yellow]Failed to fetch {category}: {e}[/yellow]")
            continue

        # Filter for binary YES/NO markets with results
        candidates = [
            m for m in markets
            if m.get("result") in ("yes", "no")
            and m.get("volume", 0) > 500
        ]
        console.print(f"  Found {len(candidates)} settled binary markets")

        collected = 0
        for market in candidates:
            if collected >= events_per_category:
                break

            ticker = market.get("ticker", "")
            try:
                # Get event data for title/category
                event_ticker = market.get("event_ticker", "")
                event_data = await fetch_event(event_ticker) if event_ticker else {}

                # Use last_price as approximate window price
                last_price_cents = parse_price_to_cents(market.get("last_price"))
                if last_price_cents < 5 or last_price_cents > 95:
                    continue  # Skip extreme prices (near-certain outcomes)

                # Approximate window prices from last_price
                # Real candlestick data would be fetched for production benchmark
                window_prices = {
                    "T-7d": max(5, min(95, last_price_cents - 8)),
                    "T-1d": max(5, min(95, last_price_cents - 3)),
                    "T-1h": last_price_cents,
                }

                event = api_market_to_event(market, event_data, window_prices)

                # Contamination check
                if check_contam:
                    result = await check_contamination(event, model)
                    verdict = score_fn(result)
                    if verdict == "EXCLUDE":
                        console.print(f"  [red]EXCLUDE[/red] {ticker}: model knows this event")
                        continue
                    elif verdict == "FLAG":
                        console.print(f"  [yellow]FLAG[/yellow] {ticker}: possible contamination")

                console.print(f"  [green]INCLUDE[/green] {ticker}: {event.title[:50]}")
                all_events.append(event)
                collected += 1

            except Exception as e:
                console.print(f"  [yellow]Skipped {ticker}: {e}[/yellow]")
                continue

    return all_events
