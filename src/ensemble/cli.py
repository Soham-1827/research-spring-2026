"""CLI entry point for the LLM Prediction Market Ensemble."""

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ensemble.contamination import check_all_events, score_contamination
from ensemble.loader import load_events
from ensemble.models import TimeWindowLabel

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
