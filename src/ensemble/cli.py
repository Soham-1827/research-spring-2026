"""CLI entry point for the LLM Prediction Market Ensemble."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

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
