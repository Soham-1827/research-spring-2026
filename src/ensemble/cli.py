"""CLI entry point for the LLM Prediction Market Ensemble."""

import typer

app = typer.Typer(name="ensemble", help="LLM Prediction Market Ensemble")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """LLM Prediction Market Ensemble - Research Framework."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
