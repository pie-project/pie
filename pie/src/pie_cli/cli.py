"""Pie CLI - Main entrypoint.

This module defines the main Typer application and registers all subcommands.
"""

import typer

from . import config, model, auth
from .serve import serve
from .run import run
from .http import http
from .doctor import doctor

app = typer.Typer(
    name="pie",
    help="Pie: Programmable Inference Engine",
    add_completion=False,
)

# Register top-level commands
app.command()(serve)
app.command()(run)
app.command()(http)
app.command()(doctor)

# Register subcommand groups
app.add_typer(config.app, name="config")
app.add_typer(model.app, name="model")
app.add_typer(auth.app, name="auth")


@app.callback()
def main() -> None:
    """Pie CLI - CLI for the Pie Inference Engine."""
    pass


if __name__ == "__main__":
    app()
