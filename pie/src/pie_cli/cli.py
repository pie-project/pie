"""Pie CLI - Main entrypoint.

This module defines the main Typer application and registers all subcommands.
"""

import typer

from pie_cli import config
from pie_cli.commands import serve, run, http
from pie_cli.commands.model import app as model_app
from pie_cli.commands.auth import app as auth_app
from pie_cli.commands.doctor import doctor

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
app.add_typer(model_app, name="model")
app.add_typer(auth_app, name="auth")

if __name__ == "__main__":
    app()
