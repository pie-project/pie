"""Bakery CLI Console.

This module provides a shared Rich console instance for the Bakery CLI.
"""

from rich.console import Console
from rich.theme import Theme

# Define a custom theme closer to Control's aesthetic if needed,
# for now default is likely fine or we can match specific styles.
# Control uses [green]✓[/green] and [red]✗[/red] explicitly.

theme = Theme(
    {
        "info": "dim",
        "warning": "yellow",
        "error": "red",
        "success": "green",
    }
)

console = Console(theme=theme)
