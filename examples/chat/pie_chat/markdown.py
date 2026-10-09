import io
import shutil

from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from rich.console import Console
from rich.markdown import Markdown as RichMarkdown


class Markdown:
    def __init__(self) -> None:
        self.text = ""


def render(text: str) -> list[tuple[str, str]]:
    width = max(20, shutil.get_terminal_size((80, 24)).columns - 2)
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=True, color_system="truecolor", width=width)
    console.print(RichMarkdown(text), end="")
    return to_formatted_text(ANSI(buffer.getvalue()))
