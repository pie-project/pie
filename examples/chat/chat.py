"""A terminal chat with the look of the pie-cli demo.

This file is only the interface: a prompt, a streamed reply and a status bar.
The reply comes from a placeholder backend until the engine is wired in.

    python examples/chat/chat.py
"""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import AsyncIterator

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style

ACCENT = "\033[38;5;173m"  # the warm orange of the demo
DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"
WIDTH = 66
MODEL = "placeholder"

STYLE = Style.from_dict({"bottom-toolbar": "#9a9a9a bg:#1c1c1e", "placeholder": "#666666"})


class PlaceholderBackend:
    """Stands in for the engine: streams a fixed reply, one word at a time."""

    REPLY = (
        "This is a placeholder reply. The engine is not connected yet, so every answer "
        "is the same text. Once it is wired in, tokens will stream here as they are generated."
    )

    async def reply(self, _prompt: str) -> AsyncIterator[str]:
        for word in self.REPLY.split(" "):
            await asyncio.sleep(0.04)
            yield word + " "


def banner() -> None:
    lines = [
        f"{ACCENT}✻{RESET} {BOLD}pie chat{RESET}",
        "",
        f"{DIM}Ask a question, and the answer streams in as it is written.{RESET}",
        f"{DIM}model {MODEL} · this Mac{RESET}",
    ]
    print(f"{ACCENT}╭" + "─" * WIDTH + f"╮{RESET}")
    for line in lines:
        visible = len(line.replace(ACCENT, "").replace(DIM, "").replace(BOLD, "").replace(RESET, ""))
        print(f"{ACCENT}│{RESET} " + line + " " * (WIDTH - 1 - visible) + f"{ACCENT}│{RESET}")
    print(f"{ACCENT}╰" + "─" * WIDTH + f"╯{RESET}")
    print(f"{DIM}  Enter sends. /new starts over. Ctrl-D quits.{RESET}\n")


def toolbar(streaming: bool) -> HTML:
    if streaming:
        return HTML(" ✍  answering…")
    return HTML(" Enter sends · /new starts over · Ctrl-D quits")


async def run(backend: PlaceholderBackend) -> None:
    session: PromptSession[str] = PromptSession(style=STYLE)
    streaming = False

    def prompt_mark() -> HTML:
        return HTML("<style fg='#d97757'><b>❯</b></style> ")

    def placeholder() -> HTML:
        return HTML("<placeholder>Ask anything…</placeholder>")

    banner()
    with patch_stdout(raw=True):
        while True:
            try:
                text = await session.prompt_async(
                    prompt_mark,
                    placeholder=placeholder,
                    bottom_toolbar=lambda: toolbar(streaming),
                    refresh_interval=0.2,
                )
            except (EOFError, KeyboardInterrupt):
                break

            text = text.strip()
            if not text:
                continue
            if text == "/new":
                print(f"{DIM}  (new conversation){RESET}\n")
                continue

            print(f"{BOLD}❯{RESET} {text}")
            streaming = True
            print(f"{ACCENT}●{RESET} ", end="", flush=True)
            async for piece in backend.reply(text):
                print(piece, end="", flush=True)
            print("\n")
            streaming = False


def main() -> None:
    asyncio.run(run(PlaceholderBackend()))


if __name__ == "__main__":
    main()
