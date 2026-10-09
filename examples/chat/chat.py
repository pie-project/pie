"""A full-screen terminal chat with the look of the pie-cli demo.

The screen clears on start. The conversation fills the top, and the input box
stays at the bottom with a status line under it. The reply comes from a
placeholder backend until the engine is connected.

    python examples/chat/chat.py
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame

ACCENT = "#d97757"  # the warm orange of the demo
DIM = "#8a8a8a"
MODEL = "placeholder"

STYLE = Style.from_dict({
    "accent": ACCENT,
    "dim": DIM,
    "bold": "bold",
    "status": f"{DIM} bg:#1c1c1e",
    "banner": ACCENT,
    "input-frame": ACCENT,
    "placeholder": "#666666",
})


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


class Chat:
    def __init__(self, backend: PlaceholderBackend) -> None:
        self.backend = backend
        self.transcript: list[tuple[str, str]] = []  # (style, text) pieces, in order
        self.streaming = False
        self.app: Application | None = None

        self.input = Buffer(multiline=False)
        self.input_window = Window(
            content=BufferControl(buffer=self.input),
            height=1,
            dont_extend_height=True,
        )
        self.output = Window(
            content=FormattedTextControl(self.render_transcript, get_cursor_position=self.end_of_transcript),
            wrap_lines=True,
            always_hide_cursor=True,
        )

    # ---- what is drawn -------------------------------------------------

    def banner(self) -> list[tuple[str, str]]:
        return [
            ("class:banner", "✻ "), ("class:bold", "pie chat\n"),
            ("class:dim", f"model {MODEL} · this Mac\n"),
            ("class:dim", "Ask a question, and the answer streams in as it is written.\n\n"),
        ]

    def render_transcript(self) -> list[tuple[str, str]]:
        pieces = self.banner()
        pieces.extend(self.transcript)
        return pieces

    def end_of_transcript(self):
        from prompt_toolkit.data_structures import Point

        text = "".join(t for _, t in self.render_transcript())
        return Point(x=0, y=text.count("\n"))

    def status(self) -> list[tuple[str, str]]:
        hint = " answering…" if self.streaming else " Enter sends · /new starts over · Ctrl-D quits"
        return [("class:status", hint.ljust(200))]

    # ---- what happens ----------------------------------------------------

    def add(self, style: str, text: str) -> None:
        self.transcript.append((style, text))
        if self.app:
            self.app.invalidate()

    async def send(self, text: str) -> None:
        self.add("class:bold", f"❯ {text}\n")
        self.streaming = True
        self.add("class:accent", "● ")
        async for piece in self.backend.reply(text):
            self.add("", piece)
        self.add("", "\n\n")
        self.streaming = False
        if self.app:
            self.app.invalidate()

    def on_enter(self, buffer: Buffer) -> bool:
        text = buffer.text.strip()
        buffer.reset()
        if not text:
            return False
        if text == "/new":
            self.transcript.clear()
            self.add("class:dim", "(new conversation)\n\n")
            return False
        if self.app:
            self.app.create_background_task(self.send(text))
        return False

    # ---- layout --------------------------------------------------------

    def build(self) -> Application:
        bindings = KeyBindings()

        @bindings.add("c-d")
        def _(event):
            event.app.exit()

        self.input.accept_handler = self.on_enter

        box = Frame(self.input_window, style="class:input-frame")
        status = Window(content=FormattedTextControl(self.status), height=1, style="class:status")

        root = HSplit([self.output, box, status])
        self.app = Application(
            layout=Layout(root, focused_element=self.input_window),
            key_bindings=bindings,
            style=STYLE,
            full_screen=True,
            mouse_support=False,
        )
        return self.app


def main() -> None:
    os.system("cls" if os.name == "nt" else "clear")  # start on a clean screen, as the Claude CLI does
    Chat(PlaceholderBackend()).build().run()


if __name__ == "__main__":
    main()
