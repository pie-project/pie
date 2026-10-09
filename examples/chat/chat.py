"""A full-screen terminal chat with the look of the pie-cli demo.

The screen clears on start. The conversation fills the top, and the input box
stays at the bottom with a status line under it.

Replies come from a running `pie serve` through its OpenAI-compatible endpoint
(`/v1/chat/completions`, streamed). Start the engine first, then:

    python examples/chat/chat.py                 # starts `pie serve` if it is not running
    python examples/chat/chat.py --no-engine     # only connect, never start an engine
    python examples/chat/chat.py --placeholder   # fixed text, no engine needed

A local address (127.0.0.1 or localhost) that nothing is listening on gets a
`pie serve` started for this chat. The chat stops that engine on exit. An engine
that was already running is left alone.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import AsyncIterator
from pathlib import Path
from urllib.parse import urlsplit

from mascots import ANIMALS
from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame

ACCENT = "#d97757"  # the warm orange of the demo
DIM = "#8a8a8a"
MODEL = "default"

STYLE = Style.from_dict({
    "accent": ACCENT,
    "dim": DIM,
    "bold": "bold",
    "status": f"{DIM} bg:#1c1c1e",
    "banner": ACCENT,
    "error": "#e06c75",
    "input-frame": ACCENT,
    "placeholder": "#666666",
})


class EngineBackend:
    """Sends the conversation to a running `pie serve` and streams the reply."""

    def __init__(self, url: str) -> None:
        self.endpoint = url.rstrip("/") + "/v1/chat/completions"
        self.history: list[dict[str, str]] = []

    def reset(self) -> None:
        self.history.clear()

    async def reply(self, text: str) -> AsyncIterator[str]:
        self.history.append({"role": "user", "content": text})
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        pieces: list[str] = []

        def pump() -> None:
            # Blocking HTTP read on a worker thread; deltas hop back to the event loop.
            # Sampling settings match the chat-completion inferlet's defaults. Without them
            # the engine's default sampler produces garbled text with this small model.
            body = json.dumps({
                "model": MODEL,
                "messages": self.history,
                "stream": True,
                "temperature": 0.6,
                "top_p": 0.95,
            }).encode()
            request = urllib.request.Request(
                self.endpoint, data=body, headers={"content-type": "application/json"}
            )
            try:
                with urllib.request.urlopen(request) as response:
                    for raw in response:
                        line = raw.decode("utf-8").strip()
                        if not line.startswith("data:"):
                            continue
                        payload = line[len("data:"):].strip()
                        if payload == "[DONE]":
                            break
                        choice = json.loads(payload)["choices"][0]
                        delta = choice.get("delta", {}).get("content") or ""
                        if delta:
                            loop.call_soon_threadsafe(queue.put_nowait, ("text", delta))
            except (urllib.error.URLError, OSError, ValueError, KeyError) as error:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(error)))
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

        threading.Thread(target=pump, daemon=True).start()
        while True:
            kind, value = await queue.get()
            if kind == "done":
                break
            if kind == "error":
                self.history.pop()  # drop the unanswered question so a retry stays clean
                raise RuntimeError(value)
            pieces.append(value)
            yield value
        self.history.append({"role": "assistant", "content": "".join(pieces)})


# One random animal from mascots.py per launch: a 10x8 sprite drawn with half-block characters,
# one pixel per column and two pixel rows per text line, so the banner is 4 lines tall.
MASCOT_NAME, (MASCOT_PALETTE, MASCOT_ROWS) = random.choice(list(ANIMALS.items()))


def mascot_rows() -> list[list[tuple[str, str]]]:
    """Return the mascot as rows of styled text fragments, two pixel rows per line."""
    colors = {k: f"#{r:02x}{g:02x}{b:02x}" for k, (r, g, b) in MASCOT_PALETTE.items()}
    lines = []
    for top, bottom in zip(MASCOT_ROWS[0::2], MASCOT_ROWS[1::2]):
        fragments = []
        for t, b in zip(top, bottom):
            if t in colors and b in colors:
                fragments.append((f"fg:{colors[t]} bg:{colors[b]}", "▀"))
            elif t in colors:
                fragments.append((f"fg:{colors[t]}", "▀"))
            elif b in colors:
                fragments.append((f"fg:{colors[b]}", "▄"))
            else:
                fragments.append(("", " "))
        lines.append(fragments)
    return lines


class PlaceholderBackend:
    """Stands in for the engine: streams a fixed reply, one word at a time."""

    REPLY = (
        "This is a placeholder reply. The engine is not connected, so every answer "
        "is the same text. Run without --placeholder to talk to the engine."
    )

    def reset(self) -> None:
        pass

    async def reply(self, _text: str) -> AsyncIterator[str]:
        for word in self.REPLY.split(" "):
            await asyncio.sleep(0.04)
            yield word + " "


class Chat:
    def __init__(self, backend: PlaceholderBackend) -> None:
        self.backend = backend
        self.transcript: list[tuple[str, str]] = []  # (style, text) pieces, in order
        self.streaming = False
        self.app: Application | None = None
        self.exit_armed = False  # set by the first Ctrl-C on an empty prompt

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
        """The pie mascot on the left, the title and model on the right, as in the Claude logo."""
        mascot = mascot_rows()
        info = [
            [("class:bold", "pie chat")],
            [("class:dim", f"model {MODEL} · this Mac")],
            [("class:dim", os.getcwd())],
        ]
        pieces: list[tuple[str, str]] = []
        for row in range(len(mascot)):
            pieces.extend(mascot[row])
            pieces.append(("", "   "))
            if row < len(info):
                pieces.extend(info[row])
            pieces.append(("", "\n"))
        pieces.append(("", "\n"))
        return pieces

    def render_transcript(self) -> list[tuple[str, str]]:
        pieces = self.banner()
        pieces.extend(self.transcript)
        return pieces

    def end_of_transcript(self):
        from prompt_toolkit.data_structures import Point

        text = "".join(t for _, t in self.render_transcript())
        return Point(x=0, y=text.count("\n"))

    def status(self) -> list[tuple[str, str]]:
        if self.exit_armed:
            hint = " Press Ctrl-C again to exit"
        elif self.streaming:
            hint = " answering…"
        else:
            hint = " Enter sends · /new starts over · Ctrl-C or Ctrl-D twice quits"
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
        try:
            async for piece in self.backend.reply(text):
                self.add("", piece)
            self.add("", "\n\n")
        except RuntimeError as error:
            self.add("class:error", f"\n  could not reach the engine: {error}\n")
            self.add("class:dim", "  start it with `pie serve`, then send the message again.\n\n")
        finally:
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
            self.backend.reset()
            self.add("class:dim", "(new conversation)\n\n")
            return False
        if self.app:
            self.app.create_background_task(self.send(text))
        return False

    def arm_or_exit(self, event) -> None:
        if self.exit_armed:
            event.app.exit()
            return
        self.exit_armed = True
        event.app.invalidate()
        asyncio.get_running_loop().call_later(2.0, self.disarm_exit)

    def disarm_exit(self) -> None:
        self.exit_armed = False
        if self.app:
            self.app.invalidate()

    # ---- layout --------------------------------------------------------

    def build(self) -> Application:
        bindings = KeyBindings()

        @bindings.add("c-d")
        def _(event):
            # Like the Claude CLI: on an empty prompt, one press arms the exit and a
            # second press within a couple of seconds quits. With text, it does nothing.
            if self.input.text:
                return
            self.arm_or_exit(event)

        @bindings.add("c-c")
        def _(event):
            # Clears typed text first; on an empty prompt it works like Ctrl-D.
            if self.input.text:
                self.input.reset()
                return
            self.arm_or_exit(event)

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


LOCAL_HOSTS = {"127.0.0.1", "localhost"}
START_TIMEOUT_S = 300  # the first load can take minutes while weights are read


def port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def start_engine(host: str, port: int) -> subprocess.Popen:
    """Start `pie serve` in the background and wait until it accepts connections."""
    pie = shutil.which("pie")
    if pie is None:
        raise SystemExit("the `pie` command is not on PATH; install it or use --no-engine")
    log_dir = Path(tempfile.gettempdir()) / "pie-chat"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = open(log_dir / "serve.log", "w")  # noqa: SIM115 - the engine keeps writing to it
    print(f"starting the engine (log: {log.name}) ...", flush=True)
    engine = subprocess.Popen([pie, "serve"], stdout=log, stderr=subprocess.STDOUT)

    deadline = time.monotonic() + START_TIMEOUT_S
    while not port_open(host, port):
        if engine.poll() is not None:
            raise SystemExit(f"`pie serve` exited early; see {log.name}")
        if time.monotonic() > deadline:
            engine.terminate()
            raise SystemExit(f"the engine did not start within {START_TIMEOUT_S}s; see {log.name}")
        time.sleep(1)
    print("engine ready.", flush=True)
    return engine


def stop_engine(engine: subprocess.Popen) -> None:
    engine.terminate()
    try:
        engine.wait(timeout=15)
    except subprocess.TimeoutExpired:
        engine.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal chat on a running pie engine.")
    parser.add_argument("--url", default="http://127.0.0.1:8080", help="address of `pie serve`")
    parser.add_argument("--no-engine", action="store_true", help="connect only; never start `pie serve`")
    parser.add_argument("--placeholder", action="store_true", help="fixed replies, no engine")
    args = parser.parse_args()

    engine = None
    if not args.placeholder:
        split = urlsplit(args.url)
        host, port = split.hostname or "127.0.0.1", split.port or 8080
        if not args.no_engine and host in LOCAL_HOSTS and not port_open(host, port):
            engine = start_engine(host, port)

    backend = PlaceholderBackend() if args.placeholder else EngineBackend(args.url)
    os.system("cls" if os.name == "nt" else "clear")  # start on a clean screen, as the Claude CLI does
    try:
        Chat(backend).build().run()
    finally:
        if engine is not None:
            stop_engine(engine)


if __name__ == "__main__":
    main()
