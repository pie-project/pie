import asyncio
import time
from datetime import datetime, timezone

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer

from .backend import PlaceholderBackend
from .config import (
    EXIT_WINDOW_SECONDS,
    FRAME_SECONDS,
    MODES,
    SCROLL_PAUSE_SECONDS,
    STREAM_REDRAW_SECONDS,
    WARM_DELAY_SECONDS,
)
from .markdown import Markdown


class Chat:
    def __init__(self, backend: PlaceholderBackend) -> None:
        self.backend = backend
        self.transcript: list = []
        self.streaming = False
        self.app: Application | None = None
        self.frame = 0
        self.scroll_back = 0
        self.last_scroll = 0.0
        self.content_version = 0
        self.last_draw = 0.0
        self.exit_armed = False
        self.exit_key = ""
        self.mode = 0
        self.input = Buffer(multiline=False)
        self.input.accept_handler = self.on_enter
        self.input.on_text_changed += self.on_typing
        self.warm_task: asyncio.Task | None = None

    def on_typing(self, buffer: Buffer) -> None:
        if self.warm_task is not None:
            self.warm_task.cancel()
        self.warm_task = asyncio.ensure_future(self.warm_later(buffer.text))

    async def warm_later(self, draft: str) -> None:
        try:
            await asyncio.sleep(WARM_DELAY_SECONDS)
        except asyncio.CancelledError:
            return
        if draft.strip() and not self.streaming:
            await asyncio.to_thread(self.backend.warm, draft)

    def add(self, style: str, text: str) -> None:
        self.transcript.append((style, text))
        self.content_changed()

    def content_changed(self) -> None:
        self.content_version += 1
        self.redraw()

    def redraw(self) -> None:
        self.last_draw = time.monotonic()
        if self.app:
            self.app.invalidate()

    def redraw_while_streaming(self) -> None:
        if time.monotonic() - self.last_draw >= STREAM_REDRAW_SECONDS:
            self.redraw()

    def scroll(self, lines: int) -> None:
        self.scroll_back = max(0, self.scroll_back + lines)
        self.last_scroll = time.monotonic()
        self.redraw()

    def cycle_mode(self) -> None:
        self.mode = (self.mode + 1) % len(MODES)
        self.redraw()

    async def send(self, text: str) -> None:
        self.scroll_back = 0
        self.add("class:bold", f"❯ {text}\n\n")
        self.streaming = True
        self.add("class:accent", "● ")
        answer = Markdown()
        self.transcript.append(answer)
        started = time.monotonic()
        try:
            async for piece in self.backend.reply(text):
                answer.text += piece
                self.content_version += 1
                self.redraw_while_streaming()
            self.add("", "\n\n")
            elapsed = round(time.monotonic() - started)
            done_at = datetime.now(timezone.utc).astimezone().strftime("%-I:%M %p")
            self.add("class:dim", f"✻ Done for {elapsed}s · done {done_at}\n\n")
        except RuntimeError as error:
            self.add("class:error", f"\n  could not reach the engine: {error}\n")
            self.add("class:dim", "  start it with `pie serve`, then send the message again.\n\n")
        finally:
            self.streaming = False
            self.redraw()

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

    def press_exit_key(self, key: str) -> bool:
        if self.exit_armed and self.exit_key == key:
            return True
        self.exit_armed = True
        self.exit_key = key
        self.redraw()
        asyncio.get_running_loop().call_later(EXIT_WINDOW_SECONDS, self.disarm_exit)
        return False

    def disarm_exit(self) -> None:
        self.exit_armed = False
        self.exit_key = ""
        self.redraw()

    async def animate(self) -> None:
        while True:
            await asyncio.sleep(FRAME_SECONDS)
            if time.monotonic() - self.last_scroll < SCROLL_PAUSE_SECONDS:
                continue
            self.frame += 1
            self.redraw()
