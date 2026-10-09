import os
import shutil

from prompt_toolkit.application import Application
from prompt_toolkit.data_structures import Point
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.mouse_events import MouseEventType

from .chat import Chat
from .config import MODEL_NAME, MODES, PROMPT_COLOUR, STYLE
from .engine import engine_version
from .markdown import Markdown, render
from .mascot import mascot_rows

ENGINE_VERSION = engine_version()
MASCOT_WIDTH = 10
PAGE_LINES = 10
WHEEL_LINES = 3
GAP = " "


def banner(chat: Chat) -> list[tuple[str, str]]:
    mascot = mascot_rows(chat.frame)
    info = [
        [],
        [("class:bold", "Pie Code"), ("class:dim", f" v{ENGINE_VERSION}")],
        [("class:dim", f"{MODEL_NAME} · this Mac")],
        [("class:dim", os.getcwd())],
    ]
    blank = [("", " " * MASCOT_WIDTH)]
    pieces: list[tuple[str, str]] = []
    for row in range(max(len(mascot), len(info))):
        pieces.extend(mascot[row] if row < len(mascot) else blank)
        pieces.append(("", GAP))
        if row < len(info):
            pieces.extend(info[row])
        pieces.append(("", "\n"))
    pieces.append(("", "\n"))
    return pieces


def wheel(chat: Chat):
    def handler(mouse_event):
        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            chat.scroll(WHEEL_LINES)
        elif mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            chat.scroll(-WHEEL_LINES)
    return handler


_cache = {"version": None, "lines": []}
RESERVED_ROWS = 5


def split_lines(pieces: list[tuple]) -> list[list[tuple]]:
    lines: list[list[tuple]] = [[]]
    for style, text, handler in pieces:
        parts = text.split("\n")
        for index, part in enumerate(parts):
            if index:
                lines.append([])
            if part:
                lines[-1].append((style, part, handler))
    return lines


def transcript_lines(chat: Chat) -> list[list[tuple]]:
    if _cache["version"] != chat.version:
        handler = wheel(chat)
        pieces = banner(chat)
        for item in chat.transcript:
            if isinstance(item, Markdown):
                pieces.extend(render(item.text))
            else:
                pieces.append(item)
        _cache.update(version=chat.version, lines=split_lines([(p[0], p[1], handler) for p in pieces]))
    return _cache["lines"]


def visible_lines(chat: Chat) -> list[list[tuple]]:
    lines = transcript_lines(chat)
    rows = max(1, shutil.get_terminal_size((80, 24)).lines - RESERVED_ROWS)
    end = max(0, len(lines) - chat.scroll_back)
    return lines[max(0, end - rows):end]


def visible_pieces(chat: Chat) -> list[tuple]:
    pieces: list[tuple] = []
    for line in visible_lines(chat):
        pieces.extend(line)
        pieces.append(("", "\n", None))
    return pieces


def cursor_at_end(chat: Chat) -> Point:
    return Point(x=0, y=max(0, len(visible_lines(chat)) - 1))


def mode_line(chat: Chat) -> list[tuple[str, str]]:
    if chat.exit_armed:
        return [("class:dim", f"  Press Ctrl-{chat.exit_key} again to exit")]
    icon, label, colour = MODES[chat.mode]
    return [(f"fg:{colour}", f"  {icon} {label}"), ("class:dim", " (shift+tab to cycle)")]


def build(chat: Chat) -> Application:
    bindings = KeyBindings()

    @bindings.add("s-tab")
    def _(event):
        chat.cycle_mode()

    @bindings.add("pageup")
    def _(event):
        chat.scroll(PAGE_LINES)

    @bindings.add("pagedown")
    def _(event):
        chat.scroll(-PAGE_LINES)

    @bindings.add("c-d")
    def _(event):
        if chat.input.text:
            return
        if chat.press_exit_key("D"):
            event.app.exit()

    @bindings.add("c-c")
    def _(event):
        if chat.input.text:
            chat.input.reset()
            return
        if chat.press_exit_key("C"):
            event.app.exit()

    input_window = Window(content=BufferControl(buffer=chat.input), height=1, dont_extend_height=True)
    output = Window(
        content=FormattedTextControl(lambda: visible_pieces(chat), get_cursor_position=lambda: cursor_at_end(chat)),
        wrap_lines=True,
        always_hide_cursor=True,
    )
    rule = Window(height=1, char="─", style="class:rule")
    prompt = Window(
        content=FormattedTextControl(lambda: [(f"fg:{PROMPT_COLOUR} bold", "❯ ")]),
        width=2,
        dont_extend_width=True,
    )
    box = HSplit([rule, VSplit([prompt, input_window]), rule])
    mode = Window(content=FormattedTextControl(lambda: mode_line(chat)), height=1)
    chat.app = Application(
        layout=Layout(HSplit([output, box, mode]), focused_element=input_window),
        key_bindings=bindings,
        style=STYLE,
        full_screen=True,
        mouse_support=True,
    )
    chat.app.ttimeoutlen = 0.01
    chat.app.timeoutlen = 0.05
    return chat.app
