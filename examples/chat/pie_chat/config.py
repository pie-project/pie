from prompt_toolkit.styles import Style

ACCENT = "#d97757"
DIM = "#8a8a8a"
PROMPT_COLOUR = "#c8cbf2"
MODEL = "default"
MODEL_NAME = "Qwen"
LOCAL_HOSTS = {"127.0.0.1", "localhost"}
DEFAULT_URL = "http://127.0.0.1:8080"
START_TIMEOUT_S = 300
FRAME_SECONDS = 1.2
EXIT_WINDOW_SECONDS = 0.75
WARM_DELAY_SECONDS = 0.4
STREAM_REDRAW_SECONDS = 0.08
SCROLL_PAUSE_SECONDS = 0.5

MODES = (
    ("⏵⏵", "auto mode on", "#ffd43b"),
    ("⏸", "manual mode on", "#9a9a9a"),
    ("⏵⏵", "accept edits on", "#a78bfa"),
    ("⏸", "plan mode on", "#5fd0b4"),
)

STYLE = Style.from_dict({
    "accent": ACCENT,
    "dim": DIM,
    "bold": "bold",
    "status": f"{DIM} bg:#1c1c1e",
    "error": "#e06c75",
    "rule": "#4a4a50",
    "placeholder": "#666666",
})
