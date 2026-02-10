"""Main Textual application for LLM serving monitor.

All widgets, theme, and constants consolidated into a single module.
"""

from dataclasses import dataclass
from datetime import datetime

from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.widgets import Footer, Static, Tree

from .data import (
    DEFAULT_CONFIG,
    Inferlet,
    MetricsProvider,
    SimulatedProvider,
    TPGroupMetrics,
)


# =============================================================================
# Theme: Centralized color palette
# =============================================================================


@dataclass(frozen=True)
class ColorPalette:
    """Immutable color definitions for the LLM Monitor theme."""

    # Background colors (darkest to lightest)
    bg_base: str = "#000000"
    bg_surface: str = "#0a0a0a"
    bg_elevated: str = "#111111"
    bg_hover: str = "#1a1a1a"

    # Primary accent (warm orange spectrum)
    primary: str = "#ff9040"
    primary_bright: str = "#ffb060"
    primary_dim: str = "#e07830"

    # Semantic status colors
    success: str = "#ff9040"
    warning: str = "#ff8030"
    error: str = "#ff5040"
    info: str = "#888888"

    # Utilization gradient
    util_low: str = "#70c070"
    util_medium: str = "#ffb060"
    util_high: str = "#ff8040"
    util_critical: str = "#ff5040"

    # Text hierarchy
    text_primary: str = "#f0f0f0"
    text_secondary: str = "#909090"
    text_muted: str = "#666666"
    text_dim: str = "#444444"

    # Graph series colors
    graph_1: str = "#ffb060"
    graph_2: str = "#ff9040"
    graph_3: str = "#ff6060"
    graph_4: str = "#60a0c0"

    # Borders and dividers
    border_default: str = "#ff9040"
    border_subtle: str = "#2a2a2a"

    # Bar chart fills
    bar_empty: str = "#1a1a1a"

    # Modern UI symbols
    bar_chars: str = "▓▒░"
    status_active: str = "◉"
    status_idle: str = "○"
    legend_bullets: tuple = ("◆", "◇", "◈", "●", "○")
    section_divider: str = "─"


THEME = ColorPalette()


def get_util_color(utilization: float) -> str:
    """Get color based on utilization percentage."""
    if utilization >= 90:
        return THEME.util_critical
    elif utilization >= 75:
        return THEME.util_high
    elif utilization >= 50:
        return THEME.util_medium
    else:
        return THEME.util_low


# =============================================================================
# Braille utilities for graph rendering
# =============================================================================

BRAILLE_BASE = 0x2800
BRAILLE_DOTS = [
    [0x01, 0x08],  # Row 0 (top)
    [0x02, 0x10],  # Row 1
    [0x04, 0x20],  # Row 2
    [0x40, 0x80],  # Row 3 (bottom)
]


def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """Normalize a value to the 0-1 range."""
    val_range = max_val - min_val
    if val_range == 0:
        return 0.5
    normalized = (value - min_val) / val_range
    return max(0.0, min(1.0, normalized))


# =============================================================================
# Widget: GraphCanvas - Braille rendering area
# =============================================================================


class GraphCanvas(Static):
    """The actual graph drawing area."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._series: list[dict] = []

    def set_series(self, series: list[dict]) -> None:
        self._series = series
        self.refresh()

    def _render_braille_graph(self, width: int, height: int) -> list[list[int]]:
        data_points_needed = width * 2
        total_dot_rows = height * 4
        grid = [[-1] * data_points_needed for _ in range(total_dot_rows)]

        for series_idx, series in enumerate(self._series):
            values = series.get("values", [])
            min_val = series.get("min_val", 0)
            max_val = series.get("max_val", 100)

            if not values:
                continue

            num_values = min(len(values), data_points_needed)
            start_col = data_points_needed - num_values
            display_values = values[-num_values:]

            prev_row = None
            for i, value in enumerate(display_values):
                col = start_col + i
                normalized = normalize_value(value, min_val, max_val)
                dot_row = int((1 - normalized) * (total_dot_rows - 1))
                dot_row = max(0, min(total_dot_rows - 1, dot_row))

                if grid[dot_row][col] == -1:
                    grid[dot_row][col] = series_idx

                if prev_row is not None and abs(dot_row - prev_row) > 1:
                    start, end = min(dot_row, prev_row), max(dot_row, prev_row)
                    for r in range(start, end + 1):
                        if grid[r][col] == -1:
                            grid[r][col] = series_idx

                prev_row = dot_row

        return grid

    def on_resize(self, event) -> None:
        self.refresh()

    def _build_axis_labels(self, height: int) -> list[tuple[str, str]]:
        """Build right-side axis labels showing max values on bottom row only."""
        if not self._series:
            return [("", "#888888") for _ in range(height)]

        # Build the bottom label with all max values
        parts = []
        for series in self._series:
            max_val = series.get("max_val", 100)
            name = series.get("name", "")
            is_integer = series.get("is_integer", False)

            if is_integer:
                max_str = f"{int(max_val)}"
            elif max_val >= 1000:
                max_str = f"{max_val/1000:.1f}k"
            else:
                max_str = f"{max_val:.0f}"

            parts.append(f"{name}:{max_str}")

        # All rows empty except last
        axis_lines = [("", "#888888") for _ in range(height - 1)]
        axis_lines.append((" ".join(parts), THEME.text_dim))

        return axis_lines

    def render(self) -> Text:
        total_width = self.size.width if self.size.width > 0 else 10
        height = self.size.height if self.size.height > 0 else 2
        total_width = max(10, total_width)
        height = max(2, height)

        text = Text()

        if total_width < 5 or height < 1:
            return text

        # Build axis label for bottom row: max TPUT and min LAT only
        axis_parts = []
        for series in self._series:
            name = series.get("name", "")
            is_integer = series.get("is_integer", False)

            # For TPUT, show max; for LAT, show min
            if name == "TPUT":
                max_val = series.get("max_val", 100)
                if max_val >= 1000:
                    val_str = f"{max_val/1000:.1f}k"
                else:
                    val_str = f"{max_val:.0f}"
                axis_parts.append(f"maxTPUT:{val_str} t/s")
            elif name == "LAT":
                min_val = series.get("min_val", 0)
                val_str = f"{int(min_val)}" if is_integer else f"{min_val:.0f}"
                axis_parts.append(f"minLAT:{val_str} ms")
        axis_label = " ".join(axis_parts)
        axis_len = len(axis_label) + 2  # +2 for spacing

        # Graph uses full width except last row reserves space for axis
        graph_width = total_width
        last_row_graph_width = max(10, total_width - axis_len)

        grid = self._render_braille_graph(graph_width, height)
        data_points_needed = graph_width * 2
        colors = [s.get("color", "#888888") for s in self._series]

        for char_row in range(height):
            # Last row uses shorter width to fit axis label
            row_width = last_row_graph_width if char_row == height - 1 else graph_width

            for char_col in range(row_width):
                braille = BRAILLE_BASE
                char_series = -1

                for dot_row in range(4):
                    for dot_col in range(2):
                        grid_row = char_row * 4 + dot_row
                        grid_col = char_col * 2 + dot_col

                        if grid_col < data_points_needed and grid_row < len(grid):
                            series_idx = grid[grid_row][grid_col]
                            if series_idx >= 0:
                                braille |= BRAILLE_DOTS[dot_row][dot_col]
                                if char_series == -1:
                                    char_series = series_idx

                if braille == BRAILLE_BASE:
                    text.append(" ")
                else:
                    color = colors[char_series] if char_series >= 0 else "#888888"
                    text.append(chr(braille), style=color)

            # Add axis label on last row
            if char_row == height - 1:
                text.append(f"  {axis_label}", style=THEME.text_dim)

            if char_row < height - 1:
                text.append("\n")

        return text


# =============================================================================
# Widget: GraphLegend - Legend with current values
# =============================================================================


class GraphLegend(Static):
    """Legend showing current values."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._series: list[dict] = []

    def set_series(self, series: list[dict]):
        self._series = series
        self.refresh()

    def render(self) -> Table:
        table = Table(
            box=None,
            expand=True,
            show_header=False,
            padding=(0, 1),
        )

        bullets = THEME.legend_bullets

        table.add_column(ratio=1, justify="left", no_wrap=True)
        table.add_column(ratio=1, justify="left", no_wrap=True)

        def format_value(current: float, suffix: str, is_integer: bool = False) -> str:
            if is_integer:
                return f"{int(current)}{suffix}"
            elif current >= 10000:
                return f"{current/1000:.1f}k{suffix}"
            elif current >= 1000:
                return f"{current:.0f}{suffix}"
            else:
                return f"{current:.1f}{suffix}"

        def build_cell(idx: int, series: dict) -> Text:
            name = series.get("name", "")
            color = series.get("color", "#888888")
            current = series.get("current", 0)
            suffix = series.get("suffix", "")
            values = series.get("values", [])
            is_integer = series.get("is_integer", False)
            extra_info = series.get("extra_info", "")
            hide_avg = series.get("hide_avg", False)

            val_str = format_value(current, suffix, is_integer)

            bullet = bullets[idx % len(bullets)]
            cell = Text()
            cell.append(f"{bullet} ", style=color)
            cell.append(f"{name} ", style=THEME.text_muted)
            cell.append(val_str, style=color)

            if values and len(values) > 1 and not hide_avg:
                avg = sum(values) / len(values)
                avg_str = format_value(avg, suffix, is_integer)
                cell.append(f" (avg: ", style=THEME.text_dim)
                cell.append(f"{avg_str}", style=THEME.text_dim)
                cell.append(")", style=THEME.text_dim)

            if extra_info:
                cell.append(f" {extra_info}", style=THEME.text_dim)
            return cell

        if len(self._series) >= 4:
            table.add_row(
                build_cell(1, self._series[1]), build_cell(0, self._series[0])
            )
            table.add_row(
                build_cell(2, self._series[2]), build_cell(3, self._series[3])
            )
        elif self._series:
            for i in range(0, len(self._series), 2):
                row = [build_cell(i, self._series[i])]
                if i + 1 < len(self._series):
                    row.append(build_cell(i + 1, self._series[i + 1]))
                table.add_row(*row)

        return table


# =============================================================================
# Widget: CombinedGraph - Multi-line graph with legend
# =============================================================================


class CombinedGraph(Vertical):
    """Multi-line Braille graph with legend at bottom."""

    DEFAULT_CSS = """
    CombinedGraph {
        height: 100%;
    }
    
    CombinedGraph GraphCanvas {
        height: 1fr;
    }
    
    CombinedGraph GraphLegend {
        height: 2;
    }
    """

    def __init__(self, title: str = "", height: int = 6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._title = title

    def compose(self) -> ComposeResult:
        yield GraphLegend(id="graph-legend")
        yield GraphCanvas(id="graph-canvas")

    def set_series(self, series: list[dict]):
        canvas = self.query_one("#graph-canvas", GraphCanvas)
        canvas.set_series(series)
        legend = self.query_one("#graph-legend", GraphLegend)
        legend.set_series(series)


# =============================================================================
# Widget: ConfigPanel - Configuration display
# =============================================================================


class ConfigPanel(ScrollableContainer):
    """Displays server configuration in a compact format."""

    DEFAULT_CSS = """
    ConfigPanel {
        width: 100%;
        height: 100%;
        overflow-y: auto;
    }
    
    ConfigPanel Static {
        width: 100%;
    }
    """

    def __init__(self, config: dict | None = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._config = config or DEFAULT_CONFIG

    def compose(self) -> ComposeResult:
        yield Static(id="config-content")

    def on_mount(self) -> None:
        self._render_config()

    def update_config(self, config: dict) -> None:
        self._config = config
        self._render_config()

    def _render_config(self) -> None:
        text = Text()

        def add_item(label: str, value: str, highlight: bool = False) -> None:
            text.append(f"{label} ", style=f"bold {THEME.primary}")
            if highlight:
                text.append(f"{value}\n", style=THEME.primary_bright)
            else:
                text.append(f"{value}\n", style=THEME.text_secondary)

        def add_bool_item(
            label: str, value: bool, on_text: str = "on", off_text: str = "off"
        ) -> None:
            text.append(f"{label} ", style=f"bold {THEME.primary}")
            text.append(
                f"{on_text if value else off_text}\n",
                style=THEME.success if value else THEME.text_dim,
            )

        add_item("host", self._config.get("host", "127.0.0.1"))
        add_item("port", str(self._config.get("port", 8080)))
        add_bool_item(
            "auth", self._config.get("auth", {}).get("enabled", False), "enabled", "disabled"
        )
        add_item("repo", self._config.get("hf_repo", "qwen3-32b"), highlight=True)
        add_item("device", str(self._config.get("device", [0])))
        add_item("tp_size", str(self._config.get("tensor_parallel_size", 1)))
        add_item("dtype", self._config.get("activation_dtype", "bfloat16"))
        add_item("kv_page", str(self._config.get("kv_page_size", 16)))
        add_item("batch", str(self._config.get("max_batch_tokens", 10240)))
        add_item("mem", f"{self._config.get('gpu_mem_utilization', 0.8):.0%}")
        add_bool_item("cuda_graphs", self._config.get("use_cuda_graphs", False))

        static = self.query_one("#config-content", Static)
        static.update(text)


# =============================================================================
# Widget: InferletsTable - Running inferlets table
# =============================================================================


class InferletsTable(ScrollableContainer):
    """Table displaying active inferlets."""

    DEFAULT_CSS = """
    InferletsTable {
        width: 100%;
        height: 100%;
    }
    
    InferletsTable Static {
        width: 100%;
        height: auto;
    }
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._inferlets: list[Inferlet] = []

    def compose(self) -> ComposeResult:
        yield Static(id="table-content")

    def _get_kv_bar(self, kv_cache: float) -> Text:
        if kv_cache <= 0:
            return Text("—", style=THEME.text_dim)

        bar_width = 8
        filled = int((kv_cache / 100) * bar_width)
        mid = 1 if filled < bar_width and kv_cache > 0 else 0
        empty = bar_width - filled - mid

        bar = "▓" * filled + "▒" * mid + "░" * empty
        color = get_util_color(kv_cache)

        text = Text()
        text.append(bar, style=color)
        text.append(f" {kv_cache:.0f}%", style=color)
        return text

    def update_inferlets(self, inferlets: list[Inferlet]) -> None:
        self._inferlets = inferlets

        table = Table(
            expand=True,
            box=None,
            show_header=True,
            header_style=THEME.primary,
            row_styles=["", THEME.bg_elevated],
        )

        table.add_column("ID", ratio=1)
        table.add_column("Program", ratio=2)
        table.add_column("User", ratio=1)
        table.add_column("Status", ratio=1)
        table.add_column("Elapsed", ratio=1)
        table.add_column("KV usage", ratio=2)

        for inf in inferlets:
            if inf.status == "running":
                status = Text(f"{THEME.status_active} running", style=THEME.success)
            else:
                status = Text(f"{THEME.status_idle} idle", style=THEME.text_dim)

            table.add_row(
                Text(inf.id, style=THEME.text_secondary),
                Text(inf.program, style=THEME.primary_bright),
                Text(inf.user, style=THEME.text_secondary),
                status,
                Text(inf.elapsed, style=THEME.text_secondary),
                self._get_kv_bar(inf.kv_cache),
            )

        static = self.query_one("#table-content", Static)
        static.update(table)


# =============================================================================
# Widget: TPGroupTree - Workers tree view
# =============================================================================


class TPGroupTree(Tree):
    """Tree displaying TP groups and GPU utilization."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__("Resources", *args, **kwargs)
        self.show_root = False
        self.guide_depth = 3

    def _make_bar(self, util: float, width: int = 10) -> str:
        filled = int((util / 100) * width)
        mid = 1 if filled < width and util > 0 else 0
        empty = width - filled - mid
        return "▓" * filled + "▒" * mid + "░" * empty

    def update_tp_groups(self, tp_groups: list[TPGroupMetrics]) -> None:
        self.root.remove_children()

        for tp in tp_groups:
            tp_label = Text()
            tp_label.append(f"GRP{tp.tp_id}", style=THEME.primary)

            tp_node = self.root.add(tp_label, expand=True)

            for gpu in tp.gpus:
                util = gpu.utilization
                gpu_color = get_util_color(util)

                gpu_label = Text()
                gpu_label.append(f"GPU{gpu.gpu_id}  ", style=THEME.text_secondary)
                gpu_label.append(self._make_bar(util, 8), style=gpu_color)
                gpu_label.append(f" {util:.0f}%", style=gpu_color)
                gpu_label.append(
                    f"  {gpu.memory_used_gb:.1f}/{gpu.memory_total_gb:.0f}G",
                    style=THEME.text_muted,
                )

                tp_node.add_leaf(gpu_label)

            if tp != tp_groups[-1]:
                self.root.add_leaf(Text(""))


# =============================================================================
# Widget: StatusBar - Title with live indicator
# =============================================================================


class StatusBar(Static):
    """Title bar showing Pie branding with configurable status indicator.

    Supports two modes:
    - Live mode: pulsing dot with "LIVE" text
    - Loading mode: custom text with optional progress bar
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pulse = True
        self._mode = "live"  # "live" or "loading"
        self._loading_text = ""
        self._progress = None  # None or 0.0-1.0

    def on_mount(self):
        self._update_display()

    def pulse(self):
        """Toggle the live indicator pulse (only in live mode)."""
        self._pulse = not self._pulse
        self._update_display()

    def set_live(self):
        """Switch to live mode with pulsing indicator."""
        self._mode = "live"
        self._update_display()

    def set_loading(self, text: str, progress: float | None = None):
        """Switch to loading mode with custom text and optional progress.

        Args:
            text: Status text (e.g., "Loading weights...")
            progress: Optional progress 0.0-1.0 for progress bar
        """
        self._mode = "loading"
        self._loading_text = text
        self._progress = progress
        self._update_display()

    def _make_progress_bar(self, progress: float, width: int = 10) -> str:
        """Create a simple progress bar."""
        filled = int(progress * width)
        empty = width - filled
        return "▓" * filled + "░" * empty

    def _update_display(self):
        timestamp = datetime.now().strftime("%H:%M:%S")

        if self._mode == "live":
            pulse_char = "●" if self._pulse else "○"
            status_part = f"[{THEME.success}]{pulse_char}[/] [{THEME.text_dim}]LIVE[/]"
        else:
            # Loading mode
            if self._progress is not None:
                bar = self._make_progress_bar(self._progress)
                pct = int(self._progress * 100)
                status_part = f"[{THEME.warning}]{bar}[/] [{THEME.text_muted}]{self._loading_text} {pct}%[/]"
            else:
                status_part = (
                    f"[{THEME.warning}]◐[/] [{THEME.text_muted}]{self._loading_text}[/]"
                )

        self.update(
            f"[{THEME.primary}]◈ Pie[/] [{THEME.text_dim}]v0.2.0[/] "
            f"[{THEME.text_muted}]Monitor[/]  "
            f"{status_part}  "
            f"[{THEME.text_dim}]{timestamp}[/]"
        )


# =============================================================================
# Main Application
# =============================================================================


class LLMMonitorApp(App):
    """Professional monitoring dashboard for LLM serving systems."""

    CSS = f"""
    Screen {{
        background: {THEME.bg_base};
    }}
    
    Footer {{
        background: {THEME.bg_elevated};
    }}
    
    #status-bar {{
        height: 1;
        background: {THEME.bg_surface};
        color: {THEME.text_secondary};
        padding: 0 2;
    }}
    
    #main-container {{
        height: 100%;
        padding: 0 1;
    }}
    
    #top-row {{
        height: 1fr;
        min-height: 10;
        margin: 1 0;
    }}
    
    #config-panel {{
        width: 1fr;
        background: {THEME.bg_surface};
        border: round {THEME.border_subtle};
        border-title-color: {THEME.text_muted};
        padding: 0 1;
    }}
    
    #graph-panel {{
        width: 2fr;
        background: {THEME.bg_surface};
        border: round {THEME.border_default};
        border-title-color: {THEME.primary};
        padding: 0 1;
        margin-left: 1;
    }}
    
    #bottom-row {{
        height: 1fr;
        min-height: 10;
        margin-bottom: 1;
    }}
    
    #tp-panel {{
        width: 1fr;
        background: {THEME.bg_base};
        border: round {THEME.border_default};
        border-title-color: {THEME.primary};
        padding: 0 1;
    }}
    
    #inferlets-panel {{
        width: 2fr;
        background: {THEME.bg_surface};
        border: round {THEME.border_default};
        border-title-color: {THEME.primary};
        padding: 0 1;
        margin-left: 1;
    }}
    
    TPGroupTree {{
        height: 100%;
        background: {THEME.bg_base};
        scrollbar-background: {THEME.bg_base};
    }}
    
    ConfigPanel {{
        height: 100%;
        background: {THEME.bg_base};
    }}
    
    InferletsTable {{
        height: 100%;
        width: 100%;
        background: {THEME.bg_base};
    }}
    
    InferletsTable > .datatable--header {{
        background: {THEME.bg_elevated};
        color: {THEME.primary};
    }}
    
    InferletsTable > .datatable--cursor {{
        background: {THEME.bg_hover};
    }}
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    ENABLE_COMMAND_PALETTE = False

    def __init__(
        self,
        provider: MetricsProvider | None = None,
        num_gpus: int = 8,
        num_tp_groups: int = 4,
        refresh_rate: float = 0.5,
    ):
        super().__init__()
        self.refresh_rate = refresh_rate
        # Use provided provider or create simulated one
        if provider is not None:
            self.provider = provider
        else:
            self.provider = SimulatedProvider(
                num_gpus=num_gpus, num_tp_groups=num_tp_groups
            )

    def compose(self) -> ComposeResult:
        yield StatusBar(id="status-bar")

        with Container(id="main-container"):
            with Horizontal(id="top-row"):
                with Vertical(id="config-panel") as panel:
                    panel.border_title = "Configuration"
                    yield ConfigPanel(id="config")

                with Vertical(id="graph-panel") as panel:
                    panel.border_title = "System Metrics"
                    yield CombinedGraph(title="", height=6, id="combined-graph")

            with Horizontal(id="bottom-row"):
                with Vertical(id="tp-panel") as panel:
                    panel.border_title = "Workers"
                    yield TPGroupTree(id="tp-tree")

                with Vertical(id="inferlets-panel") as panel:
                    panel.border_title = "Inferlets"
                    yield InferletsTable(id="inferlets-table")

        yield Footer()

    def on_mount(self) -> None:
        # Update ConfigPanel with provider config
        if hasattr(self.provider, "config") and self.provider.config:
            config_panel = self.query_one("#config", ConfigPanel)
            config_panel.update_config(self.provider.config)

        self.set_interval(self.refresh_rate, self.update_metrics)
        self.update_metrics()

    def update_metrics(self) -> None:
        data = self.provider.get_metrics()

        page_info = f"({data.kv_pages_used}/{data.kv_pages_total} pages)"

        combined = self.query_one("#combined-graph", CombinedGraph)
        combined.set_series(
            [
                {
                    "name": "KV",
                    "color": THEME.graph_1,
                    "values": self.provider.kv_cache_history,
                    "current": data.kv_cache_usage,
                    "min_val": 0,
                    "max_val": 100,
                    "suffix": "%",
                    "extra_info": page_info,
                    "hide_avg": True,
                },
                {
                    "name": "TPUT",
                    "color": THEME.graph_2,
                    "values": self.provider.token_tput_history,
                    "current": data.token_throughput,
                    "min_val": 0,
                    "max_val": 2500,
                    "suffix": " t/s",
                },
                {
                    "name": "LAT",
                    "color": THEME.graph_3,
                    "values": self.provider.latency_history,
                    "current": data.latency_ms,
                    "min_val": 0,
                    "max_val": 120,
                    "suffix": "ms",
                },
                {
                    "name": "BATCH",
                    "color": THEME.graph_4,
                    "values": self.provider.batch_history,
                    "current": data.active_batches,
                    "min_val": 0,
                    "max_val": 16,
                    "suffix": "",
                    "is_integer": True,
                },
            ]
        )

        tp_tree = self.query_one("#tp-tree", TPGroupTree)
        tp_tree.update_tp_groups(data.tp_groups)

        table = self.query_one("#inferlets-table", InferletsTable)
        table.update_inferlets(data.inferlets)

        status_bar = self.query_one("#status-bar", StatusBar)
        status_bar.pulse()

    def action_refresh(self) -> None:
        self.update_metrics()
