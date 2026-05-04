//! ratatui rendering + input loop for the monitor TUI.
//!
//! Layout (mirrors `pie_cli/monitor/app.py`):
//!
//! ```text
//! ┌── status bar ────────────────────────────────────────────────┐
//! ├── Configuration ─────────────┬── System Metrics ─────────────┤
//! │   host / port / repo / etc.  │   KV / TPUT / LAT / BATCH     │
//! ├── Workers ───────────────────┼── Inferlets ──────────────────┤
//! │   GRP/GPU tree               │   id / program / user / kv    │
//! └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! Refresh: 250 ms tick (snapshots are produced by the provider every
//! 1 s; the extra TUI ticks just keep the live indicator pulsing).

use std::io;
use std::time::Duration;

use anyhow::{Context, Result};
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Cell, Paragraph, Row, Table};

use super::braille::{self, Series};
use super::data::{DisplayConfig, Snapshot, SystemMetrics};
use super::provider::Provider;
use super::theme;

const TICK: Duration = Duration::from_millis(250);

/// Run the TUI against a live engine. Blocks the calling thread
/// until the user presses `q` / `Esc`, the engine dies, or stdin
/// goes away. On return the terminal is restored — caller can shut
/// down the engine without further coordination.
pub fn run(
    runtime: &tokio::runtime::Handle,
    url: String,
    token: String,
    display_cfg: DisplayConfig,
) -> Result<()> {
    enable_raw_mode().context("enter raw mode")?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).context("enter alt screen")?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).context("init terminal")?;

    let provider = Provider::spawn(runtime, url, token);

    let res = run_loop(&mut terminal, &provider, &display_cfg);

    // Restore terminal even if the loop errored.
    let _ = disable_raw_mode();
    let _ = execute!(io::stdout(), LeaveAlternateScreen);
    let _ = terminal.show_cursor();

    res
}

fn run_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    provider: &Provider,
    display_cfg: &DisplayConfig,
) -> Result<()> {
    let mut pulse = true;
    loop {
        let snapshot = provider.snapshot();
        terminal
            .draw(|f| draw(f, &snapshot, display_cfg, pulse))
            .context("draw frame")?;

        // Drain input events for up to TICK; flip pulse on timeout.
        if event::poll(TICK).context("poll terminal events")? {
            if let Event::Key(k) = event::read().context("read terminal event")? {
                if k.kind == KeyEventKind::Press
                    && matches!(k.code, KeyCode::Char('q') | KeyCode::Esc)
                {
                    return Ok(());
                }
            }
        } else {
            pulse = !pulse;
        }
    }
}

fn draw(f: &mut ratatui::Frame, snap: &Snapshot, cfg: &DisplayConfig, pulse: bool) {
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // status bar
            Constraint::Percentage(50),
            Constraint::Min(8),
            Constraint::Length(1), // footer hint
        ])
        .split(f.area());

    draw_status_bar(f, outer[0], snap, pulse);

    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Ratio(1, 3), Constraint::Ratio(2, 3)])
        .split(outer[1]);
    draw_config_panel(f, top[0], cfg);
    draw_graphs(f, top[1], snap);

    let bottom = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Ratio(1, 3), Constraint::Ratio(2, 3)])
        .split(outer[2]);
    draw_workers(f, bottom[0], &snap.metrics);
    draw_inferlets(f, bottom[1], &snap.metrics);

    draw_footer(f, outer[3]);
}

// -----------------------------------------------------------------------------
// Status bar
// -----------------------------------------------------------------------------

fn draw_status_bar(f: &mut ratatui::Frame, area: Rect, snap: &Snapshot, pulse: bool) {
    let now = chrono::Local::now().format("%H:%M:%S").to_string();
    let pulse_char = if pulse { "●" } else { "○" };
    let (status_color, status_text) = if snap.connected {
        (theme::SUCCESS, "LIVE")
    } else {
        (theme::WARNING, "RECONNECTING")
    };

    let line = Line::from(vec![
        Span::styled("◈ Pie ", Style::default().fg(theme::PRIMARY)),
        Span::styled(
            format!("v{} ", env!("CARGO_PKG_VERSION")),
            Style::default().fg(theme::TEXT_DIM),
        ),
        Span::styled("Monitor   ", Style::default().fg(theme::TEXT_MUTED)),
        Span::styled(pulse_char, Style::default().fg(status_color)),
        Span::raw(" "),
        Span::styled(status_text, Style::default().fg(theme::TEXT_DIM)),
        Span::raw("   "),
        Span::styled(now, Style::default().fg(theme::TEXT_DIM)),
    ]);
    f.render_widget(Paragraph::new(line), area);
}

// -----------------------------------------------------------------------------
// Config panel
// -----------------------------------------------------------------------------

fn draw_config_panel(f: &mut ratatui::Frame, area: Rect, cfg: &DisplayConfig) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme::BORDER_SUBTLE))
        .title(Span::styled(
            " Configuration ",
            Style::default().fg(theme::TEXT_MUTED),
        ));

    let mut lines: Vec<Line<'static>> = Vec::new();
    let mut item = |k: &'static str, v: String, highlight: bool| {
        let value_style = if highlight {
            Style::default().fg(theme::PRIMARY_BRIGHT)
        } else {
            Style::default().fg(theme::TEXT_SECONDARY)
        };
        lines.push(Line::from(vec![
            Span::styled(
                format!("{k:<10} "),
                Style::default().fg(theme::PRIMARY).add_modifier(Modifier::BOLD),
            ),
            Span::styled(v, value_style),
        ]));
    };
    item("host", cfg.host.clone(), false);
    item("port", cfg.port.to_string(), false);
    item(
        "auth",
        if cfg.auth_enabled { "enabled" } else { "disabled" }.to_string(),
        false,
    );
    item("repo", cfg.hf_repo.clone(), true);
    item("device", format!("{:?}", cfg.device), false);
    item("tp_size", cfg.tensor_parallel_size.to_string(), false);
    item("dtype", cfg.activation_dtype.clone(), false);
    item("kv_page", cfg.kv_page_size.to_string(), false);
    item("batch", cfg.max_batch_tokens.to_string(), false);

    f.render_widget(Paragraph::new(lines).block(block), area);
}

// -----------------------------------------------------------------------------
// Graph panel: legend (top 2 rows) + braille canvas
// -----------------------------------------------------------------------------

fn draw_graphs(f: &mut ratatui::Frame, area: Rect, snap: &Snapshot) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme::BORDER_DEFAULT))
        .title(Span::styled(
            " System Metrics ",
            Style::default().fg(theme::PRIMARY),
        ));
    let inner = block.inner(area);
    f.render_widget(block, area);

    if inner.height < 3 || inner.width < 10 {
        return;
    }

    let v = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(2), Constraint::Min(2)])
        .split(inner);

    draw_graph_legend(f, v[0], snap);
    draw_graph_canvas(f, v[1], snap);
}

fn draw_graph_legend(f: &mut ratatui::Frame, area: Rect, snap: &Snapshot) {
    let m = &snap.metrics;
    let row1 = Line::from(vec![
        legend_cell("◇ TPUT", format_tput(m.token_throughput), theme::GRAPH_TPUT, &snap.token_tput_history, "t/s", false),
        Span::raw("    "),
        legend_cell(
            "◆ KV",
            format!("{:.1}%", m.kv_cache_usage),
            theme::GRAPH_KV,
            &snap.kv_cache_history,
            "%",
            true,
        ),
    ]);
    let row2 = Line::from(vec![
        legend_cell(
            "◈ LAT",
            format!("{:.1}ms", m.latency_ms),
            theme::GRAPH_LAT,
            &snap.latency_history,
            "ms",
            false,
        ),
        Span::raw("    "),
        legend_cell(
            "● BATCH",
            format!("{}", m.active_batches),
            theme::GRAPH_BATCH,
            &snap.batch_history,
            "",
            false,
        ),
    ]);
    f.render_widget(Paragraph::new(vec![row1, row2]), area);
}

fn legend_cell(
    name: &'static str,
    current: String,
    color: ratatui::style::Color,
    values: &[f64],
    suffix: &'static str,
    hide_avg: bool,
) -> Span<'static> {
    let mut s = format!("{name} {current}");
    if !hide_avg && values.len() > 1 {
        let avg: f64 = values.iter().sum::<f64>() / values.len() as f64;
        s.push_str(&format!(" (avg: {:.1}{suffix})", avg));
    }
    let _ = suffix; // suffix is baked into `current`/avg already
    Span::styled(s, Style::default().fg(color))
}

fn format_tput(v: f64) -> String {
    if v >= 10_000.0 {
        format!("{:.1}k t/s", v / 1000.0)
    } else if v >= 1000.0 {
        format!("{:.0} t/s", v)
    } else {
        format!("{:.1} t/s", v)
    }
}

fn draw_graph_canvas(f: &mut ratatui::Frame, area: Rect, snap: &Snapshot) {
    let series = [
        Series {
            name: "KV",
            color: theme::GRAPH_KV,
            values: &snap.kv_cache_history,
            min_val: 0.0,
            max_val: 100.0,
            is_integer: false,
        },
        Series {
            name: "TPUT",
            color: theme::GRAPH_TPUT,
            values: &snap.token_tput_history,
            min_val: 0.0,
            max_val: 2500.0,
            is_integer: false,
        },
        Series {
            name: "LAT",
            color: theme::GRAPH_LAT,
            values: &snap.latency_history,
            min_val: 0.0,
            max_val: 120.0,
            is_integer: false,
        },
        Series {
            name: "BATCH",
            color: theme::GRAPH_BATCH,
            values: &snap.batch_history,
            min_val: 0.0,
            max_val: 16.0,
            is_integer: true,
        },
    ];
    let lines = braille::render(&series, area.width as usize, area.height as usize, 0);
    f.render_widget(Paragraph::new(lines), area);
}

// -----------------------------------------------------------------------------
// Workers (TP/GPU) panel
// -----------------------------------------------------------------------------

fn draw_workers(f: &mut ratatui::Frame, area: Rect, m: &SystemMetrics) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme::BORDER_DEFAULT))
        .title(Span::styled(
            " Workers ",
            Style::default().fg(theme::PRIMARY),
        ));

    let mut lines: Vec<Line<'static>> = Vec::new();
    if m.tp_groups.is_empty() {
        lines.push(Line::from(Span::styled(
            "(no GPUs detected — install nvidia-ml or build with PIE_PORTABLE_CUDA)",
            Style::default().fg(theme::TEXT_DIM),
        )));
    }
    for (i, tp) in m.tp_groups.iter().enumerate() {
        if i > 0 {
            lines.push(Line::from(""));
        }
        lines.push(Line::from(Span::styled(
            format!("GRP{}", tp.tp_id),
            Style::default().fg(theme::PRIMARY).add_modifier(Modifier::BOLD),
        )));
        for gpu in &tp.gpus {
            let color = theme::util_color(gpu.utilization);
            let bar = theme::ascii_bar(gpu.utilization, 8);
            lines.push(Line::from(vec![
                Span::styled(
                    format!("  GPU{}  ", gpu.gpu_id),
                    Style::default().fg(theme::TEXT_SECONDARY),
                ),
                Span::styled(bar, Style::default().fg(color)),
                Span::styled(
                    format!(" {:.0}%", gpu.utilization),
                    Style::default().fg(color),
                ),
                Span::styled(
                    format!(
                        "  {:.1}/{:.0}G",
                        gpu.memory_used_gb, gpu.memory_total_gb
                    ),
                    Style::default().fg(theme::TEXT_MUTED),
                ),
            ]));
        }
    }
    f.render_widget(Paragraph::new(lines).block(block), area);
}

// -----------------------------------------------------------------------------
// Inferlets table
// -----------------------------------------------------------------------------

fn draw_inferlets(f: &mut ratatui::Frame, area: Rect, m: &SystemMetrics) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme::BORDER_DEFAULT))
        .title(Span::styled(
            " Inferlets ",
            Style::default().fg(theme::PRIMARY),
        ));

    let header = Row::new(vec![
        Cell::from("ID"),
        Cell::from("Program"),
        Cell::from("User"),
        Cell::from("Status"),
        Cell::from("Elapsed"),
        Cell::from("KV usage"),
    ])
    .style(Style::default().fg(theme::PRIMARY));

    let rows: Vec<Row> = m
        .inferlets
        .iter()
        .map(|inf| {
            let status_cell = if inf.status == "running" {
                Cell::from(Span::styled(
                    "◉ running",
                    Style::default().fg(theme::SUCCESS),
                ))
            } else {
                Cell::from(Span::styled(
                    "○ idle",
                    Style::default().fg(theme::TEXT_DIM),
                ))
            };
            let kv_cell = if inf.kv_cache <= 0.0 {
                Cell::from(Span::styled("—", Style::default().fg(theme::TEXT_DIM)))
            } else {
                let color = theme::util_color(inf.kv_cache);
                let bar = theme::ascii_bar(inf.kv_cache, 8);
                Cell::from(Line::from(vec![
                    Span::styled(bar, Style::default().fg(color)),
                    Span::styled(
                        format!(" {:.0}%", inf.kv_cache),
                        Style::default().fg(color),
                    ),
                ]))
            };
            Row::new(vec![
                Cell::from(Span::styled(
                    inf.id.clone(),
                    Style::default().fg(theme::TEXT_SECONDARY),
                )),
                Cell::from(Span::styled(
                    inf.program.clone(),
                    Style::default().fg(theme::PRIMARY_BRIGHT),
                )),
                Cell::from(Span::styled(
                    inf.user.clone(),
                    Style::default().fg(theme::TEXT_SECONDARY),
                )),
                status_cell,
                Cell::from(Span::styled(
                    inf.elapsed.clone(),
                    Style::default().fg(theme::TEXT_SECONDARY),
                )),
                kv_cell,
            ])
        })
        .collect();

    let widths = [
        Constraint::Ratio(1, 8),
        Constraint::Ratio(2, 8),
        Constraint::Ratio(1, 8),
        Constraint::Ratio(1, 8),
        Constraint::Ratio(1, 8),
        Constraint::Ratio(2, 8),
    ];
    let table = Table::new(rows, widths)
        .header(header)
        .block(block);
    f.render_widget(table, area);
}

// -----------------------------------------------------------------------------
// Footer hint
// -----------------------------------------------------------------------------

fn draw_footer(f: &mut ratatui::Frame, area: Rect) {
    let line = Line::from(Span::styled(
        " q / Esc: quit ",
        Style::default().fg(theme::TEXT_DIM),
    ));
    f.render_widget(Paragraph::new(line), area);
}
