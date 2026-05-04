//! Color palette + small helpers for the TUI. Mirrors
//! `pie_cli/monitor/app.py::ColorPalette`. Hex literals from the
//! Python source converted to ratatui's `Color::Rgb`.
//!
//! A few palette entries (e.g. `TEXT_PRIMARY`) aren't yet referenced
//! by the Rust widgets — kept around so future widgets can pick
//! them up without diving back into the Python source.
#![allow(dead_code)]

use ratatui::style::Color;

pub const PRIMARY: Color = Color::Rgb(0xff, 0x90, 0x40);
pub const PRIMARY_BRIGHT: Color = Color::Rgb(0xff, 0xb0, 0x60);

pub const SUCCESS: Color = Color::Rgb(0xff, 0x90, 0x40);
pub const WARNING: Color = Color::Rgb(0xff, 0x80, 0x30);

pub const UTIL_LOW: Color = Color::Rgb(0x70, 0xc0, 0x70);
pub const UTIL_MEDIUM: Color = Color::Rgb(0xff, 0xb0, 0x60);
pub const UTIL_HIGH: Color = Color::Rgb(0xff, 0x80, 0x40);
pub const UTIL_CRITICAL: Color = Color::Rgb(0xff, 0x50, 0x40);

pub const TEXT_PRIMARY: Color = Color::Rgb(0xf0, 0xf0, 0xf0);
pub const TEXT_SECONDARY: Color = Color::Rgb(0x90, 0x90, 0x90);
pub const TEXT_MUTED: Color = Color::Rgb(0x66, 0x66, 0x66);
pub const TEXT_DIM: Color = Color::Rgb(0x44, 0x44, 0x44);

pub const GRAPH_KV: Color = Color::Rgb(0xff, 0xb0, 0x60);
pub const GRAPH_TPUT: Color = Color::Rgb(0xff, 0x90, 0x40);
pub const GRAPH_LAT: Color = Color::Rgb(0xff, 0x60, 0x60);
pub const GRAPH_BATCH: Color = Color::Rgb(0x60, 0xa0, 0xc0);

pub const BORDER_DEFAULT: Color = Color::Rgb(0xff, 0x90, 0x40);
pub const BORDER_SUBTLE: Color = Color::Rgb(0x2a, 0x2a, 0x2a);

/// Color on the green→red gradient driven by 0–100 utilization.
pub fn util_color(util: f64) -> Color {
    if util >= 90.0 {
        UTIL_CRITICAL
    } else if util >= 75.0 {
        UTIL_HIGH
    } else if util >= 50.0 {
        UTIL_MEDIUM
    } else {
        UTIL_LOW
    }
}

/// `▓▓▒░░` style bar of `width` cells filled to `pct%`. Uses the
/// same three-density characters the Python widget draws.
pub fn ascii_bar(pct: f64, width: usize) -> String {
    let filled = ((pct / 100.0) * width as f64).max(0.0) as usize;
    let filled = filled.min(width);
    let mid = if filled < width && pct > 0.0 { 1 } else { 0 };
    let empty = width.saturating_sub(filled).saturating_sub(mid);
    let mut s = String::with_capacity(width * 3);
    for _ in 0..filled {
        s.push('▓');
    }
    for _ in 0..mid {
        s.push('▒');
    }
    for _ in 0..empty {
        s.push('░');
    }
    s
}
