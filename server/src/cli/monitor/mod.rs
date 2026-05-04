//! Live monitor TUI — `pie serve --monitor`.
//!
//! Rust port of `pie_cli/monitor/`. Layout, theme, braille line graph,
//! TP/GPU tree, inferlets table all preserved. Drops the simulated
//! provider (offline-dev tooling) — we always run against a live
//! engine.

mod app;
mod braille;
pub mod data;
mod provider;
mod theme;

pub use app::run as run_tui;
