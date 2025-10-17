//! Output and logging utilities for the Pie CLI.

use crate::path;
use anyhow::{Context, Result};
use libpie::Config as EngineConfig;
use rustyline::completion::Completer;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::history::FileHistory;
use rustyline::validate::{ValidationContext, ValidationResult, Validator};
use rustyline::{Editor, ExternalPrinter, Helper};
use std::sync::Arc;
use std::{fs, io};
use tokio::sync::Mutex;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, FmtSubscriber, Layer};

pub struct MyHelper;

// To satisfy the `Helper` trait bounds, we must implement all its component traits.
// For now, we'll provide empty implementations for the ones we don't need.

impl Completer for MyHelper {
    type Candidate = String;
}

impl Hinter for MyHelper {
    type Hint = String;

    fn hint(&self, _line: &str, _pos: usize, _ctx: &rustyline::Context<'_>) -> Option<Self::Hint> {
        None // No hints for now
    }
}

impl Highlighter for MyHelper {}

impl Validator for MyHelper {
    fn validate(&self, _ctx: &mut ValidationContext) -> rustyline::Result<ValidationResult> {
        Ok(ValidationResult::Valid(None)) // No validation
    }
}

impl Helper for MyHelper {}

/// Initializes logging based on the engine configuration.
/// Returns an optional WorkerGuard that must be kept alive for file logging to work.
pub fn init_logging(config: &EngineConfig) -> Result<Option<WorkerGuard>> {
    let mut guard = None;

    // Console logger setup
    let console_filter = if config.verbose {
        // If -v is passed, show info for everything
        EnvFilter::new("info")
    } else {
        // Otherwise, use RUST_LOG or default to "warn"
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"))
    };
    let console_layer = tracing_subscriber::fmt::layer()
        .with_writer(io::stdout)
        .with_filter(console_filter);

    // File logger setup
    let file_layer = if let Some(log_path) = &config.log {
        let parent = log_path
            .parent()
            .context("Log path has no parent directory")?;
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create log directory at {:?}", parent))?;

        let file_appender = tracing_appender::rolling::never(parent, log_path.file_name().unwrap());
        let (non_blocking_writer, worker_guard) = tracing_appender::non_blocking(file_appender);

        // Save the guard to be returned
        guard = Some(worker_guard);

        let layer = tracing_subscriber::fmt::layer()
            .with_writer(non_blocking_writer)
            .with_ansi(false) // No colors in files
            .with_filter(EnvFilter::new("info")); // Log `INFO` and above to the file

        Some(layer)
    } else {
        None
    };

    // Register the layers
    tracing_subscriber::registry()
        .with(console_layer)
        .with(file_layer)
        .init();

    Ok(guard)
}

/// Initializes a simple logging subscriber for commands that don't start the engine.
pub fn init_simple_logging() -> Result<()> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(tracing::Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;
    Ok(())
}

/// Type alias for the external printer used in the interactive shell.
pub type SharedPrinter = Arc<Mutex<dyn ExternalPrinter + Send>>;

/// Creates an editor and external printer for the interactive shell with history loaded.
pub async fn create_editor_and_printer_with_history()
-> Result<(Editor<MyHelper, FileHistory>, SharedPrinter)> {
    let mut rl = Editor::new()?;
    rl.set_helper(Some(MyHelper)); // Enable our custom highlighter
    let printer: SharedPrinter = Arc::new(Mutex::new(rl.create_external_printer()?));

    let history_path = path::get_shell_history_path()?;
    let _ = rl.load_history(&history_path);

    Ok((rl, printer))
}

/// Prints a message using the shared printer, avoiding prompt corruption.
pub async fn print_with_printer(printer: &SharedPrinter, message: String) -> Result<()> {
    let mut p = printer.lock().await;
    p.print(message)
        .map_err(|e| anyhow::anyhow!("Failed to print: {}", e))?;
    Ok(())
}
