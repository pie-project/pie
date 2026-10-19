//! `--trace`: the agent trace a run or a server records, written as a `.pie`
//! file when it ends.

use std::path::{Path, PathBuf};

/// Starts the trace `flag` (else `$PIE_TRACE`) names, before the engine
/// boots so the boot's facts land in its header. Whether one started.
pub fn begin(flag: Option<&Path>) -> bool {
    let path = flag.map(Path::to_path_buf).or_else(|| {
        std::env::var_os("PIE_TRACE")
            .filter(|path| !path.is_empty())
            .map(PathBuf::from)
    });
    let Some(path) = path else {
        return false;
    };
    pie_trace::start(
        path,
        pie_trace::Meta {
            pie_version: Some(env!("CARGO_PKG_VERSION").to_string()),
            commit: option_env!("PIE_COMMIT").map(str::to_string),
            ..pie_trace::Meta::default()
        },
    );
    true
}

/// Stops the trace and writes it, off the async threads.
pub async fn end() {
    match tokio::task::spawn_blocking(pie_trace::finish).await {
        Ok(Ok(Some(written))) => eprintln!(
            "trace: {} ({} agents, {} steps, {} events, {:.1} MB)",
            crate::ui::short_path(&written.path),
            written.agents,
            written.steps,
            written.events,
            written.bytes as f64 / 1e6,
        ),
        Ok(Ok(None)) => {}
        Ok(Err(error)) => eprintln!("trace: not written: {error}"),
        Err(error) => eprintln!("trace: not written: {error}"),
    }
}
