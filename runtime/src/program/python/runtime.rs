//! Python runtime resources shared across the linker and program services.
//!
//! Loads the CPython interpreter and stdlib shared modules once at engine
//! startup (from $PIE_HOME/py-runtime/shared/*.wasm) and exposes them as
//! two variants:
//!
//! - **Full** modules — have their data segments and start functions intact.
//!   Used when instantiating non-snapshotted Python components (CPython needs
//!   to initialize from scratch) and during the snapshot creation pipeline.
//! - **Stripped** modules — have data segments, data count, and start sections
//!   removed. Used when instantiating snapshotted components so the shared
//!   modules don't clobber the pre-initialized memory image.
//!
//! Both variants are compiled once at startup; snapshot status is decided
//! per-component at instantiate time.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use wasmtime::{Engine, Module};

use crate::path;

use super::snapshot;

struct State {
    /// $PIE_HOME/py-runtime directory, if it exists on disk.
    py_runtime_dir: Option<PathBuf>,
    /// Shared modules with data segments and start sections intact.
    shared_modules_full: Vec<(String, Module)>,
    /// Shared modules with data segments and start sections stripped.
    shared_modules_stripped: Vec<(String, Module)>,
    /// Whether to apply the snapshot optimization to Python components.
    snapshot_enabled: bool,
}

static STATE: OnceLock<State> = OnceLock::new();

/// Initializes the shared Python runtime. Must be called once at startup,
/// after the Wasmtime engine is created and before the linker/program services
/// are spawned. Subsequent calls are no-ops.
pub fn init(engine: &Engine, snapshot_enabled: bool) {
    if STATE.get().is_some() {
        return;
    }

    let py_runtime_dir = {
        let dir = path::get_py_runtime_dir();
        if dir.is_dir() {
            tracing::info!("Python runtime directory: {}", dir.display());
            Some(dir)
        } else {
            tracing::info!("No Python runtime directory found at {}", dir.display());
            None
        }
    };

    let (shared_modules_full, shared_modules_stripped) =
        if let Some(ref dir) = py_runtime_dir {
            let shared_dir = dir.join("shared");
            if shared_dir.is_dir() {
                load_shared_modules(engine, &shared_dir)
            } else {
                (Vec::new(), Vec::new())
            }
        } else {
            (Vec::new(), Vec::new())
        };

    if !shared_modules_full.is_empty() {
        tracing::info!(
            "Loaded {} shared core module(s); snapshot {}",
            shared_modules_full.len(),
            if snapshot_enabled { "enabled" } else { "disabled" },
        );
    }

    let _ = STATE.set(State {
        py_runtime_dir,
        shared_modules_full,
        shared_modules_stripped,
        snapshot_enabled,
    });
}

fn state() -> &'static State {
    STATE.get().expect("python::runtime::init must be called before use")
}

/// Returns the py-runtime directory path, or None if py-runtime is not installed.
pub fn dir() -> Option<&'static Path> {
    state().py_runtime_dir.as_deref()
}

/// Returns the full (un-stripped) shared modules.
pub fn full_modules() -> &'static [(String, Module)] {
    &state().shared_modules_full
}

/// Returns the stripped (no data segments, no start sections) shared modules.
pub fn stripped_modules() -> &'static [(String, Module)] {
    &state().shared_modules_stripped
}

/// Whether the snapshot optimization is enabled for Python components.
pub fn is_snapshot_enabled() -> bool {
    state().snapshot_enabled
}

/// Whether any shared modules were loaded (i.e., py-runtime is installed).
pub fn is_available() -> bool {
    !state().shared_modules_full.is_empty()
}

/// Loads shared core modules (.wasm files) from a directory, producing both
/// full and stripped variants of each.
fn load_shared_modules(
    engine: &Engine,
    shared_dir: &Path,
) -> (Vec<(String, Module)>, Vec<(String, Module)>) {
    let mut full = Vec::new();
    let mut stripped = Vec::new();

    let entries = match fs::read_dir(shared_dir) {
        Ok(entries) => entries,
        Err(e) => {
            tracing::warn!(
                "Failed to read shared modules dir {}: {e}",
                shared_dir.display()
            );
            return (full, stripped);
        }
    };

    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!("Failed to read shared module entry: {e}");
                continue;
            }
        };
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "wasm") {
            let import_name = path.file_stem().unwrap().to_str().unwrap().to_string();
            tracing::info!(
                "Loading shared module: {} -> {}",
                path.display(),
                import_name
            );

            let bytes = match fs::read(&path) {
                Ok(b) => b,
                Err(e) => {
                    tracing::error!("Failed to read shared module {}: {e}", path.display());
                    continue;
                }
            };

            match Module::new(engine, &bytes) {
                Ok(module) => full.push((import_name.clone(), module)),
                Err(e) => {
                    tracing::error!("Failed to compile shared module {}: {e}", path.display());
                    continue;
                }
            }

            match snapshot::strip_module_data(&bytes) {
                Ok(stripped_bytes) => match Module::new(engine, &stripped_bytes) {
                    Ok(module) => stripped.push((import_name, module)),
                    Err(e) => tracing::error!(
                        "Failed to compile stripped shared module {}: {e}",
                        path.display()
                    ),
                },
                Err(e) => tracing::error!(
                    "Failed to strip shared module {}: {e}",
                    path.display()
                ),
            }
        }
    }

    (full, stripped)
}
