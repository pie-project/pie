//! Python runtime resources shared across the linker and program services.
//!
//! Tracks the CPython runtime directory and lazily loads the stdlib shared
//! modules from $PIE_HOME/py-runtime/shared/*.wasm when a Python component
//! is installed or instantiated. Non-Python components should not pay this
//! compilation cost. Loaded modules are exposed as two variants:
//!
//! - **Full** modules — have their data segments and start functions intact.
//!   Used when instantiating non-snapshotted Python components (CPython needs
//!   to initialize from scratch) and during the snapshot creation pipeline.
//! - **Stripped** modules — have data segments, data count, and start sections
//!   removed. Used when instantiating snapshotted components so the shared
//!   modules don't clobber the pre-initialized memory image.
//!
//! Both variants are compiled at most once; snapshot status is decided
//! per-component at instantiate time.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use wasmtime::{Engine, Module};

use crate::util;

use super::snapshot;

struct State {
    /// Wasmtime engine used to compile shared modules lazily.
    engine: Engine,
    /// $PIE_HOME/py-runtime directory, if it exists on disk.
    py_runtime_dir: Option<PathBuf>,
    /// Lazily compiled shared modules. Startup should not pay CPython
    /// compilation cost for non-Python inferlets.
    shared_modules: OnceLock<(Vec<(String, Module)>, Vec<(String, Module)>)>,
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
        let dir = util::get_py_runtime_dir();
        if dir.is_dir() {
            tracing::info!("Python runtime directory: {}", dir.display());
            Some(dir)
        } else {
            tracing::info!("No Python runtime directory found at {}", dir.display());
            None
        }
    };

    let _ = STATE.set(State {
        engine: engine.clone(),
        py_runtime_dir,
        shared_modules: OnceLock::new(),
        snapshot_enabled,
    });
}

fn state() -> &'static State {
    STATE
        .get()
        .expect("python::runtime::init must be called before use")
}

/// Returns the py-runtime directory path, or None if py-runtime is not installed.
pub fn dir() -> Option<&'static Path> {
    state().py_runtime_dir.as_deref()
}

/// Returns the full (un-stripped) shared modules.
pub fn full_modules() -> &'static [(String, Module)] {
    &loaded_modules().0
}

/// Returns the stripped (no data segments, no start sections) shared modules.
pub fn stripped_modules() -> &'static [(String, Module)] {
    &loaded_modules().1
}

/// Whether the snapshot optimization is enabled for Python components.
pub fn is_snapshot_enabled() -> bool {
    state().snapshot_enabled
}

/// Whether any shared modules were loaded (i.e., py-runtime is installed).
pub fn is_available() -> bool {
    state().py_runtime_dir.is_some() && !full_modules().is_empty()
}

fn loaded_modules() -> &'static (Vec<(String, Module)>, Vec<(String, Module)>) {
    let state = state();
    state.shared_modules.get_or_init(|| {
        let Some(dir) = state.py_runtime_dir.as_ref() else {
            return (Vec::new(), Vec::new());
        };
        let shared_dir = dir.join("shared");
        if !shared_dir.is_dir() {
            return (Vec::new(), Vec::new());
        }

        let loaded = load_shared_modules(&state.engine, &shared_dir);
        if !loaded.0.is_empty() {
            tracing::info!(
                "Loaded {} shared core module(s); snapshot {}",
                loaded.0.len(),
                if state.snapshot_enabled {
                    "enabled"
                } else {
                    "disabled"
                },
            );
        }
        loaded
    })
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
                Err(e) => tracing::error!("Failed to strip shared module {}: {e}", path.display()),
            }
        }
    }

    (full, stripped)
}
