//! Test inferlet build helper.
//!
//! Provides functions to build and locate test inferlet WASM components.

use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

use pie::process::ProcessId;
use pie::program::ProgramName;

/// Root directory of the test inferlets workspace.
fn inferlets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/inferlets")
}

/// Build all test inferlets. Panics on failure.
pub fn build_inferlets() {
    let status = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2"])
        .current_dir(inferlets_dir())
        .status()
        .expect("Failed to run cargo build for test inferlets");

    assert!(status.success(), "Test inferlet build failed");
}

/// Path to a compiled test inferlet WASM file.
pub fn inferlet_wasm_path(name: &str) -> PathBuf {
    // Cargo replaces hyphens with underscores in output filenames
    let filename = format!("{}.wasm", name.replace('-', "_"));
    inferlets_dir()
        .join("target/wasm32-wasip2/debug")
        .join(filename)
}

/// Read the WASM binary for a test inferlet. Builds if needed.
pub fn read_inferlet_wasm(name: &str) -> Vec<u8> {
    let path = inferlet_wasm_path(name);
    if !path.exists() {
        build_inferlets();
    }
    std::fs::read(&path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e))
}

/// Read and parse the Pie.toml manifest for a test inferlet.
pub fn read_inferlet_manifest(name: &str) -> pie::program::Manifest {
    let path = inferlets_dir().join(name).join("Pie.toml");
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));
    pie::program::Manifest::parse(&content)
        .unwrap_or_else(|e| panic!("Failed to parse {}: {}", path.display(), e))
}

/// Add and install a test inferlet in one step (async).
pub async fn add_and_install(name: &str) -> ProgramName {
    let wasm = read_inferlet_wasm(name);
    let manifest = read_inferlet_manifest(name);
    let program_name = ProgramName::parse(&format!("{name}@0.1.0")).unwrap();
    pie::program::add(wasm, manifest, true).await.unwrap();
    pie::program::install(&program_name).await.unwrap();
    program_name
}

/// Wait for a process to complete (disappear from process::list()).
/// Returns true if the process exited within the timeout, false otherwise.
pub fn wait_for_process(id: ProcessId, timeout: Duration) -> bool {
    let start = Instant::now();
    loop {
        if !pie::process::list().contains(&id) {
            return true;
        }
        if start.elapsed() > timeout {
            return false;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}
