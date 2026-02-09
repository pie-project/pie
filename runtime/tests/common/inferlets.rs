//! Test inferlet build helper.
//!
//! Provides functions to build and locate test inferlet WASM components.

use std::path::PathBuf;
use std::process::Command;

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
