//! Program management tests — verify inferlet add, install, and uninstall.
//!
//! These tests exercise the program management lifecycle using real WASM
//! inferlet components built from sources in `tests/inferlets/`.

use std::sync::{Arc, OnceLock};

mod common;
use common::{create_mock_env, MockEnv, mock_device::EchoBehavior, inferlets};

use pie::program::{self, ProgramName};

/// Shared state: MockEnv + tokio runtime (must outlive the process).
struct TestState {
    #[allow(dead_code)]
    env: MockEnv,
    rt: tokio::runtime::Runtime,
}

static STATE: OnceLock<TestState> = OnceLock::new();

fn state() -> &'static TestState {
    STATE.get_or_init(|| {
        // Build test inferlets first
        inferlets::build_inferlets();

        let rt = tokio::runtime::Runtime::new().unwrap();
        let env = create_mock_env("test-model", 1, 16, Arc::new(EchoBehavior(42)));
        let config = env.config();
        rt.block_on(async {
            pie::bootstrap::bootstrap(config).await.unwrap();
        });
        TestState { env, rt }
    })
}

/// Helper: ProgramName for a test inferlet
fn test_program_name(name: &str) -> ProgramName {
    ProgramName::parse(&format!("{name}@0.1.0"))
}

// =============================================================================
// Tests
// =============================================================================

#[test]
fn add_and_register() {
    let s = state();
    let wasm = inferlets::read_inferlet_wasm("echo");
    let manifest = inferlets::read_inferlet_manifest("echo");
    let name = test_program_name("echo");

    s.rt.block_on(async {
        program::add(wasm, manifest, false).await.unwrap();
        assert!(program::is_registered(&name).await, "program should be registered after add");
    });
}

#[test]
fn install_and_query() {
    let s = state();
    let wasm = inferlets::read_inferlet_wasm("echo");
    let manifest = inferlets::read_inferlet_manifest("echo");
    let name = test_program_name("echo");

    s.rt.block_on(async {
        program::add(wasm, manifest, true).await.unwrap();
        program::install(&name).await.unwrap();
        assert!(program::is_installed(&name).await, "program should be installed after install");
    });
}

#[test]
fn fetch_manifest_after_add() {
    let s = state();
    let wasm = inferlets::read_inferlet_wasm("error");
    let manifest = inferlets::read_inferlet_manifest("error");
    let name = test_program_name("error");

    s.rt.block_on(async {
        program::add(wasm, manifest.clone(), true).await.unwrap();
        let fetched = program::fetch_manifest(&name).await;
        assert!(fetched.is_some(), "manifest should be retrievable after add");
        let fetched = fetched.unwrap();
        assert_eq!(fetched.package.name, "error");
    });
}

#[test]
fn uninstall_removes_program() {
    let s = state();
    let wasm = inferlets::read_inferlet_wasm("error");
    let manifest = inferlets::read_inferlet_manifest("error");
    let name = test_program_name("error");

    s.rt.block_on(async {
        program::add(wasm, manifest, true).await.unwrap();
        program::install(&name).await.unwrap();
        assert!(program::is_installed(&name).await);

        let removed = program::uninstall(&name).await;
        assert!(removed, "uninstall should return true");
        assert!(!program::is_installed(&name).await, "program should no longer be installed");
        // But it should still be registered (uninstall doesn't remove from cache)
        assert!(program::is_registered(&name).await, "program should still be registered");
    });
}

#[test]
fn install_context_inferlet() {
    let s = state();
    let wasm = inferlets::read_inferlet_wasm("context");
    let manifest = inferlets::read_inferlet_manifest("context");
    let name = test_program_name("context");

    s.rt.block_on(async {
        program::add(wasm, manifest, true).await.unwrap();
        program::install(&name).await.unwrap();
        assert!(program::is_installed(&name).await);
    });
}

#[test]
fn install_all_test_inferlets() {
    let s = state();

    let inferlet_names = ["echo", "context", "error"];

    s.rt.block_on(async {
        for name in &inferlet_names {
            let wasm = inferlets::read_inferlet_wasm(name);
            let manifest = inferlets::read_inferlet_manifest(name);
            let program_name = test_program_name(name);

            program::add(wasm, manifest, true).await.unwrap();
            program::install(&program_name).await.unwrap();
            assert!(
                program::is_installed(&program_name).await,
                "{name} should be installed"
            );
        }
    });
}
