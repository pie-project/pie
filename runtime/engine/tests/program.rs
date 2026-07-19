//! Program management tests — verify inferlet add, install, and uninstall.
//!
//! These tests exercise the program management lifecycle using real WASM
//! inferlet components built from sources in `tests/inferlets/`.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use futures::future::join_all;
mod common;
use common::{MockEnv, create_mock_env, inferlets, mock_device::EchoBehavior};

use pie_engine::inferlet::process;
use pie_engine::inferlet::program::{self, ProgramName};
use tokio::sync::oneshot;

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
            pie_engine::bootstrap::bootstrap(config).await.unwrap();
        });
        TestState { env, rt }
    })
}

/// Helper: ProgramName for a test inferlet
fn test_program_name(name: &str) -> ProgramName {
    ProgramName::parse(&format!("{name}@0.1.0")).unwrap()
}

/// The tests share one engine state and mutate the SAME program names
/// (add/install/uninstall on "echo"/"error", plus the install-all sweep), so
/// they must not interleave. Poisoning is ignored: a panicked test already
/// failed; the guard only orders access.
fn serial_guard() -> std::sync::MutexGuard<'static, ()> {
    static GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GUARD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

async fn spawn_echo(name: ProgramName, input: String) -> oneshot::Receiver<Result<String, String>> {
    let (response, result) = oneshot::channel();
    process::spawn("test-user".into(), name, input, None, false, Some(response))
        .expect("spawn echo inferlet");
    result
}

// =============================================================================
// Tests
// =============================================================================

#[test]
fn add_and_register() {
    let _serial = serial_guard();
    let s = state();
    let wasm = inferlets::read_inferlet_wasm("echo");
    let manifest = inferlets::read_inferlet_manifest("echo");
    let name = test_program_name("echo");

    s.rt.block_on(async {
        program::add(wasm, manifest, false).await.unwrap();
        assert!(
            program::is_registered(&name).await,
            "program should be registered after add"
        );
    });
}

#[test]
fn install_and_query() {
    let _serial = serial_guard();
    let s = state();
    let wasm = inferlets::read_inferlet_wasm("echo");
    let manifest = inferlets::read_inferlet_manifest("echo");
    let name = test_program_name("echo");

    s.rt.block_on(async {
        program::add(wasm, manifest, true).await.unwrap();
        program::install(&name).await.unwrap();
        assert!(
            program::is_installed(&name).await,
            "program should be installed after install"
        );
    });
}

#[test]
fn fetch_manifest_after_add() {
    let _serial = serial_guard();
    let s = state();
    let wasm = inferlets::read_inferlet_wasm("error");
    let manifest = inferlets::read_inferlet_manifest("error");
    let name = test_program_name("error");

    s.rt.block_on(async {
        program::add(wasm, manifest.clone(), true).await.unwrap();
        let fetched = program::fetch_manifest(&name).await;
        assert!(
            fetched.is_some(),
            "manifest should be retrievable after add"
        );
        let fetched = fetched.unwrap();
        assert_eq!(fetched.package.name, "error");
    });
}

#[test]
fn uninstall_removes_program() {
    let _serial = serial_guard();
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
        assert!(
            !program::is_installed(&name).await,
            "program should no longer be installed"
        );
        // But it should still be registered (uninstall doesn't remove from cache)
        assert!(
            program::is_registered(&name).await,
            "program should still be registered"
        );
    });
}

#[test]
fn install_context_inferlet() {
    let _serial = serial_guard();
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
    let _serial = serial_guard();
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

#[test]
fn concurrent_instantiation_and_reinstall_use_current_component() {
    let _serial = serial_guard();
    let s = state();
    let wasm = inferlets::read_inferlet_wasm("echo");
    let manifest = inferlets::read_inferlet_manifest("echo");
    let name = test_program_name("echo");

    s.rt.block_on(async {
        program::add(wasm.clone(), manifest.clone(), true)
            .await
            .unwrap();
        program::install(&name).await.unwrap();

        let receivers =
            join_all((0..64).map(|index| spawn_echo(name.clone(), format!("fleet-{index}")))).await;
        let results = tokio::time::timeout(Duration::from_secs(10), join_all(receivers))
            .await
            .expect("concurrent echo fleet timed out");
        for (index, result) in results.into_iter().enumerate() {
            assert_eq!(
                result.expect("process result sender dropped").unwrap(),
                format!("fleet-{index}")
            );
        }

        program::add(wasm, manifest, true).await.unwrap();
        program::install(&name).await.unwrap();
        let result = spawn_echo(name.clone(), "after-reinstall".into()).await;
        assert_eq!(
            result
                .await
                .expect("reinstalled process result sender dropped")
                .unwrap(),
            "after-reinstall"
        );
    });
}

#[test]
fn python_full_variant_instantiates_without_store_bound_modules() {
    let _serial = serial_guard();
    let s = state();
    let wasm = inferlets::read_inferlet_wasm("echo");
    let mut manifest = inferlets::read_inferlet_manifest("echo");
    manifest.package.name = "echo-python-full".into();
    manifest
        .runtime
        .insert("python-runtime".into(), "test-runtime".into());
    let name = test_program_name("echo-python-full");

    s.rt.block_on(async {
        program::add(wasm, manifest, true).await.unwrap();
        program::install(&name).await.unwrap();
        for input in ["python-full-first", "python-full-second"] {
            let result = spawn_echo(name.clone(), input.into()).await;
            assert_eq!(
                result
                    .await
                    .expect("process result sender dropped")
                    .unwrap(),
                input
            );
        }
    });
}
