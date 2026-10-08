//! C-ABI shim embedding the Pie engine in an iOS app.
//!
//! Mirrors the boot path of `pie run` (server/src/cli/run_cmd.rs): load a
//! config TOML, boot a one-shot engine via `serve::start_engine`, drive one
//! inferlet over the local WebSocket with `pie-client`, collect its output,
//! and tear the engine down. The whole transcript comes back as a single
//! heap-allocated C string the caller must release with `pie_ios_free`.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::Path;

use anyhow::{Context, Result};
use pie_client::client::{Client, ProcessEvent};
use pie_server::{config, serve};

/// Boot Pie, run one inferlet to completion, shut down, return transcript.
///
/// All arguments are NUL-terminated UTF-8 paths/strings. Returns a malloc'd
/// C string (transcript, or a message starting with "PIE ERROR:").
///
/// # Safety
/// All pointers must be valid NUL-terminated C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_ios_run(
    config_path: *const c_char,
    wasm_path: *const c_char,
    manifest_path: *const c_char,
    inferlet_id: *const c_char,
    input_json: *const c_char,
) -> *mut c_char {
    let arg = |p: *const c_char| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
    let out = run_impl(
        &arg(config_path),
        &arg(wasm_path),
        &arg(manifest_path),
        &arg(inferlet_id),
        &arg(input_json),
    )
    .unwrap_or_else(|e| format!("PIE ERROR: {e:#}"));
    CString::new(out.replace('\0', ""))
        .expect("NULs stripped above")
        .into_raw()
}

/// Release a string returned by `pie_ios_run`.
///
/// # Safety
/// `s` must be a pointer previously returned by `pie_ios_run` (or null).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_ios_free(s: *mut c_char) {
    if !s.is_null() {
        drop(unsafe { CString::from_raw(s) });
    }
}

/// Per-chunk callback: NUL-terminated UTF-8 chunk + the caller's ctx.
/// Invoked on the thread that called `pie_ios_run_stream`.
pub type PieStreamCb = unsafe extern "C" fn(chunk: *const c_char, ctx: *mut c_void);

use std::os::raw::c_void;

/// Streaming variant of `pie_ios_run`: inferlet stdout/stderr chunks are
/// delivered to `cb` as they arrive (token deltas, for inferlets that
/// print as they generate); the final return value (or error message)
/// comes back as the function result.
///
/// # Safety
/// String pointers must be valid NUL-terminated C strings; `cb` must be
/// callable for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_ios_run_stream(
    config_path: *const c_char,
    wasm_path: *const c_char,
    manifest_path: *const c_char,
    inferlet_id: *const c_char,
    input_json: *const c_char,
    cb: PieStreamCb,
    cb_ctx: *mut c_void,
) -> *mut c_char {
    let arg = |p: *const c_char| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
    let ctx_addr = cb_ctx as usize;
    let emit = move |s: &str| {
        if let Ok(c) = CString::new(s.replace('\0', "")) {
            unsafe { cb(c.as_ptr(), ctx_addr as *mut c_void) };
        }
    };
    let out = stream_impl(
        &arg(config_path),
        &arg(wasm_path),
        &arg(manifest_path),
        &arg(inferlet_id),
        &arg(input_json),
        &emit,
    )
    .unwrap_or_else(|e| format!("PIE ERROR: {e:#}"));
    CString::new(out.replace('\0', ""))
        .expect("NULs stripped above")
        .into_raw()
}

/// Upper bound on one inferlet run. A wedged driver or a forward pass
/// that never returns would otherwise block the caller's serial queue
/// forever with nothing on screen; past this the call returns an error
/// the app can show. Generous: a phone-class CPU decoding 110 tokens
/// after a long prefill is well under a minute.
const TURN_DEADLINE: std::time::Duration = std::time::Duration::from_secs(240);

/// Programs already installed in this process, by wasm path. The wasm
/// ships inside the app bundle and cannot change without a relaunch, so
/// re-uploading (and, with force-overwrite, uninstalling and recompiling)
/// it on every turn is pure latency.
static INSTALLED: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

fn stream_impl(
    config_path: &str,
    wasm: &str,
    manifest: &str,
    inferlet: &str,
    input: &str,
    emit: &dyn Fn(&str),
) -> Result<String> {
    let g = engine_globals(config_path)?;
    g.runtime.block_on(async move {
        let turn = async {
            let client = Client::connect(&g.url).await.context("ws connect")?;
            client.auth_by_token(&g.token).await.context("auth")?;
            let already = INSTALLED
                .lock()
                .map(|v| v.iter().any(|p| p == wasm))
                .unwrap_or(false);
            if !already {
                client
                    .add_program(Path::new(wasm), Path::new(manifest), true)
                    .await
                    .context("add_program")?;
                if let Ok(mut v) = INSTALLED.lock() {
                    v.push(wasm.to_string());
                }
            }
            let mut process = client
                .launch_process(inferlet.to_string(), input.to_string(), true, None)
                .await
                .context("launch_process")?;
            loop {
                match process.recv().await.context("event recv")? {
                    // Only stdout is speakable by contract; stderr is
                    // diagnostics and belongs in the console log.
                    ProcessEvent::Stdout(s) => emit(&s),
                    ProcessEvent::Stderr(s) => eprintln!("[inferlet stderr] {}", s.trim_end()),
                    ProcessEvent::Return(s) => return Ok(s),
                    ProcessEvent::Error(e) => anyhow::bail!("inferlet errored: {e}"),
                    _ => {}
                }
            }
        };
        match tokio::time::timeout(TURN_DEADLINE, turn).await {
            Ok(result) => result,
            Err(_) => anyhow::bail!(
                "turn did not finish within {} s (engine stalled) — relaunch the app",
                TURN_DEADLINE.as_secs()
            ),
        }
    })
}

/// The engine boots once per process (global tracing subscriber and
/// driver channels are process-wide) and stays warm across runs — the
/// natural shape for an app embedding Pie anyway.
struct EngineGlobals {
    runtime: tokio::runtime::Runtime,
    url: String,
    token: String,
}

static ENGINE: std::sync::OnceLock<EngineGlobals> = std::sync::OnceLock::new();
static ENGINE_INIT: std::sync::Mutex<()> = std::sync::Mutex::new(());
/// A boot that failed part-way leaves process-global state behind
/// (driver threads that may still hold the weights, driver-channel
/// indices, the tracing subscriber). Booting again on top of that is a
/// coin flip between a double-loaded model and an abort, so a failed
/// boot is final for the process and says so; the app relaunches.
static BOOT_FAILURE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

fn engine_globals(config_path: &str) -> Result<&'static EngineGlobals> {
    if let Some(g) = ENGINE.get() {
        return Ok(g);
    }
    let _guard = ENGINE_INIT.lock().expect("engine init lock");
    if let Some(g) = ENGINE.get() {
        return Ok(g);
    }
    if let Some(earlier) = BOOT_FAILURE.get() {
        anyhow::bail!("engine boot failed earlier in this process ({earlier}); relaunch the app to retry");
    }

    match boot_engine(config_path) {
        Ok(globals) => {
            let _ = ENGINE.set(globals);
            Ok(ENGINE.get().expect("just set"))
        }
        Err(err) => {
            let message = format!("{err:#}");
            eprintln!("[pie] engine boot failed: {message}");
            let _ = BOOT_FAILURE.set(message);
            Err(err)
        }
    }
}

fn boot_engine(config_path: &str) -> Result<EngineGlobals> {
    let mut cfg = config::Config::from_toml_file(Path::new(config_path))
        .with_context(|| format!("loading config {config_path}"))?;
    // Port 0: let the OS pick, same as `pie run`'s one-shot engine.
    cfg.server.port = 0;

    let runtime = serve::build_runtime(&cfg)?;
    let engine = runtime
        .block_on(serve::start_engine(cfg))
        .context("start_engine")?;
    let url = engine.url.clone();
    let token = engine.token.clone();
    // Keep the engine (driver threads, server task) alive for the process
    // lifetime — iOS reclaims the process wholesale. EngineHandle holds
    // raw pointers and isn't Sync, so it can't live in the static.
    std::mem::forget(engine);

    Ok(EngineGlobals {
        runtime,
        url,
        token,
    })
}

fn run_impl(
    config_path: &str,
    wasm: &str,
    manifest: &str,
    inferlet: &str,
    input: &str,
) -> Result<String> {
    let g = engine_globals(config_path)?;
    let mut log = format!("engine up at {}\n", g.url);
    let result = g.runtime.block_on(drive(
        &g.url, &g.token, wasm, manifest, inferlet, input, &mut log,
    ));
    match result {
        Ok(()) => Ok(log),
        Err(e) => {
            log.push_str(&format!("\nERROR: {e:#}\n"));
            Ok(log)
        }
    }
}

async fn drive(
    url: &str,
    token: &str,
    wasm: &str,
    manifest: &str,
    inferlet: &str,
    input: &str,
    log: &mut String,
) -> Result<()> {
    let client = Client::connect(url).await.context("ws connect")?;
    client.auth_by_token(token).await.context("auth")?;
    client
        .add_program(Path::new(wasm), Path::new(manifest), true)
        .await
        .context("add_program")?;
    log.push_str("inferlet installed\n");

    let mut process = client
        .launch_process(inferlet.to_string(), input.to_string(), true, None)
        .await
        .context("launch_process")?;
    log.push_str(&format!("launched (pid={})\n----\n", process.id()));

    loop {
        match process.recv().await.context("event recv")? {
            ProcessEvent::Stdout(s) | ProcessEvent::Stderr(s) => log.push_str(&s),
            ProcessEvent::Return(s) => {
                log.push_str(&format!("----\nreturn: {s}\n"));
                return Ok(());
            }
            ProcessEvent::Error(e) => anyhow::bail!("inferlet errored: {e}"),
            _ => {}
        }
    }
}
