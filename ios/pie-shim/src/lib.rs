//! C-ABI shim embedding the Pie engine in an iOS app.
//!
//! Mirrors the boot path of `pie run` (src/ops/run.rs): parse a standalone
//! config TOML with `pie::derive::derive_standalone`, compose the
//! controller, gateway and worker in-process with `pie::run_standalone`,
//! and drive one inferlet turn per call over the loopback WebSocket with
//! `pie-client`. The engine boots once and stays warm for the life of the
//! process. Each call installs the component on first sight, launches it,
//! streams its stdout to the caller as it is produced, and returns its
//! return value as a heap-allocated C string the caller releases with
//! `pie_ios_free`.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::panic::{self, AssertUnwindSafe};
use std::path::Path;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use client::client::{Client, Process, ProcessEvent};
use tokio::time::{Instant, timeout, timeout_at};

/// Per-chunk callback: a NUL-terminated UTF-8 chunk plus the caller's ctx.
/// Invoked on the thread that called `pie_ios_run_stream`.
pub type PieStreamCb = unsafe extern "C" fn(chunk: *const c_char, ctx: *mut c_void);

/// Run one inferlet turn: boot the engine on first use, install the
/// component at `wasm_path` on first sight, launch it with `input_json`,
/// deliver every stdout chunk to `cb` as it arrives, and return the
/// inferlet's return value.
///
/// `version` may be null. When given it must equal the version the
/// component's `pie.package` section declares (the engine refuses the
/// install otherwise), and it names the version of a component that has
/// no such section.
/// The result is a malloc'd C string: the return value, or a message
/// starting with "PIE ERROR:" on any failure, a panic on this thread
/// included. Release it with `pie_ios_free`.
///
/// # Safety
/// `config_path`, `wasm_path` and `input_json` must be valid
/// NUL-terminated C strings; `version` must be one or null; `cb` must be
/// callable for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_ios_run_stream(
    config_path: *const c_char,
    wasm_path: *const c_char,
    version: *const c_char,
    input_json: *const c_char,
    cb: PieStreamCb,
    cb_ctx: *mut c_void,
) -> *mut c_char {
    // SAFETY: the caller guarantees each non-null pointer is a
    // NUL-terminated string that outlives this call.
    let arg = |p: *const c_char| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
    let version = (!version.is_null()).then(|| arg(version));
    let emit = move |chunk: &str| {
        if let Ok(c) = CString::new(chunk.replace('\0', "")) {
            // SAFETY: `cb` is callable for the duration of the call and `c`
            // outlives the callback.
            unsafe { cb(c.as_ptr(), cb_ctx) };
        }
    };
    // Unwinding into the C caller aborts the app with nothing on screen;
    // an error string at least says why.
    let out = catching(|| {
        stream_impl(
            &arg(config_path),
            &arg(wasm_path),
            version.as_deref(),
            &arg(input_json),
            &emit,
        )
    })
    .unwrap_or_else(|e| format!("PIE ERROR: {e:#}"));
    CString::new(out.replace('\0', ""))
        .expect("NULs stripped above")
        .into_raw()
}

/// Release a string returned by `pie_ios_run_stream`.
///
/// # Safety
/// `s` must be a pointer previously returned by `pie_ios_run_stream` (or
/// null) and must not be used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_ios_free(s: *mut c_char) {
    if !s.is_null() {
        // SAFETY: `s` came from `CString::into_raw` in `pie_ios_run_stream`.
        drop(unsafe { CString::from_raw(s) });
    }
}

/// Upper bound on one inferlet turn. A wedged engine or a forward pass
/// that never returns would otherwise block the caller's serial queue
/// forever with nothing on screen; past this the call returns an error
/// the app can show. Generous: a phone GPU decoding 120 tokens after a
/// long prefill is well under a minute.
const TURN_DEADLINE: Duration = Duration::from_secs(240);

/// How long a turn waits for the engine to acknowledge a termination or
/// for its connection to close before moving on.
const CLOSE_GRACE: Duration = Duration::from_secs(5);

/// Tracing filter when `RUST_LOG` is unset. tarpc logs every RPC at INFO,
/// which would flood the app's log mirror of stderr.
const DEFAULT_LOG_FILTER: &str = "info,tarpc=warn";

/// Identity and username this shim presents to the engine. The embedded
/// gateway accepts a bare name without a key, as it does for `pie run`.
const CLIENT_IDENTITY: &str = "pie-ios";

/// Programs already installed in this process: wasm path to the name the
/// engine gave it (from the component's `pie.package` section when it has
/// one, else file stem plus version). The wasm ships inside the app bundle
/// and cannot change without a relaunch, so re-uploading (and, with
/// force-overwrite, uninstalling and recompiling) it on every turn is
/// pure latency.
static INSTALLED: Mutex<Vec<(String, String)>> = Mutex::new(Vec::new());

/// Runs `f`, reporting a panic as an error instead of unwinding. Tasks on
/// the tokio workers are already contained by tokio; this covers the
/// calling thread, which boots the engine and polls the turn.
fn catching<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
    match panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "non-string panic payload".to_string());
            Err(anyhow!("panic: {message}"))
        }
    }
}

fn stream_impl(
    config_path: &str,
    wasm: &str,
    version: Option<&str>,
    input: &str,
    emit: &dyn Fn(&str),
) -> Result<String> {
    let g = engine_globals(config_path)?;
    g.runtime
        .block_on(turn(&g.ws_url, wasm, version, input, emit))
}

/// One client connection per turn, exactly as `pie run` does it: connect,
/// identify, drive, close. Closing joins the client's reader and writer
/// tasks so a long session does not accumulate them.
///
/// The deadline spans the whole turn. When it fires, the launched process
/// is terminated (best effort) rather than left running: the phone config
/// allows only a handful of concurrent processes, and a stalled one would
/// hold its seat while the app shows the error.
async fn turn(
    ws_url: &str,
    wasm: &str,
    version: Option<&str>,
    input: &str,
    emit: &dyn Fn(&str),
) -> Result<String> {
    let deadline = Instant::now() + TURN_DEADLINE;
    let client = match timeout_at(deadline, connect(ws_url)).await {
        Ok(client) => client?,
        Err(_) => return Err(stalled()),
    };
    // The launched process lives out here so that a deadline firing
    // mid-turn still knows what to terminate.
    let mut process = None;
    let drive = drive(&client, &mut process, wasm, version, input, emit);
    let outcome = match timeout_at(deadline, drive).await {
        Ok(outcome) => outcome,
        Err(_) => {
            if let Some(process) = &process {
                match timeout(CLOSE_GRACE, client.terminate_process(process.id())).await {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => tracing::warn!(error = %e, "terminating the stalled process"),
                    Err(_) => tracing::warn!(
                        "the engine did not acknowledge termination within {} s",
                        CLOSE_GRACE.as_secs()
                    ),
                }
            }
            Err(stalled())
        }
    };
    // The process holds a reference to the connection, and the client's
    // writer task only ends once every reference is gone.
    drop(process);
    // Bounded so that a close that hangs cannot eat into the next turn.
    match timeout(CLOSE_GRACE, client.close()).await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => tracing::warn!(error = %e, "closing the client connection"),
        Err(_) => tracing::warn!(
            "client connection did not close within {} s",
            CLOSE_GRACE.as_secs()
        ),
    }
    outcome
}

fn stalled() -> anyhow::Error {
    anyhow!(
        "turn did not finish within {} s (engine stalled) - relaunch the app",
        TURN_DEADLINE.as_secs()
    )
}

async fn connect(ws_url: &str) -> Result<Client> {
    let client = Client::connect_with_identity(ws_url, CLIENT_IDENTITY)
        .await
        .context("connect to the embedded engine")?;
    client
        .authenticate(CLIENT_IDENTITY, &None)
        .await
        .context("authenticate")?;
    Ok(client)
}

async fn drive(
    client: &Client,
    slot: &mut Option<Process>,
    wasm: &str,
    version: Option<&str>,
    input: &str,
    emit: &dyn Fn(&str),
) -> Result<String> {
    let program = installed_program(client, wasm, version).await?;
    let process = slot.insert(
        client
            .launch_process(program.clone(), input.to_string(), true)
            .await
            .with_context(|| format!("launching {program}"))?,
    );
    loop {
        match process.recv().await.context("reading process output")? {
            // Only stdout is speakable by contract; everything else is
            // diagnostics and belongs in the console log.
            ProcessEvent::Stdout(text) => emit(&text),
            ProcessEvent::Stderr(text) => eprintln!("[inferlet stderr] {}", text.trim_end()),
            ProcessEvent::Message(text) => eprintln!("[inferlet message] {}", text.trim_end()),
            // Nowhere on a phone to put a file the inferlet sends back.
            ProcessEvent::File(file) => eprintln!(
                "[inferlet file] dropped {} ({} bytes)",
                file.file_name("unnamed"),
                file.data.len()
            ),
            ProcessEvent::Return(value) => return Ok(value),
            ProcessEvent::Error(message) => return Err(anyhow!("inferlet errored: {message}")),
        }
    }
}

/// The installed name of the component at `wasm`, uploading it the first
/// time this process sees that path.
async fn installed_program(client: &Client, wasm: &str, version: Option<&str>) -> Result<String> {
    let known = INSTALLED.lock().ok().and_then(|installed| {
        installed
            .iter()
            .find(|(path, _)| path == wasm)
            .map(|(_, name)| name.clone())
    });
    if let Some(name) = known {
        return Ok(name);
    }
    let bytes = std::fs::read(wasm).with_context(|| format!("reading {wasm}"))?;
    let file = Path::new(wasm)
        .file_name()
        .and_then(|f| f.to_str())
        .with_context(|| format!("{wasm} has no file name"))?;
    let name = client
        .add_program_bytes(&bytes, file, version, true)
        .await
        .with_context(|| format!("installing {file}"))?;
    tracing::info!(program = %name, "installed inferlet");
    if let Ok(mut installed) = INSTALLED.lock() {
        installed.push((wasm.to_string(), name.clone()));
    }
    Ok(name)
}

/// The engine boots once per process (the tracing subscriber and the
/// Metal device are process-wide) and stays warm across turns: the
/// natural shape for an app embedding Pie anyway.
struct EngineGlobals {
    runtime: tokio::runtime::Runtime,
    /// The embedded gateway's client endpoint, on the loopback port the OS
    /// picked.
    ws_url: String,
    /// Keeps the controller, the gateway task and the worker alive for the
    /// life of the process; iOS reclaims the process wholesale.
    _pie: pie::StandaloneHandle,
}

static ENGINE: OnceLock<EngineGlobals> = OnceLock::new();
static ENGINE_INIT: Mutex<()> = Mutex::new(());
/// A boot that failed or panicked part-way leaves process-global state
/// behind (a worker that may still hold the weights, the tracing
/// subscriber, the crypto provider). Booting again on top of that is a
/// coin flip between a double-loaded model and an abort, so a failed boot
/// is final for the process and says so; the app relaunches.
static BOOT_FAILURE: OnceLock<String> = OnceLock::new();

fn engine_globals(config_path: &str) -> Result<&'static EngineGlobals> {
    if let Some(g) = ENGINE.get() {
        return Ok(g);
    }
    let _guard = ENGINE_INIT.lock().expect("engine init lock");
    if let Some(g) = ENGINE.get() {
        return Ok(g);
    }
    if let Some(earlier) = BOOT_FAILURE.get() {
        anyhow::bail!(
            "engine boot failed earlier in this process ({earlier}); relaunch the app to retry"
        );
    }

    match catching(|| boot_engine(config_path)) {
        Ok(globals) => Ok(ENGINE.get_or_init(|| globals)),
        Err(err) => {
            let message = format!("{err:#}");
            tracing::error!("engine boot failed: {message}");
            let _ = BOOT_FAILURE.set(message);
            Err(err)
        }
    }
}

fn boot_engine(config_path: &str) -> Result<EngineGlobals> {
    install_tracing();
    let content = std::fs::read_to_string(config_path)
        .with_context(|| format!("reading config {config_path}"))?;
    let (controller, gateway, mut worker) = pie::derive::derive_standalone(&content)?;
    // Port 0: the OS picks a free loopback port, so a port left bound by
    // an earlier process can never block the boot.
    worker.server.port = 0;
    // The runtime a worker daemon gets (worker::serve::build_runtime),
    // sized from [server] worker_threads.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(worker.server.worker_threads)
        .enable_all()
        .build()
        .context("building tokio runtime")?;
    let handle = runtime
        .block_on(pie::run_standalone(controller, gateway, worker))
        .context("boot the engine")?;
    let ws_url = format!("ws://{}/v1/ws", handle.listen_addr);
    tracing::info!(%ws_url, "pie engine up");
    Ok(EngineGlobals {
        runtime,
        ws_url,
        _pie: handle,
    })
}

/// `run_standalone` installs no tracing subscriber (the pie binary does
/// that in its bootstrap), so the shim does, once. The app mirrors stderr
/// into its log file, which is the only place engine diagnostics can go
/// on a phone. `try_init` makes a second call harmless.
fn install_tracing() {
    use tracing_subscriber::EnvFilter;
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(DEFAULT_LOG_FILTER));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();
}
