//! C-ABI shim embedding the Pie engine in an iOS app.
//!
//! Mirrors the boot path of `pie run` (src/ops/run.rs): parse a standalone
//! config TOML with `pie::derive::derive_standalone`, compose the
//! controller, gateway and worker in-process with `pie::run_standalone`,
//! and drive one inferlet turn per call over the loopback WebSocket with
//! `pie-client`. The engine boots once and stays warm for the life of the
//! process. Each call installs the component on first sight, launches it,
//! streams its reply text (stdout) and reasoning text (session messages)
//! to the caller as they are produced, and returns its return value as a
//! heap-allocated C string the caller releases with `pie_ios_free`.
//!
//! Every turn carries a caller-chosen id, so `pie_ios_cancel` can stop it
//! from any thread: in flight, by terminating its process on the engine,
//! or before it has started, so that it never launches. A voice app needs
//! both: the user talks over a reply that is still being generated, and the
//! next question must not wait behind the one it replaces.

use std::collections::VecDeque;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::panic::{self, AssertUnwindSafe};
use std::path::Path;
use std::pin::pin;
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use client::client::{Client, Process, ProcessEvent};
use tokio::sync::watch;
use tokio::time::{Instant, timeout, timeout_at};

/// Per-chunk callback: what the chunk is (`PIE_CHUNK_REPLY` or
/// `PIE_CHUNK_REASONING`), the chunk as NUL-terminated UTF-8, and the
/// caller's ctx. Invoked on the thread that called `pie_ios_run_stream`.
pub type PieStreamCb = unsafe extern "C" fn(kind: i32, chunk: *const c_char, ctx: *mut c_void);

/// Chunk kind for reply text: the inferlet's stdout, which by the
/// voice-chat contract is speakable as it arrives.
pub const PIE_CHUNK_REPLY: i32 = 0;

/// Chunk kind for reasoning text: the inferlet's session messages. Never
/// to be spoken.
pub const PIE_CHUNK_REASONING: i32 = 1;

/// What `pie_ios_run_stream` returns, exactly, for a cancelled turn.
pub const PIE_CANCELLED: &str = "PIE CANCELLED";

/// Run one inferlet turn: boot the engine on first use, install the
/// component at `wasm_path` on first sight, launch it with `input_json`,
/// deliver every stdout chunk (kind 0) and session message (kind 1) to `cb`
/// as it arrives, and return the inferlet's return value.
///
/// `version` may be null. When given it must equal the version the
/// component's `pie.package` section declares (the engine refuses the
/// install otherwise), and it names the version of a component that has
/// no such section.
///
/// `turn_id` names the turn for `pie_ios_cancel`. Ids must be unique for
/// the life of the process; a cancel that arrived for this id before the
/// call makes it return at once, without launching anything.
///
/// The result is a malloc'd C string: the return value; exactly
/// `PIE CANCELLED` when the turn was cancelled, before or during (any text
/// already streamed was delivered through `cb`); or a message starting with
/// `PIE ERROR: ` on any failure, a panic on this thread included. Release
/// it with `pie_ios_free`.
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
    turn_id: u64,
    cb: PieStreamCb,
    cb_ctx: *mut c_void,
) -> *mut c_char {
    // SAFETY: the caller guarantees each non-null pointer is a
    // NUL-terminated string that outlives this call.
    let arg = |p: *const c_char| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
    let version = (!version.is_null()).then(|| arg(version));
    let emit = move |kind: i32, chunk: &str| {
        if let Ok(c) = CString::new(chunk.replace('\0', "")) {
            // SAFETY: `cb` is callable for the duration of the call and `c`
            // outlives the callback.
            unsafe { cb(kind, c.as_ptr(), cb_ctx) };
        }
    };
    // Unwinding into the C caller aborts the app with nothing on screen;
    // an error string at least says why.
    let out = catching(|| {
        stream_impl(
            turn_id,
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

/// Stop the turn `turn_id`.
///
/// A turn in flight has its process terminated on the engine and returns
/// `PIE CANCELLED` as soon as the engine acknowledges, a few milliseconds.
/// A turn still booting the engine or installing its component (both only
/// ever on the first turn of the process) stops right after that step,
/// before it launches. A turn not started yet returns `PIE CANCELLED`
/// without launching when it starts; the ids of the most recent
/// `REMEMBERED_IDS` such cancels are kept for that. On a turn that has
/// already returned this does nothing.
///
/// Never blocks on the engine: it flips the turn's state and wakes it, so
/// it is safe to call from any thread, the main thread included, while
/// `pie_ios_run_stream` is blocked on another.
#[unsafe(no_mangle)]
pub extern "C" fn pie_ios_cancel(turn_id: u64) {
    // Unwinding into the C caller aborts the app. A cancel that panicked
    // is one that did not happen, and the turn's deadline still bounds it.
    let _ = panic::catch_unwind(|| cancel(turn_id));
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

/// How long the embedded controller waits for a heartbeat before it
/// evicts a node: longer than any suspension an app comes back from (see
/// `boot_engine`).
const EMBEDDED_HEARTBEAT_TIMEOUT: Duration = Duration::from_secs(365 * 24 * 60 * 60);

/// Tracing filter when `RUST_LOG` is unset. tarpc logs every RPC at INFO,
/// which would flood the app's log mirror of stderr.
const DEFAULT_LOG_FILTER: &str = "info,tarpc=warn";

/// Identity and username this shim presents to the engine. The embedded
/// gateway accepts a bare name without a key, as it does for `pie run`.
const CLIENT_IDENTITY: &str = "pie-ios";

/// How many ids the turn registry remembers of turns cancelled before they
/// started, and of turns that have returned. A voice app has a turn or two
/// queued at most, so this only has to outlast that queue.
const REMEMBERED_IDS: usize = 64;

/// Programs already installed in this process: wasm path to the name the
/// engine gave it (from the component's `pie.package` section when it has
/// one, else file stem plus version). The wasm ships inside the app bundle
/// and cannot change without a relaunch, so re-uploading (and, with
/// force-overwrite, uninstalling and recompiling) it on every turn is
/// pure latency.
static INSTALLED: Mutex<Vec<(String, String)>> = Mutex::new(Vec::new());

/// Turns by id, for `pie_ios_cancel`, which may run on any thread while
/// the turn it names blocks another. Held only to flip state, never across
/// a call into the engine, so a cancel never waits on one.
static TURNS: Mutex<Turns> = Mutex::new(Turns {
    live: Vec::new(),
    early: VecDeque::new(),
    finished: VecDeque::new(),
    serial: 0,
});

struct Turns {
    /// Turns registered and not yet returned.
    live: Vec<LiveTurn>,
    /// Ids cancelled before any turn registered them, oldest first. The
    /// turn that registers one takes it out and returns at once.
    early: VecDeque<u64>,
    /// Ids of turns that have returned, oldest first, so that a cancel
    /// arriving just after its turn finished stays the no-op it should be
    /// instead of reading as a cancel-before-start.
    finished: VecDeque<u64>,
    /// Tells registrations apart, so a turn removes exactly its own entry.
    serial: u64,
}

struct LiveTurn {
    id: u64,
    serial: u64,
    /// Flipped to true by a cancel; the turn watches the other end.
    cancel: watch::Sender<bool>,
}

/// Every critical section on the registry is a few pushes and removals
/// that cannot be left half done, so a panic elsewhere while it was held
/// leaves nothing inconsistent behind; recover the guard rather than turn
/// every later turn and cancel into a panic.
fn turns() -> MutexGuard<'static, Turns> {
    TURNS.lock().unwrap_or_else(PoisonError::into_inner)
}

fn remember(ids: &mut VecDeque<u64>, id: u64) {
    if ids.len() == REMEMBERED_IDS {
        ids.pop_front();
    }
    ids.push_back(id);
}

fn cancel(turn_id: u64) {
    let mut turns = turns();
    let mut live = false;
    for turn in turns.live.iter().filter(|turn| turn.id == turn_id) {
        // Works whether or not the turn is waiting on it yet: a turn reads
        // the current value before it waits.
        turn.cancel.send_replace(true);
        live = true;
    }
    if live || turns.finished.contains(&turn_id) || turns.early.contains(&turn_id) {
        return;
    }
    remember(&mut turns.early, turn_id);
}

/// One turn's place in the registry, from before the engine boots until
/// the call returns (or unwinds).
struct Registration {
    id: u64,
    serial: u64,
    cancel: watch::Receiver<bool>,
}

impl Registration {
    /// Registers `id`, or `None` when it was cancelled before it started.
    fn new(id: u64) -> Option<Self> {
        let mut turns = turns();
        if let Some(at) = turns.early.iter().position(|&early| early == id) {
            turns.early.remove(at);
            remember(&mut turns.finished, id);
            return None;
        }
        let (tx, rx) = watch::channel(false);
        turns.serial += 1;
        let serial = turns.serial;
        turns.live.push(LiveTurn {
            id,
            serial,
            cancel: tx,
        });
        Some(Self {
            id,
            serial,
            cancel: rx,
        })
    }

    fn is_cancelled(&self) -> bool {
        *self.cancel.borrow()
    }
}

impl Drop for Registration {
    fn drop(&mut self) {
        let mut turns = turns();
        turns.live.retain(|turn| turn.serial != self.serial);
        remember(&mut turns.finished, self.id);
    }
}

/// Resolves once the turn is cancelled, and never if it is not.
async fn cancelled(mut signal: watch::Receiver<bool>) {
    // The registry holds the sender until the turn returns, so `wait_for`
    // fails only once nothing is waiting on it any more.
    if signal.wait_for(|&cancelled| cancelled).await.is_err() {
        std::future::pending::<()>().await;
    }
}

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

/// How a turn that did not fail ended.
enum Outcome {
    /// The inferlet's return value.
    Returned(String),
    Cancelled,
}

fn stream_impl(
    turn_id: u64,
    config_path: &str,
    wasm: &str,
    version: Option<&str>,
    input: &str,
    emit: &dyn Fn(i32, &str),
) -> Result<String> {
    // Registered before the boot, which can take seconds on the first turn,
    // so a cancel during it is not lost.
    let Some(registration) = Registration::new(turn_id) else {
        return Ok(PIE_CANCELLED.to_string());
    };
    let g = engine_globals(config_path)?;
    if registration.is_cancelled() {
        return Ok(PIE_CANCELLED.to_string());
    }
    let outcome = g.runtime.block_on(turn(
        &g.ws_url,
        wasm,
        version,
        input,
        emit,
        &registration.cancel,
    ))?;
    Ok(match outcome {
        Outcome::Returned(value) => value,
        Outcome::Cancelled => PIE_CANCELLED.to_string(),
    })
}

/// One client connection per turn, exactly as `pie run` does it: connect,
/// identify, drive, close. Closing joins the client's reader and writer
/// tasks so a long session does not accumulate them.
///
/// The deadline spans the whole turn. When it fires, or the turn is
/// cancelled, the launched process is terminated rather than left running:
/// the phone config allows only a handful of concurrent processes, and one
/// still generating a reply nobody wants holds its seat and the GPU while
/// the next turn waits behind it.
async fn turn(
    ws_url: &str,
    wasm: &str,
    version: Option<&str>,
    input: &str,
    emit: &dyn Fn(i32, &str),
    cancel: &watch::Receiver<bool>,
) -> Result<Outcome> {
    let deadline = Instant::now() + TURN_DEADLINE;
    let client = match timeout_at(deadline, connect(ws_url)).await {
        Ok(client) => client?,
        Err(_) => return Err(stalled()),
    };
    // The launched process lives out here so that a deadline firing or a
    // cancel arriving mid-turn still knows what to terminate.
    let mut process = None;
    let drive = drive(&client, &mut process, wasm, version, input, emit, cancel);
    let outcome = match timeout_at(deadline, drive).await {
        Ok(Ok(Outcome::Cancelled)) => {
            if let Some(process) = &process {
                terminate(&client, process, "cancelled").await;
            }
            Ok(Outcome::Cancelled)
        }
        Ok(outcome) => outcome,
        Err(_) => {
            if let Some(process) = &process {
                terminate(&client, process, "stalled").await;
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

/// Best effort, bounded by `CLOSE_GRACE`. The engine acknowledges as soon
/// as it has signalled the process; the fires it already has in flight
/// settle on their own.
async fn terminate(client: &Client, process: &Process, why: &str) {
    match timeout(CLOSE_GRACE, client.terminate_process(process.id())).await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => tracing::warn!(error = %e, "terminating the {why} process"),
        Err(_) => tracing::warn!(
            "the engine did not acknowledge terminating the {why} process within {} s",
            CLOSE_GRACE.as_secs()
        ),
    }
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
    emit: &dyn Fn(i32, &str),
    cancel: &watch::Receiver<bool>,
) -> Result<Outcome> {
    if *cancel.borrow() {
        return Ok(Outcome::Cancelled);
    }
    // The install is not interrupted: abandoning an upload part-way would
    // leave the engine compiling a program this process has no record of,
    // and the next turn needs it anyway. It happens once per process.
    let program = installed_program(client, wasm, version).await?;
    // Nor is the launch, a loopback round trip: once it returns the process
    // has an id to terminate, while a launch abandoned mid-flight would
    // leave one running that this turn cannot name.
    if *cancel.borrow() {
        return Ok(Outcome::Cancelled);
    }
    let process = slot.insert(
        client
            .launch_process(program.clone(), input.to_string(), true)
            .await
            .with_context(|| format!("launching {program}"))?,
    );
    let mut stop = pin!(cancelled(cancel.clone()));
    loop {
        // A cancel wins a tie: the reply it interrupts is unwanted even if
        // its last event is already here.
        let event = tokio::select! {
            biased;
            () = &mut stop => return Ok(Outcome::Cancelled),
            event = process.recv() => event.context("reading process output")?,
        };
        match event {
            ProcessEvent::Stdout(text) => emit(PIE_CHUNK_REPLY, &text),
            ProcessEvent::Message(text) => emit(PIE_CHUNK_REASONING, &text),
            // Diagnostics; the app mirrors stderr into its log file.
            ProcessEvent::Stderr(text) => eprintln!("[inferlet stderr] {}", text.trim_end()),
            // Nowhere on a phone to put a file the inferlet sends back.
            ProcessEvent::File(file) => eprintln!(
                "[inferlet file] dropped {} ({} bytes)",
                file.file_name("unnamed"),
                file.data.len()
            ),
            ProcessEvent::Return(value) => return Ok(Outcome::Returned(value)),
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
    let (mut controller, gateway, mut worker) = pie::derive::derive_standalone(&content)?;
    // Port 0: the OS picks a free loopback port, so a port left bound by
    // an earlier process can never block the boot.
    worker.server.port = 0;
    // The controller evicts a node whose heartbeats stop for the heartbeat
    // timeout, and a worker it has evicted aborts the process to be
    // restarted by its supervisor. Here the controller and the worker live
    // in one process that iOS suspends whole whenever the app is in the
    // background: every heartbeat stops at once, and on resume the
    // controller's eviction tick and the worker's next heartbeat race. When
    // the tick wins, the worker is evicted and the app dies the moment the
    // user comes back to it. Nothing in this process can be partitioned
    // from the rest, so eviction only ever reflects a suspension; it is
    // pushed out of reach.
    controller.heartbeat_timeout = EMBEDDED_HEARTBEAT_TIMEOUT;
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
