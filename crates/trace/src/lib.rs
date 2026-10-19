//! Agent traces: which GPU work, KV pages and waits belong to which inferlet.
//!
//! Only the runtime knows which rows of a batched step came from which
//! process, so it is the one place an agent's share of the device can be
//! measured. This crate is the recorder the runtime, the scheduler and the
//! engines report into while a trace is on, and the writer that turns what
//! they reported into a `.pie` file (`TRACE_FORMAT.md` at the repository
//! root): one Chrome/Perfetto JSON object with a few extra keys.
//!
//! Off by default. Every hook starts with one relaxed atomic load and returns
//! when no trace was started. While one is on, a hook appends a small record
//! to an in-memory log under a mutex; nothing is formatted or written until
//! [`finish`], so no step ever waits on I/O.
//!
//! Counts and timings only: the log has no field a prompt, an output or a
//! tool's arguments could ride in. Agents are named by their program, tools
//! by the name the inferlet answered them under.

mod render;

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Instant, SystemTime};

pub use render::{Rendered, render};

/// The window `gpu_busy` and `gpu_share` are measured over.
pub const WINDOW_US: u64 = 10_000;

/// An agent: the runtime's process id, as a number.
pub type AgentKey = u128;

/// What the file's `meta` says about the run. Unknown values are written as
/// `null`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Meta {
    pub pie_version: Option<String>,
    pub commit: Option<String>,
    /// RFC 3339, UTC; [`start`] fills it.
    pub started_at: Option<String>,
    pub model: Option<String>,
    pub scheme: Option<String>,
    pub device: Option<String>,
    pub hardware: Hardware,
    pub kv_page_bytes: Option<u64>,
    pub kv_pages_total: Option<u64>,
    pub weights_bytes: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Hardware {
    pub name: Option<String>,
    pub memory_gb: Option<f64>,
    pub bandwidth_gbs: Option<f64>,
}

/// One agent's rows in one step.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Part {
    pub agent: AgentKey,
    /// Rows of lanes that carry more than one row.
    pub prefill: u32,
    /// Rows of one-row lanes.
    pub decode: u32,
    /// KV tokens the agent's prefill lanes already held when they ran.
    pub cached: u32,
    /// Distinct KV pages the agent's lanes in this step read or wrote.
    pub pages: u32,
    /// When the earliest of the agent's requests in this step was submitted
    /// (trace clock), when known.
    pub queued_since: Option<u64>,
}

impl Part {
    #[must_use]
    pub fn tokens(&self) -> u32 {
        self.prefill + self.decode
    }
}

/// What the recorder logs, in the trace clock (microseconds since [`start`]).
#[derive(Debug, Clone, PartialEq)]
pub enum Record {
    AgentStart {
        t: u64,
        agent: AgentKey,
        program: String,
    },
    AgentEnd {
        t: u64,
        agent: AgentKey,
    },
    /// A step the scheduler posted to the engine.
    Step {
        id: u64,
        t: u64,
        parts: Vec<Part>,
    },
    /// The host saw the step's frame settle.
    StepRetired {
        id: u64,
        t: u64,
    },
    /// A device submission's real start and end, attributed to a step.
    Gpu {
        step: u64,
        start: u64,
        end: u64,
    },
    /// Device memory allocated.
    Memory {
        t: u64,
        bytes: u64,
    },
    /// The agent fed a tool's result back: it waited on the tool since its
    /// last device work.
    Tool {
        t: u64,
        agent: AgentKey,
        name: String,
    },
    Evict {
        t: u64,
        agent: AgentKey,
        pages: u32,
    },
}

struct State {
    path: PathBuf,
    epoch: Instant,
    meta: Meta,
    log: Vec<Record>,
    next_step: u64,
    /// `(engine frame, step in frame)` to trace step.
    frames: HashMap<(u64, u32), u64>,
    /// Steps submitted to the engine and not yet given a device span.
    awaiting_gpu: BTreeSet<u64>,
    last_memory_us: Option<u64>,
}

static ENABLED: AtomicBool = AtomicBool::new(false);
static STATE: Mutex<Option<State>> = Mutex::new(None);

fn with_state<T>(f: impl FnOnce(&mut State) -> T) -> Option<T> {
    let mut guard = STATE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    guard.as_mut().map(f)
}

/// Whether a trace is being recorded. Every hook checks this first.
#[inline]
#[must_use]
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Starts recording, to be written to `path` by [`finish`]. A second start
/// while one is on is ignored.
pub fn start(path: impl Into<PathBuf>, mut meta: Meta) {
    let mut guard = STATE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if guard.is_some() {
        return;
    }
    if meta.started_at.is_none() {
        meta.started_at = Some(rfc3339_utc(SystemTime::now()));
    }
    *guard = Some(State {
        path: path.into(),
        epoch: Instant::now(),
        meta,
        log: Vec::with_capacity(1 << 16),
        next_step: 1,
        frames: HashMap::new(),
        awaiting_gpu: BTreeSet::new(),
        last_memory_us: None,
    });
    ENABLED.store(true, Ordering::Relaxed);
}

/// Starts recording when `$PIE_TRACE` names a file and nothing started one.
pub fn start_from_env(meta: Meta) -> Option<PathBuf> {
    let path = std::env::var_os("PIE_TRACE").filter(|p| !p.is_empty())?;
    let path = PathBuf::from(path);
    start(path.clone(), meta);
    Some(path)
}

/// The trace clock: microseconds since [`start`]; 0 when off.
#[must_use]
pub fn now_us() -> u64 {
    if !enabled() {
        return 0;
    }
    with_state(|s| elapsed_us(s.epoch)).unwrap_or(0)
}

fn elapsed_us(epoch: Instant) -> u64 {
    u64::try_from(epoch.elapsed().as_micros()).unwrap_or(u64::MAX)
}

/// Fills in what a component knows about the run.
pub fn set_meta(f: impl FnOnce(&mut Meta)) {
    if !enabled() {
        return;
    }
    with_state(|s| f(&mut s.meta));
}

fn push(f: impl FnOnce(u64) -> Record) {
    if !enabled() {
        return;
    }
    with_state(|s| {
        let t = elapsed_us(s.epoch);
        s.log.push(f(t));
    });
}

pub fn agent_start(agent: AgentKey, program: &str) {
    // A program name, not anything the program said.
    let program = program.split('@').next().unwrap_or(program).to_string();
    push(|t| Record::AgentStart { t, agent, program });
}

pub fn agent_end(agent: AgentKey) {
    push(|t| Record::AgentEnd { t, agent });
}

/// The scheduler posted a step carrying `parts`; returns the step's id (0
/// when off).
pub fn step_posted(parts: Vec<Part>) -> u64 {
    if !enabled() {
        return 0;
    }
    with_state(|s| {
        let id = s.next_step;
        s.next_step += 1;
        let t = elapsed_us(s.epoch);
        s.log.push(Record::Step { id, t, parts });
        id
    })
    .unwrap_or(0)
}

/// The engine took `steps` (in order) as frame `frame`.
pub fn frame_submitted(frame: u64, steps: &[u64]) {
    if !enabled() {
        return;
    }
    with_state(|s| {
        for (index, &step) in steps.iter().enumerate() {
            if step == 0 {
                continue;
            }
            let index = u32::try_from(index).unwrap_or(u32::MAX);
            s.frames.insert((frame, index), step);
            s.awaiting_gpu.insert(step);
        }
    });
}

pub fn step_retired(steps: &[u64]) {
    if !enabled() {
        return;
    }
    with_state(|s| {
        let t = elapsed_us(s.epoch);
        for &id in steps.iter().filter(|&&id| id != 0) {
            s.log.push(Record::StepRetired { id, t });
        }
    });
}

/// A device submission ran from `start_s` to `end_s` on the host clock the
/// device stamps in (seconds), read as `host_now_s` now. `at` is the engine's
/// `(frame, step)` when it knows it; otherwise the span goes to the oldest
/// submitted step still without one.
pub fn gpu_span_host(at: Option<(u64, u32)>, start_s: f64, end_s: f64, host_now_s: f64) {
    if !enabled() || start_s <= 0.0 || end_s.is_nan() || end_s < start_s {
        return;
    }
    with_state(|s| {
        let now = elapsed_us(s.epoch) as f64;
        let offset = now - host_now_s * 1e6;
        let start = (start_s * 1e6 + offset).max(0.0) as u64;
        let end = (end_s * 1e6 + offset).max(0.0) as u64;
        gpu_span_locked(s, at, start, end);
    });
}

/// [`gpu_span_host`] with the span already on the trace clock.
pub fn gpu_span(at: Option<(u64, u32)>, start: u64, end: u64) {
    if !enabled() {
        return;
    }
    with_state(|s| gpu_span_locked(s, at, start, end));
}

fn gpu_span_locked(s: &mut State, at: Option<(u64, u32)>, start: u64, end: u64) {
    let step = match at.and_then(|key| s.frames.remove(&key)) {
        Some(step) => Some(step),
        None => s.awaiting_gpu.iter().next().copied(),
    };
    let Some(step) = step else {
        return;
    };
    s.awaiting_gpu.remove(&step);
    s.log.push(Record::Gpu {
        step,
        start,
        end: end.max(start),
    });
}

/// Device memory allocated now; kept at most once per [`WINDOW_US`].
pub fn memory(bytes: u64) {
    if !enabled() {
        return;
    }
    with_state(|s| {
        let t = elapsed_us(s.epoch);
        if s.last_memory_us
            .is_some_and(|last| t.saturating_sub(last) < WINDOW_US)
        {
            return;
        }
        s.last_memory_us = Some(t);
        s.log.push(Record::Memory { t, bytes });
    });
}

/// Whether [`memory`] would keep a sample now: lets a caller skip reading
/// the device's counter.
#[must_use]
pub fn memory_due() -> bool {
    if !enabled() {
        return false;
    }
    with_state(|s| {
        s.last_memory_us
            .is_none_or(|last| elapsed_us(s.epoch).saturating_sub(last) >= WINDOW_US)
    })
    .unwrap_or(false)
}

/// The agent fed the result of tool `name` back.
pub fn tool(agent: AgentKey, name: &str) {
    let name = name.to_string();
    push(|t| Record::Tool { t, agent, name });
}

pub fn evict(agent: AgentKey, pages: u32) {
    push(|t| Record::Evict { t, agent, pages });
}

/// What [`finish`] wrote.
#[derive(Debug, Clone)]
pub struct Written {
    pub path: PathBuf,
    pub bytes: usize,
    pub agents: usize,
    pub steps: usize,
    pub events: usize,
}

/// Stops recording and writes the file. `None` when no trace was on.
pub fn finish() -> std::io::Result<Option<Written>> {
    ENABLED.store(false, Ordering::Relaxed);
    let state = STATE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take();
    let Some(state) = state else {
        return Ok(None);
    };
    let rendered = render(&state.meta, &state.log);
    let text = serde_json::to_string(&rendered.json).map_err(std::io::Error::other)?;
    write_file(&state.path, text.as_bytes())?;
    Ok(Some(Written {
        path: state.path,
        bytes: text.len(),
        agents: rendered.agents,
        steps: rendered.steps,
        events: rendered.events,
    }))
}

fn write_file(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, bytes)
}

/// The memory bandwidth Apple states for a chip, by the name Metal gives it
/// and its GPU cores where one name covers two bandwidths.
#[must_use]
pub fn apple_bandwidth_gbs(name: &str, gpu_cores: Option<u32>) -> Option<f64> {
    let chip = name.trim().strip_prefix("Apple ").unwrap_or(name.trim());
    let cores = gpu_cores.unwrap_or(0);
    let gbs = match chip {
        "M1" => 68.25,
        "M1 Pro" => 200.0,
        "M1 Max" => 400.0,
        "M1 Ultra" => 800.0,
        "M2" => 100.0,
        "M2 Pro" => 200.0,
        "M2 Max" => 400.0,
        "M2 Ultra" => 800.0,
        "M3" => 100.0,
        "M3 Pro" => 150.0,
        "M3 Max" if cores == 30 => 300.0,
        "M3 Max" => 400.0,
        "M3 Ultra" => 819.0,
        "M4" => 120.0,
        "M4 Pro" => 273.0,
        "M4 Max" if cores == 32 => 410.0,
        "M4 Max" => 546.0,
        "M5" => 153.0,
        "M5 Pro" => 307.0,
        "M5 Max" if cores == 32 => 460.0,
        "M5 Max" => 614.0,
        _ => return None,
    };
    Some(gbs)
}

/// `2026-10-18T12:00:00Z` for `at`.
#[must_use]
pub fn rfc3339_utc(at: SystemTime) -> String {
    let secs = at
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let days = i64::try_from(secs / 86_400).unwrap_or(0);
    let rem = secs % 86_400;
    // Howard Hinnant's civil-from-days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rem / 3_600,
        (rem / 60) % 60,
        rem % 60
    )
}

#[cfg(test)]
mod tests;
