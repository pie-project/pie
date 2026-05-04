//! Live metrics provider — polls a running pie engine via
//! `pie-client` and (optionally) NVML for GPU stats.
//!
//! Mirrors `pie_cli/monitor/provider.py::PieMetricsProvider`. Runs as
//! a tokio background task; the TUI reads `Snapshot`s from the shared
//! `Mutex` on each refresh tick.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use serde::Deserialize;
use tokio::task::JoinHandle;

use pie_client::client::Client;

use super::data::{GpuMetrics, Inferlet, Snapshot, SystemMetrics, TpGroupMetrics};

const MAX_HISTORY: usize = 500;
const POLL_INTERVAL: Duration = Duration::from_millis(1000);

/// Spawn the polling task on `handle`. The returned `Provider` shares
/// a `Mutex<Snapshot>` with the task; the TUI reads from it on each
/// frame. Stop the task by dropping the `Provider` (signals
/// `stop_flag`) — the task observes the flag at the next tick.
pub struct Provider {
    snapshot: Arc<Mutex<Snapshot>>,
    stop_flag: Arc<std::sync::atomic::AtomicBool>,
    task: Option<JoinHandle<()>>,
}

impl Provider {
    pub fn spawn(
        handle: &tokio::runtime::Handle,
        url: String,
        token: String,
    ) -> Self {
        let snapshot = Arc::new(Mutex::new(Snapshot::default()));
        let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let s2 = Arc::clone(&snapshot);
        let f2 = Arc::clone(&stop_flag);
        let task = handle.spawn(async move {
            poll_loop(url, token, s2, f2).await;
        });

        Provider {
            snapshot,
            stop_flag,
            task: Some(task),
        }
    }

    /// Cheap snapshot read — clones the latest poll result. Called
    /// from the TUI render thread on every tick.
    pub fn snapshot(&self) -> Snapshot {
        self.snapshot.lock().unwrap().clone()
    }
}

impl Drop for Provider {
    fn drop(&mut self) {
        self.stop_flag
            .store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(t) = self.task.take() {
            t.abort();
        }
    }
}

async fn poll_loop(
    url: String,
    token: String,
    snapshot: Arc<Mutex<Snapshot>>,
    stop: Arc<std::sync::atomic::AtomicBool>,
) {
    // NVML initialized once per loop (not per outer-reconnect-cycle):
    // libnvidia-ml's handle is stable across the engine's lifetime.
    let nvml = NvmlContext::try_new();

    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
        let connect_result = Client::connect(&url).await;
        let client = match connect_result {
            Ok(c) => c,
            Err(e) => {
                tracing::debug!("monitor connect failed: {e}");
                tokio::time::sleep(Duration::from_millis(1000)).await;
                continue;
            }
        };
        if let Err(e) = client.auth_by_token(&token).await {
            tracing::debug!("monitor auth failed: {e}");
            tokio::time::sleep(Duration::from_millis(1000)).await;
            continue;
        }
        snapshot.lock().unwrap().connected = true;

        let mut state = PollState::default();
        while !stop.load(std::sync::atomic::Ordering::Relaxed) {
            match poll_once(&client, &mut state, &nvml).await {
                Ok(snap) => *snapshot.lock().unwrap() = snap,
                Err(e) => {
                    tracing::debug!("monitor poll error: {e}");
                    break;
                }
            }
            tokio::time::sleep(POLL_INTERVAL).await;
        }

        snapshot.lock().unwrap().connected = false;
    }
}

#[derive(Default)]
struct PollState {
    last_poll: Option<Instant>,
    prev_total_tokens: u64,
    estimated_tput: f64,
    estimated_latency_ms: f64,
    kv_cache_history: Vec<f64>,
    token_tput_history: Vec<f64>,
    latency_history: Vec<f64>,
    batch_history: Vec<f64>,
}

async fn poll_once(
    client: &Client,
    state: &mut PollState,
    nvml: &Option<NvmlContext>,
) -> Result<Snapshot> {
    // Engine model_status query — same call site Python's provider uses.
    let raw = client.query("model_status", String::new()).await?;
    let stats: serde_json::Value = serde_json::from_str(&raw)?;
    let stats_obj = stats.as_object();

    let mut kv_pages_used: u64 = 0;
    let mut kv_pages_total: u64 = 0;
    let mut total_tokens: u64 = 0;
    let mut avg_latency_us: u64 = 0;
    if let Some(map) = stats_obj {
        for (key, value) in map.iter() {
            let v = value.as_u64().unwrap_or_else(|| {
                value.as_i64().unwrap_or(0).max(0) as u64
            });
            if key.ends_with(".kv_pages_used") {
                kv_pages_used = kv_pages_used.saturating_add(v);
            } else if key.ends_with(".kv_pages_total") {
                kv_pages_total = kv_pages_total.saturating_add(v);
            } else if key.ends_with(".total_tokens_processed") {
                total_tokens = total_tokens.saturating_add(v);
            } else if key.ends_with(".avg_batch_latency_us") {
                avg_latency_us = avg_latency_us.max(v);
            }
        }
    }

    // Throughput: instantaneous derivative against last poll.
    let now = Instant::now();
    if let Some(last) = state.last_poll {
        let dt = now.duration_since(last).as_secs_f64();
        if dt > 0.0 && total_tokens > state.prev_total_tokens {
            state.estimated_tput =
                (total_tokens - state.prev_total_tokens) as f64 / dt;
        } else if total_tokens == state.prev_total_tokens {
            state.estimated_tput = 0.0;
        }
    }
    state.last_poll = Some(now);
    state.prev_total_tokens = total_tokens;

    if avg_latency_us > 0 {
        state.estimated_latency_ms = avg_latency_us as f64 / 1000.0;
    }

    let kv_cache_pct = if kv_pages_total > 0 {
        (kv_pages_used as f64 / kv_pages_total as f64) * 100.0
    } else {
        0.0
    };

    let gpu_metrics = nvml
        .as_ref()
        .map(NvmlContext::sample)
        .unwrap_or_default();
    // One TP group per visible GPU until the engine side surfaces a
    // real grouping over RPC. Same shape Python's provider produces.
    let tp_groups: Vec<TpGroupMetrics> = gpu_metrics
        .iter()
        .map(|g| TpGroupMetrics {
            tp_id: g.gpu_id,
            utilization: g.utilization,
            gpus: vec![g.clone()],
        })
        .collect();

    let processes = client.list_processes().await.unwrap_or_default();
    let inferlets: Vec<Inferlet> = processes
        .into_iter()
        .take(50)
        .map(parse_inferlet)
        .collect();

    push_history(&mut state.kv_cache_history, kv_cache_pct);
    push_history(&mut state.token_tput_history, state.estimated_tput);
    push_history(&mut state.latency_history, state.estimated_latency_ms);
    push_history(&mut state.batch_history, 0.0);

    Ok(Snapshot {
        metrics: SystemMetrics {
            kv_cache_usage: kv_cache_pct,
            kv_pages_used,
            kv_pages_total: if kv_pages_total > 0 {
                kv_pages_total
            } else {
                600
            },
            token_throughput: state.estimated_tput,
            latency_ms: state.estimated_latency_ms,
            active_batches: 0,
            tp_groups,
            inferlets,
        },
        kv_cache_history: state.kv_cache_history.clone(),
        token_tput_history: state.token_tput_history.clone(),
        latency_history: state.latency_history.clone(),
        batch_history: state.batch_history.clone(),
        connected: true,
    })
}

fn push_history(history: &mut Vec<f64>, value: f64) {
    history.push(value);
    if history.len() > MAX_HISTORY {
        history.remove(0);
    }
}

/// Best-effort parse of a `list_processes` entry. The runtime returns
/// strings (UUIDs) today — Python's monitor accepts either string or
/// dict for forward-compat with a richer payload. We do the same:
/// tolerate JSON objects and bare strings.
fn parse_inferlet(raw: String) -> Inferlet {
    if let Ok(p) = serde_json::from_str::<ProcessJson>(&raw) {
        let id = p.id.unwrap_or_default();
        let id_short = if id.len() > 12 { id[..12].to_string() } else { id };
        let elapsed = p
            .elapsed_secs
            .map(|s| {
                if s == 0 {
                    "-".to_string()
                } else {
                    format!("{}m{}s", s / 60, s % 60)
                }
            })
            .unwrap_or_else(|| "-".to_string());
        return Inferlet {
            id: id_short,
            program: p.program.unwrap_or_else(|| "inferlet".into()),
            user: p.username.unwrap_or_else(|| "user".into()),
            status: "running".to_string(),
            elapsed,
            kv_cache: 0.0,
        };
    }
    // Bare UUID string fallback.
    let id_short = if raw.len() > 12 { raw[..12].to_string() } else { raw };
    Inferlet {
        id: id_short,
        program: "inferlet".to_string(),
        user: "user".to_string(),
        status: "running".to_string(),
        elapsed: "-".to_string(),
        kv_cache: 0.0,
    }
}

#[derive(Deserialize)]
struct ProcessJson {
    id: Option<String>,
    program: Option<String>,
    username: Option<String>,
    elapsed_secs: Option<u64>,
}

// -----------------------------------------------------------------------------
// NVML wrapper — optional, falls back to "no GPUs" gracefully.
// -----------------------------------------------------------------------------

struct NvmlContext {
    nvml: nvml_wrapper::Nvml,
    count: u32,
}

impl NvmlContext {
    fn try_new() -> Option<Self> {
        match nvml_wrapper::Nvml::init() {
            Ok(n) => {
                let count = n.device_count().ok()?;
                Some(Self { nvml: n, count })
            }
            Err(e) => {
                tracing::debug!("NVML init failed (no GPU panel): {e}");
                None
            }
        }
    }

    fn sample(&self) -> Vec<GpuMetrics> {
        let mut out = Vec::with_capacity(self.count as usize);
        for i in 0..self.count {
            let Ok(d) = self.nvml.device_by_index(i) else { continue };
            let util = d.utilization_rates().ok();
            let mem = d.memory_info().ok();
            let utilization = util.map(|u| u.gpu as f64).unwrap_or(0.0);
            let (used_gb, total_gb) = mem
                .map(|m| {
                    (
                        m.used as f64 / (1024.0 * 1024.0 * 1024.0),
                        m.total as f64 / (1024.0 * 1024.0 * 1024.0),
                    )
                })
                .unwrap_or((0.0, 0.0));
            out.push(GpuMetrics {
                gpu_id: i as usize,
                tp_group: 0,
                utilization,
                memory_used_gb: used_gb,
                memory_total_gb: total_gb,
            });
        }
        out
    }
}
