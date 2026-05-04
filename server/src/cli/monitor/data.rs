//! Data structs the monitor TUI consumes — mirrors
//! `pie_cli/monitor/data.py`. Every value is plain Rust scalars or
//! `String` / `Vec` so the provider can write them under a Mutex
//! without lifetime gymnastics.
//!
//! Several fields aren't yet rendered by the Rust TUI but are kept
//! for parity with the Python schema so a future widget can light
//! them up without changing the provider contract.
#![allow(dead_code)]

#[derive(Debug, Clone, Default)]
pub struct GpuMetrics {
    pub gpu_id: usize,
    pub tp_group: usize,
    pub utilization: f64,
    pub memory_used_gb: f64,
    pub memory_total_gb: f64,
}

impl GpuMetrics {
    pub fn memory_percent(&self) -> f64 {
        if self.memory_total_gb <= 0.0 {
            0.0
        } else {
            (self.memory_used_gb / self.memory_total_gb) * 100.0
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct TpGroupMetrics {
    pub tp_id: usize,
    pub utilization: f64,
    pub gpus: Vec<GpuMetrics>,
}

#[derive(Debug, Clone)]
pub struct Inferlet {
    pub id: String,
    pub program: String,
    pub user: String,
    pub status: String,
    pub elapsed: String,
    pub kv_cache: f64,
}

#[derive(Debug, Clone, Default)]
pub struct SystemMetrics {
    pub kv_cache_usage: f64,
    pub kv_pages_used: u64,
    pub kv_pages_total: u64,
    pub token_throughput: f64,
    pub latency_ms: f64,
    pub active_batches: u64,
    pub tp_groups: Vec<TpGroupMetrics>,
    pub inferlets: Vec<Inferlet>,
}

/// Snapshot the TUI reads on each tick. Wraps the live metrics with
/// the per-series history rings the graph widget needs. Owned by the
/// provider (under a `Mutex`); cloned cheaply on read.
#[derive(Debug, Clone, Default)]
pub struct Snapshot {
    pub metrics: SystemMetrics,
    pub kv_cache_history: Vec<f64>,
    pub token_tput_history: Vec<f64>,
    pub latency_history: Vec<f64>,
    pub batch_history: Vec<f64>,
    /// `false` until the first successful poll lands.
    pub connected: bool,
}

/// Display config the TUI's left-hand panel renders. Sourced from the
/// loaded `config::Config` at boot time and never updated.
#[derive(Debug, Clone, Default)]
pub struct DisplayConfig {
    pub host: String,
    pub port: u16,
    pub auth_enabled: bool,
    pub hf_repo: String,
    pub device: Vec<String>,
    pub tensor_parallel_size: u32,
    pub activation_dtype: String,
    pub kv_page_size: u32,
    pub max_batch_tokens: u32,
}
