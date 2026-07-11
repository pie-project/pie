//! # Inference Module
//!
//! Forward pass management for model execution.
//!
//! Each model gets a dedicated InferenceService that:
//! - Translates logical KV page IDs to physical page IDs
//! - Routes requests to per-driver BatchSchedulers based on page affinity
//!
//! Batch scheduling, RPC execution, and response notification are handled
//! by individual BatchScheduler instances (one per driver).

pub mod batch;
pub mod contention;
pub mod execute;
pub mod paging;
mod policy;
pub mod request;
pub mod scheduler;
pub mod stats;

use tokio::sync::oneshot;

use crate::inference::paging::PhysicalPageId;
use crate::driver::{DriverId, SchedulerLimits};
use crate::service::{Service, ServiceHandler};
use anyhow::Result;
use scheduler::BatchScheduler;
use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;

pub use stats::{BUBBLE_HIST_UPPER_US, SchedulerStats};
/// Aggregated inference stats for a single model (across all drivers).
///
/// Always-on counters live at the top; per-domain probe averages
/// (currently just `fire`) are nested so the shape mirrors the probe
/// hierarchy in `crate::probe::*`.
#[derive(Debug, Default, serde::Serialize)]
pub struct InferenceStats {
    pub total_batches: u64,
    pub total_tokens_processed: u64,
    pub total_requests_processed: u64,
    pub max_forward_requests_observed: u64,
    /// Histogram buckets (1, 2-3, 4-7, 8-15, 16-31, 32-63, 64-127, 128+).
    pub batch_size_hist: [u64; 8],
    pub last_batch_latency_us: u64,
    pub cumulative_batch_latency_us: u64,
    pub avg_batch_latency_us: u64,

    /// Fire-domain probe averages. Values are 0 when built without
    /// `profile-fire`. Mirrors `crate::probe::fire::FireProbes`.
    pub fire: FireStats,

    /// Per-identity (C3) co-batch density: `(program_identity_hash, fires, rows)`,
    /// merged across drivers, sorted by rows desc. Per-identity fire density =
    /// `rows / fires` (the pentathlon per-identity-batch-density probe).
    pub per_identity_batch: Vec<(u64, u64, u64)>,
    /// Distinct-identity fire-records dropped because the bounded per-identity
    /// table saturated. `> 0` ⇒ the fleet has more identities than the table caps;
    /// `per_identity_batch` is then a bounded sample (fail-loud, never silent).
    pub identities_dropped: u64,
    /// Inter-batch bubble histogram (one count per fire), bucketed by
    /// [`BUBBLE_HIST_UPPER_US`] — for the p50/p99 bubble gate (masterplan M1).
    /// HOST PROXY (device-idle stamped at the Rust enqueue point → over-counts by
    /// the host submit/handshake delay). See [`Self::bubble_p50`].
    pub bubble_us_hist: [u64; BUBBLE_HIST_UPPER_US.len()],
}

impl InferenceStats {
    /// Inter-batch bubble p50 (µs) from the scheduler's host-side histogram.
    pub fn bubble_p50(&self) -> u64 {
        self.bubble_percentile(0.50)
    }

    /// Inter-batch bubble p99 (µs).
    pub fn bubble_p99(&self) -> u64 {
        self.bubble_percentile(0.99)
    }

    /// Alias retained for callers that explicitly request the host proxy.
    pub fn bubble_p50_proxy(&self) -> u64 {
        Self::hist_percentile(&self.bubble_us_hist, 0.50)
    }

    fn bubble_percentile(&self, q: f64) -> u64 {
        Self::hist_percentile(&self.bubble_us_hist, q)
    }

    /// The upper bound of the bucket containing the `q`-quantile of `hist`
    /// (0 if the histogram is empty).
    fn hist_percentile(hist: &[u64; BUBBLE_HIST_UPPER_US.len()], q: f64) -> u64 {
        let total: u64 = hist.iter().sum();
        if total == 0 {
            return 0;
        }
        let target = (total as f64 * q).ceil() as u64;
        let mut cum = 0u64;
        for (i, &count) in hist.iter().enumerate() {
            cum += count;
            if cum >= target {
                return BUBBLE_HIST_UPPER_US[i];
            }
        }
        *BUBBLE_HIST_UPPER_US.last().unwrap()
    }
}

#[derive(Debug, Default, serde::Serialize)]
pub struct FireStats {
    pub avg_inter_fire_us: u64,
    pub avg_post_dispatch_to_fire_us: u64,
    pub avg_recv_block_wait_us: u64,
    pub avg_guest_roundtrip_us: u64,
    pub avg_service_queue_us: u64,
    pub accumulate: AccumulateStats,
    pub pre_dispatch: PreDispatchStats,
    pub execute: ExecuteStats,
    pub post_dispatch: PostDispatchStats,
    pub quorum: QuorumStats,
}

/// Quorum-rule probe averages/counters (overview §7.2; thrust-2 §3 F1–F6).
/// Zero unless built with `profile-fire`. Populated by the quorum core
/// (thrust-2 phase S5); scaffolding lands in S0. Mirrors
/// `crate::probe::fire::QuorumProbes`.
#[derive(Debug, Default, serde::Serialize)]
pub struct QuorumStats {
    /// Mean device idle between a batch retiring and the next launching (F1).
    pub avg_inter_batch_bubble_us: u64,
    /// Mean last-ready → enqueue latency (F1 quorum completion).
    pub avg_quorum_latency_us: u64,
    /// Idle-escape (F2) fire count; divide by `total_batches` for escape rate.
    pub escape_fires: u64,
    /// Depth-2 submit-ahead (G3 bubble) fire count; divide by `total_batches`
    /// for the submit-ahead rate (steady-state decode-fleet bubble-filler).
    pub submit_ahead_fires: u64,
    /// Mean cold-hold window per cold-hold fire (F3 occupancy).
    pub avg_cold_hold_us: u64,
    /// Count of fires through the cold-hold path (F3).
    pub cold_hold_fires: u64,
    /// Dummy-run / readiness-miss count (M3 gate: rate < 1%).
    pub readiness_miss: u64,
    /// Wait-for-all wave (M-AB): mean active_pipelines (wait-set size) sampled
    /// at each WaitAll fire. ≈ fleet width ⇒ persistent wait-set (waves should
    /// be dense); ≈1 ⇒ transient/singleton. 0 if no WaitAll fire.
    pub avg_active_pipelines_at_fire: u64,
    /// Wait-for-all wave: mean stragglers fired without (deadline holds). >0 ⇒
    /// waves hold to the deadline then fire partial; ≈0 ⇒ all-ready fires.
    pub avg_missing_at_fire: u64,
    /// Count of WaitAll wave fires (denominator for the two averages above).
    pub wave_fires: u64,
}

#[derive(Debug, Default, serde::Serialize)]
pub struct AccumulateStats {
    pub avg_accum_loop_us: u64,
}

#[derive(Debug, Default, serde::Serialize)]
pub struct PreDispatchStats {
    pub avg_fire_prepare_us: u64,
}

#[derive(Debug, Default, serde::Serialize)]
pub struct ExecuteStats {
    pub avg_total_us: u64,
    pub avg_batch_build_us: u64,
    pub avg_driver_fire_us: u64,
}

#[derive(Debug, Default, serde::Serialize)]
pub struct PostDispatchStats {
    pub avg_context_tick_us: u64,
    pub avg_stats_update_us: u64,
}

// =============================================================================
// Public API
// =============================================================================

static SERVICE: Service<Message> = Service::new();

/// Spawns the inference service for the single model.
pub struct InferenceShutdownHandle;

impl InferenceShutdownHandle {
    pub async fn shutdown(self) -> Result<()> {
        SERVICE.shutdown().await
    }
}

pub async fn spawn(
    driver_indices: &[usize],
    page_size: u32,
    request_timeout_secs: u64,
) -> Result<InferenceShutdownHandle> {
    // Fetch driver info before entering the sync closure.
    let driver_ids: Vec<DriverId> = driver_indices.to_vec();
    let mut driver_batch_limits = Vec::with_capacity(driver_indices.len());
    for &driver_idx in driver_indices {
        let info = crate::driver::get_spec(driver_idx)
            .await
            .unwrap_or_else(|e| panic!("Failed to get driver info for index {driver_idx}: {e}"));
        driver_batch_limits.push(info.scheduler_limits());
    }

    SERVICE.spawn(move || {
        InferenceService::new(
            driver_ids,
            driver_batch_limits,
            page_size,
            request_timeout_secs,
        )
    })?;
    Ok(InferenceShutdownHandle)
}

pub fn submit_async(
    request: crate::driver::LaunchPlan,
    driver_idx: usize,
    instance_id: u64,
    physical_page_ids: Vec<PhysicalPageId>,
    last_page_len: u32,
    program_identity_hashes: Vec<u64>,
    pipeline_id: Option<crate::process::ProcessId>,
    completion: crate::driver::InstanceCompletion,
) -> Result<()> {
    let submitted_at_us = crate::inference::scheduler::now_micros();
    crate::driver::scheduler_handle(driver_idx)?.submit_with_identity(
        request,
        instance_id,
        completion,
        physical_page_ids,
        last_page_len,
        program_identity_hashes,
        pipeline_id,
        submitted_at_us,
    )
}

pub fn submit_prebuilt_async(
    request: crate::driver::LaunchPlan,
    driver_idx: usize,
    instance_id: u64,
    physical_page_ids: Vec<PhysicalPageId>,
    last_page_len: u32,
    program_identity_hashes: Vec<u64>,
    completion: crate::driver::InstanceCompletion,
) -> Result<()> {
    crate::driver::scheduler_handle(driver_idx)?.submit_prebuilt(
        request,
        instance_id,
        completion,
        physical_page_ids,
        last_page_len,
        program_identity_hashes,
    )
}

/// Returns aggregated inference stats for the model (lock-free, non-blocking).
pub async fn get_stats() -> InferenceStats {
    let (tx, rx) = oneshot::channel();
    let _ = SERVICE.send(Message::GetStats { response: tx });
    rx.await.unwrap_or_default()
}

// =============================================================================
// Inference Service
// =============================================================================

/// The inference service handles forward pass operations.
///
/// Routes requests to the appropriate per-driver `BatchScheduler`
/// based on physical page affinity from the context service.
struct InferenceService {
    num_drivers: usize,
    schedulers: Vec<BatchScheduler>,
    scheduler_stats: Vec<Arc<SchedulerStats>>,
}

impl std::fmt::Debug for InferenceService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceService").finish()
    }
}

impl InferenceService {
    fn new(
        driver_ids: Vec<DriverId>,
        driver_batch_limits: Vec<SchedulerLimits>,
        page_size: u32,
        request_timeout_secs: u64,
    ) -> Self {
        let num_drivers = driver_ids.len();
        let schedulers: Vec<BatchScheduler> = driver_ids
            .iter()
            .enumerate()
            .map(|(driver_idx, &driver_id)| {
                let limits = driver_batch_limits[driver_idx];
                BatchScheduler::new(
                    driver_id,
                    driver_idx,
                    page_size,
                    limits,
                    request_timeout_secs,
                )
            })
            .collect();

        let scheduler_stats: Vec<_> = schedulers.iter().map(|s| s.stats().clone()).collect();

        InferenceService {
            num_drivers,
            schedulers,
            scheduler_stats,
        }
    }

    /// Aggregate stats from all per-driver schedulers.
    fn aggregate_stats(&self) -> InferenceStats {
        let mut total_batches = 0u64;
        let mut total_tokens = 0u64;
        let mut total_requests = 0u64;
        let mut max_forward_requests = 0u64;
        let mut hist = [0u64; 8];
        let mut bubble_hist = [0u64; BUBBLE_HIST_UPPER_US.len()];
        // Per-identity (C3) co-batch density, merged across drivers by hash.
        let mut per_identity: std::collections::HashMap<u64, (u64, u64)> =
            std::collections::HashMap::new();
        let mut identities_dropped = 0u64;
        let mut last_latency = 0u64;
        let mut cumulative_latency = 0u64;
        // Per-driver sums of probe atomics. Walked in the same shape
        // as InferenceStats.fire / FireProbes so the relationship is
        // self-evident.
        let mut fire_inter = 0u64;
        let mut fire_post_dispatch_to_fire = 0u64;
        let mut fire_recv_block_wait = 0u64;
        let mut fire_guest_roundtrip = 0u64;
        let mut fire_service_queue = 0u64;
        let mut fire_accumulate_accum_loop = 0u64;
        let mut fire_pre_dispatch_fire_prepare = 0u64;
        let mut fire_execute_total = 0u64;
        let mut fire_execute_batch_build = 0u64;
        let mut fire_execute_driver_fire = 0u64;
        let mut fire_post_dispatch_context_tick = 0u64;
        let mut fire_post_dispatch_stats_update = 0u64;
        let mut q_inter_batch_bubble = 0u64;
        let mut q_quorum_latency = 0u64;
        let mut q_escape_fires = 0u64;
        let mut q_submit_ahead_fires = 0u64;
        let mut q_cold_hold = 0u64;
        let mut q_cold_hold_fires = 0u64;
        let mut q_readiness_miss = 0u64;
        let mut q_wave_active_sum = 0u64;
        let mut q_wave_missing_sum = 0u64;
        let mut q_wave_fires = 0u64;
        for s in &self.scheduler_stats {
            total_batches += s.total_batches.load(Relaxed);
            total_tokens += s.total_tokens_processed.load(Relaxed);
            total_requests += s.total_requests_processed.load(Relaxed);
            max_forward_requests =
                max_forward_requests.max(s.max_forward_requests_observed.load(Relaxed));
            for (dst, src) in hist.iter_mut().zip(s.batch_size_hist.iter()) {
                *dst += src.load(Relaxed);
            }
            last_latency = last_latency.max(s.last_batch_latency_us.load(Relaxed));
            cumulative_latency += s.cumulative_latency_us.load(Relaxed);
            let f = &s.fire;
            fire_inter += f.inter_fire_us.load(Relaxed);
            fire_post_dispatch_to_fire += f.post_dispatch_to_fire_us.load(Relaxed);
            fire_recv_block_wait += f.recv_block_wait_us.load(Relaxed);
            fire_guest_roundtrip += f.guest_roundtrip_us.load(Relaxed);
            fire_service_queue += f.service_queue_us.load(Relaxed);
            fire_accumulate_accum_loop += f.accumulate.accum_loop_us.load(Relaxed);
            fire_pre_dispatch_fire_prepare += f.pre_dispatch.fire_prepare_us.load(Relaxed);
            fire_execute_total += f.execute.total_us.load(Relaxed);
            fire_execute_batch_build += f.execute.batch_build_us.load(Relaxed);
            fire_execute_driver_fire += f.execute.driver_fire_us.load(Relaxed);
            fire_post_dispatch_context_tick += f.post_dispatch.context_tick_us.load(Relaxed);
            fire_post_dispatch_stats_update += f.post_dispatch.stats_update_us.load(Relaxed);
            q_inter_batch_bubble += f.quorum.inter_batch_bubble_us.load(Relaxed);
            q_quorum_latency += f.quorum.quorum_latency_us.load(Relaxed);
            q_escape_fires += f.quorum.escape_fires.load(Relaxed);
            q_submit_ahead_fires += f.quorum.submit_ahead_fires.load(Relaxed);
            q_cold_hold += f.quorum.cold_hold_us.load(Relaxed);
            q_cold_hold_fires += f.quorum.cold_hold_fires.load(Relaxed);
            q_readiness_miss += f.quorum.readiness_miss.load(Relaxed);
            q_wave_active_sum += f.quorum.wave_active_sum.load(Relaxed);
            q_wave_missing_sum += f.quorum.wave_missing_sum.load(Relaxed);
            q_wave_fires += f.quorum.wave_fires.load(Relaxed);
            // Merge this driver's bounded per-identity table by hash.
            for (i, hslot) in s.per_identity_hash.iter().enumerate() {
                let h = hslot.load(Relaxed);
                if h != 0 {
                    let e = per_identity.entry(h).or_insert((0u64, 0u64));
                    e.0 += s.per_identity_fires[i].load(Relaxed);
                    e.1 += s.per_identity_rows[i].load(Relaxed);
                }
            }
            identities_dropped += s.identities_dropped.load(Relaxed);
            for (dst, src) in bubble_hist.iter_mut().zip(s.bubble_us_hist.iter()) {
                *dst += src.load(Relaxed);
            }
        }

        let mut per_identity_batch: Vec<(u64, u64, u64)> = per_identity
            .into_iter()
            .map(|(h, (fires, rows))| (h, fires, rows))
            .collect();
        per_identity_batch.sort_by(|a, b| b.2.cmp(&a.2)); // rows desc
        let avg = |value: u64| {
            if total_batches > 0 {
                value / total_batches
            } else {
                0
            }
        };
        // Inter-fire is sampled starting at the 2nd batch (first one has
        // no prior to diff against), so divide by max(total_batches-1, 1)
        // to get a stable mean.
        let avg_pair = |value: u64| {
            if total_batches > 1 {
                value / (total_batches - 1)
            } else {
                0
            }
        };
        InferenceStats {
            total_batches,
            total_tokens_processed: total_tokens,
            total_requests_processed: total_requests,
            max_forward_requests_observed: max_forward_requests,
            batch_size_hist: hist,
            last_batch_latency_us: last_latency,
            cumulative_batch_latency_us: cumulative_latency,
            avg_batch_latency_us: avg(cumulative_latency),
            fire: FireStats {
                avg_inter_fire_us: avg_pair(fire_inter),
                avg_post_dispatch_to_fire_us: avg_pair(fire_post_dispatch_to_fire),
                avg_recv_block_wait_us: avg_pair(fire_recv_block_wait),
                avg_guest_roundtrip_us: avg_pair(fire_guest_roundtrip),
                avg_service_queue_us: avg_pair(fire_service_queue),
                accumulate: AccumulateStats {
                    avg_accum_loop_us: avg(fire_accumulate_accum_loop),
                },
                pre_dispatch: PreDispatchStats {
                    avg_fire_prepare_us: avg(fire_pre_dispatch_fire_prepare),
                },
                execute: ExecuteStats {
                    avg_total_us: avg(fire_execute_total),
                    avg_batch_build_us: avg(fire_execute_batch_build),
                    avg_driver_fire_us: avg(fire_execute_driver_fire),
                },
                post_dispatch: PostDispatchStats {
                    avg_context_tick_us: avg(fire_post_dispatch_context_tick),
                    avg_stats_update_us: avg(fire_post_dispatch_stats_update),
                },
                quorum: QuorumStats {
                    avg_inter_batch_bubble_us: avg(q_inter_batch_bubble),
                    avg_quorum_latency_us: avg(q_quorum_latency),
                    escape_fires: q_escape_fires,
                    submit_ahead_fires: q_submit_ahead_fires,
                    avg_cold_hold_us: if q_cold_hold_fires > 0 {
                        q_cold_hold / q_cold_hold_fires
                    } else {
                        0
                    },
                    cold_hold_fires: q_cold_hold_fires,
                    readiness_miss: q_readiness_miss,
                    avg_active_pipelines_at_fire: if q_wave_fires > 0 {
                        q_wave_active_sum / q_wave_fires
                    } else {
                        0
                    },
                    avg_missing_at_fire: if q_wave_fires > 0 {
                        q_wave_missing_sum / q_wave_fires
                    } else {
                        0
                    },
                    wave_fires: q_wave_fires,
                },
            },
            per_identity_batch,
            identities_dropped,
            bubble_us_hist: bubble_hist,
        }
    }
}

// =============================================================================
// ServiceHandler Implementation
// =============================================================================

#[derive(Debug)]
enum Message {
    GetStats {
        response: oneshot::Sender<InferenceStats>,
    },
}

impl ServiceHandler for InferenceService {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::GetStats { response } => {
                let _ = response.send(self.aggregate_stats());
            }
        }
    }
}
