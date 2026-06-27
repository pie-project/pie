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

mod adaptive_policy;
pub mod forward_prepare;
pub mod request;
pub mod scheduler;
pub mod structured;

use tokio::sync::oneshot;

use crate::arena::PhysicalPageId;
use crate::driver::{DriverId, SchedulerLimits};
use crate::service::{Service, ServiceHandler};
use anyhow::Result;
use scheduler::BatchScheduler;
use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;

pub use scheduler::{SchedulerStats, SYSTEM_SPEC_DRAFT_POS_BUCKETS};
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

    pub system_spec_draft_tokens_proposed: u64,
    pub system_spec_draft_tokens_accepted: u64,
    pub system_spec_draft_tokens_proposed_per_pos: [u64; SYSTEM_SPEC_DRAFT_POS_BUCKETS],
    pub system_spec_draft_tokens_accepted_per_pos: [u64; SYSTEM_SPEC_DRAFT_POS_BUCKETS],
}

#[derive(Debug, Default, serde::Serialize)]
pub struct FireStats {
    pub avg_inter_fire_us: u64,
    pub avg_post_dispatch_to_fire_us: u64,
    pub accumulate: AccumulateStats,
    pub pre_dispatch: PreDispatchStats,
    pub execute: ExecuteStats,
    pub post_dispatch: PostDispatchStats,
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
    pub response_dispatch: ResponseDispatchStats,
    pub driver_cuda: DriverCudaStats,
}

#[derive(Debug, Default, serde::Serialize)]
pub struct ResponseDispatchStats {
    pub avg_total_us: u64,
    pub direct_count: u64,
    pub chunk_count: u64,
}

#[derive(Debug, Default, serde::Serialize)]
pub struct DriverCudaStats {
    pub avg_ipc_submit_us: u64,
    pub avg_gpu_wait_us: u64,
    pub avg_ipc_recv_us: u64,
    pub avg_wire_parse_us: u64,
    pub avg_plan_us: u64,
    pub avg_h2d_us: u64,
    pub avg_kernel_launch_us: u64,
    pub avg_sync_us: u64,
    pub avg_response_build_us: u64,
    pub sum_sync_us: u64,
    pub sum_kernel_launch_us: u64,
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
pub async fn spawn(
    driver_indices: &[usize],
    page_size: u32,
    request_timeout_secs: u64,
) {
    // Fetch driver info before entering the sync closure.
    let driver_ids: Vec<DriverId> = driver_indices.to_vec();
    let mut driver_batch_limits = Vec::with_capacity(driver_indices.len());
    for &driver_idx in driver_indices {
        let info = crate::driver::get_spec(driver_idx)
            .await
            .unwrap_or_else(|e| panic!("Failed to get driver info for index {driver_idx}: {e}"));
        driver_batch_limits.push(info.scheduler_limits());
    }

    SERVICE
        .spawn(move || {
            InferenceService::new(
                driver_ids,
                driver_batch_limits,
                page_size,
                request_timeout_secs,
            )
        })
        .expect("Failed to spawn inference service")
}



pub fn submit_async(
    request: pie_driver_abi::ForwardRequest,
    driver_idx: usize,
    physical_page_ids: Vec<PhysicalPageId>,
    last_page_len: u32,
) -> Result<oneshot::Receiver<Result<ForwardOutput>>> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(
        Message::Submit {
            request,
            driver_idx,
            physical_page_ids,
            last_page_len,
            response: tx,
        },
    )?;
    Ok(rx)
}

/// Internal forward result shape passed from the scheduler to a waiting
/// inferlet. Normal decode returns a single token per request; carrying that
/// directly avoids allocating a one-request `ForwardResponse` for every token.
#[derive(Debug)]
pub enum ForwardOutput {
    Token(u32),
    Tokens(Vec<u32>),
    Response(pie_driver_abi::ForwardResponse),
}

impl ForwardOutput {
    pub(crate) fn first_token(&self) -> Option<u32> {
        match self {
            Self::Token(t) => Some(*t),
            Self::Tokens(tokens) => tokens.first().copied(),
            Self::Response(resp) => resp.tokens.first().copied(),
        }
    }

    #[cfg(test)]
    pub(crate) fn from_response(resp: pie_driver_abi::ForwardResponse) -> Self {
        Self::Response(resp)
    }
}

impl From<pie_driver_abi::ForwardResponse> for ForwardOutput {
    fn from(resp: pie_driver_abi::ForwardResponse) -> Self {
        Self::Response(resp)
    }
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
        let mut last_latency = 0u64;
        let mut cumulative_latency = 0u64;
        // Per-driver sums of probe atomics. Walked in the same shape
        // as InferenceStats.fire / FireProbes so the relationship is
        // self-evident.
        let mut fire_inter = 0u64;
        let mut fire_post_dispatch_to_fire = 0u64;
        let mut fire_accumulate_accum_loop = 0u64;
        let mut fire_pre_dispatch_fire_prepare = 0u64;
        let mut fire_execute_total = 0u64;
        let mut fire_execute_batch_build = 0u64;
        let mut fire_execute_driver_fire = 0u64;
        let mut fire_execute_response_dispatch_total = 0u64;
        let mut fire_execute_response_dispatch_direct = 0u64;
        let mut fire_execute_response_dispatch_chunk = 0u64;
        let mut dc_ipc_submit = 0u64;
        let mut dc_gpu_wait = 0u64;
        let mut dc_ipc_recv = 0u64;
        let mut dc_wire_parse = 0u64;
        let mut dc_plan = 0u64;
        let mut dc_h2d = 0u64;
        let mut dc_kernel_launch = 0u64;
        let mut dc_sync = 0u64;
        let mut dc_response_build = 0u64;
        let mut fire_post_dispatch_context_tick = 0u64;
        let mut fire_post_dispatch_stats_update = 0u64;
        let mut system_spec_draft_tokens_proposed = 0u64;
        let mut system_spec_draft_tokens_accepted = 0u64;
        let mut system_spec_draft_tokens_proposed_per_pos =
            [0u64; SYSTEM_SPEC_DRAFT_POS_BUCKETS];
        let mut system_spec_draft_tokens_accepted_per_pos =
            [0u64; SYSTEM_SPEC_DRAFT_POS_BUCKETS];

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
            fire_accumulate_accum_loop += f.accumulate.accum_loop_us.load(Relaxed);
            fire_pre_dispatch_fire_prepare += f.pre_dispatch.fire_prepare_us.load(Relaxed);
            fire_execute_total += f.execute.total_us.load(Relaxed);
            fire_execute_batch_build += f.execute.batch_build_us.load(Relaxed);
            fire_execute_driver_fire += f.execute.driver_fire_us.load(Relaxed);
            fire_execute_response_dispatch_total += f.execute.response_dispatch.total_us.load(Relaxed);
            fire_execute_response_dispatch_direct += f.execute.response_dispatch.direct_count.load(Relaxed);
            fire_execute_response_dispatch_chunk += f.execute.response_dispatch.chunk_count.load(Relaxed);
            let dc = &s.driver_cuda;
            dc_ipc_submit += dc.ipc_submit_us.load(Relaxed);
            dc_gpu_wait += dc.gpu_wait_us.load(Relaxed);
            dc_ipc_recv += dc.ipc_recv_us.load(Relaxed);
            dc_wire_parse += dc.wire_parse_us.load(Relaxed);
            dc_plan += dc.plan_us.load(Relaxed);
            dc_h2d += dc.h2d_us.load(Relaxed);
            dc_kernel_launch += dc.kernel_launch_us.load(Relaxed);
            dc_sync += dc.sync_us.load(Relaxed);
            dc_response_build += dc.response_build_us.load(Relaxed);
            fire_post_dispatch_context_tick += f.post_dispatch.context_tick_us.load(Relaxed);
            fire_post_dispatch_stats_update += f.post_dispatch.stats_update_us.load(Relaxed);
            system_spec_draft_tokens_proposed += s.system_spec_draft_tokens_proposed.load(Relaxed);
            system_spec_draft_tokens_accepted += s.system_spec_draft_tokens_accepted.load(Relaxed);
            for (dst, src) in system_spec_draft_tokens_proposed_per_pos
                .iter_mut()
                .zip(s.system_spec_draft_tokens_proposed_per_pos.iter())
            {
                *dst += src.load(Relaxed);
            }
            for (dst, src) in system_spec_draft_tokens_accepted_per_pos
                .iter_mut()
                .zip(s.system_spec_draft_tokens_accepted_per_pos.iter())
            {
                *dst += src.load(Relaxed);
            }
        }

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
                    response_dispatch: ResponseDispatchStats {
                        avg_total_us: avg(fire_execute_response_dispatch_total),
                        direct_count: fire_execute_response_dispatch_direct,
                        chunk_count: fire_execute_response_dispatch_chunk,
                    },
                    driver_cuda: DriverCudaStats {
                        avg_ipc_submit_us: avg(dc_ipc_submit),
                        avg_gpu_wait_us: avg(dc_gpu_wait),
                        avg_ipc_recv_us: avg(dc_ipc_recv),
                        avg_wire_parse_us: avg(dc_wire_parse),
                        avg_plan_us: avg(dc_plan),
                        avg_h2d_us: avg(dc_h2d),
                        avg_kernel_launch_us: avg(dc_kernel_launch),
                        avg_sync_us: avg(dc_sync),
                        avg_response_build_us: avg(dc_response_build),
                        sum_sync_us: dc_sync,
                        sum_kernel_launch_us: dc_kernel_launch,
                    },
                },
                post_dispatch: PostDispatchStats {
                    avg_context_tick_us: avg(fire_post_dispatch_context_tick),
                    avg_stats_update_us: avg(fire_post_dispatch_stats_update),
                },
            },
            system_spec_draft_tokens_proposed,
            system_spec_draft_tokens_accepted,
            system_spec_draft_tokens_proposed_per_pos,
            system_spec_draft_tokens_accepted_per_pos,
        }
    }
}

// =============================================================================
// ServiceHandler Implementation
// =============================================================================

/// Messages handled by InferenceService.
#[derive(Debug)]
enum Message {
    /// Submit a pre-resolved forward pass to the scheduler.
    /// All context operations must be done by the caller before sending this.
    Submit {
        request: pie_driver_abi::ForwardRequest,
        driver_idx: usize,
        physical_page_ids: Vec<PhysicalPageId>,
        last_page_len: u32,
        response: oneshot::Sender<Result<ForwardOutput>>,
    },
    GetStats {
        response: oneshot::Sender<InferenceStats>,
    },
}

impl ServiceHandler for InferenceService {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::Submit {
                request,
                driver_idx,
                physical_page_ids,
                last_page_len,
                response,
            } => {
                let idx = driver_idx.min(self.num_drivers.saturating_sub(1));
                let _ = self.schedulers[idx].handle().submit(
                    request,
                    response,
                    physical_page_ids,
                    last_page_len,
                );
            }
            Message::GetStats { response } => {
                let _ = response.send(self.aggregate_stats());
            }
        }
    }
}
