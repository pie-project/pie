//! Per-driver batch scheduler.
//!
//! Each `BatchScheduler` owns its own RPC client, scheduling policy,
//! and tokio task. It accepts pre-translated forward pass requests,
//! accumulates them into batches, and fires them based on adaptive
//! scheduling decisions.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use anyhow::Result;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc, oneshot};

use crate::context::pagestore::PhysicalPageId;
use crate::driver::{self, DriverId, SchedulerLimits};

use super::adaptive_policy::{AdaptivePolicy, EagerPolicy, GreedyPolicy};
use super::{ForwardOutput, request};

mod chunked;

use chunked::ChunkContinuation;

fn scheduler_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PIE_SCHED_TRACE").is_some())
}

fn sched_epoch() -> Instant {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    *EPOCH.get_or_init(Instant::now)
}

fn now_micros() -> u64 {
    sched_epoch().elapsed().as_micros() as u64
}

// =============================================================================
// Scheduling Policy Trait
// =============================================================================

/// Pluggable scheduling policy.
///
/// A policy receives event callbacks (`on_arrival`, `on_complete`,
/// `on_fired`) and returns a [`Decision`] when asked whether to fire
/// the current batch.
pub(super) trait SchedulingPolicy: Send {
    /// A request was added to the accumulator.
    fn on_arrival(&mut self);

    /// A batch finished executing. `latency` is the wall-clock time
    /// the forward pass took on the driver.
    fn on_complete(&mut self, latency: Duration);

    /// The current batch was fired. `fired_size` is the number of
    /// requests in the batch — policies use it to learn the steady-
    /// state cohort size and avoid firing partial batches in the next
    /// cycle.
    fn on_fired(&mut self, fired_size: usize);

    /// Decide whether to fire or wait, given the current batch size.
    /// `&mut self` so policies can ratchet internal state (e.g.,
    /// AdaptivePolicy's `cohort_high_water`) on every poll.
    fn decide(&mut self, current_batch_size: usize, prefill_cohort: bool) -> Decision;
}

// =============================================================================
// Scheduling Decision
// =============================================================================

/// The outcome of a scheduling policy decision.
pub(super) enum Decision {
    /// Fire the current batch immediately.
    Fire,
    /// Wait for more requests, up to the given duration.
    Wait(Duration),
}

// =============================================================================
// SchedulerStats (lock-free snapshot for monitoring)
// =============================================================================

pub const SYSTEM_SPEC_DRAFT_POS_BUCKETS: usize = 32;

/// Cumulative stats exposed for monitoring. Updated atomically after each batch.
#[derive(Debug, Default)]
pub struct SchedulerStats {
    pub total_batches: AtomicU64,
    pub total_tokens_processed: AtomicU64,
    /// Total request count across all batches (sum of batch sizes).
    /// Divide by `total_batches` for mean batch size in requests.
    pub total_requests_processed: AtomicU64,
    /// Largest forward request count ever fired by this scheduler.
    pub max_forward_requests_observed: AtomicU64,
    /// Coarse histogram of batch sizes. Buckets:
    /// [0]=1, [1]=2-3, [2]=4-7, [3]=8-15, [4]=16-31,
    /// [5]=32-63, [6]=64-127, [7]=128+.
    pub batch_size_hist: [AtomicU64; 8],
    pub last_batch_latency_us: AtomicU64,
    pub cumulative_latency_us: AtomicU64,
    pub cumulative_permit_wait_us: AtomicU64,
    pub cumulative_fire_prepare_us: AtomicU64,
    pub cumulative_execute_batch_us: AtomicU64,
    pub cumulative_batch_build_us: AtomicU64,
    pub cumulative_driver_fire_us: AtomicU64,
    pub cumulative_response_dispatch_us: AtomicU64,
    pub cumulative_context_tick_submit_us: AtomicU64,
    pub cumulative_stats_update_us: AtomicU64,
    /// Time between consecutive `tokio::spawn(execute_batch)` calls.
    /// Sum is excluding the first batch; divide by (total_batches-1) for mean.
    pub cumulative_inter_fire_us: AtomicU64,
    /// Time from end of response dispatch (fire N) to spawn of fire N+1.
    /// This is the "rendezvous gap": chain extender wake + main loop drain
    /// + cohort fill.
    pub cumulative_post_dispatch_to_fire_us: AtomicU64,
    /// Time spent in the main run loop's non-blocking accumulator pass —
    /// per-iter try_recv + prepare + would_exceed + push, until the first
    /// `try_recv` returns Empty (or batch full / stashed). Sums across
    /// every batch.
    pub cumulative_accum_loop_us: AtomicU64,
    /// Internal: micros since `sched_epoch()` at last fire spawn.
    pub last_fire_spawn_micros: AtomicU64,
    /// Internal: micros since `sched_epoch()` at end of response dispatch.
    pub last_dispatch_end_micros: AtomicU64,
    pub system_spec_draft_tokens_proposed: AtomicU64,
    pub system_spec_draft_tokens_accepted: AtomicU64,
    pub system_spec_draft_tokens_proposed_per_pos:
        [AtomicU64; SYSTEM_SPEC_DRAFT_POS_BUCKETS],
    pub system_spec_draft_tokens_accepted_per_pos:
        [AtomicU64; SYSTEM_SPEC_DRAFT_POS_BUCKETS],
}

#[derive(Debug, Default, Clone, Copy)]
struct BatchExecutionTiming {
    total_us: u64,
    batch_build_us: u64,
    driver_fire_us: u64,
    response_dispatch_us: u64,
    system_spec_draft_tokens_proposed: u64,
    system_spec_draft_tokens_accepted: u64,
    system_spec_draft_tokens_proposed_per_pos: [u64; SYSTEM_SPEC_DRAFT_POS_BUCKETS],
    system_spec_draft_tokens_accepted_per_pos: [u64; SYSTEM_SPEC_DRAFT_POS_BUCKETS],
}

// =============================================================================
// PendingRequest
// =============================================================================

/// A forward pass request bundled with its response channel and physical pages.
struct PendingRequest {
    request: pie_bridge::ForwardRequest,
    completion: Completion,
    physical_page_ids: Vec<PhysicalPageId>,
    last_page_len: u32,
}

enum Completion {
    Direct(oneshot::Sender<Result<ForwardOutput>>),
    Chunk {
        continuation: ChunkContinuation,
        sampler_slots: Vec<usize>,
    },
    /// Self-perpetuating speculation chain. The dispatch loop routes
    /// these to the per-driver `ChainExtPool` instead of waking 256
    /// individual per-context tokio tasks; the pool worker then forwards
    /// this fire's output to `state.response` and submits the next stage
    /// (if eligible) via `state.scheduler_handle`.
    Chain {
        state: Box<super::speculator::ChainState>,
    },
}

impl PendingRequest {
    fn direct(
        request: pie_bridge::ForwardRequest,
        response_tx: oneshot::Sender<Result<ForwardOutput>>,
        physical_page_ids: Vec<PhysicalPageId>,
        last_page_len: u32,
    ) -> Self {
        Self {
            request,
            completion: Completion::Direct(response_tx),
            physical_page_ids,
            last_page_len,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct RequestCapacityUsage {
    forward_tokens: usize,
    page_refs: usize,
    logit_rows: usize,
    prob_rows: usize,
    sampler_rows: usize,
    logprob_labels: usize,
    user_custom_mask_bytes: usize,
    spec_custom_mask_bytes: usize,
    has_spec_drafts: bool,
    has_rs_spec_drafts: bool,
    has_dense_logit_requirement: bool,
    has_prob_sampling: bool,
    is_single_token_decode: bool,
    all_samplers_token: bool,
}

fn request_capacity_usage(req: &PendingRequest, page_size: u32) -> RequestCapacityUsage {
    let input_tokens = req.request.token_ids.len();
    let spec_tokens = req.request.spec_token_ids.len();
    let forward_tokens = input_tokens.saturating_add(spec_tokens);
    let mut sampler_rows = req.request.sampling_indices.len();
    if spec_tokens > 0 {
        sampler_rows = sampler_rows.saturating_add(spec_tokens.saturating_add(1));
    }
    let mut all_samplers_token = true;
    let mut has_prob_sampling = false;
    for sampler in &req.request.samplers {
        if !is_token_sampler(sampler) {
            all_samplers_token = false;
        }
        if sampler_needs_prob_rows(sampler) {
            has_prob_sampling = true;
        }
    }
    let has_dense_logit_requirement = req.request.has_user_mask
        || !req.request.logit_masks.is_empty()
        || spec_tokens > 0
        || !all_samplers_token;
    let is_single_token_decode = input_tokens == 1
        && spec_tokens == 0
        && req.request.single_token_mode
        && !req.request.has_user_mask;
    let page_refs = req.physical_page_ids.len();
    let spec_custom_mask_bytes =
        packed_mask_bytes(forward_tokens, page_refs, req.last_page_len, page_size);
    let user_custom_mask_bytes = if req.request.has_user_mask && input_tokens > 1 {
        packed_mask_bytes(input_tokens, page_refs, req.last_page_len, page_size)
    } else {
        0
    };

    RequestCapacityUsage {
        forward_tokens,
        page_refs,
        logit_rows: 0,
        prob_rows: 0,
        sampler_rows,
        logprob_labels: request_logprob_labels(&req.request),
        user_custom_mask_bytes,
        spec_custom_mask_bytes,
        has_spec_drafts: spec_tokens > 0,
        has_rs_spec_drafts: spec_tokens > 0 && !req.request.rs_slot_ids.is_empty(),
        has_dense_logit_requirement,
        has_prob_sampling,
        is_single_token_decode,
        all_samplers_token,
    }
}

fn is_token_sampler(sampler: &pie_bridge::Sampler) -> bool {
    matches!(
        sampler,
        pie_bridge::Sampler::Multinomial { .. }
            | pie_bridge::Sampler::TopK { .. }
            | pie_bridge::Sampler::TopP { .. }
            | pie_bridge::Sampler::MinP { .. }
            | pie_bridge::Sampler::TopKTopP { .. }
    )
}

fn sampler_needs_prob_rows(sampler: &pie_bridge::Sampler) -> bool {
    match sampler {
        pie_bridge::Sampler::TopK { temperature, k } => *temperature > 0.0 && *k > 0,
        pie_bridge::Sampler::TopP { temperature, p } => *temperature > 0.0 && *p < 1.0,
        pie_bridge::Sampler::TopKTopP { temperature, k, p } => {
            *temperature > 0.0 && (*k > 0 || *p < 1.0)
        }
        _ => false,
    }
}

fn packed_mask_bytes(
    query_tokens: usize,
    page_refs: usize,
    last_page_len: u32,
    page_size: u32,
) -> usize {
    if query_tokens == 0 || page_refs == 0 || page_size == 0 {
        return 0;
    }
    let kv_len = page_refs
        .saturating_sub(1)
        .saturating_mul(page_size as usize)
        .saturating_add(last_page_len as usize);
    query_tokens.saturating_mul(kv_len).saturating_add(7) / 8
}

fn request_logprob_labels(req: &pie_bridge::ForwardRequest) -> usize {
    req.samplers
        .iter()
        .map(|s| match s {
            pie_bridge::Sampler::Logprob { .. } => 1,
            pie_bridge::Sampler::Logprobs { token_ids } => token_ids.len(),
            _ => 0,
        })
        .sum()
}

// =============================================================================
// BatchAccumulator
// =============================================================================

/// Accumulates pending requests into a batch.
///
/// Pure synchronous struct — no async, no channels. Can be tested
/// independently from the scheduling loop.
struct BatchAccumulator {
    requests: Vec<PendingRequest>,
    total_tokens: usize,
    total_pages: usize,
    total_logit_rows: usize,
    total_prob_rows: usize,
    total_sampler_rows: usize,
    total_logprob_labels: usize,
    total_user_custom_mask_bytes: usize,
    total_spec_custom_mask_bytes: usize,
    has_spec_drafts: bool,
    has_rs_spec_drafts: bool,
    has_dense_logit_requirement: bool,
    has_prob_sampling: bool,
    all_single_token_decode: bool,
    all_samplers_token: bool,
    page_size: u32,
    limits: SchedulerLimits,
}

impl BatchAccumulator {
    fn new(limits: SchedulerLimits, page_size: u32) -> Self {
        Self {
            requests: Vec::new(),
            total_tokens: 0,
            total_pages: 0,
            total_logit_rows: 0,
            total_prob_rows: 0,
            total_sampler_rows: 0,
            total_logprob_labels: 0,
            total_user_custom_mask_bytes: 0,
            total_spec_custom_mask_bytes: 0,
            has_spec_drafts: false,
            has_rs_spec_drafts: false,
            has_dense_logit_requirement: false,
            has_prob_sampling: false,
            all_single_token_decode: true,
            all_samplers_token: true,
            page_size,
            limits,
        }
    }

    fn projected_rows(
        &self,
        extra: Option<&RequestCapacityUsage>,
    ) -> (usize, usize, bool, bool, bool) {
        let total_tokens = self
            .total_tokens
            .saturating_add(extra.map(|usage| usage.forward_tokens).unwrap_or(0));
        let total_sampler_rows = self
            .total_sampler_rows
            .saturating_add(extra.map(|usage| usage.sampler_rows).unwrap_or(0));
        let has_dense_logit_requirement = self.has_dense_logit_requirement
            || extra
                .map(|usage| usage.has_dense_logit_requirement)
                .unwrap_or(false);
        let has_prob_sampling =
            self.has_prob_sampling || extra.map(|usage| usage.has_prob_sampling).unwrap_or(false);
        let all_samplers_token =
            self.all_samplers_token && extra.map(|usage| usage.all_samplers_token).unwrap_or(true);
        let all_single_token_decode = self.all_single_token_decode
            && extra
                .map(|usage| usage.is_single_token_decode)
                .unwrap_or(true);
        let compact_logit_rows = !all_single_token_decode
            && !has_dense_logit_requirement
            && !has_prob_sampling
            && all_samplers_token
            && total_sampler_rows > 0
            && total_sampler_rows < total_tokens;
        let logit_rows = if compact_logit_rows {
            total_sampler_rows
        } else {
            total_tokens
        };
        let prob_rows = if has_prob_sampling { total_tokens } else { 0 };
        (
            logit_rows,
            prob_rows,
            has_dense_logit_requirement,
            has_prob_sampling,
            all_single_token_decode,
        )
    }

    fn push(&mut self, req: PendingRequest) {
        let usage = request_capacity_usage(&req, self.page_size);
        self.push_with(req, usage);
    }

    fn push_with(&mut self, req: PendingRequest, mut usage: RequestCapacityUsage) {
        let (logit_rows, prob_rows, _, _, _) = self.projected_rows(Some(&usage));
        usage.logit_rows = logit_rows;
        usage.prob_rows = prob_rows;
        self.total_tokens = self.total_tokens.saturating_add(usage.forward_tokens);
        self.total_pages = self.total_pages.saturating_add(usage.page_refs);
        self.total_logit_rows = usage.logit_rows;
        self.total_prob_rows = usage.prob_rows;
        self.total_sampler_rows = self.total_sampler_rows.saturating_add(usage.sampler_rows);
        self.total_logprob_labels = self
            .total_logprob_labels
            .saturating_add(usage.logprob_labels);
        self.total_user_custom_mask_bytes = self
            .total_user_custom_mask_bytes
            .saturating_add(usage.user_custom_mask_bytes);
        self.total_spec_custom_mask_bytes = self
            .total_spec_custom_mask_bytes
            .saturating_add(usage.spec_custom_mask_bytes);
        self.has_spec_drafts |= usage.has_spec_drafts;
        self.has_rs_spec_drafts |= usage.has_rs_spec_drafts;
        self.has_dense_logit_requirement |= usage.has_dense_logit_requirement;
        self.has_prob_sampling |= usage.has_prob_sampling;
        self.all_single_token_decode &= usage.is_single_token_decode;
        self.all_samplers_token &= usage.all_samplers_token;
        self.requests.push(req);
    }

    fn single_request_limit_error(&self, req: &PendingRequest) -> Option<String> {
        let usage = request_capacity_usage(req, self.page_size);
        let (logit_rows, prob_rows, _, _, _) =
            BatchAccumulator::new(self.limits, self.page_size).projected_rows(Some(&usage));
        if usage.forward_tokens > self.limits.max_forward_tokens {
            return Some(format!(
                "forward request has {} forward tokens, exceeding driver limit {}",
                usage.forward_tokens, self.limits.max_forward_tokens
            ));
        }

        if usage.page_refs > self.limits.max_page_refs {
            return Some(format!(
                "forward request has {} page refs, exceeding driver limit {}",
                usage.page_refs, self.limits.max_page_refs
            ));
        }

        if logit_rows > self.limits.max_logit_rows {
            return Some(format!(
                "forward request needs {} logit rows, exceeding driver limit {}",
                logit_rows, self.limits.max_logit_rows
            ));
        }

        if prob_rows > self.limits.max_prob_rows {
            return Some(format!(
                "forward request needs {} probability rows, exceeding driver limit {}",
                prob_rows, self.limits.max_prob_rows
            ));
        }

        if usage.sampler_rows > self.limits.max_sampler_rows {
            return Some(format!(
                "forward request has {} sampler rows, exceeding driver limit {}",
                usage.sampler_rows, self.limits.max_sampler_rows
            ));
        }

        if usage.logprob_labels > self.limits.max_logprob_labels {
            return Some(format!(
                "forward request has {} logprob labels, exceeding driver limit {}",
                usage.logprob_labels, self.limits.max_logprob_labels
            ));
        }

        let custom_mask_bytes = if usage.has_spec_drafts {
            usage.spec_custom_mask_bytes
        } else {
            usage.user_custom_mask_bytes
        };
        if custom_mask_bytes > self.limits.max_custom_mask_bytes {
            return Some(format!(
                "forward request needs {custom_mask_bytes} custom mask bytes, exceeding driver limit {}",
                self.limits.max_custom_mask_bytes
            ));
        }

        if self.limits.max_forward_requests == 0 {
            return Some("driver max forward requests is zero".to_string());
        }

        None
    }

    fn would_exceed(&self, req: &PendingRequest) -> bool {
        if self.requests.is_empty() {
            return false;
        }
        let usage = request_capacity_usage(req, self.page_size);
        self.would_exceed_with(&usage)
    }

    fn would_exceed_with(&self, usage: &RequestCapacityUsage) -> bool {
        if self.requests.is_empty() {
            return false;
        }
        if self.has_rs_spec_drafts || usage.has_rs_spec_drafts {
            return true;
        }
        let next_has_spec = self.has_spec_drafts || usage.has_spec_drafts;
        let next_custom_mask_bytes = if next_has_spec {
            self.total_spec_custom_mask_bytes
                .saturating_add(usage.spec_custom_mask_bytes)
        } else {
            self.total_user_custom_mask_bytes
                .saturating_add(usage.user_custom_mask_bytes)
        };
        let (next_logit_rows, next_prob_rows, _, _, _) = self.projected_rows(Some(usage));
        self.requests.len() + 1 > self.limits.max_forward_requests
            || self.total_tokens.saturating_add(usage.forward_tokens)
                > self.limits.max_forward_tokens
            || self.total_pages.saturating_add(usage.page_refs) > self.limits.max_page_refs
            || next_logit_rows > self.limits.max_logit_rows
            || next_prob_rows > self.limits.max_prob_rows
            || self.total_sampler_rows.saturating_add(usage.sampler_rows)
                > self.limits.max_sampler_rows
            || self
                .total_logprob_labels
                .saturating_add(usage.logprob_labels)
                > self.limits.max_logprob_labels
            || next_custom_mask_bytes > self.limits.max_custom_mask_bytes
    }

    fn would_exceed_reason(&self, req: &PendingRequest) -> Option<String> {
        if self.requests.is_empty() {
            return None;
        }
        let usage = request_capacity_usage(req, self.page_size);
        let next_has_spec = self.has_spec_drafts || usage.has_spec_drafts;
        let next_custom_mask_bytes = if next_has_spec {
            self.total_spec_custom_mask_bytes
                .saturating_add(usage.spec_custom_mask_bytes)
        } else {
            self.total_user_custom_mask_bytes
                .saturating_add(usage.user_custom_mask_bytes)
        };
        let (next_logit_rows, next_prob_rows, _, _, _) = self.projected_rows(Some(&usage));
        let checks = [
            (
                "requests",
                self.requests.len().saturating_add(1),
                self.limits.max_forward_requests,
            ),
            (
                "tokens",
                self.total_tokens.saturating_add(usage.forward_tokens),
                self.limits.max_forward_tokens,
            ),
            (
                "pages",
                self.total_pages.saturating_add(usage.page_refs),
                self.limits.max_page_refs,
            ),
            ("logit_rows", next_logit_rows, self.limits.max_logit_rows),
            ("prob_rows", next_prob_rows, self.limits.max_prob_rows),
            (
                "sampler_rows",
                self.total_sampler_rows.saturating_add(usage.sampler_rows),
                self.limits.max_sampler_rows,
            ),
            (
                "logprob_labels",
                self.total_logprob_labels
                    .saturating_add(usage.logprob_labels),
                self.limits.max_logprob_labels,
            ),
            (
                "custom_mask_bytes",
                next_custom_mask_bytes,
                self.limits.max_custom_mask_bytes,
            ),
        ];
        checks
            .into_iter()
            .find(|(_, have, limit)| have > limit)
            .map(|(name, have, limit)| {
                format!(
                    "{name} {have}>{limit} pending_tokens={} pending_pages={} pending_sampler_rows={} pending_has_spec={} pending_dense={} pending_prob={}",
                    usage.forward_tokens,
                    usage.page_refs,
                    usage.sampler_rows,
                    usage.has_spec_drafts,
                    usage.has_dense_logit_requirement,
                    usage.has_prob_sampling,
                )
            })
    }

    fn is_full(&self) -> bool {
        let active_custom_mask_bytes = if self.has_spec_drafts {
            self.total_spec_custom_mask_bytes
        } else {
            self.total_user_custom_mask_bytes
        };
        self.requests.len() >= self.limits.max_forward_requests
            || self.has_rs_spec_drafts
            || self.total_tokens >= self.limits.max_forward_tokens
            || self.total_pages >= self.limits.max_page_refs
            || self.total_logit_rows >= self.limits.max_logit_rows
            || self.total_prob_rows >= self.limits.max_prob_rows
            || self.total_sampler_rows >= self.limits.max_sampler_rows
            || self.total_logprob_labels >= self.limits.max_logprob_labels
            || active_custom_mask_bytes >= self.limits.max_custom_mask_bytes
    }

    fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    fn len(&self) -> usize {
        self.requests.len()
    }

    fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    fn should_prefill_coalesce(&self) -> bool {
        !self.has_spec_drafts && self.total_tokens > self.requests.len()
    }

    fn take(&mut self) -> Vec<PendingRequest> {
        self.total_tokens = 0;
        self.total_pages = 0;
        self.total_logit_rows = 0;
        self.total_prob_rows = 0;
        self.total_sampler_rows = 0;
        self.total_logprob_labels = 0;
        self.total_user_custom_mask_bytes = 0;
        self.total_spec_custom_mask_bytes = 0;
        self.has_spec_drafts = false;
        self.has_rs_spec_drafts = false;
        self.has_dense_logit_requirement = false;
        self.has_prob_sampling = false;
        self.all_single_token_decode = true;
        self.all_samplers_token = true;
        std::mem::take(&mut self.requests)
    }
}

fn prepare_pending_for_batch(
    batch: &BatchAccumulator,
    pending: PendingRequest,
) -> Option<PendingRequest> {
    prepare_pending_with_usage(batch, pending).map(|(p, _)| p)
}

/// Same as `prepare_pending_for_batch` but also returns the computed
/// `RequestCapacityUsage`, avoiding a recompute when the caller will
/// immediately consult `would_exceed_with` + `push_with`.
///
/// The pure-decode fast path skips `maybe_start_chunking` and
/// `single_request_limit_error`: for `single_token_mode` requests with
/// 1 token, no spec drafts, no user mask, and no logit masks, both are
/// no-ops (chunk_size is never reached; per-request limits hold trivially
/// given the BatchAccumulator's `would_exceed_with` check will still gate
/// page_refs / sampler_rows etc. when batching).
fn prepare_pending_with_usage(
    batch: &BatchAccumulator,
    pending: PendingRequest,
) -> Option<(PendingRequest, RequestCapacityUsage)> {
    if is_pure_decode_pending(&pending) {
        let usage = request_capacity_usage(&pending, batch.page_size);
        let limits = batch.limits;
        // Fields that COULD still trip the single-request limit for decode:
        // page_refs (long-context decode) and logprob_labels. Token/sampler/
        // mask limits hold trivially for 1-token single_token_mode requests.
        if usage.page_refs <= limits.max_page_refs
            && usage.logprob_labels <= limits.max_logprob_labels
            && limits.max_forward_requests > 0
        {
            return Some((pending, usage));
        }
        // Fall through to the slow path so the proper error message is
        // surfaced via `single_request_limit_error`.
    }
    let pending = match pending.maybe_start_chunking(batch.limits, batch.page_size) {
        Ok(pending) => pending,
        Err((pending, msg)) => {
            pending.send_error(msg);
            return None;
        }
    };
    if let Some(msg) = batch.single_request_limit_error(&pending) {
        pending.send_error(msg);
        return None;
    }
    let usage = request_capacity_usage(&pending, batch.page_size);
    Some((pending, usage))
}

#[inline]
fn is_pure_decode_pending(p: &PendingRequest) -> bool {
    matches!(&p.completion, Completion::Direct(_))
        && p.request.token_ids.len() == 1
        && p.request.spec_token_ids.is_empty()
        && p.request.single_token_mode
        && !p.request.has_user_mask
        && p.request.logit_masks.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits(max_requests: usize, max_tokens: usize, max_pages: usize) -> SchedulerLimits {
        SchedulerLimits {
            max_forward_requests: max_requests,
            max_forward_tokens: max_tokens,
            max_page_refs: max_pages,
            max_logit_rows: usize::MAX,
            max_prob_rows: usize::MAX,
            max_sampler_rows: usize::MAX,
            max_custom_mask_bytes: usize::MAX,
            max_logprob_labels: usize::MAX,
        }
    }

    fn pending(tokens: usize, page_refs: usize) -> PendingRequest {
        let (tx, _rx) = oneshot::channel();
        PendingRequest::direct(
            pie_bridge::ForwardRequest {
                token_ids: vec![0; tokens],
                ..Default::default()
            },
            tx,
            vec![0; page_refs],
            1,
        )
    }

    fn with_spec(mut req: PendingRequest, spec_tokens: usize) -> PendingRequest {
        req.request.spec_token_ids = vec![1; spec_tokens];
        req.request.spec_position_ids = vec![1; spec_tokens];
        req.request.spec_indptr = vec![0, spec_tokens as u32];
        req
    }

    fn with_sampler_rows(mut req: PendingRequest, sampler_rows: usize) -> PendingRequest {
        req.request.sampling_indices = vec![0; sampler_rows];
        req
    }

    fn with_samplers(
        mut req: PendingRequest,
        indices: Vec<u32>,
        samplers: Vec<pie_bridge::Sampler>,
    ) -> PendingRequest {
        req.request.sampling_indices = indices;
        req.request.samplers = samplers;
        req
    }

    #[test]
    fn accumulator_splits_by_forward_tokens() {
        let mut batch = BatchAccumulator::new(limits(8, 6, 100), 16);
        batch.push(pending(4, 1));
        assert!(!batch.would_exceed(&pending(2, 1)));
        assert!(batch.would_exceed(&pending(3, 1)));
    }

    #[test]
    fn accumulator_splits_by_forward_requests() {
        let mut batch = BatchAccumulator::new(limits(2, 100, 100), 16);
        batch.push(pending(1, 1));
        assert!(!batch.would_exceed(&pending(1, 1)));
        batch.push(pending(1, 1));
        assert!(batch.is_full());
        assert!(batch.would_exceed(&pending(1, 1)));
    }

    #[test]
    fn accumulator_splits_by_page_refs() {
        let mut batch = BatchAccumulator::new(limits(8, 100, 5), 16);
        batch.push(pending(1, 3));
        assert!(!batch.would_exceed(&pending(1, 2)));
        assert!(batch.would_exceed(&pending(1, 3)));
    }

    #[test]
    fn accumulator_rejects_single_request_over_limit() {
        let batch = BatchAccumulator::new(limits(8, 6, 5), 16);
        assert!(batch.single_request_limit_error(&pending(7, 1)).is_some());
        assert!(batch.single_request_limit_error(&pending(1, 6)).is_some());
        assert!(batch.single_request_limit_error(&pending(6, 5)).is_none());
    }

    #[test]
    fn accumulator_counts_speculative_tokens() {
        let mut batch = BatchAccumulator::new(limits(8, 6, 100), 16);
        batch.push(with_spec(pending(4, 1), 2));
        assert!(batch.is_full());
        assert!(batch.would_exceed(&pending(1, 1)));

        let batch = BatchAccumulator::new(limits(8, 6, 100), 16);
        assert!(
            batch
                .single_request_limit_error(&with_spec(pending(5, 1), 2))
                .is_some()
        );
    }

    #[test]
    fn accumulator_splits_by_sampler_rows() {
        let mut capped = limits(8, 100, 100);
        capped.max_sampler_rows = 3;
        let mut batch = BatchAccumulator::new(capped, 16);
        batch.push(with_sampler_rows(pending(1, 1), 2));
        assert!(!batch.would_exceed(&with_sampler_rows(pending(1, 1), 1)));
        assert!(batch.would_exceed(&with_sampler_rows(pending(1, 1), 2)));
        assert!(
            batch
                .single_request_limit_error(&with_sampler_rows(pending(1, 1), 4))
                .is_some()
        );
    }

    #[test]
    fn accumulator_counts_spec_verification_sampler_rows() {
        let mut capped = limits(8, 100, 100);
        capped.max_sampler_rows = 3;
        let batch = BatchAccumulator::new(capped, 16);
        let req = with_spec(with_sampler_rows(pending(1, 1), 1), 2);
        assert!(batch.single_request_limit_error(&req).is_some());
    }

    #[test]
    fn accumulator_splits_by_custom_mask_bytes() {
        let mut capped = limits(8, 100, 100);
        capped.max_custom_mask_bytes = 31;
        let mut batch = BatchAccumulator::new(capped, 16);

        // 2 query rows x 64 KV positions = 128 bits = 16 bytes.
        let mut user_mask = pending(2, 4);
        user_mask.last_page_len = 16;
        user_mask.request.has_user_mask = true;
        batch.push(user_mask);

        let mut next = pending(2, 4);
        next.last_page_len = 16;
        next.request.has_user_mask = true;
        assert!(batch.would_exceed(&next));
    }

    #[test]
    fn adding_spec_request_counts_existing_requests_for_spec_mask_path() {
        let mut capped = limits(8, 100, 100);
        capped.max_custom_mask_bytes = 31;
        let mut batch = BatchAccumulator::new(capped, 16);

        let mut existing = pending(2, 4);
        existing.last_page_len = 16;
        batch.push(existing);

        let mut spec = with_spec(pending(1, 4), 1);
        spec.last_page_len = 16;
        assert!(batch.would_exceed(&spec));
    }

    #[test]
    fn accumulator_rejects_logprob_label_over_limit() {
        let mut capped = limits(8, 100, 100);
        capped.max_logprob_labels = 2;
        let batch = BatchAccumulator::new(capped, 16);
        let mut req = pending(1, 1);
        req.request.samplers = vec![pie_bridge::Sampler::Logprobs {
            token_ids: vec![1, 2, 3],
        }];
        assert!(batch.single_request_limit_error(&req).is_some());
    }

    #[test]
    fn accumulator_allows_compact_prefill_logit_rows() {
        let mut capped = limits(8, 8, 100);
        capped.max_logit_rows = 2;
        let batch = BatchAccumulator::new(capped, 16);
        let req = with_samplers(
            pending(4, 1),
            vec![3],
            vec![pie_bridge::Sampler::TopP {
                temperature: 0.0,
                p: 1.0,
            }],
        );
        assert!(batch.single_request_limit_error(&req).is_none());
    }

    #[test]
    fn accumulator_splits_by_compact_logit_rows() {
        let mut capped = limits(8, 100, 100);
        capped.max_logit_rows = 2;
        let mut batch = BatchAccumulator::new(capped, 16);
        let req = || {
            with_samplers(
                pending(4, 1),
                vec![3],
                vec![pie_bridge::Sampler::TopP {
                    temperature: 0.0,
                    p: 1.0,
                }],
            )
        };
        batch.push(req());
        assert!(!batch.would_exceed(&req()));
        batch.push(req());
        assert!(batch.would_exceed(&req()));
    }

    #[test]
    fn accumulator_rejects_dense_logit_over_limit() {
        let mut capped = limits(8, 100, 100);
        capped.max_logit_rows = 3;
        let batch = BatchAccumulator::new(capped, 16);
        let req = with_samplers(pending(4, 1), vec![3], vec![pie_bridge::Sampler::RawLogits]);
        assert!(batch.single_request_limit_error(&req).is_some());
    }

    #[test]
    fn accumulator_rejects_probability_rows_over_limit() {
        let mut capped = limits(8, 100, 100);
        capped.max_logit_rows = 8;
        capped.max_prob_rows = 3;
        let batch = BatchAccumulator::new(capped, 16);
        let req = with_samplers(
            pending(4, 1),
            vec![3],
            vec![pie_bridge::Sampler::TopP {
                temperature: 1.0,
                p: 0.9,
            }],
        );
        assert!(batch.single_request_limit_error(&req).is_some());
    }

    #[test]
    fn taking_batch_does_not_drop_stashed_request_shape() {
        let mut batch = BatchAccumulator::new(limits(8, 6, 5), 16);
        batch.push(pending(4, 2));
        let stashed = pending(4, 4);
        assert!(batch.would_exceed(&stashed));
        let fired = batch.take();
        assert_eq!(fired.len(), 1);
        assert!(batch.is_empty());
        assert!(!batch.would_exceed(&stashed));
    }
}

// =============================================================================
// SchedulerHandle
// =============================================================================

/// Cloneable submit handle. Used by the speculator's chain extender
/// (spawned outside the scheduler's `run` loop) to resubmit
/// pre-staged forward passes.
///
/// Backed by a sync crossbeam_channel rather than tokio mpsc so the
/// receiving main loop (sync OS thread) can recv with futex-level
/// wake latency (~5-15 µs) instead of tokio's task-wake roundtrip
/// (~100-200 µs).
#[derive(Clone)]
pub(crate) struct SchedulerHandle {
    tx: crossbeam::channel::Sender<PendingRequest>,
}

impl SchedulerHandle {
    pub fn submit(
        &self,
        request: pie_bridge::ForwardRequest,
        response_tx: oneshot::Sender<Result<ForwardOutput>>,
        physical_page_ids: Vec<PhysicalPageId>,
        last_page_len: u32,
    ) -> Result<()> {
        self.tx
            .send(PendingRequest::direct(
                request,
                response_tx,
                physical_page_ids,
                last_page_len,
            ))
            .map_err(|_| anyhow::anyhow!("scheduler channel closed"))?;
        Ok(())
    }

    /// Submit a forward pass that participates in a speculation chain.
    /// The dispatch loop routes the output to the per-driver pool worker
    /// instead of waking a dedicated chain-extender task.
    pub fn submit_chain(
        &self,
        request: pie_bridge::ForwardRequest,
        state: Box<super::speculator::ChainState>,
        physical_page_ids: Vec<PhysicalPageId>,
        last_page_len: u32,
    ) -> Result<()> {
        self.tx
            .send(PendingRequest {
                request,
                completion: Completion::Chain { state },
                physical_page_ids,
                last_page_len,
            })
            .map_err(|_| anyhow::anyhow!("scheduler channel closed"))?;
        Ok(())
    }
}

// =============================================================================
// BatchScheduler
// =============================================================================

/// Per-driver batch scheduler.
///
/// Owns an RPC client, a scheduling policy, and a tokio task that
/// runs the batch accumulation and firing loop.
pub(crate) struct BatchScheduler {
    tx: crossbeam::channel::Sender<PendingRequest>,
    stats: Arc<SchedulerStats>,
    chain_pool: Arc<super::speculator::ChainExtPool>,
}

/// Default size of the chain-extender pool per driver. Sweep on L40
/// (30 tokio workers) found 30 optimal: 8/16/24 starve per-worker by
/// serializing too many jobs; 36/40/64/128 over-fan out and contend
/// for runtime workers. The sweet spot is one pool task per runtime
/// worker — dispatch's notify burst resolves in a single scheduling
/// round.
const CHAIN_EXT_POOL_SIZE: usize = 30;

impl BatchScheduler {
    /// Spawn a new batch scheduler for a single driver.
    ///
    /// The RPC connection is owned by the driver service; the scheduler
    /// only stores the driver index for routing calls.
    pub fn new(
        driver_id: DriverId,
        driver_idx: usize,
        page_size: u32,
        limits: SchedulerLimits,
        request_timeout_secs: u64,
        batch_policy: String,
    ) -> Self {
        let (tx, rx) = crossbeam::channel::unbounded::<PendingRequest>();
        let submit_tx = tx.clone();
        let stats = Arc::new(SchedulerStats::default());
        let chain_pool = Arc::new(super::speculator::ChainExtPool::new(CHAIN_EXT_POOL_SIZE));

        // Run the main scheduling loop on a dedicated OS thread with
        // crossbeam channels. Why: tokio's mpsc/select! wake-pickup
        // path takes ~100-200 µs because the receiver's waker has to be
        // scheduled onto a runtime worker. crossbeam's recv/select uses
        // futex parking directly — wake latency drops to ~5-15 µs.
        // execute_batch tasks still spawn on the shared tokio runtime
        // (captured via Handle) so they keep multi-worker parallelism
        // for the GPU/IPC and response dispatch.
        let rt_handle = tokio::runtime::Handle::current();
        let stats_for_loop = stats.clone();
        let chain_pool_for_loop = chain_pool.clone();
        let batch_policy_for_loop = batch_policy.clone();
        std::thread::Builder::new()
            .name(format!("pie-sched-{driver_idx}"))
            .spawn(move || {
                Self::run(
                    driver_id,
                    driver_idx,
                    rx,
                    submit_tx,
                    page_size,
                    limits,
                    request_timeout_secs,
                    batch_policy_for_loop,
                    stats_for_loop,
                    chain_pool_for_loop,
                    rt_handle,
                );
            })
            .expect("spawn pie-sched thread");

        Self { tx, stats, chain_pool }
    }

    /// Get the chain extender pool handle. Cold submits use this to
    /// route the first fire's output through the pool instead of spawning
    /// a per-context chain-extender task.
    pub fn chain_pool(&self) -> Arc<super::speculator::ChainExtPool> {
        self.chain_pool.clone()
    }

    /// Get a handle to the cumulative scheduler stats (lock-free).
    pub fn stats(&self) -> &Arc<SchedulerStats> {
        &self.stats
    }

    /// Submit a pre-translated forward pass request.
    pub fn submit(
        &self,
        request: pie_bridge::ForwardRequest,
        response_tx: oneshot::Sender<Result<ForwardOutput>>,
        physical_page_ids: Vec<PhysicalPageId>,
        last_page_len: u32,
    ) -> Result<()> {
        self.tx
            .send(PendingRequest::direct(
                request,
                response_tx,
                physical_page_ids,
                last_page_len,
            ))
            .map_err(|_| anyhow::anyhow!("scheduler channel closed"))?;
        Ok(())
    }

    /// Cloneable handle for tasks that need to submit outside the
    /// scheduler's `run` loop (e.g., the speculator chain extender).
    pub(crate) fn handle(&self) -> SchedulerHandle {
        SchedulerHandle {
            tx: self.tx.clone(),
        }
    }

    // =========================================================================
    // Internal: Scheduling Loop
    // =========================================================================

    /// Main scheduling loop for a single driver. Sync OS thread —
    /// recv/select use futex parking (no tokio waker overhead).
    fn run(
        driver_id: DriverId,
        driver_idx: usize,
        req_rx: crossbeam::channel::Receiver<PendingRequest>,
        submit_tx: crossbeam::channel::Sender<PendingRequest>,
        page_size: u32,
        limits: SchedulerLimits,
        request_timeout_secs: u64,
        batch_policy: String,
        stats: Arc<SchedulerStats>,
        chain_pool: Arc<super::speculator::ChainExtPool>,
        rt_handle: tokio::runtime::Handle,
    ) {
        let request_timeout = Duration::from_secs(request_timeout_secs);

        // Per-driver state
        let mut batch = BatchAccumulator::new(limits, page_size);
        // Policy selection from config — see `adaptive_policy.rs` for the
        // design rationale. The per-model `[model.scheduler].batch_policy`
        // setting picks one of "adaptive", "eager", "greedy".
        let mut policy: Box<dyn SchedulingPolicy> = match batch_policy.as_str() {
            "greedy" => Box::new(GreedyPolicy::new()),
            "eager" => Box::new(EagerPolicy::new(limits.max_forward_requests, driver_idx)),
            "adaptive" => Box::new(AdaptivePolicy::new(limits.max_forward_requests, driver_idx)),
            other => panic!(
                "Unknown scheduler.batch_policy {other:?}; expected one of \
                'adaptive' | 'eager' | 'greedy'"
            ),
        };
        // Only one in-flight batch at a time to prevent pipelined KV cache corruption.
        // AtomicBool gate: set true on Fire, cleared by execute_batch task
        // after driver_fire returns (mirrors permit drop in the prior
        // tokio Semaphore design).
        use std::sync::atomic::AtomicBool;
        let in_flight = Arc::new(AtomicBool::new(false));

        // Channel for batch completion latency feedback to the policy.
        let (latency_tx, latency_rx) = crossbeam::channel::unbounded::<Duration>();
        let mut next_pending: Option<PendingRequest> = None;

        'run_loop: loop {
            // Drain completed batch latencies (non-blocking)
            while let Ok(latency) = latency_rx.try_recv() {
                policy.on_complete(latency);
            }

            // Wait for first request if batch is empty. crossbeam's
            // recv() parks via futex — far lower wake latency than
            // tokio's mpsc waker path.
            while batch.is_empty() {
                let pending = if let Some(pending) = next_pending.take() {
                    pending
                } else {
                    match req_rx.recv() {
                        Ok(p) => p,
                        Err(_) => break 'run_loop,
                    }
                };
                let Some(pending) = prepare_pending_for_batch(&batch, pending) else {
                    continue;
                };
                policy.on_arrival();
                batch.push(pending);
            }

            // Accumulate more requests (non-blocking). If a request is
            // already stashed for the next batch, fire the current batch
            // before reading more; overwriting the stash would drop that
            // request's response channel.
            let accum_start = Instant::now();
            while next_pending.is_none() {
                let pending = match req_rx.try_recv() {
                    Ok(p) => p,
                    Err(crossbeam::channel::TryRecvError::Empty) => break,
                    Err(crossbeam::channel::TryRecvError::Disconnected) => {
                        break 'run_loop;
                    }
                };
                let Some((pending, usage)) = prepare_pending_with_usage(&batch, pending)
                else {
                    continue;
                };
                if batch.would_exceed_with(&usage) {
                    if scheduler_trace_enabled() {
                        let reason = batch
                            .would_exceed_reason(&pending)
                            .unwrap_or_else(|| "unknown".to_string());
                        eprintln!(
                            "[pie-sched-trace] driver={} stash current_requests={} current_tokens={} reason={}",
                            driver_idx,
                            batch.len(),
                            batch.total_tokens(),
                            reason,
                        );
                    }
                    next_pending = Some(pending);
                    break;
                }
                policy.on_arrival();
                batch.push_with(pending, usage);
                if batch.is_full() {
                    break;
                }
            }
            stats
                .cumulative_accum_loop_us
                .fetch_add(accum_start.elapsed().as_micros() as u64, Relaxed);

            // Ask the policy what to do
            let decision = if next_pending.is_some() {
                Decision::Fire
            } else {
                policy.decide(batch.len(), batch.should_prefill_coalesce())
            };
            match decision {
                Decision::Fire => {
                    // Acquire the in-flight gate. AdaptivePolicy already
                    // gates on this (returns Wait while in_flight=true),
                    // so the CAS should usually succeed on first try. The
                    // spin loop is defensive in case the policy is greedy
                    // or there's a race.
                    let permit_wait_start = Instant::now();
                    while in_flight
                        .compare_exchange(
                            false,
                            true,
                            std::sync::atomic::Ordering::AcqRel,
                            std::sync::atomic::Ordering::Acquire,
                        )
                        .is_err()
                    {
                        std::thread::yield_now();
                    }
                    let permit_wait_us =
                        permit_wait_start.elapsed().as_micros() as u64;

                    // The policy may decide to fire while the previous GPU
                    // batch is still in flight. Do one last non-blocking
                    // drain after the permit opens so requests that arrived
                    // during that wait are coalesced into this batch instead
                    // of being stranded behind a tiny stale fire.
                    let fire_prepare_start = Instant::now();
                    while next_pending.is_none() && !batch.is_full() {
                        let Ok(pending) = req_rx.try_recv() else {
                            break;
                        };
                        if let Some(msg) = batch.single_request_limit_error(&pending) {
                            pending.send_error(msg);
                            continue;
                        }
                        if batch.would_exceed(&pending) {
                            if scheduler_trace_enabled() {
                                let reason = batch
                                    .would_exceed_reason(&pending)
                                    .unwrap_or_else(|| "unknown".to_string());
                                eprintln!(
                                    "[pie-sched-trace] driver={} stash current_requests={} current_tokens={} reason={}",
                                    driver_idx,
                                    batch.len(),
                                    batch.total_tokens(),
                                    reason,
                                );
                            }
                            next_pending = Some(pending);
                            break;
                        }
                        policy.on_arrival();
                        batch.push(pending);
                    }

                    let total_tokens = batch.total_tokens();
                    if scheduler_trace_enabled() {
                        eprintln!(
                            "[pie-sched-trace] driver={} fire requests={} tokens={} prefill_like={} stashed={}",
                            driver_idx,
                            batch.len(),
                            total_tokens,
                            batch.should_prefill_coalesce(),
                            next_pending.is_some(),
                        );
                    }
                    let requests_to_fire = batch.take();
                    policy.on_fired(requests_to_fire.len());

                    // Collect batch context IDs for accurate rent charging.
                    // Per-request shape stores the single context_id in
                    // `context_ids[0]`.
                    let batch_ctx_ids: Vec<u64> = requests_to_fire
                        .iter()
                        .map(|r| r.request.context_ids[0])
                        .collect();
                    let batch_size = batch_ctx_ids.len() as u64;
                    let fire_prepare_us = fire_prepare_start.elapsed().as_micros() as u64;
                    stats
                        .cumulative_permit_wait_us
                        .fetch_add(permit_wait_us, Relaxed);
                    stats
                        .cumulative_fire_prepare_us
                        .fetch_add(fire_prepare_us, Relaxed);

                    // Inter-fire instrumentation: time between consecutive spawns,
                    // and the post-dispatch-to-next-fire gap (rendezvous window).
                    let now_us = now_micros();
                    let last_spawn = stats.last_fire_spawn_micros.swap(now_us, Relaxed);
                    if last_spawn != 0 {
                        stats
                            .cumulative_inter_fire_us
                            .fetch_add(now_us.saturating_sub(last_spawn), Relaxed);
                    }
                    let last_dispatch_end = stats.last_dispatch_end_micros.load(Relaxed);
                    if last_dispatch_end != 0 {
                        stats
                            .cumulative_post_dispatch_to_fire_us
                            .fetch_add(now_us.saturating_sub(last_dispatch_end), Relaxed);
                    }

                    // Spawn batch execution on the shared multi-thread
                    // tokio runtime (captured rt_handle). The async task
                    // clears `in_flight` itself when its driver_fire
                    // returns, mirroring the prior tokio Semaphore permit
                    // drop semantics.
                    let latency_tx_clone = latency_tx.clone();
                    let stats_clone = stats.clone();
                    let timeout = request_timeout;
                    let submit_tx_clone = submit_tx.clone();

                    let stats_for_probe = stats_clone.clone();
                    let chain_pool_clone = chain_pool.clone();
                    let in_flight_clone = in_flight.clone();
                    rt_handle.spawn(async move {
                        let start = Instant::now();
                        let timing = Self::execute_batch(
                            driver_idx,
                            requests_to_fire,
                            driver_id,
                            page_size,
                            timeout,
                            Some(in_flight_clone),
                            Some(submit_tx_clone),
                            Some(stats_for_probe),
                            Some(chain_pool_clone),
                        )
                        .await;
                        let latency = start.elapsed();
                        let _ = latency_tx_clone.send(latency);

                        // Advance market clock for this driver: prices, rent, dividends.
                        // Pass batch context IDs so tick only charges contexts
                        // that were in this batch (not stale pinned contexts).
                        let tick_submit_start = Instant::now();
                        crate::context::tick(driver_idx, latency.as_secs_f64(), batch_ctx_ids);
                        let tick_submit_us = tick_submit_start.elapsed().as_micros() as u64;

                        // Update cumulative atomic counters (consumed by external
                        // monitoring; ignored by the policy).
                        let stats_update_start = Instant::now();
                        stats_clone.total_batches.fetch_add(1, Relaxed);
                        stats_clone
                            .total_tokens_processed
                            .fetch_add(total_tokens as u64, Relaxed);
                        stats_clone
                            .total_requests_processed
                            .fetch_add(batch_size, Relaxed);
                        stats_clone
                            .max_forward_requests_observed
                            .fetch_max(batch_size, Relaxed);
                        let bucket = match batch_size {
                            0 | 1 => 0,
                            2..=3 => 1,
                            4..=7 => 2,
                            8..=15 => 3,
                            16..=31 => 4,
                            32..=63 => 5,
                            64..=127 => 6,
                            _ => 7,
                        };
                        stats_clone.batch_size_hist[bucket].fetch_add(1, Relaxed);
                        stats_clone
                            .last_batch_latency_us
                            .store(latency.as_micros() as u64, Relaxed);
                        stats_clone
                            .cumulative_latency_us
                            .fetch_add(latency.as_micros() as u64, Relaxed);
                        stats_clone
                            .cumulative_execute_batch_us
                            .fetch_add(timing.total_us, Relaxed);
                        stats_clone
                            .cumulative_batch_build_us
                            .fetch_add(timing.batch_build_us, Relaxed);
                        stats_clone
                            .cumulative_driver_fire_us
                            .fetch_add(timing.driver_fire_us, Relaxed);
                        stats_clone
                            .cumulative_response_dispatch_us
                            .fetch_add(timing.response_dispatch_us, Relaxed);
                        stats_clone
                            .cumulative_context_tick_submit_us
                            .fetch_add(tick_submit_us, Relaxed);
                        stats_clone
                            .system_spec_draft_tokens_proposed
                            .fetch_add(timing.system_spec_draft_tokens_proposed, Relaxed);
                        stats_clone
                            .system_spec_draft_tokens_accepted
                            .fetch_add(timing.system_spec_draft_tokens_accepted, Relaxed);
                        for (counter, value) in stats_clone
                            .system_spec_draft_tokens_proposed_per_pos
                            .iter()
                            .zip(timing.system_spec_draft_tokens_proposed_per_pos)
                        {
                            if value != 0 {
                                counter.fetch_add(value, Relaxed);
                            }
                        }
                        for (counter, value) in stats_clone
                            .system_spec_draft_tokens_accepted_per_pos
                            .iter()
                            .zip(timing.system_spec_draft_tokens_accepted_per_pos)
                        {
                            if value != 0 {
                                counter.fetch_add(value, Relaxed);
                            }
                        }
                        stats_clone
                            .cumulative_stats_update_us
                            .fetch_add(stats_update_start.elapsed().as_micros() as u64, Relaxed);
                    });
                }
                Decision::Wait(wait_duration) => {
                    crossbeam::channel::select! {
                        recv(req_rx) -> maybe_req => {
                            match maybe_req {
                                Ok(pending) => {
                                    let Some(pending) = prepare_pending_for_batch(&batch, pending)
                                    else {
                                        continue;
                                    };
                                    if batch.would_exceed(&pending) {
                                        if scheduler_trace_enabled() {
                                            let reason = batch
                                                .would_exceed_reason(&pending)
                                                .unwrap_or_else(|| "unknown".to_string());
                                            eprintln!(
                                                "[pie-sched-trace] driver={} stash current_requests={} current_tokens={} reason={}",
                                                driver_idx,
                                                batch.len(),
                                                batch.total_tokens(),
                                                reason,
                                            );
                                        }
                                        next_pending = Some(pending);
                                        continue;
                                    }
                                    policy.on_arrival();
                                    batch.push(pending);
                                }
                                Err(_) => break 'run_loop, // channel closed
                            }
                        }
                        recv(latency_rx) -> latency => {
                            if let Ok(l) = latency {
                                policy.on_complete(l);
                            }
                        }
                        default(wait_duration) => {}
                    }
                }
            }
        }

        // Shutdown: fire remaining batch on the shared runtime and let
        // it complete in the background. The scheduler thread itself is
        // exiting; we don't wait for the final fire to finish.
        if !batch.is_empty() {
            let requests = batch.take();
            let _ = rt_handle.spawn(Self::execute_batch(
                driver_idx,
                requests,
                driver_id,
                page_size,
                request_timeout,
                None,
                None,
                None,
                None,
            ));
        }
    }

    /// Execute a batch of forward pass requests via the driver service.
    async fn execute_batch(
        driver_idx: usize,
        requests: Vec<PendingRequest>,
        driver_id: DriverId,
        page_size: u32,
        _timeout: Duration,
        in_flight: Option<Arc<std::sync::atomic::AtomicBool>>,
        submit_tx: Option<crossbeam::channel::Sender<PendingRequest>>,
        stats_for_probe: Option<Arc<SchedulerStats>>,
        chain_pool: Option<Arc<super::speculator::ChainExtPool>>,
    ) -> BatchExecutionTiming {
        let batch_start = Instant::now();
        let build_start = Instant::now();
        // Detect if ANY request carries system spec drafts. The
        // common case (256-conc decode) has none, so we skip the
        // per-request Vec build + position-histogram loop.
        let any_spec = requests.iter().any(|req| !req.request.spec_token_ids.is_empty());
        let system_spec_proposed_per_req: Vec<usize> = if any_spec {
            requests
                .iter()
                .map(|req| req.request.spec_token_ids.len())
                .collect()
        } else {
            Vec::new()
        };
        let system_spec_draft_tokens_proposed =
            system_spec_proposed_per_req.iter().sum::<usize>() as u64;
        let mut system_spec_draft_tokens_accepted = 0u64;
        let mut system_spec_draft_tokens_proposed_per_pos =
            [0u64; SYSTEM_SPEC_DRAFT_POS_BUCKETS];
        let mut system_spec_draft_tokens_accepted_per_pos =
            [0u64; SYSTEM_SPEC_DRAFT_POS_BUCKETS];
        if any_spec {
            for proposed in &system_spec_proposed_per_req {
                for pos in 0..(*proposed).min(SYSTEM_SPEC_DRAFT_POS_BUCKETS) {
                    system_spec_draft_tokens_proposed_per_pos[pos] += 1;
                }
            }
        }

        // Build batched request — a single `pie_bridge::ForwardRequest`
        // populated by folding each per-request shape into the batch.
        let elide_decode_masks = requests.iter().all(|req| {
            req.request.single_token_mode
                && !req.request.has_user_mask
                && req.request.token_ids.len() <= 1
                && req.request.spec_token_ids.is_empty()
        });
        let mut batch_req = request::new_batched_forward_request_with_capacity(requests.len());
        for req in &requests {
            request::append_request_with_options(
                &mut batch_req,
                &req.request,
                &req.physical_page_ids,
                req.last_page_len,
                page_size,
                elide_decode_masks,
            );
        }
        let batch_build_us = build_start.elapsed().as_micros() as u64;

        // Send via driver service (typed call handles serialization + timeout)
        let driver_fire_start = Instant::now();
        let result = driver::fire_batch(driver_idx, batch_req).await;
        let driver_fire_us = driver_fire_start.elapsed().as_micros() as u64;
        // Release the in-flight gate so the main scheduler loop can fire
        // the next batch immediately (mirrors prior tokio Semaphore
        // permit drop). Mid-execute_batch, before response_dispatch.
        if let Some(ref flag) = in_flight {
            flag.store(false, std::sync::atomic::Ordering::Release);
        }

        let response_start = Instant::now();
        match result {
            Ok(batch_resp) => {
                let n_results = batch_resp.num_requests as usize;
                if n_results != requests.len() {
                    let msg = format!(
                        "batch response count mismatch from driver {driver_id}: \
                         expected {}, got {n_results}",
                        requests.len()
                    );
                    tracing::error!(
                        driver = driver_id,
                        expected = requests.len(),
                        got = n_results,
                        "Batch response count mismatch",
                    );
                    for req in requests {
                        req.send_result::<ForwardOutput>(
                            Err(anyhow::anyhow!(msg.clone())),
                            None,
                            page_size,
                        );
                    }
                } else {
                    let has_chunked = requests
                        .iter()
                        .any(|req| matches!(req.completion, Completion::Chunk { .. }));
                    let token_payload_only = !has_chunked
                        && batch_resp.dists_ids.is_empty()
                        && batch_resp.dists_probs.is_empty()
                        && batch_resp.logits_bytes.is_empty()
                        && batch_resp.logprobs_values.is_empty()
                        && batch_resp.entropies.is_empty()
                        && batch_resp.spec_tokens.is_empty()
                        && batch_resp.tokens_indptr.len() >= requests.len() + 1;

                    // Send oneshot replies first, defer drop of the
                    // request husks. Each PendingRequest's drop is
                    // ~3-4 µs (22-Vec ForwardRequest), and doing it
                    // inline pushes the 256th chain extender's wake
                    // out by ~1.2 ms — directly extending the gap.
                    let mut deferred_drop: Vec<(
                        pie_bridge::ForwardRequest,
                        Vec<PhysicalPageId>,
                    )> = Vec::with_capacity(n_results);
                    if token_payload_only {
                        for (r, req) in requests.into_iter().enumerate() {
                            let lo = batch_resp.tokens_indptr[r] as usize;
                            let hi = batch_resp.tokens_indptr[r + 1] as usize;
                            if system_spec_proposed_per_req
                                .get(r)
                                .copied()
                                .unwrap_or_default()
                                > 0
                            {
                                let accepted = hi.saturating_sub(lo).saturating_sub(1);
                                system_spec_draft_tokens_accepted += accepted as u64;
                                for pos in 0..accepted.min(SYSTEM_SPEC_DRAFT_POS_BUCKETS) {
                                    system_spec_draft_tokens_accepted_per_pos[pos] += 1;
                                }
                            }
                            let output = if hi == lo + 1 {
                                ForwardOutput::Token(batch_resp.tokens[lo])
                            } else {
                                ForwardOutput::Tokens(batch_resp.tokens[lo..hi].to_vec())
                            };
                            let PendingRequest {
                                request,
                                completion,
                                physical_page_ids,
                                last_page_len: _,
                            } = req;
                            match completion {
                                Completion::Direct(tx) => {
                                    tx.send(Ok(output)).ok();
                                    deferred_drop.push((request, physical_page_ids));
                                }
                                Completion::Chunk { .. } => {
                                    let req = PendingRequest {
                                        request,
                                        completion,
                                        physical_page_ids,
                                        last_page_len: 0,
                                    };
                                    req.send_result(Ok(output), submit_tx.as_ref(), page_size);
                                }
                                Completion::Chain { state } => {
                                    if let Some(pool) = chain_pool.as_ref() {
                                        let ctx_id = state.prev_request.context_ids
                                            .first().copied().unwrap_or(0);
                                        pool.submit(ctx_id, super::speculator::ChainExtJob {
                                            state,
                                            output: Ok(output),
                                            enqueued_us: super::speculator::now_micros(),
                                        });
                                    }
                                    deferred_drop.push((request, physical_page_ids));
                                }
                            }
                        }
                    } else {
                        for (r, req) in requests.into_iter().enumerate() {
                            let per_req = request::extract_per_request(&batch_resp, r);
                            if system_spec_proposed_per_req
                                .get(r)
                                .copied()
                                .unwrap_or_default()
                                > 0
                            {
                                let accepted = per_req.tokens.len().saturating_sub(1);
                                system_spec_draft_tokens_accepted += accepted as u64;
                                for pos in 0..accepted.min(SYSTEM_SPEC_DRAFT_POS_BUCKETS) {
                                    system_spec_draft_tokens_accepted_per_pos[pos] += 1;
                                }
                            }
                            let output = ForwardOutput::Response(per_req);
                            let PendingRequest {
                                request,
                                completion,
                                physical_page_ids,
                                last_page_len: _,
                            } = req;
                            match completion {
                                Completion::Direct(tx) => {
                                    tx.send(Ok(output)).ok();
                                    deferred_drop.push((request, physical_page_ids));
                                }
                                Completion::Chunk { .. } => {
                                    let req = PendingRequest {
                                        request,
                                        completion,
                                        physical_page_ids,
                                        last_page_len: 0,
                                    };
                                    req.send_result(Ok(output), submit_tx.as_ref(), page_size);
                                }
                                Completion::Chain { state } => {
                                    if let Some(pool) = chain_pool.as_ref() {
                                        let ctx_id = state.prev_request.context_ids
                                            .first().copied().unwrap_or(0);
                                        pool.submit(ctx_id, super::speculator::ChainExtJob {
                                            state,
                                            output: Ok(output),
                                            enqueued_us: super::speculator::now_micros(),
                                        });
                                    }
                                    deferred_drop.push((request, physical_page_ids));
                                }
                            }
                        }
                    }
                    if let Some(s) = stats_for_probe.as_ref() {
                        s.last_dispatch_end_micros.store(now_micros(), Relaxed);
                    }
                    if !deferred_drop.is_empty() {
                        // Dedicated blocking pool so the chain-extender
                        // wake-up wave doesn't compete with this dealloc
                        // task for a worker thread.
                        tokio::task::spawn_blocking(move || drop(deferred_drop));
                    }
                }
            }
            Err(e) => {
                tracing::error!("fire_batch failed for driver {}: {:?}", driver_id, e);
                for req in requests {
                    req.send_result::<ForwardOutput>(
                        Err(anyhow::anyhow!(
                            "fire_batch failed for driver {driver_id}: {e:#}"
                        )),
                        None,
                        page_size,
                    );
                }
            }
        }
        BatchExecutionTiming {
            total_us: batch_start.elapsed().as_micros() as u64,
            batch_build_us,
            driver_fire_us,
            response_dispatch_us: response_start.elapsed().as_micros() as u64,
            system_spec_draft_tokens_proposed,
            system_spec_draft_tokens_accepted,
            system_spec_draft_tokens_proposed_per_pos,
            system_spec_draft_tokens_accepted_per_pos,
        }
    }
}
