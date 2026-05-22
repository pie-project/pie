//! Per-driver batch scheduler.
//!
//! Each `BatchScheduler` owns its own RPC client, scheduling policy,
//! and tokio task. It accepts pre-translated forward pass requests,
//! accumulates them into batches, and fires them based on adaptive
//! scheduling decisions.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use anyhow::Result;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc, oneshot};

use crate::context::pagestore::PhysicalPageId;
use crate::driver::{self, DriverId, SchedulerLimits};

use super::adaptive_policy::{AdaptivePolicy, EagerPolicy, GreedyPolicy};
use super::{ForwardOutput, request};

mod chunked;

use chunked::ChunkContinuation;

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
    pub last_request_queue_us: AtomicU64,
    pub cumulative_request_queue_us: AtomicU64,
    pub last_batch_queue_us: AtomicU64,
    pub cumulative_batch_queue_us: AtomicU64,
    pub last_permit_wait_us: AtomicU64,
    pub cumulative_permit_wait_us: AtomicU64,
    pub last_batch_build_us: AtomicU64,
    pub cumulative_batch_build_us: AtomicU64,
    pub last_driver_forward_us: AtomicU64,
    pub cumulative_driver_forward_us: AtomicU64,
    pub last_response_fanout_us: AtomicU64,
    pub cumulative_response_fanout_us: AtomicU64,
    pub last_response_classify_us: AtomicU64,
    pub cumulative_response_classify_us: AtomicU64,
    pub last_response_token_output_build_us: AtomicU64,
    pub cumulative_response_token_output_build_us: AtomicU64,
    pub last_response_direct_send_us: AtomicU64,
    pub cumulative_response_direct_send_us: AtomicU64,
    pub last_response_chunk_send_us: AtomicU64,
    pub cumulative_response_chunk_send_us: AtomicU64,
    pub last_response_extract_us: AtomicU64,
    pub cumulative_response_extract_us: AtomicU64,
    pub last_response_error_us: AtomicU64,
    pub cumulative_response_error_us: AtomicU64,
}

// =============================================================================
// PendingRequest
// =============================================================================

/// A forward pass request bundled with its response channel and physical pages.
struct PendingRequest {
    request: pie_bridge::ForwardRequest,
    completion: Completion,
    physical_page_ids: PhysicalPageSpan,
    last_page_len: u32,
    enqueued_at: Instant,
}

#[derive(Clone, Debug)]
struct PhysicalPageSpan {
    pages: Arc<[PhysicalPageId]>,
    len: usize,
}

impl PhysicalPageSpan {
    fn from_vec(pages: Vec<PhysicalPageId>) -> Self {
        let len = pages.len();
        Self {
            pages: Arc::from(pages),
            len,
        }
    }

    pub(crate) fn from_shared_prefix(pages: Arc<[PhysicalPageId]>, len: usize) -> Self {
        debug_assert!(len <= pages.len());
        Self { pages, len }
    }

    #[inline]
    fn as_slice(&self) -> &[PhysicalPageId] {
        &self.pages[..self.len]
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }
}

impl std::ops::Deref for PhysicalPageSpan {
    type Target = [PhysicalPageId];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl PartialEq<Vec<PhysicalPageId>> for PhysicalPageSpan {
    fn eq(&self, other: &Vec<PhysicalPageId>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

enum Completion {
    Direct(oneshot::Sender<Result<ForwardOutput>>),
    Chunk {
        continuation: ChunkContinuation,
        sampler_slots: Vec<usize>,
    },
}

type ForwardOutputSender = oneshot::Sender<Result<ForwardOutput>>;

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
            physical_page_ids: PhysicalPageSpan::from_vec(physical_page_ids),
            last_page_len,
            enqueued_at: Instant::now(),
        }
    }

    fn direct_shared(
        request: pie_bridge::ForwardRequest,
        response_tx: oneshot::Sender<Result<ForwardOutput>>,
        physical_page_ids: Arc<[PhysicalPageId]>,
        physical_page_len: usize,
        last_page_len: u32,
    ) -> Self {
        Self {
            request,
            completion: Completion::Direct(response_tx),
            physical_page_ids: PhysicalPageSpan::from_shared_prefix(
                physical_page_ids,
                physical_page_len,
            ),
            last_page_len,
            enqueued_at: Instant::now(),
        }
    }

    #[inline]
    fn context_id(&self) -> u64 {
        self.request.context_ids.first().copied().unwrap_or(0)
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct BatchExecutionTiming {
    batch_build_us: u64,
    driver_forward_us: u64,
    response_fanout_us: u64,
    response_classify_us: u64,
    response_token_output_build_us: u64,
    response_direct_send_us: u64,
    response_chunk_send_us: u64,
    response_extract_us: u64,
    response_error_us: u64,
}

fn duration_us(d: Duration) -> u64 {
    d.as_micros().min(u128::from(u64::MAX)) as u64
}

fn fanout_breakdown_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PIE_SCHED_FANOUT_BREAKDOWN").is_some())
}

fn preextract_direct_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("PIE_SCHED_PREEXTRACT_DIRECT")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(true)
    })
}

#[inline]
fn add_elapsed_us(dst: &mut u64, start: Instant) {
    *dst = dst.saturating_add(duration_us(start.elapsed()));
}

fn send_forward_output(
    req: PendingRequest,
    output: ForwardOutput,
    submit_tx: Option<&mpsc::WeakUnboundedSender<PendingRequest>>,
    page_size: u32,
    timing: &mut BatchExecutionTiming,
    breakdown: bool,
) {
    let PendingRequest {
        request,
        completion,
        physical_page_ids,
        last_page_len,
        enqueued_at,
    } = req;

    match completion {
        Completion::Direct(tx) => {
            let send_start = breakdown.then(Instant::now);
            tx.send(Ok(output)).ok();
            if let Some(start) = send_start {
                add_elapsed_us(&mut timing.response_direct_send_us, start);
            }
        }
        completion => {
            let req = PendingRequest {
                request,
                completion,
                physical_page_ids,
                last_page_len,
                enqueued_at,
            };
            let send_start = breakdown.then(Instant::now);
            req.send_result(Ok(output), submit_tx, page_size);
            if let Some(start) = send_start {
                add_elapsed_us(&mut timing.response_chunk_send_us, start);
            }
        }
    }
}

fn extract_direct_targets(requests: Vec<PendingRequest>) -> Vec<ForwardOutputSender> {
    let mut targets = Vec::with_capacity(requests.len());
    for req in requests {
        let PendingRequest { completion, .. } = req;
        if let Completion::Direct(tx) = completion {
            targets.push(tx);
        }
    }
    targets
}

fn send_direct_forward_output(
    tx: ForwardOutputSender,
    result: Result<ForwardOutput>,
    timing: &mut BatchExecutionTiming,
    breakdown: bool,
) {
    let send_start = breakdown.then(Instant::now);
    tx.send(result).ok();
    if let Some(start) = send_start {
        add_elapsed_us(&mut timing.response_direct_send_us, start);
    }
}

fn queue_timing_us(requests: &[PendingRequest], fire_at: Instant) -> (u64, u64) {
    let mut sum = 0u64;
    let mut max = 0u64;
    for req in requests {
        let q = duration_us(fire_at.saturating_duration_since(req.enqueued_at));
        sum = sum.saturating_add(q);
        max = max.max(q);
    }
    (sum, max)
}

fn reserve_batch_request(
    batch: &mut pie_bridge::ForwardRequest,
    requests: &[PendingRequest],
) -> bool {
    let mut token_ids = 0usize;
    let mut position_ids = 0usize;
    let mut kv_pages = 0usize;
    let mut rs_slots = 0usize;
    let mut masks_if_not_elided = 0usize;
    let mut logit_masks = 0usize;
    let mut sampling_indices = 0usize;
    let mut samplers = 0usize;
    let mut adapter_bindings = 0usize;
    let mut spec_tokens = 0usize;
    let mut spec_positions = 0usize;
    let mut output_spec_flags = 0usize;
    let mut context_ids = 0usize;
    let mut elide_decode_masks = true;

    for req in requests {
        let fwd = &req.request;
        token_ids += fwd.token_ids.len();
        position_ids += fwd.position_ids.len();
        kv_pages += req.physical_page_ids.len();
        rs_slots += fwd.rs_slot_ids.len();
        logit_masks += fwd.logit_masks.len();
        sampling_indices += fwd.sampling_indices.len();
        samplers += fwd.samplers.len();
        adapter_bindings += fwd.adapter_bindings.len();
        spec_tokens += fwd.spec_token_ids.len();
        spec_positions += fwd.spec_position_ids.len();
        output_spec_flags += fwd.output_spec_flags.len();
        context_ids += fwd.context_ids.len();

        elide_decode_masks &= fwd.single_token_mode
            && !fwd.has_user_mask
            && fwd.token_ids.len() <= 1
            && fwd.spec_token_ids.is_empty();
        masks_if_not_elided += if fwd.masks.is_empty() && !fwd.position_ids.is_empty() {
            fwd.position_ids.len()
        } else {
            fwd.masks.len()
        };
    }

    let request_count = requests.len();
    batch.token_ids.reserve(token_ids);
    batch.position_ids.reserve(position_ids);
    batch.kv_page_indices.reserve(kv_pages);
    batch.kv_page_indptr.reserve(request_count);
    batch.kv_last_page_lens.reserve(request_count);
    batch.qo_indptr.reserve(request_count);
    batch.rs_slot_ids.reserve(rs_slots);
    batch.rs_slot_flags.reserve(rs_slots);
    if !elide_decode_masks {
        batch.masks.reserve(masks_if_not_elided);
    }
    batch.mask_indptr.reserve(request_count);
    batch.logit_masks.reserve(logit_masks);
    batch.logit_mask_indptr.reserve(request_count);
    batch.sampling_indices.reserve(sampling_indices);
    batch.sampling_indptr.reserve(request_count);
    batch.samplers.reserve(samplers);
    batch.sampler_indptr.reserve(request_count);
    batch.adapter_bindings.reserve(adapter_bindings);
    batch.spec_token_ids.reserve(spec_tokens);
    batch.spec_position_ids.reserve(spec_positions);
    batch.spec_indptr.reserve(request_count);
    batch.output_spec_flags.reserve(output_spec_flags);
    batch.context_ids.reserve(context_ids);
    elide_decode_masks
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
    let has_output_spec = req.request.output_spec_flags.iter().any(|&enabled| enabled);
    let has_dense_logit_requirement = req.request.has_user_mask
        || !req.request.logit_masks.is_empty()
        || spec_tokens > 0
        || has_output_spec
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
        .map(request_logprob_labels_for_sampler)
        .sum()
}

fn request_logprob_labels_for_sampler(sampler: &pie_bridge::Sampler) -> usize {
    match sampler {
        pie_bridge::Sampler::Logprob { .. } => 1,
        pie_bridge::Sampler::Logprobs { token_ids } => token_ids.len(),
        _ => 0,
    }
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
        let mut usage = request_capacity_usage(&req, self.page_size);
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
        let next_has_spec = self.has_spec_drafts || usage.has_spec_drafts;
        let next_custom_mask_bytes = if next_has_spec {
            self.total_spec_custom_mask_bytes
                .saturating_add(usage.spec_custom_mask_bytes)
        } else {
            self.total_user_custom_mask_bytes
                .saturating_add(usage.user_custom_mask_bytes)
        };
        let (next_logit_rows, next_prob_rows, _, _, _) = self.projected_rows(Some(&usage));
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

    fn is_full(&self) -> bool {
        let active_custom_mask_bytes = if self.has_spec_drafts {
            self.total_spec_custom_mask_bytes
        } else {
            self.total_user_custom_mask_bytes
        };
        self.requests.len() >= self.limits.max_forward_requests
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
    Some(pending)
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
#[derive(Clone)]
pub(crate) struct SchedulerHandle {
    tx: mpsc::UnboundedSender<PendingRequest>,
}

impl SchedulerHandle {
    pub fn submit(
        &self,
        request: pie_bridge::ForwardRequest,
        response_tx: oneshot::Sender<Result<ForwardOutput>>,
        physical_page_ids: Vec<PhysicalPageId>,
        last_page_len: u32,
    ) -> Result<()> {
        self.tx.send(PendingRequest::direct(
            request,
            response_tx,
            physical_page_ids,
            last_page_len,
        ))?;
        Ok(())
    }

    pub(crate) fn submit_shared(
        &self,
        request: pie_bridge::ForwardRequest,
        response_tx: oneshot::Sender<Result<ForwardOutput>>,
        physical_page_ids: Arc<[PhysicalPageId]>,
        physical_page_len: usize,
        last_page_len: u32,
    ) -> Result<()> {
        self.tx.send(PendingRequest::direct_shared(
            request,
            response_tx,
            physical_page_ids,
            physical_page_len,
            last_page_len,
        ))?;
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
    tx: mpsc::UnboundedSender<PendingRequest>,
    stats: Arc<SchedulerStats>,
}

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
        let (tx, rx) = mpsc::unbounded_channel();
        let submit_tx = tx.downgrade();
        let stats = Arc::new(SchedulerStats::default());
        tokio::spawn(Self::run(
            driver_id,
            driver_idx,
            rx,
            submit_tx,
            page_size,
            limits,
            request_timeout_secs,
            batch_policy,
            stats.clone(),
        ));

        Self { tx, stats }
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
        self.tx.send(PendingRequest::direct(
            request,
            response_tx,
            physical_page_ids,
            last_page_len,
        ))?;
        Ok(())
    }

    pub(crate) fn submit_shared(
        &self,
        request: pie_bridge::ForwardRequest,
        response_tx: oneshot::Sender<Result<ForwardOutput>>,
        physical_page_ids: Arc<[PhysicalPageId]>,
        physical_page_len: usize,
        last_page_len: u32,
    ) -> Result<()> {
        self.tx.send(PendingRequest::direct_shared(
            request,
            response_tx,
            physical_page_ids,
            physical_page_len,
            last_page_len,
        ))?;
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

    /// Main scheduling loop for a single driver.
    async fn run(
        driver_id: DriverId,
        driver_idx: usize,
        mut req_rx: mpsc::UnboundedReceiver<PendingRequest>,
        submit_tx: mpsc::WeakUnboundedSender<PendingRequest>,
        page_size: u32,
        limits: SchedulerLimits,
        request_timeout_secs: u64,
        batch_policy: String,
        stats: Arc<SchedulerStats>,
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
        let in_flight = Arc::new(Semaphore::new(1));

        // Channel for batch completion latency feedback to the policy.
        let (latency_tx, mut latency_rx) = mpsc::unbounded_channel::<Duration>();
        let mut next_pending: Option<PendingRequest> = None;

        'run_loop: loop {
            // Drain completed batch latencies (non-blocking)
            while let Ok(latency) = latency_rx.try_recv() {
                policy.on_complete(latency);
            }

            // Wait for first request if batch is empty
            while batch.is_empty() {
                let pending = if let Some(pending) = next_pending.take() {
                    pending
                } else {
                    let Some(pending) = req_rx.recv().await else {
                        break 'run_loop;
                    };
                    pending
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
            while next_pending.is_none() {
                let Ok(pending) = req_rx.try_recv() else {
                    break;
                };
                let Some(pending) = prepare_pending_for_batch(&batch, pending) else {
                    continue;
                };
                if batch.would_exceed(&pending) {
                    next_pending = Some(pending);
                    break;
                }
                policy.on_arrival();
                batch.push(pending);
                if batch.is_full() {
                    break;
                }
            }

            // Ask the policy what to do
            let decision = if next_pending.is_some() {
                Decision::Fire
            } else {
                policy.decide(batch.len(), batch.should_prefill_coalesce())
            };
            match decision {
                Decision::Fire => {
                    // Acquire a permit (may wait if at in-flight limit).
                    let permit_wait_start = Instant::now();
                    let permit = in_flight
                        .clone()
                        .acquire_owned()
                        .await
                        .expect("semaphore closed");
                    let permit_wait_us = duration_us(permit_wait_start.elapsed());

                    // The policy may decide to fire while the previous GPU
                    // batch is still in flight. Do one last non-blocking
                    // drain after the permit opens so requests that arrived
                    // during that wait are coalesced into this batch instead
                    // of being stranded behind a tiny stale fire.
                    while next_pending.is_none() && !batch.is_full() {
                        let Ok(pending) = req_rx.try_recv() else {
                            break;
                        };
                        if let Some(msg) = batch.single_request_limit_error(&pending) {
                            pending.send_error(msg);
                            continue;
                        }
                        if batch.would_exceed(&pending) {
                            next_pending = Some(pending);
                            break;
                        }
                        policy.on_arrival();
                        batch.push(pending);
                    }

                    let total_tokens = batch.total_tokens();
                    let fire_at = Instant::now();
                    let requests_to_fire = batch.take();
                    let (request_queue_us, batch_queue_us) =
                        queue_timing_us(&requests_to_fire, fire_at);
                    policy.on_fired(requests_to_fire.len());
                    if std::env::var_os("PIE_SCHED_TRACE").is_some() {
                        eprintln!(
                            "[sched-batch driver={driver_idx} requests={} tokens={total_tokens}]",
                            requests_to_fire.len()
                        );
                    }

                    // Collect batch context IDs for accurate rent charging.
                    // Per-request shape stores the single context_id in
                    // `context_ids[0]`.
                    let batch_ctx_ids: Vec<u64> = requests_to_fire
                        .iter()
                        .map(PendingRequest::context_id)
                        .collect();
                    let batch_size = batch_ctx_ids.len() as u64;

                    // Spawn batch execution
                    let latency_tx_clone = latency_tx.clone();
                    let stats_clone = stats.clone();
                    let timeout = request_timeout;
                    let submit_tx_clone = submit_tx.clone();

                    tokio::spawn(async move {
                        let start = Instant::now();
                        let execution = Self::execute_batch(
                            driver_idx,
                            requests_to_fire,
                            driver_id,
                            page_size,
                            timeout,
                            Some(permit),
                            Some(submit_tx_clone),
                        )
                        .await;
                        let latency = start.elapsed();
                        latency_tx_clone.send(latency).ok();

                        // Advance market clock for this driver: prices, rent, dividends.
                        // Pass batch context IDs so tick only charges contexts
                        // that were in this batch (not stale pinned contexts).
                        crate::context::tick(driver_idx, latency.as_secs_f64(), batch_ctx_ids);

                        // Update cumulative atomic counters (consumed by external
                        // monitoring; ignored by the policy).
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
                        let denom = batch_size.max(1);
                        stats_clone
                            .last_request_queue_us
                            .store(request_queue_us / denom, Relaxed);
                        stats_clone
                            .cumulative_request_queue_us
                            .fetch_add(request_queue_us, Relaxed);
                        stats_clone
                            .last_batch_queue_us
                            .store(batch_queue_us, Relaxed);
                        stats_clone
                            .cumulative_batch_queue_us
                            .fetch_add(batch_queue_us, Relaxed);
                        stats_clone
                            .last_permit_wait_us
                            .store(permit_wait_us, Relaxed);
                        stats_clone
                            .cumulative_permit_wait_us
                            .fetch_add(permit_wait_us, Relaxed);
                        stats_clone
                            .last_batch_build_us
                            .store(execution.batch_build_us, Relaxed);
                        stats_clone
                            .cumulative_batch_build_us
                            .fetch_add(execution.batch_build_us, Relaxed);
                        stats_clone
                            .last_driver_forward_us
                            .store(execution.driver_forward_us, Relaxed);
                        stats_clone
                            .cumulative_driver_forward_us
                            .fetch_add(execution.driver_forward_us, Relaxed);
                        stats_clone
                            .last_response_fanout_us
                            .store(execution.response_fanout_us, Relaxed);
                        stats_clone
                            .cumulative_response_fanout_us
                            .fetch_add(execution.response_fanout_us, Relaxed);
                        stats_clone
                            .last_response_classify_us
                            .store(execution.response_classify_us, Relaxed);
                        stats_clone
                            .cumulative_response_classify_us
                            .fetch_add(execution.response_classify_us, Relaxed);
                        stats_clone
                            .last_response_token_output_build_us
                            .store(execution.response_token_output_build_us, Relaxed);
                        stats_clone
                            .cumulative_response_token_output_build_us
                            .fetch_add(execution.response_token_output_build_us, Relaxed);
                        stats_clone
                            .last_response_direct_send_us
                            .store(execution.response_direct_send_us, Relaxed);
                        stats_clone
                            .cumulative_response_direct_send_us
                            .fetch_add(execution.response_direct_send_us, Relaxed);
                        stats_clone
                            .last_response_chunk_send_us
                            .store(execution.response_chunk_send_us, Relaxed);
                        stats_clone
                            .cumulative_response_chunk_send_us
                            .fetch_add(execution.response_chunk_send_us, Relaxed);
                        stats_clone
                            .last_response_extract_us
                            .store(execution.response_extract_us, Relaxed);
                        stats_clone
                            .cumulative_response_extract_us
                            .fetch_add(execution.response_extract_us, Relaxed);
                        stats_clone
                            .last_response_error_us
                            .store(execution.response_error_us, Relaxed);
                        stats_clone
                            .cumulative_response_error_us
                            .fetch_add(execution.response_error_us, Relaxed);
                    });
                }
                Decision::Wait(wait_duration) => {
                    tokio::select! {
                        _ = tokio::time::sleep(wait_duration) => {},
                        maybe_req = req_rx.recv() => {
                            if let Some(pending) = maybe_req {
                                let Some(pending) = prepare_pending_for_batch(&batch, pending) else {
                                    continue;
                                };
                                if batch.would_exceed(&pending) {
                                    next_pending = Some(pending);
                                    continue;
                                }
                                policy.on_arrival();
                                batch.push(pending);
                            } else {
                                break; // channel closed
                            }
                        },
                        latency = latency_rx.recv() => {
                            if let Some(l) = latency {
                                policy.on_complete(l);
                            }
                        },
                    }
                }
            }
        }

        // Shutdown: fire remaining batch
        if !batch.is_empty() {
            let requests = batch.take();
            let _ = Self::execute_batch(
                driver_idx,
                requests,
                driver_id,
                page_size,
                request_timeout,
                None,
                None,
            )
            .await;
        }
    }

    /// Execute a batch of forward pass requests via the driver service.
    async fn execute_batch(
        driver_idx: usize,
        requests: Vec<PendingRequest>,
        driver_id: DriverId,
        page_size: u32,
        _timeout: Duration,
        mut permit: Option<OwnedSemaphorePermit>,
        submit_tx: Option<mpsc::WeakUnboundedSender<PendingRequest>>,
    ) -> BatchExecutionTiming {
        let mut timing = BatchExecutionTiming::default();
        let request_count = requests.len();
        let build_start = Instant::now();
        // Build batched request — a single `pie_bridge::ForwardRequest`
        // populated by folding each per-request shape into the batch.
        let mut batch_req = request::new_batched_forward_request();
        let elide_decode_masks = reserve_batch_request(&mut batch_req, &requests);
        for req in &requests {
            request::append_request_with_options(
                &mut batch_req,
                &req.request,
                req.physical_page_ids.as_slice(),
                req.last_page_len,
                page_size,
                elide_decode_masks,
            );
        }
        timing.batch_build_us = duration_us(build_start.elapsed());

        let mut requests = Some(requests);
        let direct_targets_task = if preextract_direct_enabled()
            && requests.as_ref().is_some_and(|requests| {
                requests
                    .iter()
                    .all(|req| matches!(req.completion, Completion::Direct(_)))
            }) {
            let requests = requests.take().expect("requests must be present");
            Some(tokio::task::spawn_blocking(move || {
                extract_direct_targets(requests)
            }))
        } else {
            None
        };

        // Send via driver service (typed call handles serialization + timeout)
        let driver_start = Instant::now();
        let result = driver::fire_batch(driver_idx, batch_req).await;
        timing.driver_forward_us = duration_us(driver_start.elapsed());
        drop(permit.take());

        let response_start = Instant::now();
        let breakdown = fanout_breakdown_enabled();
        let mut direct_targets = if let Some(task) = direct_targets_task {
            match task.await {
                Ok(targets) if targets.len() == request_count => Some(targets),
                Ok(targets) => {
                    tracing::error!(
                        driver = driver_id,
                        expected = request_count,
                        got = targets.len(),
                        "Pre-extracted direct response target count mismatch",
                    );
                    Some(targets)
                }
                Err(e) => {
                    tracing::error!(
                        driver = driver_id,
                        error = %e,
                        "Direct response target pre-extraction task failed",
                    );
                    Some(Vec::new())
                }
            }
        } else {
            None
        };

        match result {
            Ok(batch_resp) => {
                let n_results = batch_resp.num_requests as usize;
                if n_results != request_count {
                    let msg = format!(
                        "batch response count mismatch from driver {driver_id}: \
                         expected {}, got {n_results}",
                        request_count
                    );
                    tracing::error!(
                        driver = driver_id,
                        expected = request_count,
                        got = n_results,
                        "Batch response count mismatch",
                    );
                    let error_start = breakdown.then(Instant::now);
                    if let Some(targets) = direct_targets.take() {
                        for tx in targets {
                            send_direct_forward_output(
                                tx,
                                Err(anyhow::anyhow!(msg.clone())),
                                &mut timing,
                                breakdown,
                            );
                        }
                    } else {
                        for req in requests.take().expect("requests must be present") {
                            req.send_result::<ForwardOutput>(
                                Err(anyhow::anyhow!(msg.clone())),
                                None,
                                page_size,
                            );
                        }
                    }
                    if let Some(start) = error_start {
                        add_elapsed_us(&mut timing.response_error_us, start);
                    }
                    timing.response_fanout_us = duration_us(response_start.elapsed());
                    return timing;
                }

                let classify_start = breakdown.then(Instant::now);
                let has_chunked = direct_targets.is_none()
                    && requests
                        .as_ref()
                        .expect("requests must be present")
                        .iter()
                        .any(|req| matches!(req.completion, Completion::Chunk { .. }));
                let token_payload_only = !has_chunked
                    && batch_resp.dists_ids.is_empty()
                    && batch_resp.dists_probs.is_empty()
                    && batch_resp.logits_bytes.is_empty()
                    && batch_resp.logprobs_values.is_empty()
                    && batch_resp.entropies.is_empty()
                    && batch_resp.tokens_indptr.len() > request_count;
                if let Some(start) = classify_start {
                    add_elapsed_us(&mut timing.response_classify_us, start);
                }

                if token_payload_only {
                    if let Some(targets) = direct_targets.take() {
                        for (r, tx) in targets.into_iter().enumerate() {
                            let build_start = Instant::now();
                            let lo = batch_resp.tokens_indptr[r] as usize;
                            let hi = batch_resp.tokens_indptr[r + 1] as usize;
                            let output = if hi == lo + 1 {
                                ForwardOutput::Token(batch_resp.tokens[lo])
                            } else {
                                ForwardOutput::Tokens(batch_resp.tokens[lo..hi].to_vec())
                            };
                            if breakdown {
                                add_elapsed_us(
                                    &mut timing.response_token_output_build_us,
                                    build_start,
                                );
                            }
                            send_direct_forward_output(tx, Ok(output), &mut timing, breakdown);
                        }
                    } else if breakdown {
                        for (r, req) in requests
                            .take()
                            .expect("requests must be present")
                            .into_iter()
                            .enumerate()
                        {
                            let build_start = Instant::now();
                            let lo = batch_resp.tokens_indptr[r] as usize;
                            let hi = batch_resp.tokens_indptr[r + 1] as usize;
                            let output = if hi == lo + 1 {
                                ForwardOutput::Token(batch_resp.tokens[lo])
                            } else {
                                ForwardOutput::Tokens(batch_resp.tokens[lo..hi].to_vec())
                            };
                            add_elapsed_us(&mut timing.response_token_output_build_us, build_start);
                            send_forward_output(
                                req,
                                output,
                                submit_tx.as_ref(),
                                page_size,
                                &mut timing,
                                true,
                            );
                        }
                    } else {
                        for (r, req) in requests
                            .take()
                            .expect("requests must be present")
                            .into_iter()
                            .enumerate()
                        {
                            let lo = batch_resp.tokens_indptr[r] as usize;
                            let hi = batch_resp.tokens_indptr[r + 1] as usize;
                            let output = if hi == lo + 1 {
                                ForwardOutput::Token(batch_resp.tokens[lo])
                            } else {
                                ForwardOutput::Tokens(batch_resp.tokens[lo..hi].to_vec())
                            };
                            send_forward_output(
                                req,
                                output,
                                submit_tx.as_ref(),
                                page_size,
                                &mut timing,
                                false,
                            );
                        }
                    }
                } else {
                    if let Some(targets) = direct_targets.take() {
                        for (r, tx) in targets.into_iter().enumerate() {
                            let extract_start = breakdown.then(Instant::now);
                            let per_req = request::extract_per_request(&batch_resp, r);
                            if let Some(start) = extract_start {
                                add_elapsed_us(&mut timing.response_extract_us, start);
                            }
                            send_direct_forward_output(
                                tx,
                                Ok(ForwardOutput::Response(per_req)),
                                &mut timing,
                                breakdown,
                            );
                        }
                    } else if breakdown {
                        for (r, req) in requests
                            .take()
                            .expect("requests must be present")
                            .into_iter()
                            .enumerate()
                        {
                            // Extract this request's slice from the batched
                            // response. The api layer (build_wit_output)
                            // walks samplers + the single-request response
                            // to construct the WIT Output.
                            let extract_start = Instant::now();
                            let per_req = request::extract_per_request(&batch_resp, r);
                            add_elapsed_us(&mut timing.response_extract_us, extract_start);
                            send_forward_output(
                                req,
                                ForwardOutput::Response(per_req),
                                submit_tx.as_ref(),
                                page_size,
                                &mut timing,
                                true,
                            );
                        }
                    } else {
                        for (r, req) in requests
                            .take()
                            .expect("requests must be present")
                            .into_iter()
                            .enumerate()
                        {
                            // Extract this request's slice from the batched
                            // response. The api layer (build_wit_output)
                            // walks samplers + the single-request response
                            // to construct the WIT Output.
                            let per_req = request::extract_per_request(&batch_resp, r);
                            req.send_result(
                                Ok(ForwardOutput::Response(per_req)),
                                submit_tx.as_ref(),
                                page_size,
                            );
                        }
                    }
                }
            }
            Err(e) => {
                tracing::error!("fire_batch failed for driver {}: {:?}", driver_id, e);
                let error_start = breakdown.then(Instant::now);
                let msg = format!("fire_batch failed for driver {driver_id}: {e:#}");
                if let Some(targets) = direct_targets.take() {
                    for tx in targets {
                        send_direct_forward_output(
                            tx,
                            Err(anyhow::anyhow!(msg.clone())),
                            &mut timing,
                            breakdown,
                        );
                    }
                } else {
                    for req in requests.take().expect("requests must be present") {
                        req.send_result::<ForwardOutput>(
                            Err(anyhow::anyhow!(msg.clone())),
                            None,
                            page_size,
                        );
                    }
                }
                if let Some(start) = error_start {
                    add_elapsed_us(&mut timing.response_error_us, start);
                }
            }
        }
        timing.response_fanout_us = duration_us(response_start.elapsed());
        timing
    }
}
