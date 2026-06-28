//! Per-driver batch scheduler.
//!
//! Each `BatchScheduler` owns its own RPC client, scheduling policy,
//! and tokio task. It accepts pre-translated forward pass requests,
//! accumulates them into batches, and fires them greedily (one policy
//! under FCFS).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use anyhow::Result;
use tokio::sync::oneshot;

use crate::arena::PhysicalPageId;
use crate::driver::{self, DriverId, SchedulerLimits};

use super::adaptive_policy::RunAheadPolicy;
use super::{ForwardOutput, request};

mod chunked;

use chunked::ChunkContinuation;

fn scheduler_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PIE_SCHED_TRACE").is_some())
}

/// Test-only deterministic batch-accumulation hold (µs). When set, after the
/// first request the scheduler blocks up to this long for more requests to
/// arrive before firing, so concurrent requests reliably co-batch into one fire
/// (a deterministic `forward_R >= 2` for the merged-path verify). Default unset
/// → today's fire-on-arrival, zero production impact. This is the test-lever
/// ancestor of #10's production accumulation-window admission policy.
fn scheduler_accum_hold_us() -> Option<u64> {
    static HOLD: OnceLock<Option<u64>> = OnceLock::new();
    *HOLD.get_or_init(|| {
        std::env::var("PIE_SCHED_ACCUM_HOLD_US")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|&us| us > 0)
    })
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
    /// A request was added to the accumulator. `program_identity_hashes` are the
    /// per-program `program_identity_hash`es carried by this request (one per
    /// program in its sampler pass; empty for plain decode) — run-ahead policies
    /// union them into the window's distinct-program set to drive the #10
    /// dedup-aware accumulation. Other policies ignore them.
    fn on_arrival(&mut self, program_identity_hashes: &[u64]);

    /// A batch finished executing. `latency` is the wall-clock time
    /// the forward pass took on the driver.
    fn on_complete(&mut self, latency: Duration);

    /// The current batch was fired. `fired_size` is the number of
    /// requests in the batch — policies use it to learn the steady-
    /// state cohort size and avoid firing partial batches in the next
    /// cycle.
    fn on_fired(&mut self, fired_size: usize);

    /// The current batch was submitted (enqueued). `submission_latency` is the
    /// host-side enqueue duration; run-ahead policies EWMA it into the
    /// `lead_time` (how far ahead of an in-flight batch's completion to fire so
    /// the next enqueue lands just-in-time). Default no-op for non-run-ahead
    /// policies, whose fire is synchronous (no separate submission phase).
    fn on_submitted(&mut self, _submission_latency: Duration) {}

    /// Decide whether to fire or wait, given the current batch size.
    /// `&mut self` so policies can update internal state on every poll.
    fn decide(&mut self, current_batch_size: usize) -> Decision;

    /// The number of DISTINCT programs (`program_identity_hash`) accumulated in
    /// the current not-yet-fired window — the #10 distinct-count witness (read at
    /// the fire trace so the verify can assert dedup: N-same-grammar ⇒ 1, and the
    /// distinct-burst cap: N-distinct ⇒ N). Default `0` for policies that don't
    /// track it; the run-ahead policy returns its live set size.
    fn distinct_program_count(&self) -> usize {
        0
    }
}

// =============================================================================
// Scheduling Decision
// =============================================================================

/// The outcome of a scheduling policy decision.
pub(super) enum Decision {
    /// Fire the current batch immediately.
    Fire,
    /// Wait for more requests, up to the given duration. Greedy-only under
    /// FCFS never constructs this; collapsing the policy abstraction is a
    /// deferred follow-up.
    #[allow(dead_code)]
    Wait(Duration),
}

// =============================================================================
// SchedulerStats (lock-free snapshot for monitoring)
// =============================================================================

pub const SYSTEM_SPEC_DRAFT_POS_BUCKETS: usize = 32;

/// Cumulative stats exposed for monitoring. Updated atomically after each batch.
#[derive(Debug, Default)]
pub struct SchedulerStats {
    // ── Always-on counters (no Instant::now needed). ────────────────────────
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
    pub system_spec_draft_tokens_proposed: AtomicU64,
    pub system_spec_draft_tokens_accepted: AtomicU64,
    pub system_spec_draft_tokens_proposed_per_pos:
        [AtomicU64; SYSTEM_SPEC_DRAFT_POS_BUCKETS],
    pub system_spec_draft_tokens_accepted_per_pos:
        [AtomicU64; SYSTEM_SPEC_DRAFT_POS_BUCKETS],

    // ── Fire-domain probes (gated behind `profile-fire` feature). ───────────
    //
    // Hierarchy + invariants documented in `crate::probe::fire`. Writers
    // use the `probe_fire!` macro from that module so the fetch_add
    // disappears when the feature is off. The struct itself is always
    // defined so callers and readers compile uniformly.
    pub fire: crate::probe::fire::FireProbes,

    // ── Driver-fire phase breakdown (gated behind `profile-driver-cuda`). ──
    //
    // Decomposes the `fire.execute.driver_fire_us` bucket into Rust
    // (ipc_submit / gpu_wait / ipc_recv) and C++ host phases (wire_parse
    // / plan / h2d / kernel_launch / sync / response_build). See
    // `crate::probe::driver_cuda` for the plumbing.
    pub driver_cuda: crate::probe::driver_cuda::DriverCudaProbes,
}

/// Out-of-band data execute_batch reports back to the run loop. Per-fire
/// *timing* probes are no longer in this struct — they're recorded
/// directly into `stats.fire.*` via `probe_fire!`. What's left here is
/// genuine fire-output data (spec-decoding draft counters) that the run
/// loop then folds into the spec-domain atomics.
#[derive(Debug, Default, Clone, Copy)]
struct BatchExecutionTiming {
    system_spec_draft_tokens_proposed: u64,
    system_spec_draft_tokens_accepted: u64,
    system_spec_draft_tokens_proposed_per_pos: [u64; SYSTEM_SPEC_DRAFT_POS_BUCKETS],
    system_spec_draft_tokens_accepted_per_pos: [u64; SYSTEM_SPEC_DRAFT_POS_BUCKETS],
}

/// Completion feedback the spawned fire task sends back to the scheduler loop
/// so the run-ahead policy can update its timing EWMAs. `forward_latency` (the
/// off-thread GPU/driver wait) feeds `on_complete` and pops the in-flight FIFO;
/// `submission_latency` (the host batch-build/enqueue done on the scheduler
/// thread) feeds `on_submitted` (the lead time the fire is brought forward by).
struct FireCompletion {
    forward_latency: Duration,
    submission_latency: Duration,
}

// =============================================================================
// PendingRequest
// =============================================================================

/// A forward pass request bundled with its response channel and physical pages.
struct PendingRequest {
    request: pie_driver_abi::ForwardRequest,
    completion: Completion,
    physical_page_ids: Vec<PhysicalPageId>,
    last_page_len: u32,
    /// #10: per-program `program_identity_hash` for this request (one per program
    /// in its sampler pass; empty for plain decode). Computed once at attach
    /// (host-side, before carrier encoding) and threaded to the policy's
    /// distinct-program set via `on_arrival` — runtime-side only, never on the
    /// wire `ForwardRequest`.
    program_identity_hashes: Vec<u64>,
}

enum Completion {
    Direct(oneshot::Sender<Result<ForwardOutput>>),
    Chunk {
        continuation: ChunkContinuation,
        sampler_slots: Vec<usize>,
    },
}

impl PendingRequest {
    fn direct(
        request: pie_driver_abi::ForwardRequest,
        response_tx: oneshot::Sender<Result<ForwardOutput>>,
        physical_page_ids: Vec<PhysicalPageId>,
        last_page_len: u32,
        program_identity_hashes: Vec<u64>,
    ) -> Self {
        Self {
            request,
            completion: Completion::Direct(response_tx),
            physical_page_ids,
            last_page_len,
            program_identity_hashes,
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
    for i in 0..req.request.n_samplers() {
        let sampler = req.request.sampler_at(i).expect("slot in range");
        if !is_token_sampler(&sampler) {
            all_samplers_token = false;
        }
        if sampler_needs_prob_rows(&sampler) {
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
        has_dense_logit_requirement,
        has_prob_sampling,
        is_single_token_decode,
        all_samplers_token,
    }
}

fn is_token_sampler(sampler: &pie_driver_abi::Sampler) -> bool {
    matches!(
        sampler,
        pie_driver_abi::Sampler::Multinomial { .. }
            | pie_driver_abi::Sampler::TopK { .. }
            | pie_driver_abi::Sampler::TopP { .. }
            | pie_driver_abi::Sampler::MinP { .. }
            | pie_driver_abi::Sampler::TopKTopP { .. }
    )
}

fn sampler_needs_prob_rows(sampler: &pie_driver_abi::Sampler) -> bool {
    match sampler {
        pie_driver_abi::Sampler::TopK { temperature, k } => *temperature > 0.0 && *k > 0,
        pie_driver_abi::Sampler::TopP { temperature, p } => *temperature > 0.0 && *p < 1.0,
        pie_driver_abi::Sampler::TopKTopP { temperature, k, p } => {
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

fn request_logprob_labels(req: &pie_driver_abi::ForwardRequest) -> usize {
    // Logprob → 1 label, Logprobs → its token_ids count (from the CSR), else 0.
    (0..req.n_samplers())
        .map(|s| match req.sampler_kinds[s] {
            pie_driver_abi::PIE_SAMPLER_LOGPROB => 1,
            pie_driver_abi::PIE_SAMPLER_LOGPROBS => (req.sampler_token_ids_indptr[s + 1]
                - req.sampler_token_ids_indptr[s])
                as usize,
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
        let logit_rows = if total_sampler_rows == 0 {
            // Pure KV-fill / encode fire (e.g. a multimodal image-token
            // prefill): no token is sampled, so the driver computes no
            // logits. Without this, the fire would project `total_tokens`
            // logit rows and trip `max_logit_rows` for large image spans.
            0
        } else if compact_logit_rows {
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

    /// Run-ahead token-carryover separation (one-step #6, R10): a candidate
    /// forward `t+1` whose `next_input_producer_links` references the
    /// `pipeline_source_link` of a forward `t` ALREADY in this batch must NOT
    /// co-batch with it. `t+1`'s pre-forward `next-inputs` inject reads `t`'s
    /// sampled token (a *prior* fire's retained buffer); `t` samples
    /// post-forward, so co-batching would read a not-yet-sampled token → `t`
    /// and `t+1` must fire in separate, ordered batches (`t` first). This is
    /// purely the token-carryover data-dependency — not a KV/working-set
    /// concern. Batch-local set-membership on the link ids (`0` = not a source;
    /// links are 1-based). Depth ≤1 makes this a simple membership check; the
    /// parity-phase machinery (§2.3) is the deferred deeper-run-ahead path.
    fn would_depend_on_batch(&self, req: &PendingRequest) -> bool {
        let deps = &req.request.next_input_producer_links;
        if deps.is_empty() || self.requests.is_empty() {
            return false;
        }
        self.requests.iter().any(|in_batch| {
            let source_link = in_batch.request.pipeline_source_link;
            source_link != 0 && deps.contains(&source_link)
        })
    }

    fn would_exceed_with(&self, usage: &RequestCapacityUsage) -> bool {
        if self.requests.is_empty() {
            return false;
        }
        // rs_cache spec-decode (MTP for hybrid GDN models) no longer needs a
        // per-batch cap: the driver runs a frozen verify (committed slot stays
        // at its pre-verify value) and a single batched repair forward over
        // [input | accepted] to advance state. There is no per-request snapshot
        // buffer, so rs-spec batches grow to the normal forward limits below,
        // exactly like non-spec batches.
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
            // rs-spec batches (frozen verify + batched repair) carry no
            // per-request buffer, so they fill to the normal forward limits
            // like any other batch — no rs-spec-specific early fire.
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
    matches!(&p.completion, Completion::Direct(_)) && p.request.token_ids.len() == 1
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
            pie_driver_abi::ForwardRequest {
                token_ids: vec![0; tokens],
                ..Default::default()
            },
            tx,
            vec![0; page_refs],
            1,
            Vec::new(),
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
        samplers: Vec<pie_driver_abi::Sampler>,
    ) -> PendingRequest {
        req.request.sampling_indices = indices;
        req.request.set_samplers(&samplers);
        req
    }

    fn with_pipeline_source(mut req: PendingRequest, link: u32) -> PendingRequest {
        req.request.pipeline_source_link = link;
        req
    }

    fn with_next_input(mut req: PendingRequest, producer_link: u32) -> PendingRequest {
        req.request.next_input_producer_links = vec![producer_link];
        req
    }

    #[test]
    fn accumulator_separates_run_ahead_carryover_dependency() {
        // One-step run-ahead (R10): a forward `t` samples its token under link
        // L=1; a forward `t+1` that injects L=1 via `next-inputs` MUST NOT
        // co-batch with it (it would read `t`'s not-yet-sampled token). `t+1` is
        // stashed for the next fire; an unrelated/plain request is unaffected.
        let mut batch = BatchAccumulator::new(limits(8, 100, 100), 16);
        batch.push(with_pipeline_source(pending(1, 1), 1));
        assert!(batch.would_depend_on_batch(&with_next_input(pending(1, 1), 1)));
        assert!(!batch.would_depend_on_batch(&with_next_input(pending(1, 1), 2)));
        assert!(!batch.would_depend_on_batch(&pending(1, 1)));
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
        req.request.set_samplers(&[pie_driver_abi::Sampler::Logprobs {
            token_ids: vec![1, 2, 3],
        }]);
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
            vec![pie_driver_abi::Sampler::TopP {
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
                vec![pie_driver_abi::Sampler::TopP {
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
        let req = with_samplers(pending(4, 1), vec![3], vec![pie_driver_abi::Sampler::RawLogits]);
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
            vec![pie_driver_abi::Sampler::TopP {
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

/// Cloneable submit handle.
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
        request: pie_driver_abi::ForwardRequest,
        response_tx: oneshot::Sender<Result<ForwardOutput>>,
        physical_page_ids: Vec<PhysicalPageId>,
        last_page_len: u32,
    ) -> Result<()> {
        self.submit_with_identity(
            request,
            response_tx,
            physical_page_ids,
            last_page_len,
            Vec::new(),
        )
    }

    /// Submit carrying the request's per-program `program_identity_hash`es (the
    /// #10 distinct-count key, computed host-side at attach). Empty ⇒ plain
    /// decode. The hashes ride on `PendingRequest` (runtime-side only) and reach
    /// the policy via `on_arrival`; they are never placed on the wire request.
    pub fn submit_with_identity(
        &self,
        request: pie_driver_abi::ForwardRequest,
        response_tx: oneshot::Sender<Result<ForwardOutput>>,
        physical_page_ids: Vec<PhysicalPageId>,
        last_page_len: u32,
        program_identity_hashes: Vec<u64>,
    ) -> Result<()> {
        self.tx
            .send(PendingRequest::direct(
                request,
                response_tx,
                physical_page_ids,
                last_page_len,
                program_identity_hashes,
            ))
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
    ) -> Self {
        let (tx, rx) = crossbeam::channel::unbounded::<PendingRequest>();
        let submit_tx = tx.clone();
        let stats = Arc::new(SchedulerStats::default());

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
                    stats_for_loop,
                    rt_handle,
                );
            })
            .expect("spawn pie-sched thread");

        Self { tx, stats }
    }

    /// Get a handle to the cumulative scheduler stats (lock-free).
    pub fn stats(&self) -> &Arc<SchedulerStats> {
        &self.stats
    }

    /// Submit a pre-translated forward pass request.
    pub fn submit(
        &self,
        request: pie_driver_abi::ForwardRequest,
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
                Vec::new(),
            ))
            .map_err(|_| anyhow::anyhow!("scheduler channel closed"))?;
        Ok(())
    }

    /// Cloneable handle for tasks that need to submit outside the
    /// scheduler's `run` loop.
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
        stats: Arc<SchedulerStats>,
        rt_handle: tokio::runtime::Handle,
    ) {
        // The per-request timeout is currently unused by the run-ahead fire
        // path (the driver wait is bounded by the channel). Kept consuming the
        // param so the scheduler signature is stable for a future per-request
        // deadline.
        let _request_timeout = Duration::from_secs(request_timeout_secs);

        // Per-driver state
        let mut batch = BatchAccumulator::new(limits, page_size);
        // Run-ahead just-in-time firing (#6). Fires the next batch a lead-time
        // before the in-flight one completes so the enqueue lands as the GPU
        // goes idle. Degrades to greedy when nothing is in flight at decide
        // time (the synchronous-fire fallback).
        let mut policy: Box<dyn SchedulingPolicy> =
            Box::new(RunAheadPolicy::new(limits.max_forward_requests));
        // The fire is non-blocking: `execute_batch` is split into an ordered
        // enqueue on this thread (fixing driver-inbox order == fire order) plus
        // an off-thread wait, so building/enqueuing the next batch overlaps the
        // in-flight GPU. The in-flight depth is bounded by the policy's FIFO
        // cap (one-step run-ahead, R10).

        // Channel for batch completion feedback to the policy. Carries the
        // off-thread forward (GPU) latency + the on-thread submission latency.
        let (latency_tx, latency_rx) = crossbeam::channel::unbounded::<FireCompletion>();
        let mut next_pending: Option<PendingRequest> = None;

        'run_loop: loop {
            // Drain completed batch feedback (non-blocking): GPU latency →
            // on_complete (+ FIFO pop), submission latency → on_submitted.
            while let Ok(c) = latency_rx.try_recv() {
                policy.on_submitted(c.submission_latency);
                policy.on_complete(c.forward_latency);
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
                policy.on_arrival(&pending.program_identity_hashes);
                batch.push(pending);
            }

            // Accumulate more requests (non-blocking). If a request is
            // already stashed for the next batch, fire the current batch
            // before reading more; overwriting the stash would drop that
            // request's response channel.
            let accum_start = Instant::now();
            // Test-only deterministic co-batch hold (`PIE_SCHED_ACCUM_HOLD_US`):
            // block up to the deadline after the first request so concurrent
            // requests land in the same drain window (deterministic
            // `forward_R >= 2` for the merged-path verify). Unset → `None` →
            // today's fire-on-arrival, unchanged.
            let accum_deadline =
                scheduler_accum_hold_us().map(|us| accum_start + Duration::from_micros(us));
            while next_pending.is_none() {
                let pending = match accum_deadline
                    .and_then(|d| d.checked_duration_since(Instant::now()))
                {
                    Some(remaining) => match req_rx.recv_timeout(remaining) {
                        Ok(p) => p,
                        Err(crossbeam::channel::RecvTimeoutError::Timeout) => break,
                        Err(crossbeam::channel::RecvTimeoutError::Disconnected) => {
                            break 'run_loop;
                        }
                    },
                    None => match req_rx.try_recv() {
                        Ok(p) => p,
                        Err(crossbeam::channel::TryRecvError::Empty) => break,
                        Err(crossbeam::channel::TryRecvError::Disconnected) => {
                            break 'run_loop;
                        }
                    },
                };
                let Some((pending, usage)) = prepare_pending_with_usage(&batch, pending)
                else {
                    continue;
                };
                // Run-ahead one-step separation (R10): a forward `t+1` whose
                // token-carryover source `t` is in this batch fires in the NEXT
                // batch (after `t` samples its token) — co-batching would read a
                // not-yet-sampled token.
                if batch.would_depend_on_batch(&pending) {
                    next_pending = Some(pending);
                    break;
                }
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
                policy.on_arrival(&pending.program_identity_hashes);
                batch.push_with(pending, usage);
                if batch.is_full() {
                    break;
                }
            }
            crate::probe_fire_record!(
                stats.fire.accumulate.accum_loop_us,
                accum_start.elapsed()
            );

            // Ask the policy what to do
            let decision = if next_pending.is_some() {
                Decision::Fire
            } else {
                policy.decide(batch.len())
            };
            match decision {
                Decision::Fire => {
                    // No in-flight gate to acquire: the scheduler runs
                    // execute_batch synchronously, so we can only reach
                    // here when the previous fire has fully completed.

                    // Do one last non-blocking drain so requests that
                    // arrived between the recv loop and here are
                    // coalesced into this batch instead of being
                    // stranded behind it.
                    let fire_prepare_start = Instant::now();
                    while next_pending.is_none() && !batch.is_full() {
                        let Ok(pending) = req_rx.try_recv() else {
                            break;
                        };
                        if let Some(msg) = batch.single_request_limit_error(&pending) {
                            pending.send_error(msg);
                            continue;
                        }
                        if batch.would_depend_on_batch(&pending) {
                            next_pending = Some(pending);
                            break;
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
                        policy.on_arrival(&pending.program_identity_hashes);
                        batch.push(pending);
                    }

                    let total_tokens = batch.total_tokens();
                    if scheduler_trace_enabled() {
                        eprintln!(
                            "[pie-sched-trace] driver={} fire requests={} tokens={} prefill_like={} stashed={} distinct_programs={}",
                            driver_idx,
                            batch.len(),
                            total_tokens,
                            batch.should_prefill_coalesce(),
                            next_pending.is_some(),
                            policy.distinct_program_count(),
                        );
                    }
                    let requests_to_fire = batch.take();
                    let batch_size = requests_to_fire.len() as u64;
                    crate::probe_fire_record!(
                        stats.fire.pre_dispatch.fire_prepare_us,
                        fire_prepare_start.elapsed()
                    );

                    // Inter-fire instrumentation: time between consecutive fires,
                    // and the post-dispatch-to-next-fire gap (rendezvous window).
                    // The timestamps themselves (last_fire_spawn_micros,
                    // last_dispatch_end_micros) are always-on — cheap atomic
                    // swap/load. The accumulators are probe-gated.
                    let now_us = now_micros();
                    let last_spawn = stats.fire.last_fire_spawn_micros.swap(now_us, Relaxed);
                    if last_spawn != 0 {
                        crate::probe_fire_record!(
                            stats.fire.inter_fire_us,
                            std::time::Duration::from_micros(now_us.saturating_sub(last_spawn))
                        );
                    }
                    let last_dispatch_end = stats.fire.last_dispatch_end_micros.load(Relaxed);
                    if last_dispatch_end != 0 {
                        crate::probe_fire_record!(
                            stats.fire.post_dispatch_to_fire_us,
                            std::time::Duration::from_micros(
                                now_us.saturating_sub(last_dispatch_end)
                            )
                        );
                    }

                    // Build the batched request on the scheduler thread (this
                    // overlaps the GPU of any in-flight batch), ENQUEUE it in
                    // fire-order here (fixing driver-inbox order == fire order,
                    // so a forward `t+1` never reaches the worker before its
                    // token-carryover source `t`), and AWAIT the response
                    // off-thread so this thread is freed to collect/build the
                    // next batch.
                    let build_start = Instant::now();
                    let batch_req =
                        Self::build_batch_request(&requests_to_fire, page_size, &stats);
                    let submission_latency = build_start.elapsed();

                    match driver::fire_batch_deferred(driver_idx, batch_req) {
                        Ok(handle) => {
                            // The batch is enqueued (its order fixed) — record it
                            // as in-flight so the policy paces the next fire.
                            policy.on_fired(batch_size as usize);

                            let stats_spawn = Arc::clone(&stats);
                            let rt_handle_spawn = rt_handle.clone();
                            let submit_tx_spawn = submit_tx.clone();
                            let latency_tx_spawn = latency_tx.clone();
                            rt_handle.spawn_blocking(move || {
                                // Phase: driver_fire — block off-thread for the
                                // GPU response. (The ipc_submit probe was set on
                                // the scheduler thread during enqueue; under
                                // `profile-driver-cuda` it reads 0 here — the
                                // gpu_wait probe set in this task is accurate.)
                                let fire_start = Instant::now();
                                let fire_result = crate::probe_fire!(
                                    stats_spawn.fire.execute.driver_fire_us,
                                    {
                                        let r = handle.wait();
                                        let ipc_submit_us =
                                            crate::probe::driver_cuda::take_ipc_submit_us();
                                        let gpu_wait_us =
                                            crate::probe::driver_cuda::take_gpu_wait_us();
                                        let ipc_recv_us =
                                            crate::probe::driver_cuda::take_ipc_recv_us();
                                        if ipc_submit_us > 0 {
                                            stats_spawn
                                                .driver_cuda
                                                .ipc_submit_us
                                                .fetch_add(ipc_submit_us, Relaxed);
                                        }
                                        if gpu_wait_us > 0 {
                                            stats_spawn
                                                .driver_cuda
                                                .gpu_wait_us
                                                .fetch_add(gpu_wait_us, Relaxed);
                                        }
                                        if ipc_recv_us > 0 {
                                            stats_spawn
                                                .driver_cuda
                                                .ipc_recv_us
                                                .fetch_add(ipc_recv_us, Relaxed);
                                        }
                                        r
                                    }
                                );
                                let forward_latency = fire_start.elapsed();
                                let timing = Self::dispatch_fired_batch(
                                    fire_result,
                                    requests_to_fire,
                                    driver_id,
                                    page_size,
                                    &rt_handle_spawn,
                                    Some(submit_tx_spawn),
                                    &stats_spawn,
                                );
                                Self::record_fire_stats(
                                    &stats_spawn,
                                    &timing,
                                    forward_latency,
                                    batch_size,
                                    total_tokens,
                                );
                                let _ = latency_tx_spawn.send(FireCompletion {
                                    forward_latency,
                                    submission_latency,
                                });
                            });
                        }
                        Err(e) => {
                            // Enqueue failed (channel closed/aborted) — fail the
                            // batch's requests; nothing went in flight.
                            let msg =
                                format!("fire_batch_deferred failed for driver {driver_id}: {e:#}");
                            for req in requests_to_fire {
                                req.send_result::<ForwardOutput>(
                                    Err(anyhow::anyhow!(msg.clone())),
                                    None,
                                    page_size,
                                );
                            }
                        }
                    }
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
                                    if batch.would_depend_on_batch(&pending) {
                                        next_pending = Some(pending);
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
                                        continue;
                                    }
                                    policy.on_arrival(&pending.program_identity_hashes);
                                    batch.push(pending);
                                }
                                Err(_) => break 'run_loop, // channel closed
                            }
                        }
                        recv(latency_rx) -> completion => {
                            if let Ok(c) = completion {
                                policy.on_submitted(c.submission_latency);
                                policy.on_complete(c.forward_latency);
                            }
                        }
                        default(wait_duration) => {}
                    }
                }
            }
        }

        // Shutdown: fire the remaining batch synchronously so any
        // inferlets still awaiting responses get them before we exit.
        // ~10 ms of additional shutdown latency in the worst case.
        if !batch.is_empty() {
            let requests = batch.take();
            let _ = Self::execute_batch_blocking(
                driver_idx,
                requests,
                driver_id,
                page_size,
                &rt_handle,
                None,
                &stats,
            );
        }
    }

    /// Build the batched `pie_driver_abi::ForwardRequest` by folding each
    /// per-request shape into one batch. Runs on the scheduler thread (so it
    /// overlaps the GPU of any in-flight batch); the caller then enqueues it in
    /// fire-order via [`driver::fire_batch_deferred`].
    fn build_batch_request(
        requests: &[PendingRequest],
        page_size: u32,
        stats: &SchedulerStats,
    ) -> pie_driver_abi::ForwardRequest {
        // Build batched request — a single `pie_driver_abi::ForwardRequest`
        // populated by folding each per-request shape into the batch.
        let elide_decode_masks = requests.iter().all(|req| {
            req.request.single_token_mode
                && !req.request.has_user_mask
                && req.request.token_ids.len() <= 1
                && req.request.spec_token_ids.is_empty()
        });
        crate::probe_fire!(stats.fire.execute.batch_build_us, {
            let mut batch_req =
                request::new_batched_forward_request_with_capacity(requests.len());
            for req in requests {
                request::append_request_with_options(
                    &mut batch_req,
                    &req.request,
                    &req.physical_page_ids,
                    req.last_page_len,
                    page_size,
                    elide_decode_masks,
                );
            }
            batch_req
        })
    }

    /// Fold a completed batch's always-on counters + spec-domain accumulators
    /// into the shared stats. `latency` is the off-thread forward (GPU) wait —
    /// the dominant component of the batch's wall time under the overlapped
    /// fire (the host build/enqueue overlaps the prior in-flight batch).
    fn record_fire_stats(
        stats: &SchedulerStats,
        timing: &BatchExecutionTiming,
        latency: Duration,
        batch_size: u64,
        total_tokens: usize,
    ) {
        crate::probe_fire!(stats.fire.post_dispatch.stats_update_us, {
            stats.total_batches.fetch_add(1, Relaxed);
            stats
                .total_tokens_processed
                .fetch_add(total_tokens as u64, Relaxed);
            stats
                .total_requests_processed
                .fetch_add(batch_size, Relaxed);
            stats
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
            stats.batch_size_hist[bucket].fetch_add(1, Relaxed);
            stats
                .last_batch_latency_us
                .store(latency.as_micros() as u64, Relaxed);
            stats
                .cumulative_latency_us
                .fetch_add(latency.as_micros() as u64, Relaxed);
            stats
                .system_spec_draft_tokens_proposed
                .fetch_add(timing.system_spec_draft_tokens_proposed, Relaxed);
            stats
                .system_spec_draft_tokens_accepted
                .fetch_add(timing.system_spec_draft_tokens_accepted, Relaxed);
            for (counter, value) in stats
                .system_spec_draft_tokens_proposed_per_pos
                .iter()
                .zip(timing.system_spec_draft_tokens_proposed_per_pos)
            {
                if value != 0 {
                    counter.fetch_add(value, Relaxed);
                }
            }
            for (counter, value) in stats
                .system_spec_draft_tokens_accepted_per_pos
                .iter()
                .zip(timing.system_spec_draft_tokens_accepted_per_pos)
            {
                if value != 0 {
                    counter.fetch_add(value, Relaxed);
                }
            }
        });
    }

    /// Build + enqueue + await + dispatch a batch synchronously on the caller's
    /// thread. Used for the shutdown drain (no overlap needed); the hot path
    /// instead splits these phases across the scheduler thread and a spawned
    /// task so the GPU wait overlaps the next batch's build.
    fn execute_batch_blocking(
        driver_idx: usize,
        requests: Vec<PendingRequest>,
        driver_id: DriverId,
        page_size: u32,
        rt_handle: &tokio::runtime::Handle,
        submit_tx: Option<crossbeam::channel::Sender<PendingRequest>>,
        stats: &SchedulerStats,
    ) -> BatchExecutionTiming {
        let batch_req = Self::build_batch_request(&requests, page_size, stats);
        let fire_result = match driver::fire_batch_deferred(driver_idx, batch_req) {
            Ok(handle) => handle.wait(),
            Err(e) => Err(e),
        };
        Self::dispatch_fired_batch(
            fire_result,
            requests,
            driver_id,
            page_size,
            rt_handle,
            submit_tx,
            stats,
        )
    }

    /// Dispatch a fired batch's response to the per-request oneshots and
    /// accumulate spec-decode draft counters. `fire_result` is the awaited
    /// forward response (the GPU wait already happened off-thread). The
    /// `deferred_drop` punt still routes request-husk dealloc to the blocking
    /// pool so it does not compete with response dispatch.
    fn dispatch_fired_batch(
        fire_result: Result<pie_driver_abi::ForwardResponse>,
        requests: Vec<PendingRequest>,
        driver_id: DriverId,
        page_size: u32,
        rt_handle: &tokio::runtime::Handle,
        submit_tx: Option<crossbeam::channel::Sender<PendingRequest>>,
        stats: &SchedulerStats,
    ) -> BatchExecutionTiming {
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

        // Response dispatch: per-request oneshot fires and queueing the
        // deferred_drop Vec. The GPU wait already happened off-thread (the
        // caller awaited `FireHandle::wait` before handing us `fire_result`).
        //
        // Per-completion-type counts are accumulated into these locals
        // inside the match arms and fetch_add'd once after the loop, so
        // we don't pay a per-request atomic op on the hot path.
        let response_dispatch_start = Instant::now();
        let mut direct_count: u64 = 0;
        let mut chunk_count: u64 = 0;
        match fire_result {
            Ok(batch_resp) => {
                let wp = batch_resp.probe_wire_parse_us as u64;
                let pl = batch_resp.probe_plan_us as u64;
                let hd = batch_resp.probe_h2d_us as u64;
                let kl = batch_resp.probe_kernel_launch_us as u64;
                let sy = batch_resp.probe_sync_us as u64;
                let rb = batch_resp.probe_response_build_us as u64;
                if wp | pl | hd | kl | sy | rb != 0 {
                    stats.driver_cuda.wire_parse_us.fetch_add(wp, Relaxed);
                    stats.driver_cuda.plan_us.fetch_add(pl, Relaxed);
                    stats.driver_cuda.h2d_us.fetch_add(hd, Relaxed);
                    stats.driver_cuda.kernel_launch_us.fetch_add(kl, Relaxed);
                    stats.driver_cuda.sync_us.fetch_add(sy, Relaxed);
                    stats.driver_cuda.response_build_us.fetch_add(rb, Relaxed);
                }
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
                        && batch_resp.program_tokens.is_empty()
                        && batch_resp.tokens_indptr.len() >= requests.len() + 1;

                    // Send oneshot replies first, defer drop of the
                    // request husks. Each PendingRequest's drop is
                    // ~3-4 µs (22-Vec ForwardRequest), and doing it
                    // inline adds avoidable tail latency to response
                    // dispatch for large batches.
                    let mut deferred_drop: Vec<(
                        pie_driver_abi::ForwardRequest,
                        Vec<PhysicalPageId>,
                    )> = Vec::with_capacity(n_results);
                    // #27 cut #1 eager-D2H fast-path (a2-mode): a request that set
                    // up the `sampling_output_*` dst table had its sampled token
                    // copied DIRECTLY to the pinned output Tensor (D2H), so the
                    // driver response carries NO token (`tokens[]` empty). Resolve
                    // each oneshot with success WITHOUT extracting `tokens[..]` —
                    // the inferlet's `output()` reads the filled pinned buffer
                    // (gated on `forward_result.is_some()`); an `Err`/drop here
                    // would hit the abort path (txn drop + "no output tensor").
                    // Keyed per-request on `sampling_output_*`, which
                    // `populate_output_fastpath` sets iff it also stashed the
                    // pinned outputs (1:1 with the host pinned-read gate, so no
                    // skew). One-ahead MVP batches are all-or-nothing fast-path.
                    let all_fast_path = !requests.is_empty()
                        && requests
                            .iter()
                            .all(|req| !req.request.sampling_output_dst_ptrs.is_empty());
                    if all_fast_path {
                        for req in requests.into_iter() {
                            // Empty `Tokens` is `Some` → the host gate passes and
                            // reads the pinned buffer; the payload is ignored.
                            let output = ForwardOutput::Tokens(Vec::new());
                            let PendingRequest {
                                request,
                                completion,
                                physical_page_ids,
                                last_page_len: _,
                                program_identity_hashes,
                            } = req;
                            match completion {
                                Completion::Direct(tx) => {
                                    direct_count += 1;
                                    tx.send(Ok(output)).ok();
                                    deferred_drop.push((request, physical_page_ids));
                                }
                                Completion::Chunk { .. } => {
                                    chunk_count += 1;
                                    let req = PendingRequest {
                                        request,
                                        completion,
                                        physical_page_ids,
                                        last_page_len: 0,
                                        program_identity_hashes,
                                    };
                                    req.send_result(Ok(output), submit_tx.as_ref(), page_size);
                                }
                            }
                        }
                    } else if token_payload_only {
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
                                program_identity_hashes,
                            } = req;
                            match completion {
                                Completion::Direct(tx) => {
                                    direct_count += 1;
                                    tx.send(Ok(output)).ok();
                                    deferred_drop.push((request, physical_page_ids));
                                }
                                Completion::Chunk { .. } => {
                                    chunk_count += 1;
                                    let req = PendingRequest {
                                        request,
                                        completion,
                                        physical_page_ids,
                                        last_page_len: 0,
                                        program_identity_hashes,
                                    };
                                    req.send_result(Ok(output), submit_tx.as_ref(), page_size);
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
                                program_identity_hashes,
                            } = req;
                            match completion {
                                Completion::Direct(tx) => {
                                    direct_count += 1;
                                    tx.send(Ok(output)).ok();
                                    deferred_drop.push((request, physical_page_ids));
                                }
                                Completion::Chunk { .. } => {
                                    chunk_count += 1;
                                    let req = PendingRequest {
                                        request,
                                        completion,
                                        physical_page_ids,
                                        last_page_len: 0,
                                        program_identity_hashes,
                                    };
                                    req.send_result(Ok(output), submit_tx.as_ref(), page_size);
                                }
                            }
                        }
                    }
                    stats.fire.last_dispatch_end_micros.store(now_micros(), Relaxed);
                    if !deferred_drop.is_empty() {
                        // Dedicated blocking pool so this dealloc task
                        // does not compete with response dispatch. Use the captured
                        // `rt_handle` because we're now on the scheduler
                        // OS thread, not a tokio task — `tokio::task::
                        // spawn_blocking` would panic without an ambient
                        // runtime context.
                        rt_handle.spawn_blocking(move || drop(deferred_drop));
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
        crate::probe_fire_record!(
            stats.fire.execute.response_dispatch.total_us,
            response_dispatch_start.elapsed()
        );
        // Per-completion-type counts. Counters, not durations — three
        // atomic ops per fire regardless of batch size, so always-on
        // (no feature gate).
        if direct_count > 0 {
            stats
                .fire
                .execute
                .response_dispatch
                .direct_count
                .fetch_add(direct_count, Relaxed);
        }
        if chunk_count > 0 {
            stats
                .fire
                .execute
                .response_dispatch
                .chunk_count
                .fetch_add(chunk_count, Relaxed);
        }
        BatchExecutionTiming {
            system_spec_draft_tokens_proposed,
            system_spec_draft_tokens_accepted,
            system_spec_draft_tokens_proposed_per_pos,
            system_spec_draft_tokens_accepted_per_pos,
        }
    }
}
