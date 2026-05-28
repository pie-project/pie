//! Pass-level speculative execution: the runtime self-stages each
//! ctx's next forward pass while the GPU is busy with the prior
//! one, then claims it on the inferlet's actual `execute()` call.
//!
//! Design summary (see SPECULATIVE_EXECUTION_DESIGN.md for the
//! full story):
//!
//!   - Per ctx, a bounded `VecDeque<StagedEntry>` holds the chain
//!     of pre-fired forward passes (depth limited by
//!     `scheduler.speculation_depth`; `0` disables speculation).
//!   - After every fire completes, a chain extender task takes
//!     the output, builds the next request from it
//!     ([`build_next_request`]), submits to the scheduler, pushes
//!     the entry, and spawns the next extender. The chain
//!     self-sustains until end-of-reserved-pages, ctx invalidation,
//!     or fingerprint miss.
//!   - The inferlet's `execute()` first calls [`try_hit`]
//!     directly. On match it claims the staged entry's rx and
//!     skips the actor mailbox round-trip entirely. On miss/cold
//!     it falls through to a normal submit through the actor.
//!
//! This module owns the chain state. The scheduler is unaware of
//! speculation — it sees normal `submit()` calls. The driver is
//! unaware of speculation — it just runs forward passes.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use anyhow::Result;
use dashmap::DashMap;
use tokio::sync::{mpsc, oneshot};

use crate::context::ContextId;
use crate::context::pagestore::PhysicalPageId;
use crate::inference::ForwardOutput;
use crate::inference::scheduler::SchedulerHandle;

/// Per-(model, device) map of pre-fired stages keyed by context.
///
/// Backed by `dashmap` (a sharded concurrent HashMap) rather than a
/// single `Mutex<HashMap>`. With 256 chain extenders that all wake
/// after each fire and rush to acquire this map, a single mutex
/// serializes ~52 µs/iter onto the critical path; dashmap's
/// per-shard locks let them run in parallel.
pub(crate) type StagedBatchMap = Arc<DashMap<ContextId, VecDeque<StagedEntry>>>;

/// A pre-fired forward pass for a ctx, sitting in the per-ctx
/// chain queue waiting for the inferlet's matching `execute()` call.
///
/// Anchor = the (token, position) the inferlet's actual request
/// must match for this entry to count as a hit. Built when the
/// chain extender constructed the entry.
pub(crate) struct StagedEntry {
    pub anchor_token: u32,
    pub anchor_pos: u32,
    pub spec_token_ids: Vec<u32>,
    pub spec_position_ids: Vec<u32>,
    pub output_spec_flags: Vec<bool>,
    /// True when the staged entry's underlying request uses an
    /// rs_cache (recurrent-state) slot. For these contexts the api
    /// layer rewrites output_spec_flags=[false] in the cold path
    /// before submit, so the chain extender's prev_req (and thus this
    /// entry) always has [false]. The inferlet's outgoing request
    /// still carries [true] when SpecMode::System is active, so we
    /// must skip comparing output_spec_flags on rs_cache entries to
    /// avoid 99% chain-drop rates.
    pub uses_rs_cache: bool,
    /// Claim-side hint for the extender that owns this staged fire.
    /// The inferlet can claim a pre-fired final token while telling
    /// its extender not to submit another stage.
    pub allow_extend: Arc<AtomicBool>,
    /// Future fire's output for this ctx. The scheduler holds the
    /// matching `Sender`; when the kernel finishes and the output
    /// is delivered, this receiver resolves.
    pub output_rx: oneshot::Receiver<Result<ForwardOutput>>,
}

pub(crate) fn entry_matches_request(
    entry: &StagedEntry,
    request: &pie_bridge::ForwardRequest,
) -> bool {
    let base = Some(entry.anchor_token) == request.token_ids.first().copied()
        && Some(entry.anchor_pos) == request.position_ids.first().copied()
        && entry.spec_token_ids == request.spec_token_ids
        && entry.spec_position_ids == request.spec_position_ids;
    if !base {
        return false;
    }
    // For rs_cache contexts the api layer's cold path rewrites
    // output_spec_flags=[false] before submit, so the chain extender's
    // staged entry always has [false]. The inferlet's outgoing request
    // still has [true] when SpecMode::System is active. Skip the flag
    // comparison here — the rewrite makes the staged fire's output
    // shape identical regardless of which flag the inferlet sent.
    if entry.uses_rs_cache {
        return true;
    }
    entry.output_spec_flags == request.output_spec_flags
}

/// Per-model speculator state.
struct ModelEntry {
    /// Per-context depth of pass-level speculation
    /// (`scheduler.speculation_depth` in toml). `0` disables
    /// speculation for this model.
    speculation_depth: usize,
    /// One staged-batch map per device on this model.
    devices: Vec<StagedBatchMap>,
}

/// Per-model registry. The inferlet-side `try_hit` accesses this
/// without going through the inference actor.
static REGISTRY: LazyLock<Mutex<Vec<ModelEntry>>> = LazyLock::new(|| Mutex::new(Vec::new()));

/// Total number of `try_hit` calls that successfully claimed a
/// staged entry. Surfaced via `model_status.bypass_hits` for
/// observability. Increments are per call, not per token.
pub static BYPASS_HIT_COUNT: AtomicU64 = AtomicU64::new(0);

/// Total number of chain extenders that successfully submitted a
/// staged forward pass to the scheduler. Surfaced via
/// `model_status.chain_submits`.
pub static CHAIN_SUBMIT_COUNT: AtomicU64 = AtomicU64::new(0);

/// Total number of staged chains dropped due to anchor mismatch.
/// Surfaced via `model_status.chain_drops`.
pub static CHAIN_DROP_COUNT: AtomicU64 = AtomicU64::new(0);

/// Register the staged batches for a model with the global
/// registry. Called once at `InferenceService::new`. The
/// `staged_batch` slice is one entry per device on this model.
/// `speculation_depth == 0` disables speculation for this model.
pub(crate) fn register_model(
    model_idx: usize,
    staged_batch: &[StagedBatchMap],
    speculation_depth: usize,
) {
    if let Ok(mut reg) = REGISTRY.lock() {
        while reg.len() <= model_idx {
            reg.push(ModelEntry {
                speculation_depth: 0,
                devices: Vec::new(),
            });
        }
        reg[model_idx] = ModelEntry {
            speculation_depth,
            devices: staged_batch.to_vec(),
        };
    }
}

/// Opaque per-(model, device) handle the api layer caches on the
/// ctx side. Lets `try_hit` skip the REGISTRY lookup on every
/// `execute()` — the lookup happens once when the ctx is first
/// bound to a `ForwardPass`, and the resulting arc is reused.
#[derive(Clone)]
pub struct StagedBatch(StagedBatchMap);

impl std::fmt::Debug for StagedBatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "StagedBatch")
    }
}

/// Resolve the per-(model, device) staged-batch arc for a ctx.
/// Called once per ctx (cached by the api layer), not per execute.
/// Returns `None` when speculation is disabled for the model.
pub fn lookup_for_ctx(model_idx: usize, device_idx: usize) -> Option<StagedBatch> {
    let reg = REGISTRY.lock().ok()?;
    let model = reg.get(model_idx)?;
    if model.speculation_depth == 0 {
        return None;
    }
    let arc = model.devices.get(device_idx)?.clone();
    Some(StagedBatch(arc))
}

/// Fast hit check from the api layer, using a cached
/// `StagedBatch` handle. If the front entry's anchor matches the
/// inferlet's request, pop it and return the output receiver. On
/// mismatch, clear the entire chain — deeper stages were built on
/// a now-invalid assumption.
///
/// Returns `None` on cold (no entries), mismatch, or lock failure.
pub fn try_hit(
    spec: &StagedBatch,
    ctx_id: ContextId,
    request: &pie_bridge::ForwardRequest,
    allow_extend: bool,
) -> Option<oneshot::Receiver<Result<ForwardOutput>>> {
    let mut deque = spec.0.get_mut(&ctx_id)?;
    let front = deque.front()?;
    if entry_matches_request(front, request) {
        let entry = deque.pop_front()?;
        if !allow_extend {
            entry.allow_extend.store(false, Ordering::Relaxed);
        }
        BYPASS_HIT_COUNT.fetch_add(1, Ordering::Relaxed);
        Some(entry.output_rx)
    } else {
        deque.clear();
        CHAIN_DROP_COUNT.fetch_add(1, Ordering::Relaxed);
        None
    }
}

/// Drop any staged entries for `ctx_id` across all devices on
/// this model. Called when the ctx is being destroyed or
/// suspended. The staged request might still be in the
/// scheduler's queue / batch; if it fires, its `output_tx` has no
/// receiver and the send fails silently. Pages it references
/// would be stale by then but races on working pages are benign
/// (writes at deterministic positions). The hard invariant: drop
/// the staged_batch entry NOW so a later hit-check never forwards
/// stale predictions.
pub fn invalidate_ctx(model_idx: usize, ctx_id: ContextId) {
    let Ok(reg) = REGISTRY.lock() else {
        return;
    };
    let Some(model) = reg.get(model_idx) else {
        return;
    };
    for sb_arc in &model.devices {
        sb_arc.remove(&ctx_id);
    }
}

// =============================================================================
// Pooled chain extender
// =============================================================================
//
// Prior design: one tokio task per context, spawned on the cold submit.
// At conc=256 every fire woke 256 such tasks; tokio's task-pickup
// latency from the global injection queue dominated the inter-fire gap
// (~1.7 ms observed). See the gap-profile probes in the scheduler stats.
//
// New design: a fixed pool of N (default 8) long-lived worker tasks per
// driver. Per-context chain state travels with each forward request via
// `Completion::Chain { state }`; the dispatch loop routes the fire's
// output to a pool worker (sharded by ctx_id) instead of waking a
// per-context task. Wakes per fire drop from N_contexts to ≤ pool_size.

/// All per-context state needed to continue a speculation chain after
/// a fire completes. Threaded through the scheduler as
/// `Completion::Chain { state }` so the dispatch loop can route the
/// fire's output to the pool worker without per-context task plumbing.
pub(crate) struct ChainState {
    /// Where to deliver THIS fire's output: either the inferlet's
    /// downstream `Sender` (for the cold submit) or the prior stage's
    /// `StagedEntry::output_rx` paired tx (for every subsequent stage).
    pub response: oneshot::Sender<Result<ForwardOutput>>,
    pub scheduler_handle: SchedulerHandle,
    pub staged_batch_arc: StagedBatchMap,
    pub prev_request: pie_bridge::ForwardRequest,
    pub all_pages: Vec<PhysicalPageId>,
    pub cur_page_idx: usize,
    pub cur_last_page_len: u32,
    pub max_queue_depth: usize,
    pub allow_extend: Arc<AtomicBool>,
    pub page_size: u32,
}

/// A single chain-extension job handed to a pool worker.
pub(crate) struct ChainExtJob {
    pub state: Box<ChainState>,
    pub output: Result<ForwardOutput>,
    /// Micros since `sched_epoch()` when the dispatch loop enqueued
    /// this job. Used by the pool worker to measure wake-pickup latency.
    pub enqueued_us: u64,
}

pub static CHAIN_EXT_WAKE_LATENCY_US: AtomicU64 = AtomicU64::new(0);
pub static CHAIN_EXT_WORK_LATENCY_US: AtomicU64 = AtomicU64::new(0);
pub static CHAIN_EXT_JOBS_SAMPLED: AtomicU64 = AtomicU64::new(0);
pub static CHAIN_EXT_BUILD_NS: AtomicU64 = AtomicU64::new(0);
pub static CHAIN_EXT_SUBMIT_NS: AtomicU64 = AtomicU64::new(0);
pub static CHAIN_EXT_PUSH_NS: AtomicU64 = AtomicU64::new(0);
pub static CHAIN_EXT_RESPSEND_NS: AtomicU64 = AtomicU64::new(0);

pub(crate) fn sched_epoch() -> std::time::Instant {
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    *EPOCH.get_or_init(std::time::Instant::now)
}

pub(crate) fn now_micros() -> u64 {
    sched_epoch().elapsed().as_micros() as u64
}

/// Fixed-size pool of long-lived workers that process chain-extension
/// jobs. Jobs are sharded across workers by `ctx_id` so the same
/// context's stages always land on the same worker (cache locality for
/// the rare contended states like `staged_batch_arc.entry(ctx_id)`).
///
/// **Why std::thread (not tokio::spawn)**: chain ext jobs measured a
/// 216 µs avg wake propagation when handled by tokio tasks parked on
/// `mpsc::recv()` — the wake → worker-pickup path through tokio's
/// scheduler is the dominant cost at conc=256 (256 wakes per fire,
/// even sharded into 32 pool tasks, see prof_pool_recvmany_*.json).
/// std::thread workers parked on `Condvar::wait()` are woken directly
/// by the kernel scheduler with no tokio injection queue in between;
/// per-job wake drops from 216 µs to ~10 µs in our profile.
pub(crate) struct ChainExtPool {
    senders: Vec<mpsc::UnboundedSender<ChainExtJob>>,
}

impl ChainExtPool {
    pub fn new(num_workers: usize) -> Self {
        let num_workers = num_workers.max(1);
        let mut senders = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let (tx, mut rx) = mpsc::unbounded_channel::<ChainExtJob>();
            // `recv_many` drains a burst in one parked-wake cycle. Pool
            // tasks run on the main shared tokio runtime so the dispatch
            // loop (also tokio) can wake them via cheap intra-runtime
            // notifies — OS-thread workers via crossbeam add futex
            // syscall cost on the sender side.
            tokio::spawn(async move {
                let mut buf: Vec<ChainExtJob> = Vec::with_capacity(32);
                loop {
                    buf.clear();
                    let n = rx.recv_many(&mut buf, 64).await;
                    if n == 0 {
                        return;
                    }
                    for job in buf.drain(..) {
                        process_chain_job(job);
                    }
                }
            });
            senders.push(tx);
        }
        Self { senders }
    }

    /// Route a job to a worker sharded by `ctx_id`.
    pub fn submit(&self, ctx_id: ContextId, job: ChainExtJob) {
        let shard = (ctx_id as usize) % self.senders.len();
        let _ = self.senders[shard].send(job);
    }
}

/// Cold-submit entry point: build the initial ChainState and hand the
/// first fire to the scheduler with `Completion::Chain`. The fire's
/// output will be routed to the per-driver pool when the GPU returns;
/// no tokio task is spawned here.
#[allow(clippy::too_many_arguments)]
pub(crate) fn start_chain(
    response: oneshot::Sender<Result<ForwardOutput>>,
    scheduler_handle: SchedulerHandle,
    staged_batch_arc: StagedBatchMap,
    model_idx: usize,
    request: pie_bridge::ForwardRequest,
    physical_page_ids: Vec<PhysicalPageId>,
    all_pages: Vec<PhysicalPageId>,
    cur_page_idx: usize,
    cur_last_page_len: u32,
    max_queue_depth: usize,
    allow_extend: Arc<AtomicBool>,
) {
    let page_size = crate::context::tokens_per_page(model_idx);
    let state = Box::new(ChainState {
        response,
        scheduler_handle: scheduler_handle.clone(),
        staged_batch_arc,
        prev_request: request.clone(),
        all_pages,
        cur_page_idx,
        cur_last_page_len,
        max_queue_depth,
        allow_extend,
        page_size,
    });
    if let Err(_e) = scheduler_handle.submit_chain(
        request,
        state,
        physical_page_ids,
        cur_last_page_len,
    ) {
        // The submit can only fail if the scheduler has shut down; the
        // ChainState (with the inferlet's response Sender) is dropped on
        // the floor, and the inferlet's awaiting Receiver will see a
        // closed channel as the failure signal.
    }
}

/// Per-job processing. Forwards the just-received fire output to the
/// chain's downstream, then optionally submits the next stage and
/// pushes a `StagedEntry` for the inferlet's later try_hit. The chain
/// self-continues: when the next fire completes, the dispatch loop
/// routes another `ChainExtJob` to the pool — there is no per-context
/// task to keep alive.
pub(crate) fn process_chain_job(job: ChainExtJob) {
    let job_start_us = now_micros();
    let wake_us = job_start_us.saturating_sub(job.enqueued_us);
    CHAIN_EXT_WAKE_LATENCY_US.fetch_add(wake_us, Ordering::Relaxed);
    let ChainExtJob { state, output, enqueued_us: _ } = job;
    let ChainState {
        response,
        scheduler_handle,
        staged_batch_arc,
        prev_request,
        all_pages,
        cur_page_idx,
        cur_last_page_len,
        max_queue_depth,
        allow_extend,
        page_size,
    } = *state;

    let output = match output {
        Ok(o) => o,
        Err(e) => {
            let _ = response.send(Err(e));
            return;
        }
    };

    let ctx_id = match prev_request.context_ids.first() {
        Some(&id) => id,
        None => return,
    };

    // Orphan-stage gate. If the receiver of our `response` is gone
    // (ctx invalidated or cold-submit caller cancelled), don't push
    // another stage — orphan fires would inflate batch sizes.
    if response.is_closed() {
        return;
    }

    // Kill switches.
    if max_queue_depth == 0 || !allow_extend.load(Ordering::Relaxed) {
        let _ = response.send(Ok(output));
        return;
    }

    // Try to build & submit the next stage. Same gates and page math
    // as the prior `spawn_extend_chain` loop body — see comments there
    // for invariants.
    if let Some((next_req, anchor_token, anchor_pos)) =
        build_next_request(&prev_request, &output)
    {
        if let (Some(&prev_pos), true) =
            (prev_request.position_ids.last(), page_size > 0)
        {
            if let Some(pos_advance) = anchor_pos.checked_sub(prev_pos) {
                if pos_advance > 0 {
                    let lpl0 = cur_last_page_len.saturating_sub(1);
                    let total = lpl0.saturating_add(pos_advance);
                    let page_delta = total / page_size;
                    let next_lpl = (total % page_size) + 1;
                    let next_page_idx =
                        cur_page_idx.saturating_add(page_delta as usize);
                    let spec_tokens = next_req.spec_token_ids.len() as u32;
                    let writable_total = total.saturating_add(spec_tokens);
                    let writable_page_delta = writable_total / page_size;
                    let writable_page_idx =
                        cur_page_idx.saturating_add(writable_page_delta as usize);
                    if next_page_idx < all_pages.len()
                        && writable_page_idx < all_pages.len()
                    {
                        let next_pages: Vec<PhysicalPageId> =
                            all_pages[..=writable_page_idx].to_vec();
                        let queued = staged_batch_arc
                            .get(&ctx_id)
                            .map(|deque| deque.len())
                            .unwrap_or(0);
                        let should_extend = queued <= max_queue_depth;
                        if should_extend {
                            let (final_tx_next, final_rx_next) = oneshot::channel();
                            let next_allow_extend = Arc::new(AtomicBool::new(true));
                            let next_state = Box::new(ChainState {
                                response: final_tx_next,
                                scheduler_handle: scheduler_handle.clone(),
                                staged_batch_arc: staged_batch_arc.clone(),
                                prev_request: next_req.clone(),
                                all_pages,
                                cur_page_idx: next_page_idx,
                                cur_last_page_len: next_lpl,
                                max_queue_depth,
                                allow_extend: next_allow_extend.clone(),
                                page_size,
                            });
                            if scheduler_handle
                                .submit_chain(
                                    next_req.clone(),
                                    next_state,
                                    next_pages,
                                    next_lpl,
                                )
                                .is_ok()
                            {
                                CHAIN_SUBMIT_COUNT.fetch_add(1, Ordering::Relaxed);
                                staged_batch_arc
                                    .entry(ctx_id)
                                    .or_default()
                                    .push_back(StagedEntry {
                                        anchor_token,
                                        anchor_pos,
                                        spec_token_ids: next_req.spec_token_ids.clone(),
                                        spec_position_ids: next_req
                                            .spec_position_ids
                                            .clone(),
                                        output_spec_flags: next_req
                                            .output_spec_flags
                                            .clone(),
                                        uses_rs_cache: !next_req.rs_slot_ids.is_empty(),
                                        allow_extend: next_allow_extend,
                                        output_rx: final_rx_next,
                                    });
                            }
                        }
                    }
                }
            }
        }
    }

    let _ = response.send(Ok(output));

    let work_us = now_micros().saturating_sub(job_start_us);
    CHAIN_EXT_WORK_LATENCY_US.fetch_add(work_us, Ordering::Relaxed);
    CHAIN_EXT_JOBS_SAMPLED.fetch_add(1, Ordering::Relaxed);
}

// =============================================================================
// Eligibility + next-request builder
// =============================================================================

/// Why a request was deemed structurally ineligible for the
/// speculation chain. Surfaces in telemetry so operators can see
/// which cases their workload bumps into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    /// More than one sampling slot, or not at the last input pos.
    NotDecodeShape,
    /// Sampler isn't a token-producing variant (Dist / Logprobs / …).
    ProbeSampler,
    /// Request carries a custom attention mask.
    UserMaskPresent,
    /// Request carries a logit-postprocess mask.
    LogitMaskPresent,
}

/// Structural check the runtime applies before opening (or extending)
/// a speculation chain. `Ok(())` means the request is eligible.
///
/// Stochastic samplers (temperature > 0, no seed) are accepted: the
/// chain's pre-fired sample is itself a draw from the requested
/// distribution, which is exactly what the inferlet asked for.
pub fn evaluate_request_shape(req: &pie_bridge::ForwardRequest) -> Result<(), SkipReason> {
    use pie_bridge::Sampler;

    // Structural shape: exactly one sampling slot at the last input
    // position, with a sampler attached.
    if req.sampling_indices.len() != 1 || req.samplers.len() != 1 {
        return Err(SkipReason::NotDecodeShape);
    }
    let last_input_pos = req.position_ids.len().saturating_sub(1) as u32;
    if req.sampling_indices[0] != last_input_pos {
        return Err(SkipReason::NotDecodeShape);
    }

    // Probe variants emit non-Token outputs (distributions, logits,
    // etc.), so the next pass shape doesn't match the 1-token decode
    // we'd pre-fire.
    match &req.samplers[0] {
        Sampler::RawLogits
        | Sampler::Dist { .. }
        | Sampler::Logprob { .. }
        | Sampler::Logprobs { .. }
        | Sampler::Entropy
        | Sampler::Embedding => return Err(SkipReason::ProbeSampler),

        Sampler::Multinomial { .. }
        | Sampler::TopK { .. }
        | Sampler::TopP { .. }
        | Sampler::MinP { .. }
        | Sampler::TopKTopP { .. } => {}
    }

    if !req.logit_masks.is_empty() {
        return Err(SkipReason::LogitMaskPresent);
    }
    if req.has_user_mask {
        return Err(SkipReason::UserMaskPresent);
    }

    Ok(())
}

/// Build the next-cycle ForwardRequest from a just-completed
/// per-request request + response, ready to submit as a pre-staged
/// speculation fire. Returns `None` if the input isn't spec-eligible
/// (failed `evaluate_request_shape`) or the response has no first
/// token. On success, also returns the `(anchor_token, anchor_pos)`
/// pair used for fingerprint matching when the inferlet's actual
/// next call arrives.
///
/// The returned request:
///   - is a single-token decode at the position after all returned
///     tokens, with input = the last sampled token
///   - carries the same samplers / adapter as the prior call
///   - leaves masks empty so the scheduler can route it through the
///     single-token decode path
///   - carries forward system-speculative drafts returned by the prior
///     driver pass, so pass-level speculation remains orthogonal to
///     driver/system speculation.
pub fn build_next_request(
    prev_req: &pie_bridge::ForwardRequest,
    prev_resp: &ForwardOutput,
) -> Option<(pie_bridge::ForwardRequest, u32, u32)> {
    if evaluate_request_shape(prev_req).is_err() {
        return None;
    }
    let (sampled_token, pos_advance, spec_token_ids, spec_position_ids) = match prev_resp {
        ForwardOutput::Token(token) => (*token, 1u32, Vec::new(), Vec::new()),
        ForwardOutput::Tokens(tokens) => {
            let token = *tokens.last()?;
            let advance = u32::try_from(tokens.len()).ok()?;
            (token, advance, Vec::new(), Vec::new())
        }
        ForwardOutput::Response(resp) => {
            let token = *resp.tokens.last()?;
            let advance = u32::try_from(resp.tokens.len()).ok()?;
            let (spec_tokens, spec_positions) = if resp.spec_indptr.len() >= 2 {
                let lo = resp.spec_indptr[0] as usize;
                let hi = resp.spec_indptr[1] as usize;
                (
                    resp.spec_tokens.get(lo..hi).unwrap_or(&[]).to_vec(),
                    resp.spec_positions.get(lo..hi).unwrap_or(&[]).to_vec(),
                )
            } else {
                (resp.spec_tokens.clone(), resp.spec_positions.clone())
            };
            (token, advance, spec_tokens, spec_positions)
        }
    };
    let last_pos = *prev_req.position_ids.last()?;
    let next_pos = last_pos.checked_add(pos_advance)?;
    let context_id = *prev_req.context_ids.first()?;
    let output_spec_flags = if prev_req.output_spec_flags.is_empty() {
        vec![false]
    } else {
        prev_req.output_spec_flags.clone()
    };
    let rs_slot_flags = vec![0; prev_req.rs_slot_ids.len()];
    let next_req = pie_bridge::ForwardRequest {
        token_ids: vec![sampled_token],
        position_ids: vec![next_pos],
        kv_page_indices: Vec::new(),
        kv_page_indptr: vec![0],
        kv_last_page_lens: Vec::new(),
        qo_indptr: vec![0, 1],
        rs_slot_ids: prev_req.rs_slot_ids.clone(),
        rs_slot_flags,
        masks: Vec::new(),
        mask_indptr: vec![0, 0],
        logit_masks: Vec::new(),
        logit_mask_indptr: vec![0, 0],
        sampling_indices: vec![0],
        sampling_indptr: vec![0, 1],
        samplers: prev_req.samplers.clone(),
        sampler_indptr: vec![0, 1],
        adapter_bindings: prev_req.adapter_bindings.clone(),
        spec_indptr: vec![0, spec_token_ids.len() as u32],
        spec_token_ids,
        spec_position_ids,
        output_spec_flags,
        context_ids: vec![context_id],
        single_token_mode: true,
        has_user_mask: false,
    };
    Some((next_req, sampled_token, next_pos))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pie_bridge::Sampler;

    fn req_with(
        positions: Vec<u32>,
        sampling_indices: Vec<u32>,
        samplers: Vec<Sampler>,
    ) -> pie_bridge::ForwardRequest {
        let n = positions.len() as u32;
        let n_sampling = sampling_indices.len() as u32;
        let n_samplers = samplers.len() as u32;
        pie_bridge::ForwardRequest {
            token_ids: vec![0; positions.len()],
            position_ids: positions,
            kv_page_indices: Vec::new(),
            kv_page_indptr: vec![0],
            kv_last_page_lens: Vec::new(),
            qo_indptr: vec![0, n],
            rs_slot_ids: Vec::new(),
            rs_slot_flags: Vec::new(),
            masks: Vec::new(),
            mask_indptr: vec![0, 0],
            logit_masks: Vec::new(),
            logit_mask_indptr: vec![0, 0],
            sampling_indices,
            sampling_indptr: vec![0, n_sampling],
            samplers,
            sampler_indptr: vec![0, n_samplers],
            adapter_bindings: vec![pie_bridge::AdapterBinding {
                adapter_id: -1,
                seed: -1,
            }],
            spec_token_ids: Vec::new(),
            spec_position_ids: Vec::new(),
            spec_indptr: vec![0, 0],
            output_spec_flags: vec![false],
            context_ids: vec![0],
            single_token_mode: true,
            has_user_mask: false,
        }
    }

    fn argmax() -> Sampler {
        Sampler::Multinomial {
            temperature: 0.0,
            seed: 0,
        }
    }

    fn token_resp(token: u32) -> ForwardOutput {
        ForwardOutput::Token(token)
    }

    #[test]
    fn rule_accepts_argmax_decode() {
        let req = req_with(vec![10], vec![0], vec![argmax()]);
        assert_eq!(evaluate_request_shape(&req), Ok(()));
    }

    #[test]
    fn rule_accepts_stochastic_sampler() {
        // Temperature > 0 with no seed: the chain still pre-fires —
        // its sample is a valid draw from the requested distribution.
        let req = req_with(
            vec![10],
            vec![0],
            vec![Sampler::Multinomial {
                temperature: 0.7,
                seed: 0,
            }],
        );
        assert_eq!(evaluate_request_shape(&req), Ok(()));
    }

    #[test]
    fn rule_rejects_probe_sampler() {
        let req = req_with(vec![10], vec![0], vec![Sampler::Entropy]);
        assert_eq!(evaluate_request_shape(&req), Err(SkipReason::ProbeSampler));
    }

    #[test]
    fn rule_accepts_system_spec_request() {
        let mut req = req_with(vec![10], vec![0], vec![argmax()]);
        req.output_spec_flags = vec![true];
        req.spec_token_ids = vec![11, 12];
        req.spec_position_ids = vec![11, 12];
        req.spec_indptr = vec![0, 2];
        assert_eq!(evaluate_request_shape(&req), Ok(()));
    }

    #[test]
    fn rule_rejects_multi_slot() {
        let req = req_with(vec![10, 11], vec![0, 1], vec![argmax(), argmax()]);
        assert_eq!(
            evaluate_request_shape(&req),
            Err(SkipReason::NotDecodeShape)
        );
    }

    #[test]
    fn build_next_request_decode() {
        let req = req_with(vec![10], vec![0], vec![argmax()]);
        let resp = token_resp(42);
        let (next, anchor_token, anchor_pos) = build_next_request(&req, &resp).expect("eligible");
        assert_eq!(anchor_token, 42);
        assert_eq!(anchor_pos, 11);
        assert_eq!(next.token_ids, vec![42]);
        assert_eq!(next.position_ids, vec![11]);
        assert!(next.masks.is_empty());
        assert_eq!(next.mask_indptr, vec![0, 0]);
    }

    #[test]
    fn build_next_request_carries_system_spec_drafts() {
        let mut req = req_with(vec![10], vec![0], vec![argmax()]);
        req.output_spec_flags = vec![true];
        req.rs_slot_ids = vec![7];
        req.rs_slot_flags = vec![1];
        let resp = ForwardOutput::from_response(pie_bridge::ForwardResponse {
            num_requests: 1,
            tokens: vec![42, 43, 44],
            tokens_indptr: vec![0, 3],
            spec_tokens: vec![45, 46],
            spec_positions: vec![14, 15],
            spec_indptr: vec![0, 2],
            ..Default::default()
        });
        let (next, anchor_token, anchor_pos) = build_next_request(&req, &resp).expect("eligible");
        assert_eq!(anchor_token, 44);
        assert_eq!(anchor_pos, 13);
        assert_eq!(next.token_ids, vec![44]);
        assert_eq!(next.position_ids, vec![13]);
        assert_eq!(next.spec_token_ids, vec![45, 46]);
        assert_eq!(next.spec_position_ids, vec![14, 15]);
        assert_eq!(next.spec_indptr, vec![0, 2]);
        assert_eq!(next.output_spec_flags, vec![true]);
        assert_eq!(next.rs_slot_ids, vec![7]);
        assert_eq!(next.rs_slot_flags, vec![0]);
    }

    #[test]
    fn build_next_request_none_on_empty_tokens() {
        let req = req_with(vec![10], vec![0], vec![argmax()]);
        let resp = ForwardOutput::from_response(pie_bridge::ForwardResponse {
            num_requests: 1,
            ..Default::default()
        });
        assert!(build_next_request(&req, &resp).is_none());
    }
}
