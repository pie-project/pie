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
//!     (lock-free). On match it claims the staged entry's rx and
//!     skips the actor mailbox round-trip entirely. On miss/cold
//!     it falls through to a normal submit through the actor.
//!
//! This module owns the chain state. The scheduler is unaware of
//! speculation — it sees normal `submit()` calls. The driver is
//! unaware of speculation — it just runs forward passes.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use tokio::sync::oneshot;

use crate::context::ContextId;
use crate::context::pagestore::PhysicalPageId;
use crate::inference::scheduler::SchedulerHandle;

/// A pre-fired forward pass for a ctx, sitting in the per-ctx
/// chain queue waiting for the inferlet's matching `execute()` call.
///
/// Anchor = the (token, position) the inferlet's actual request
/// must match for this entry to count as a hit. Built when the
/// chain extender constructed the entry.
pub(crate) struct StagedEntry {
    pub anchor_token: u32,
    pub anchor_pos: u32,
    /// Future fire's output for this ctx. The scheduler holds the
    /// matching `Sender`; when the kernel finishes and the output
    /// is delivered, this receiver resolves.
    pub output_rx: oneshot::Receiver<pie_bridge::ForwardResponse>,
}

/// Per-model speculator state.
struct ModelEntry {
    /// Per-context depth of pass-level speculation
    /// (`scheduler.speculation_depth` in toml). `0` disables
    /// speculation for this model.
    speculation_depth: usize,
    /// One staged-batch map per device on this model.
    devices: Vec<Arc<Mutex<HashMap<ContextId, VecDeque<StagedEntry>>>>>,
}

/// Per-model registry. The lock-free `try_hit` accesses this
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
    staged_batch: &[Arc<Mutex<HashMap<ContextId, VecDeque<StagedEntry>>>>],
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

/// `true` when this model has speculation enabled
/// (`scheduler.speculation_depth > 0`).
fn is_spec_enabled(model_idx: usize) -> bool {
    REGISTRY
        .lock()
        .ok()
        .and_then(|reg| reg.get(model_idx).map(|m| m.speculation_depth > 0))
        .unwrap_or(false)
}

/// Opaque per-(model, device) handle the api layer caches on the
/// ctx side. Lets `try_hit` skip the REGISTRY lookup on every
/// `execute()` — the lookup happens once when the ctx is first
/// bound to a `ForwardPass`, and the resulting arc is reused.
#[derive(Clone)]
pub struct StagedBatch(Arc<Mutex<HashMap<ContextId, VecDeque<StagedEntry>>>>);

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

/// Lock-free hit check from the api layer, using a cached
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
) -> Option<oneshot::Receiver<pie_bridge::ForwardResponse>> {
    let mut sb = spec.0.lock().ok()?;
    let deque = sb.get_mut(&ctx_id)?;
    let front = deque.front()?;
    let req_token = request.token_ids.first().copied();
    let req_pos = request.position_ids.first().copied();
    if Some(front.anchor_token) == req_token && Some(front.anchor_pos) == req_pos {
        let entry = deque.pop_front()?;
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
        if let Ok(mut sb) = sb_arc.lock() {
            sb.remove(&ctx_id);
        }
    }
}

/// Spawn a post-fire chain extender for a ctx. Awaits the
/// just-submitted request's output, then builds the next stage
/// (if eligible and within `max_queue_depth`), submits it to the
/// scheduler, pushes a `StagedEntry` to `staged_batch_arc`, and
/// recursively spawns the next extender so the chain continues
/// as outputs arrive.
///
/// The `response` channel is the inferlet's response channel for
/// the CURRENT call (the one that's about to receive `sched_rx`'s
/// output). Each subsequent stage's `final_tx` lives in its
/// StagedEntry, awaiting the next inferlet hit.
///
/// `all_pages` is the full list of physical pages the ctx had at
/// cold-submit time. The chain extender uses the prefix
/// `all_pages[..=cur_page_idx]`: when `cur_page_idx`'s page fills,
/// it advances to the next page in the list. When all pages are
/// full, the chain terminates and the next inferlet submit will
/// re-pin (allocating fresh pages).
///
/// Termination conditions:
///   - Output's leading slot isn't a `Token` (e.g., probe sampler)
///   - Request fails `evaluate_request_shape` (non-deterministic
///     sampler, custom mask, etc.)
///   - The ctx has run out of pre-allocated pages
///   - Ctx's deque is already at `max_queue_depth` entries
///   - `sched_rx` errored (scheduler dropped the channel)
pub(crate) fn spawn_extend_chain(
    sched_rx: oneshot::Receiver<pie_bridge::ForwardResponse>,
    response: oneshot::Sender<pie_bridge::ForwardResponse>,
    scheduler_handle: SchedulerHandle,
    staged_batch_arc: Arc<Mutex<HashMap<ContextId, VecDeque<StagedEntry>>>>,
    model_idx: usize,
    prev_request: pie_bridge::ForwardRequest,
    all_pages: Vec<PhysicalPageId>,
    cur_page_idx: usize,
    cur_last_page_len: u32,
    max_queue_depth: usize,
) {
    tokio::spawn(async move {
        let output = match sched_rx.await {
            Ok(o) => o,
            Err(_) => return,
        };
        let ctx_id = match prev_request.context_ids.first() {
            Some(&id) => id,
            None => return,
        };

        // Orphan-stage gate. If the receiver of our `response`
        // channel is gone, this stage's StagedEntry was dropped
        // before any inferlet could hit it — either because the
        // ctx was destroyed (`invalidate_ctx` emptied the deque
        // mid-flight) or because the cold-submit caller cancelled.
        // Continuing to extend past this point produces orphan
        // stages that fire on the GPU but no inferlet claims,
        // inflating batch sizes. Bail out before doing any further
        // chain work.
        if response.is_closed() {
            return;
        }

        // Kill switch: forward the cold-submit output and skip
        // pushing/recursing. The next inferlet submit will miss
        // try_hit (deque empty), go through the actor, and land
        // here again — same forwarding, no chain ever forms.
        if !is_spec_enabled(model_idx) {
            let _ = response.send(output);
            return;
        }

        if let Some((next_req, anchor_token, anchor_pos)) =
            build_next_request(&prev_request, &output)
        {
            let page_size = crate::context::tokens_per_page(model_idx);
            // Advance one write slot: either grow within the current
            // page, or roll over to the next pre-allocated page.
            let (next_page_idx, next_lpl) = if cur_last_page_len + 1 <= page_size {
                (cur_page_idx, cur_last_page_len + 1)
            } else {
                (cur_page_idx + 1, 1)
            };
            if next_page_idx < all_pages.len() {
                let next_pages: Vec<PhysicalPageId> = all_pages[..=next_page_idx].to_vec();
                let should_extend = match staged_batch_arc.lock() {
                    Ok(sb) => sb.get(&ctx_id).map_or(0, |d| d.len()) < max_queue_depth,
                    Err(_) => false,
                };
                if should_extend {
                    let (sched_tx_next, sched_rx_next) = oneshot::channel();
                    let (final_tx_next, final_rx_next) = oneshot::channel();
                    if scheduler_handle
                        .submit(next_req.clone(), sched_tx_next, next_pages, next_lpl)
                        .is_ok()
                    {
                        CHAIN_SUBMIT_COUNT.fetch_add(1, Ordering::Relaxed);
                        if let Ok(mut sb) = staged_batch_arc.lock() {
                            sb.entry(ctx_id).or_default().push_back(StagedEntry {
                                anchor_token,
                                anchor_pos,
                                output_rx: final_rx_next,
                            });
                        }
                        spawn_extend_chain(
                            sched_rx_next,
                            final_tx_next,
                            scheduler_handle.clone(),
                            staged_batch_arc.clone(),
                            model_idx,
                            next_req,
                            all_pages,
                            next_page_idx,
                            next_lpl,
                            max_queue_depth,
                        );
                    }
                }
            }
        }

        // Forward this stage's output to its claimant (inferlet
        // or the upstream forwarder spawned by `try_hit`).
        let _ = response.send(output);
    });
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
///   - is a single-token decode at `last_pos + 1` with input =
///     the sampled token
///   - carries the same samplers / adapter as the prior call
///   - has no custom masks (kernel synthesizes causal)
///   - has no speculative drafts (those are a property of the
///     specific request, not the chain — propagating them would
///     amount to predicting the inferlet's draft strategy)
pub fn build_next_request(
    prev_req: &pie_bridge::ForwardRequest,
    prev_resp: &pie_bridge::ForwardResponse,
) -> Option<(pie_bridge::ForwardRequest, u32, u32)> {
    if evaluate_request_shape(prev_req).is_err() {
        return None;
    }
    let sampled_token = *prev_resp.tokens.first()?;
    let last_pos = *prev_req.position_ids.last()?;
    let next_pos = last_pos.checked_add(1)?;
    let context_id = *prev_req.context_ids.first()?;
    let next_req = pie_bridge::ForwardRequest {
        token_ids: vec![sampled_token],
        position_ids: vec![next_pos],
        kv_page_indices: Vec::new(),
        kv_page_indptr: vec![0],
        kv_last_page_lens: Vec::new(),
        qo_indptr: vec![0, 1],
        masks: Vec::new(),
        mask_indptr: vec![0, 0],
        logit_masks: Vec::new(),
        logit_mask_indptr: vec![0, 0],
        sampling_indices: vec![0],
        sampling_indptr: vec![0, 1],
        samplers: prev_req.samplers.clone(),
        sampler_indptr: vec![0, 1],
        adapter_bindings: prev_req.adapter_bindings.clone(),
        spec_token_ids: Vec::new(),
        spec_position_ids: Vec::new(),
        spec_indptr: vec![0, 0],
        output_spec_flags: vec![false],
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

    fn token_resp(token: u32) -> pie_bridge::ForwardResponse {
        pie_bridge::ForwardResponse {
            num_requests: 1,
            tokens: vec![token],
            tokens_indptr: vec![0, 1],
            ..Default::default()
        }
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
    }

    #[test]
    fn build_next_request_none_on_empty_tokens() {
        let req = req_with(vec![10], vec![0], vec![argmax()]);
        let resp = pie_bridge::ForwardResponse {
            num_requests: 1,
            ..Default::default()
        };
        assert!(build_next_request(&req, &resp).is_none());
    }
}
