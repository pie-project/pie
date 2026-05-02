//! Per-device batch scheduler.
//!
//! Each `BatchScheduler` owns its own RPC client, scheduling policy,
//! and tokio task. It accepts pre-translated forward pass requests,
//! accumulates them into batches, and fires them based on adaptive
//! scheduling decisions.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};

use anyhow::Result;
use tokio::sync::{mpsc, oneshot, Semaphore};


use crate::device::DeviceId;
use crate::context::pagestore::PhysicalPageId;

use crate::device;

use super::adaptive_policy::{AdaptivePolicy, EagerPolicy, GreedyPolicy};
use super::request::{
    BatchedForwardPassRequest, BatchedForwardPassResponse, ForwardPassOutput, ForwardPassRequest,
    ForwardPassResponse, Sampler, SlotOutput,
};

/// Build the per-slot output list from the worker's parallel-fields response.
///
/// Walks the request's samplers in slot order, pulling one item from the
/// matching response field per slot. This preserves the 1:1 mapping between
/// `pass.sampler(...)` calls and returned slots — even when sampler types
/// are mixed (e.g. multinomial + entropy on the same position).
///
/// Spec-mode requests are detected via a token-count mismatch (the verifier
/// produces a token sequence whose length is unrelated to the inferlet's
/// sampler count); in that case all slots collapse to `Token` entries.
fn build_slot_output(samplers: &[Sampler], resp: ForwardPassResponse) -> ForwardPassOutput {
    let spec_tokens = resp.spec_tokens;
    let spec_positions = resp.spec_positions;

    let expected_token_slots = samplers
        .iter()
        .filter(|s| {
            matches!(
                s,
                Sampler::Multinomial { .. }
                    | Sampler::TopK { .. }
                    | Sampler::TopP { .. }
                    | Sampler::MinP { .. }
                    | Sampler::TopKTopP { .. }
            )
        })
        .count();

    let is_spec_walk = !resp.tokens.is_empty()
        && resp.tokens.len() != expected_token_slots
        && resp.dists.is_empty()
        && resp.logits.is_empty()
        && resp.logprobs.is_empty()
        && resp.entropies.is_empty();

    if is_spec_walk {
        let slots = resp.tokens.into_iter().map(SlotOutput::Token).collect();
        return ForwardPassOutput { slots, spec_tokens, spec_positions };
    }

    let mut tok_iter = resp.tokens.into_iter();
    let mut dist_iter = resp.dists.into_iter();
    let mut logit_iter = resp.logits.into_iter();
    let mut lp_iter = resp.logprobs.into_iter();
    let mut ent_iter = resp.entropies.into_iter();

    let slots: Vec<SlotOutput> = samplers
        .iter()
        .filter_map(|s| match s {
            Sampler::Multinomial { .. }
            | Sampler::TopK { .. }
            | Sampler::TopP { .. }
            | Sampler::MinP { .. }
            | Sampler::TopKTopP { .. } => tok_iter.next().map(SlotOutput::Token),
            Sampler::Dist { .. } => dist_iter
                .next()
                .map(|(ids, ps)| SlotOutput::Distribution(ids, ps)),
            Sampler::RawLogits => logit_iter.next().map(SlotOutput::Logits),
            Sampler::Logprob { .. } | Sampler::Logprobs { .. } => {
                lp_iter.next().map(SlotOutput::Logprobs)
            }
            Sampler::Entropy => ent_iter.next().map(SlotOutput::Entropy),
            // Embedding is reserved but not currently produced by the worker.
            Sampler::Embedding => None,
        })
        .collect();

    ForwardPassOutput { slots, spec_tokens, spec_positions }
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
    /// the forward pass took on the device.
    fn on_complete(&mut self, latency: Duration);

    /// The current batch was fired.
    fn on_fired(&mut self);

    /// Decide whether to fire or wait, given the current batch size.
    fn decide(&self, current_batch_size: usize) -> Decision;
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
    pub last_batch_latency_us: AtomicU64,
    pub cumulative_latency_us: AtomicU64,

}

// =============================================================================
// PendingRequest
// =============================================================================

/// A forward pass request bundled with its response channel and physical pages.
struct PendingRequest {
    request: ForwardPassRequest,
    response_tx: oneshot::Sender<ForwardPassOutput>,
    physical_page_ids: Vec<PhysicalPageId>,
    last_page_len: u32,
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
    max_batch_size: usize,
    max_batch_tokens: usize,
}

impl BatchAccumulator {
    fn new(max_batch_size: usize, max_batch_tokens: usize) -> Self {
        Self {
            requests: Vec::new(),
            total_tokens: 0,
            max_batch_size,
            max_batch_tokens,
        }
    }

    /// Push a request, rejecting when adding would exceed
    /// `max_batch_tokens` and the batch is non-empty.
    ///
    /// Returns `Err(req)` so the caller can stash the rejected request
    /// as the seed of the next batch (after firing the current one).
    /// Empty-batch pushes always succeed: a single oversize prefill
    /// can't be made smaller than itself, so we let it through and let
    /// the device path surface the error per pie #322. True
    /// chunked-prefill (splitting a long prompt across forward passes)
    /// is a separate, larger change.
    fn try_push(&mut self, req: PendingRequest) -> Result<(), PendingRequest> {
        let req_tokens = req.request.tokens.len();
        if !self.requests.is_empty()
            && self
                .total_tokens
                .saturating_add(req_tokens)
                > self.max_batch_tokens
        {
            return Err(req);
        }
        self.total_tokens += req_tokens;
        self.requests.push(req);
        Ok(())
    }

    fn is_full(&self) -> bool {
        self.requests.len() >= self.max_batch_size
            || self.total_tokens >= self.max_batch_tokens
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

    fn take(&mut self) -> Vec<PendingRequest> {
        self.total_tokens = 0;
        std::mem::take(&mut self.requests)
    }
}

// =============================================================================
// BatchScheduler
// =============================================================================

/// Per-device batch scheduler.
///
/// Owns an RPC client, a scheduling policy, and a tokio task that
/// runs the batch accumulation and firing loop.
pub(crate) struct BatchScheduler {
    tx: mpsc::UnboundedSender<PendingRequest>,
    stats: Arc<SchedulerStats>,
}

impl BatchScheduler {
    /// Spawn a new batch scheduler for a single device.
    ///
    /// The RPC connection is owned by the device service; the scheduler
    /// only stores the device index for routing calls.
    pub fn new(
        device_id: DeviceId,
        device_idx: usize,
        page_size: u32,
        max_batch_size: usize,
        max_batch_tokens: usize,
        request_timeout_secs: u64,
        policy: Option<String>,
    ) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let stats = Arc::new(SchedulerStats::default());
        tokio::spawn(Self::run(
            device_id, device_idx, rx,
            page_size,
            max_batch_size, max_batch_tokens,
            request_timeout_secs,
            policy,
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
        request: ForwardPassRequest,
        response_tx: oneshot::Sender<ForwardPassOutput>,
        physical_page_ids: Vec<PhysicalPageId>,
        last_page_len: u32,
    ) -> Result<()> {
        self.tx.send(PendingRequest {
            request,
            response_tx,
            physical_page_ids,
            last_page_len,
        })?;
        Ok(())
    }

    // =========================================================================
    // Internal: Scheduling Loop
    // =========================================================================

    /// Main scheduling loop for a single device.
    async fn run(
        device_id: DeviceId,
        device_idx: usize,
        mut req_rx: mpsc::UnboundedReceiver<PendingRequest>,
        page_size: u32,
        max_batch_size: usize,
        max_batch_tokens: usize,
        request_timeout_secs: u64,
        policy_from_config: Option<String>,
        stats: Arc<SchedulerStats>,
    ) {
        let request_timeout = Duration::from_secs(request_timeout_secs);

        // Per-device state
        let mut batch = BatchAccumulator::new(max_batch_size, max_batch_tokens);
        // Policy selection from config — see `adaptive_policy.rs` for the
        // design rationale. The per-model `[model.X.scheduler.policy]`
        // setting picks one of "adaptive" (default), "eager", "greedy".
        let policy_name = policy_from_config.unwrap_or_else(|| "adaptive".to_string());
        let mut policy: Box<dyn SchedulingPolicy> = match policy_name.as_str() {
            "greedy" => Box::new(GreedyPolicy::new()),
            "eager" => Box::new(EagerPolicy::new(max_batch_size, device_idx)),
            "adaptive" => Box::new(AdaptivePolicy::new(max_batch_size, device_idx)),
            other => panic!(
                "Unknown scheduler.policy {other:?}; expected one of \
                'adaptive' | 'eager' | 'greedy'"
            ),
        };
        // Only one in-flight batch at a time to prevent pipelined KV cache corruption.
        let in_flight = Arc::new(Semaphore::new(1));

        // Channel for batch completion latency feedback to the policy.
        let (latency_tx, mut latency_rx) = mpsc::unbounded_channel::<Duration>();

        // Stash for a request that didn't fit in the current batch's
        // token budget. Seeded back into the next batch as soon as the
        // current one fires, so we never lose admission order.
        let mut held_for_next_batch: Option<PendingRequest> = None;

        loop {
            // Drain completed batch latencies (non-blocking)
            while let Ok(latency) = latency_rx.try_recv() {
                policy.on_complete(latency);
            }

            // Wait for first request if batch is empty. If a request
            // was held back from the previous batch (token budget
            // overflow), seed with it instead of blocking on recv.
            if batch.is_empty() {
                if let Some(held) = held_for_next_batch.take() {
                    // try_push on an empty batch always succeeds — the
                    // overflow check only fires when batch is non-empty.
                    // Use `.expect` (NOT debug_assert!) so the call still
                    // executes in release builds; debug_assert! discards
                    // the entire expression under optimization.
                    batch
                        .try_push(held)
                        .ok()
                        .expect("seed try_push on empty batch must succeed");
                } else {
                    let Some(pending) = req_rx.recv().await else {
                        break;
                    };
                    policy.on_arrival();
                    batch
                        .try_push(pending)
                        .ok()
                        .expect("seed try_push on empty batch must succeed");
                }
            }

            // Accumulate more requests (non-blocking). Stop on a
            // token-budget overflow and stash the rejected request to
            // seed the next batch — the in-progress accumulator is
            // about to fire (forced below), so the held request will
            // be re-admitted on the next loop iteration.
            while let Ok(pending) = req_rx.try_recv() {
                policy.on_arrival();
                if let Err(rejected) = batch.try_push(pending) {
                    held_for_next_batch = Some(rejected);
                    break;
                }
                if batch.is_full() {
                    break;
                }
            }

            // A held overflow forces a fire — waiting can't shrink the
            // already-accumulated batch's token count.
            let force_fire = held_for_next_batch.is_some();
            let decision = if force_fire {
                Decision::Fire
            } else {
                policy.decide(batch.len())
            };
            match decision {
                Decision::Fire => {
                    // Acquire a permit (may wait if at in-flight limit)
                    // if in_flight.available_permits() == 0 {
                    //     eprintln!("[SCHED dev={device_idx}] semaphore full, waiting for in-flight batch to complete");
                    // }
                    let permit = in_flight
                        .clone()
                        .acquire_owned()
                        .await
                        .expect("semaphore closed");

                    let total_tokens = batch.total_tokens();
                    let requests_to_fire = batch.take();
                    policy.on_fired();

                    // Collect batch context IDs for accurate rent charging.
                    let batch_ctx_ids: Vec<u64> = requests_to_fire.iter()
                        .map(|r| r.request.context_id)
                        .collect();

                    // Spawn batch execution
                    let latency_tx_clone = latency_tx.clone();
                    let stats_clone = stats.clone();
                    let timeout = request_timeout;

                    tokio::spawn(async move {
                        let start = Instant::now();
                        Self::execute_batch(
                            device_idx,
                            requests_to_fire,
                            device_id,
                            page_size,
                            timeout,
                        )
                        .await;
                        let latency = start.elapsed();

                        // Advance market clock for this device: prices, rent, dividends.
                        // Pass batch context IDs so tick only charges contexts
                        // that were in this batch (not stale pinned contexts).
                        crate::context::tick(device_idx, latency.as_secs_f64(), batch_ctx_ids);

                        // Update cumulative atomic counters (consumed by external
                        // monitoring; ignored by the policy).
                        stats_clone.total_batches.fetch_add(1, Relaxed);
                        stats_clone.total_tokens_processed.fetch_add(total_tokens as u64, Relaxed);
                        stats_clone.last_batch_latency_us.store(latency.as_micros() as u64, Relaxed);
                        stats_clone.cumulative_latency_us.fetch_add(latency.as_micros() as u64, Relaxed);

                        latency_tx_clone.send(latency).ok();
                        drop(permit); // release in-flight slot
                    });
                }
                Decision::Wait(wait_duration) => {
                    tokio::select! {
                        _ = tokio::time::sleep(wait_duration) => {}
                        maybe_req = req_rx.recv() => {
                            if let Some(pending) = maybe_req {
                                policy.on_arrival();
                                if let Err(rejected) = batch.try_push(pending) {
                                    // Hold for the next batch; the
                                    // outer loop will see force_fire
                                    // on its next iteration.
                                    held_for_next_batch = Some(rejected);
                                }
                            } else {
                                break; // channel closed
                            }
                        }
                        latency = latency_rx.recv() => {
                            if let Some(l) = latency {
                                policy.on_complete(l);
                            }
                        }
                    }
                }
            }
        }

        // Shutdown: fire remaining batch
        if !batch.is_empty() {
            let requests = batch.take();
            Self::execute_batch(
                device_idx,
                requests,
                device_id,
                page_size,
                request_timeout,
            )
            .await;
        }
    }

    /// Execute a batch of forward pass requests via the device service.
    async fn execute_batch(
        device_idx: usize,
        requests: Vec<PendingRequest>,
        device_id: DeviceId,
        page_size: u32,
        timeout: Duration,
    ) {
        // Build batched request
        let mut batch_req = BatchedForwardPassRequest::new(device_id);
        for req in &requests {
            batch_req.add_request(
                &req.request,
                &req.physical_page_ids,
                req.last_page_len,
                page_size,
            );
        }

        // Send via device service (typed call handles serialization + timeout)
        let result = device::fire_batch(device_idx, &batch_req).await;

        match result {
            Ok(batch_resp) => {
                if batch_resp.results.len() != requests.len() {
                    tracing::warn!(
                        device = device_id,
                        expected = requests.len(),
                        got = batch_resp.results.len(),
                        "Batch response count mismatch — some requests may get no output",
                    );
                }

                let mut resp_iter = batch_resp.results.into_iter();
                for req in requests {
                    if let Some(resp) = resp_iter.next() {
                        // Build the per-slot output list by walking the
                        // request's sampler types in order and pulling from
                        // the matching response field. This preserves the
                        // 1:1 mapping between `pass.sampler(...)` calls and
                        // returned slots — even when types are mixed.
                        let output = build_slot_output(&req.request.samplers, resp);
                        if output.slots.is_empty()
                            && output.spec_tokens.is_empty()
                            && !req.request.sampling_indices.is_empty()
                        {
                            eprintln!(
                                "FP_NONE_FOR_DECODE ctx={} samplers={} tokens={} pages={} lpl={}",
                                req.request.context_id,
                                req.request.sampling_indices.len(),
                                req.request.tokens.len(),
                                req.physical_page_ids.len(),
                                req.last_page_len,
                            );
                        }
                        req.response_tx.send(output).ok();
                    } else {
                        tracing::warn!(device = device_id, "Fewer results than requests — sending None");
                        req.response_tx.send(ForwardPassOutput::default()).ok();
                    }
                }
            }
            Err(e) => {
                tracing::error!("fire_batch failed for device {}: {:?}", device_id, e);
                for req in requests {
                    req.response_tx.send(ForwardPassOutput::default()).ok();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pending(n_tokens: usize) -> PendingRequest {
        let (tx, _rx) = oneshot::channel();
        PendingRequest {
            request: ForwardPassRequest {
                context_id: 0,
                tokens: vec![0; n_tokens],
                positions: (0..n_tokens as u32).collect(),
                speculative_tokens: vec![],
                speculative_positions: vec![],
                output_speculative_tokens: false,
                masks: vec![],
                has_user_mask: false,
                logit_mask: None,
                sampling_indices: vec![],
                samplers: vec![],
                adapter_id: None,
                adapter_seed: None,
            },
            response_tx: tx,
            physical_page_ids: vec![],
            last_page_len: 0,
        }
    }

    /// Helper: assert a try_push succeeded (PendingRequest doesn't
    /// impl Debug, so we can't use `.unwrap()` directly).
    fn must_push(acc: &mut BatchAccumulator, req: PendingRequest) {
        assert!(acc.try_push(req).is_ok(), "try_push expected to accept");
    }

    #[test]
    fn try_push_accepts_when_under_token_budget() {
        // Two 100-token requests fit in a 256-token batch.
        let mut acc = BatchAccumulator::new(/*max_size=*/ 8, /*max_tokens=*/ 256);
        must_push(&mut acc, make_pending(100));
        must_push(&mut acc, make_pending(100));
        assert_eq!(acc.len(), 2);
        assert_eq!(acc.total_tokens(), 200);
    }

    #[test]
    fn try_push_rejects_when_overflow_and_batch_non_empty() {
        // 200 tokens already; a 100-token request would overshoot 256.
        // Reject and return the request to the caller.
        let mut acc = BatchAccumulator::new(8, 256);
        must_push(&mut acc, make_pending(200));
        let oversize = make_pending(100);
        let oversize_tokens = oversize.request.tokens.len();
        match acc.try_push(oversize) {
            Ok(()) => panic!("expected rejection"),
            Err(rejected) => {
                assert_eq!(rejected.request.tokens.len(), oversize_tokens);
            }
        }
        assert_eq!(acc.total_tokens(), 200, "rejected request must not bump counter");
        assert_eq!(acc.len(), 1, "rejected request must not be added");
    }

    #[test]
    fn try_push_accepts_oversize_seed_when_batch_empty() {
        // A single request larger than max_batch_tokens still seeds an
        // empty batch — pie has no chunked prefill, so the alternative
        // (silent drop) is worse than letting the device path surface
        // the error per pie #322.
        let mut acc = BatchAccumulator::new(8, 256);
        must_push(&mut acc, make_pending(1024));
        assert_eq!(acc.len(), 1);
        assert!(acc.is_full(), "oversize seed should mark batch full");
    }

    #[test]
    fn try_push_rejects_at_exact_overflow_boundary() {
        // Boundary: 200 + 57 = 257 > 256 → reject.
        // Boundary: 200 + 56 = 256 = 256 → accept (=, not >).
        let mut acc = BatchAccumulator::new(8, 256);
        must_push(&mut acc, make_pending(200));
        assert!(acc.try_push(make_pending(57)).is_err(), "257 > 256 should reject");
        must_push(&mut acc, make_pending(56));
        assert_eq!(acc.total_tokens(), 256);
    }

    #[test]
    fn is_full_fires_on_size_or_token_limit() {
        // Size limit only.
        let mut acc = BatchAccumulator::new(2, 1024);
        must_push(&mut acc, make_pending(10));
        assert!(!acc.is_full());
        must_push(&mut acc, make_pending(10));
        assert!(acc.is_full(), "len=2 hits max_batch_size=2");

        // Token limit only.
        let mut acc = BatchAccumulator::new(8, 256);
        must_push(&mut acc, make_pending(256));
        assert!(acc.is_full(), "total_tokens=max_batch_tokens hits limit");
    }
}
