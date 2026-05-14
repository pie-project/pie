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

use crate::device;
use crate::context::pagestore::PhysicalPageId;

use super::adaptive_policy;
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
    /// A request was added to the accumulator. `ctx_id` identifies
    /// the context that produced this request; policies that don't
    /// track per-context state may ignore it.
    fn on_arrival(&mut self, ctx_id: u64);

    /// A batch finished executing. `latency` is the wall-clock time
    /// the forward pass took on the device.
    fn on_complete(&mut self, latency: Duration);

    /// The current batch was fired. `fired_ctx_ids` lists every
    /// context that contributed an entry to the batch; policies that
    /// track per-context state (e.g., hot-aware) use this to know
    /// which contexts are expected to re-queue imminently.
    fn on_fired(&mut self, fired_ctx_ids: &[u64]);

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
    /// Total request count across all batches (sum of batch sizes).
    /// Divide by `total_batches` for mean batch size in requests.
    pub total_requests_processed: AtomicU64,
    /// Largest batch size (in requests) ever fired by this scheduler.
    pub max_batch_size_observed: AtomicU64,
    /// Coarse histogram of batch sizes. Buckets:
    /// [0]=1, [1]=2-3, [2]=4-7, [3]=8-15, [4]=16-31,
    /// [5]=32-63, [6]=64-127, [7]=128+.
    pub batch_size_hist: [AtomicU64; 8],
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

    fn push(&mut self, req: PendingRequest) {
        self.total_tokens += req.request.tokens.len();
        self.requests.push(req);
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
/// Clonable submit handle for use from spawned tasks (e.g., the
/// post-fire chain extender).
#[derive(Clone)]
pub(crate) struct SchedulerHandle {
    tx: mpsc::UnboundedSender<PendingRequest>,
}

impl SchedulerHandle {
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
}

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
        model_idx: usize,
        device_idx: usize,
        page_size: u32,
        max_batch_size: usize,
        max_batch_tokens: usize,
        request_timeout_secs: u64,
        batch_policy: String,
    ) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let stats = Arc::new(SchedulerStats::default());
        tokio::spawn(Self::run(
            model_idx, device_idx, rx,
            page_size,
            max_batch_size, max_batch_tokens,
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

    /// Clonable handle that can submit from spawned tasks
    /// (e.g., the post-fire chain extender).
    pub(crate) fn handle(&self) -> SchedulerHandle {
        SchedulerHandle { tx: self.tx.clone() }
    }

    // =========================================================================
    // Internal: Scheduling Loop
    // =========================================================================

    /// Main scheduling loop for a single device.
    async fn run(
        _model_idx: usize,
        device_idx: usize,
        mut req_rx: mpsc::UnboundedReceiver<PendingRequest>,
        page_size: u32,
        max_batch_size: usize,
        max_batch_tokens: usize,
        _request_timeout_secs: u64,
        batch_policy: String,
        stats: Arc<SchedulerStats>,
    ) {
        let mut batch = BatchAccumulator::new(max_batch_size, max_batch_tokens);
        let hot_window_ms: u64 = std::env::var("PIE_HOT_WINDOW_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(5);
        let mut policy: Box<dyn SchedulingPolicy> = adaptive_policy::make_policy(
            &batch_policy,
            max_batch_size,
            device_idx,
            Duration::from_millis(hot_window_ms),
        );
        // Only one in-flight batch at a time to prevent pipelined KV cache corruption.
        let in_flight = Arc::new(Semaphore::new(1));

        // Channel for batch completion latency feedback to the policy.
        let (latency_tx, mut latency_rx) = mpsc::unbounded_channel::<Duration>();

        loop {
            // Drain completed batch latencies (non-blocking)
            while let Ok(latency) = latency_rx.try_recv() {
                policy.on_complete(latency);
            }

            // Wait for first request if batch is empty
            if batch.is_empty() {
                let Some(pending) = req_rx.recv().await else {
                    break;
                };
                policy.on_arrival(pending.request.context_id);
                batch.push(pending);
            }

            // Accumulate more requests (non-blocking)
            while let Ok(pending) = req_rx.try_recv() {
                policy.on_arrival(pending.request.context_id);
                batch.push(pending);
                if batch.is_full() {
                    break;
                }
            }

            // Ask the policy what to do
            match policy.decide(batch.len()) {
                Decision::Fire => {
                    // Acquire a permit (may wait if at in-flight limit)
                    let permit = in_flight
                        .clone()
                        .acquire_owned()
                        .await
                        .expect("semaphore closed");

                    let total_tokens = batch.total_tokens();
                    let requests_to_fire = batch.take();

                    // Collect batch context IDs for accurate rent
                    // charging and for the policy's `on_fired` hook
                    // (hot-aware uses it to track the just-fired
                    // cohort; the others ignore it).
                    let batch_ctx_ids: Vec<u64> = requests_to_fire.iter()
                        .map(|r| r.request.context_id)
                        .collect();
                    policy.on_fired(&batch_ctx_ids);

                    // Spawn batch execution
                    let latency_tx_clone = latency_tx.clone();
                    let stats_clone = stats.clone();

                    let batch_size = batch_ctx_ids.len() as u64;
                    tokio::spawn(async move {
                        let start = Instant::now();
                        Self::execute_batch(
                            device_idx,
                            requests_to_fire,
                            page_size,
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
                        stats_clone.total_requests_processed.fetch_add(batch_size, Relaxed);
                        stats_clone.max_batch_size_observed.fetch_max(batch_size, Relaxed);
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
                                policy.on_arrival(pending.request.context_id);
                                batch.push(pending);
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
                page_size,
            )
            .await;
        }
    }

    /// Execute a batch of forward pass requests via the device service.
    async fn execute_batch(
        device_idx: usize,
        requests: Vec<PendingRequest>,
        page_size: u32,
    ) {
        // Build batched request. `predict` is decided per-request by
        // InferenceService::handle when it forwards the submission.
        // `device_idx` doubles as the wire-protocol `device_id`:
        // they're both `usize` and the runtime only ever has one
        // device per local routing index, so they always coincide.
        let mut batch_req = BatchedForwardPassRequest::new(device_idx);
        for req in &requests {
            batch_req.add_request(
                &req.request,
                &req.physical_page_ids,
                req.last_page_len,
                page_size,
                /*predict=*/ false,
            );
        }

        // Send via device service (typed call handles serialization + timeout)
        let result = device::fire_batch(device_idx, &batch_req).await;

        match result {
            Ok(batch_resp) => {
                let BatchedForwardPassResponse { results } = batch_resp;
                if results.len() != requests.len() {
                    tracing::warn!(
                        device = device_idx,
                        expected = requests.len(),
                        got = results.len(),
                        "Batch response count mismatch — some requests may get no output",
                    );
                }

                let mut resp_iter = results.into_iter();
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
                        tracing::warn!(device = device_idx, "Fewer results than requests — sending None");
                        req.response_tx.send(ForwardPassOutput::default()).ok();
                    }
                }
            }
            Err(e) => {
                tracing::error!("fire_batch failed for device {}: {:?}", device_idx, e);
                for req in requests {
                    req.response_tx.send(ForwardPassOutput::default()).ok();
                }
            }
        }
    }

}
