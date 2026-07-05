//! Mock driver backend for integration tests.
//!
//! Implements [`DriverChannel`] directly and pre-registers one channel
//! per device through [`register_driver`] + [`install_channel`]. The
//! unified driver surface means there's no RPC server, no mqueue, and
//! no shmem ring in the test path — just a typed channel that dispatches
//! `RequestPayload::Forward` to a [`Behavior`] and treats every cold
//! method as a no-op status response.

use std::sync::atomic::AtomicU32;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use async_trait::async_trait;

use pie_engine::driver::{
    DriverChannel, DriverRequest, DriverResponse, DriverSpec, install_channel, register_driver,
};

// =============================================================================
// Behavior Trait
// =============================================================================

/// How the mock device handles a forward batch. The behavior receives
/// the batched [`pie_driver_abi::ForwardRequest`] (with indptrs delimiting
/// each per-request slice) and returns a batched
/// [`pie_driver_abi::ForwardResponse`] keyed by the same indptrs.
pub trait Behavior: Send + Sync + 'static {
    fn handle_fire_batch(&self, req: &pie_driver_abi::ForwardRequest) -> pie_driver_abi::ForwardResponse;
}

/// Helper: derive the request count from `qo_indptr`. The batched
/// shape stores `num_requests + 1` entries (one indptr per request +
/// the trailing total).
fn num_requests(req: &pie_driver_abi::ForwardRequest) -> u32 {
    req.qo_indptr.len().saturating_sub(1) as u32
}

fn total_tokens(req: &pie_driver_abi::ForwardRequest) -> usize {
    req.token_ids.len()
}

/// Build a `ForwardResponse` that emits exactly one token per request
/// — the common case for both `Echo` and `Counter` behaviors.
fn build_token_response(tokens: Vec<u32>) -> pie_driver_abi::ForwardResponse {
    let n = tokens.len() as u32;
    let tokens_indptr: Vec<u32> = (0..=n).collect();
    pie_driver_abi::ForwardResponse {
        num_requests: n,
        tokens_indptr,
        tokens,
        dists_req_indptr: vec![0; (n + 1) as usize],
        dists_kv_indptr: vec![0],
        dists_ids: Vec::new(),
        dists_probs: Vec::new(),
        logits_req_indptr: vec![0; (n + 1) as usize],
        logits_byte_indptr: vec![0],
        logits_bytes: Vec::new(),
        logprobs_req_indptr: vec![0; (n + 1) as usize],
        logprobs_val_indptr: vec![0],
        logprobs_values: Vec::new(),
        entropies_indptr: vec![0; (n + 1) as usize],
        entropies: Vec::new(),
        ..Default::default()
    }
}

// =============================================================================
// Built-in Behaviors
// =============================================================================

/// Always returns the same token for every request.
pub struct EchoBehavior(pub u32);

impl Behavior for EchoBehavior {
    fn handle_fire_batch(&self, req: &pie_driver_abi::ForwardRequest) -> pie_driver_abi::ForwardResponse {
        let n = num_requests(req) as usize;
        build_token_response(vec![self.0; n])
    }
}

// =============================================================================
// (Removed) sampling-program eval-mock executor — the sampling-IR sampler
// subsystem was retired; ptir succeeds it (outputs are channels, not declared
// OutputKinds). The pure synthetic-logits helpers below survive for any
// deterministic-logits mock use.
// =============================================================================

/// SplitMix64 + seed×column — the exact scheme validated against `sample_temp.cu`
/// (so the SAME synthetic row can later be fed to a one-row CUDA fire for an
/// eval-mock ≡ real-driver cross-check).
fn mock_hash_uniform(seed_eff: u64, j: u32) -> f32 {
    let mut x = seed_eff.wrapping_add(0x9E37_79B9_7F4A_7C15u64.wrapping_mul((j as u64) + 1));
    x ^= x >> 27;
    x = x.wrapping_mul(0x3C79_AC49_2BA7_B653);
    x ^= x >> 33;
    x = x.wrapping_mul(0x1C69_B3F7_4AC4_AE35);
    x ^= x >> 27;
    ((x >> 40) as u32 as f32 + 0.5) * (1.0 / 16_777_216.0)
}

/// Deterministic pseudo-random logits row seeded by request id, in ~[-4, 4].
/// Determinism (not model accuracy) is all `eval` needs for a plumbing gate.
pub fn synthetic_logits(req_id: u64, vocab: usize) -> Vec<f32> {
    let seed = req_id ^ 0xA5A5_A5A5u64;
    (0..vocab as u32)
        .map(|j| (mock_hash_uniform(seed, j) * 2.0 - 1.0) * 4.0)
        .collect()
}

/// (removed with the sampling-IR eval mock: logits_shape / host_value /
/// run_program_mock)


/// (removed with the sampling-IR eval mock: run_program_mock_outputs)

/// (removed with the sampling-IR eval mock: build_program_response /
/// SamplingProgramBehavior)

/// Returns sequential tokens starting from `start`.
pub struct CounterBehavior {
    next: AtomicU32,
}

impl CounterBehavior {
    pub fn new(start: u32) -> Self {
        Self {
            next: AtomicU32::new(start),
        }
    }
}

impl Behavior for CounterBehavior {
    fn handle_fire_batch(&self, req: &pie_driver_abi::ForwardRequest) -> pie_driver_abi::ForwardResponse {
        let n = num_requests(req) as usize;
        let tokens: Vec<u32> = (0..n)
            .map(|_| self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
            .collect();
        build_token_response(tokens)
    }
}

/// Wraps another behavior and adds simulated latency before responding.
pub struct DelayedBehavior<B: Behavior> {
    pub inner: B,
    pub latency: Duration,
}

impl<B: Behavior> Behavior for DelayedBehavior<B> {
    fn handle_fire_batch(&self, req: &pie_driver_abi::ForwardRequest) -> pie_driver_abi::ForwardResponse {
        std::thread::sleep(self.latency);
        self.inner.handle_fire_batch(req)
    }
}

/// Wraps another behavior and fails after N successful calls.
pub struct FailAfterBehavior<B: Behavior> {
    pub inner: B,
    remaining: AtomicU32,
}

impl<B: Behavior> FailAfterBehavior<B> {
    pub fn new(inner: B, success_count: u32) -> Self {
        Self {
            inner,
            remaining: AtomicU32::new(success_count),
        }
    }
}

impl<B: Behavior> Behavior for FailAfterBehavior<B> {
    fn handle_fire_batch(&self, req: &pie_driver_abi::ForwardRequest) -> pie_driver_abi::ForwardResponse {
        if self
            .remaining
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed)
            == 0
        {
            // Empty response to simulate failure: zero tokens, no
            // dists/logits/etc.
            pie_driver_abi::ForwardResponse {
                num_requests: 0,
                tokens_indptr: vec![0],
                ..Default::default()
            }
        } else {
            self.inner.handle_fire_batch(req)
        }
    }
}

// =============================================================================
// CallRecorder
// =============================================================================

#[derive(Debug, Clone)]
pub struct RecordedCall {
    pub device_idx: usize,
    pub method: &'static str,
    pub num_requests: usize,
    pub total_tokens: usize,
    pub timestamp: Instant,
}

#[derive(Debug, Default)]
pub struct CallRecorder {
    calls: Mutex<Vec<RecordedCall>>,
}

impl CallRecorder {
    pub fn new() -> Self {
        Self::default()
    }

    fn record(&self, call: RecordedCall) {
        self.calls.lock().unwrap().push(call);
    }

    pub fn call_count(&self) -> usize {
        self.calls.lock().unwrap().len()
    }

    pub fn calls(&self) -> Vec<RecordedCall> {
        self.calls.lock().unwrap().clone()
    }

    pub fn wait_for_calls(&self, n: usize, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        loop {
            if self.call_count() >= n {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

// =============================================================================
// MockChannel — DriverChannel impl per device.
// =============================================================================

struct MockChannel {
    device_idx: usize,
    behavior: Arc<dyn Behavior>,
    recorder: Arc<CallRecorder>,
    aborted: Mutex<bool>,
}

fn status_response(status: i32) -> DriverResponse {
    DriverResponse {
        aborted: false,
        payload: pie_driver_abi::ResponsePayload::Status(pie_driver_abi::StatusResponse { status }),
    }
}

impl MockChannel {
    fn submit_impl(&self, req: DriverRequest) -> Result<DriverResponse> {
        if *self.aborted.lock().unwrap() {
            anyhow::bail!("mock channel {}: aborted", self.device_idx);
        }
        match req.payload {
            pie_driver_abi::RequestPayload::Forward(fwd) => {
                self.recorder.record(RecordedCall {
                    device_idx: self.device_idx,
                    method: "fire_batch",
                    num_requests: num_requests(&fwd) as usize,
                    total_tokens: total_tokens(&fwd),
                    timestamp: Instant::now(),
                });
                let resp = self.behavior.handle_fire_batch(&fwd);
                Ok(DriverResponse {
                    aborted: false,
                    payload: pie_driver_abi::ResponsePayload::Forward(resp),
                })
            }
            pie_driver_abi::RequestPayload::Copy(_)
            | pie_driver_abi::RequestPayload::Adapter(_)
            | pie_driver_abi::RequestPayload::Health => Ok(status_response(0)),
        }
    }
}

#[async_trait]
impl DriverChannel for MockChannel {
    async fn submit(&self, req: DriverRequest) -> Result<DriverResponse> {
        self.submit_impl(req)
    }

    fn submit_sync(&self, req: DriverRequest) -> Result<DriverResponse> {
        self.submit_impl(req)
    }

    fn notify(&self, _req: DriverRequest) -> Result<()> {
        // Cold ops are no-ops in the mock — they always "succeed".
        Ok(())
    }

    fn abort(&self) {
        *self.aborted.lock().unwrap() = true;
    }
}

// =============================================================================
// MockBackend — owns the per-device channels.
// =============================================================================

pub struct MockBackend {
    driver_ids: Vec<usize>,
    recorder: Arc<CallRecorder>,
}

impl MockBackend {
    /// Pre-register `num_devices` mock channels with the runtime. Each
    /// channel gets a fresh [`pie_engine::driver::DriverId`] from the
    /// runtime's allocator; callers thread those IDs through their
    /// bootstrap config so the scheduler dispatches forward batches
    /// into the mocks.
    pub fn new(num_devices: usize, behavior: Arc<dyn Behavior>) -> Self {
        let recorder = Arc::new(CallRecorder::new());
        let mut driver_ids = Vec::with_capacity(num_devices);
        for device_idx in 0..num_devices {
            let channel = Arc::new(MockChannel {
                device_idx,
                behavior: Arc::clone(&behavior),
                recorder: Arc::clone(&recorder),
                aborted: Mutex::new(false),
            });
            let id = register_driver(DriverSpec {
                num_kv_pages: 64,
                limits: pie_engine::driver::SchedulerLimits {
                    max_forward_requests: 32,
                    max_forward_tokens: 4096,
                    max_page_refs: 64,
                    max_logit_rows: usize::MAX,
                    max_prob_rows: usize::MAX,
                    max_sampler_rows: usize::MAX,
                    max_custom_mask_bytes: usize::MAX,
                    max_logprob_labels: usize::MAX,
                },
            });
            install_channel(id, channel);
            driver_ids.push(id);
        }
        Self {
            driver_ids,
            recorder,
        }
    }

    /// Driver IDs allocated by [`Self::new`], one per device. Hand to
    /// the bootstrap config builder.
    pub fn driver_ids(&self) -> &[usize] {
        &self.driver_ids
    }

    pub fn recorder(&self) -> &CallRecorder {
        &self.recorder
    }
}
