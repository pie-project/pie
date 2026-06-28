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

use pie::driver::{
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
// eval-mock executor (lane L7) — runs an attached sampling-program through the
// canonical CPU `eval` interpreter over deterministic synthetic logits, so the
// SDK→host→driver `sampling-program` path gets a first green end-to-end decode
// WITHOUT the 4090. The real-driver path (numeric parity on real logits) is a
// separate gate owned by echo/delta.
// =============================================================================

use pie_driver_abi::SamplingBinding;
use pie_sampling_ir::bytecode;
use pie_sampling_ir::eval::{eval, InputBindings, Value};
use pie_sampling_ir::types::DType;
use pie_sampling_ir::{output_kinds, OutputKind};

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

/// Vocab length of the program's `Logits`-bound input slot (per the carrier
/// binding-map): the last dim of that input's declared shape.
fn logits_vocab(
    prog: &pie_sampling_ir::SamplingProgram,
    bindings: &[SamplingBinding],
) -> Option<usize> {
    bindings.iter().enumerate().find_map(|(i, b)| match b {
        SamplingBinding::Logits => prog
            .inputs
            .get(i)
            .and_then(|inp| inp.shape.last_len())
            .map(|v| v as usize),
        _ => None,
    })
}

/// Decode a submit-bound host buffer (raw LE bytes) into an `eval::Value` per the
/// program input's declared dtype.
fn host_value(dtype: DType, bytes: &[u8]) -> Value {
    match dtype {
        DType::F32 => Value::F32(
            bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect(),
        ),
        DType::I32 => Value::I32(
            bytes.chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect(),
        ),
        DType::U32 => Value::U32(
            bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect(),
        ),
        DType::Bool => Value::Bool(bytes.iter().map(|&b| b != 0).collect()),
    }
}

/// `run_program_mock`: decode → validate → bind submit-bound host inputs →
/// `eval` over `logits` → first declared output as a token id. (Tier-1 declares
/// a single `Token` output; scalar outputs would route to the entropy channel.)
pub fn run_program_mock(
    submission: &pie_driver_abi::SamplingProgramSubmission,
    logits: &[f32],
) -> Result<u32, String> {
    let outs = run_program_mock_outputs(submission, logits)?;
    match outs.first() {
        Some((OutputKind::Token, Value::I32(v))) if !v.is_empty() => Ok(v[0] as u32),
        other => Err(format!("expected Token (I32) first output, got {other:?}")),
    }
}

/// `eval` an attached program over `logits`, returning **every** declared output
/// paired with its `OutputKind` (in slot-declaration order). This is what lets
/// the mock executor marshal multi-output programs (e.g. mirostat's
/// `[Token, Scalar S]`) into the matching `ForwardResponse` channels, not just
/// the Token channel.
pub fn run_program_mock_outputs(
    submission: &pie_driver_abi::SamplingProgramSubmission,
    logits: &[f32],
) -> Result<Vec<(OutputKind, Value)>, String> {
    let prog = bytecode::decode(&submission.bytecode).map_err(|e| format!("decode: {e:?}"))?;
    prog.validate().map_err(|e| format!("validate: {e:?}"))?;
    // Build the positional input vector (`Op::Input(i)` ↔ `inputs[i]`) from the
    // carrier binding-map: a `Logits` slot binds the synthetic logits row; a
    // `Tensor` slot binds its keyed submit value, typed by the input decl.
    let mut inputs: Vec<Value> = Vec::with_capacity(prog.inputs.len());
    for (i, inp) in prog.inputs.iter().enumerate() {
        let binding = submission
            .bindings
            .get(i)
            .copied()
            .unwrap_or(SamplingBinding::Logits);
        let value = match binding {
            SamplingBinding::Logits => Value::F32(logits.to_vec()),
            // The draft-logits intrinsic (#21 mtp): the mock has no separate
            // draft buffer; it source-selects `ws.logits` draft rows (M=1 ⇒ the
            // logits row), so the mock returns the same logits value as `Logits`.
            SamplingBinding::MtpLogits => Value::F32(logits.to_vec()),
            SamplingBinding::Tensor { key } => {
                let bytes = submission
                    .inputs
                    .iter()
                    .find(|si| si.key == key)
                    .map(|si| si.bytes.as_slice())
                    .ok_or_else(|| format!("submit input key {key} missing for slot {i}"))?;
                host_value(inp.dtype, bytes)
            }
        };
        inputs.push(value);
    }
    let kinds = output_kinds(&prog).map_err(|e| format!("output_kinds: {e:?}"))?;
    let values =
        eval(&prog, &InputBindings::new(&inputs, 0)).map_err(|e| format!("eval: {e:?}"))?;
    Ok(kinds.into_iter().zip(values).collect())
}

/// Build a `ForwardResponse` for a batch of program evaluations, marshaling each
/// request's outputs into the matching channels: `Token` → tokens slot,
/// `Scalar`/`Entropy` → entropies slot (the shared per-slot scalar f32 channel,
/// per the L6/L1 marshaling contract). One Token + (optionally) one scalar per
/// request — the tier-1 mirostat/grammar shape.
fn build_program_response(
    per_request: &[Vec<(OutputKind, Value)>],
) -> pie_driver_abi::ForwardResponse {
    let n = per_request.len() as u32;
    let mut tokens: Vec<u32> = Vec::with_capacity(per_request.len());
    let mut entropies: Vec<f32> = Vec::new();
    let mut entropies_indptr: Vec<u32> = vec![0];

    for outs in per_request {
        // Token (required for a token-producing program).
        let tok = outs.iter().find_map(|(k, v)| match (k, v) {
            (OutputKind::Token, Value::I32(x)) if !x.is_empty() => Some(x[0] as u32),
            _ => None,
        });
        tokens.push(tok.unwrap_or(0));

        // Scalar / Entropy → entropies channel (0 or 1 per request).
        let scalar = outs.iter().find_map(|(k, v)| match (k, v) {
            (OutputKind::Scalar | OutputKind::Entropy, Value::F32(x)) if !x.is_empty() => Some(x[0]),
            _ => None,
        });
        match scalar {
            Some(s) => {
                entropies.push(s);
                entropies_indptr.push(entropies.len() as u32);
            }
            None => entropies_indptr.push(entropies.len() as u32),
        }
    }

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
        entropies_indptr,
        entropies,
        ..Default::default()
    }
}

/// Mock backend that executes an attached sampling-program via `eval` over
/// synthetic logits (one token + optional scalar per request). Falls back to
/// `fallback` for requests without a program.
pub struct SamplingProgramBehavior {
    pub fallback: u32,
}

impl Behavior for SamplingProgramBehavior {
    fn handle_fire_batch(&self, req: &pie_driver_abi::ForwardRequest) -> pie_driver_abi::ForwardResponse {
        let n = num_requests(req) as usize;
        // MVP/tier-1: a single program (index 0) covers the batch.
        match req.sampling_program_at(0) {
            Some(sub) => {
                let prog = bytecode::decode(&sub.bytecode).expect("decode program");
                let vocab =
                    logits_vocab(&prog, &sub.bindings).expect("program declares a Logits input");
                let per_request: Vec<Vec<(OutputKind, Value)>> = (0..n)
                    .map(|r| {
                        let logits = synthetic_logits(r as u64, vocab);
                        run_program_mock_outputs(&sub, &logits).expect("eval-mock run")
                    })
                    .collect();
                build_program_response(&per_request)
            }
            None => build_token_response(vec![self.fallback; n]),
        }
    }
}

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
    /// channel gets a fresh [`pie::driver::DriverId`] from the
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
                limits: pie::driver::SchedulerLimits {
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
