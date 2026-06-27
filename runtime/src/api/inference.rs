//! pie:core/inference - ForwardPass + sampler programs; pie:core/tensor -
//! Tensor + Program resources.

use crate::api::adapter::Adapter;
use crate::api::context::Context;
use crate::api::pie;
use crate::inference::ForwardOutput;
use crate::inference::structured::compiled_grammar::CompiledGrammar;
use crate::inference::structured::grammar::Grammar as InternalGrammar;
use crate::inference::structured::json_schema::{
    JsonSchemaOptions, builtin_json_grammar, json_schema_to_grammar,
};
use crate::inference::structured::matcher::GrammarMatcher;
use crate::inference::structured::regex::regex_to_grammar;
use crate::instance::InstanceState;
use crate::{context, inference};
use anyhow::Result;
use pie_driver_abi::Brle;
use std::mem::take;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
use tokio::sync::oneshot;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

#[derive(Debug, Clone, serde::Serialize)]
pub struct ExecuteProfileSnapshot {
    pub calls: u64,
    pub hits: u64,
    pub misses: u64,
    pub total_us: u64,
    pub prepare_us: u64,
    pub try_hit_us: u64,
    pub hit_wait_us: u64,
    pub cold_prepare_us: u64,
    pub pin_us: u64,
    pub submit_wait_us: u64,
    pub postprocess_us: u64,
}

struct ExecuteProfileStats {
    calls: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    total_us: AtomicU64,
    prepare_us: AtomicU64,
    try_hit_us: AtomicU64,
    hit_wait_us: AtomicU64,
    cold_prepare_us: AtomicU64,
    pin_us: AtomicU64,
    submit_wait_us: AtomicU64,
    postprocess_us: AtomicU64,
}

static EXECUTE_PROFILE: ExecuteProfileStats = ExecuteProfileStats {
    calls: AtomicU64::new(0),
    hits: AtomicU64::new(0),
    misses: AtomicU64::new(0),
    total_us: AtomicU64::new(0),
    prepare_us: AtomicU64::new(0),
    try_hit_us: AtomicU64::new(0),
    hit_wait_us: AtomicU64::new(0),
    cold_prepare_us: AtomicU64::new(0),
    pin_us: AtomicU64::new(0),
    submit_wait_us: AtomicU64::new(0),
    postprocess_us: AtomicU64::new(0),
};

fn execute_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PIE_PROFILE_EXECUTE").is_some())
}

fn elapsed_us(duration: Duration) -> u64 {
    duration.as_micros() as u64
}

pub fn execute_profile_snapshot() -> Option<ExecuteProfileSnapshot> {
    if !execute_profile_enabled() {
        return None;
    }
    Some(ExecuteProfileSnapshot {
        calls: EXECUTE_PROFILE.calls.load(Ordering::Relaxed),
        hits: EXECUTE_PROFILE.hits.load(Ordering::Relaxed),
        misses: EXECUTE_PROFILE.misses.load(Ordering::Relaxed),
        total_us: EXECUTE_PROFILE.total_us.load(Ordering::Relaxed),
        prepare_us: EXECUTE_PROFILE.prepare_us.load(Ordering::Relaxed),
        try_hit_us: EXECUTE_PROFILE.try_hit_us.load(Ordering::Relaxed),
        hit_wait_us: EXECUTE_PROFILE.hit_wait_us.load(Ordering::Relaxed),
        cold_prepare_us: EXECUTE_PROFILE.cold_prepare_us.load(Ordering::Relaxed),
        pin_us: EXECUTE_PROFILE.pin_us.load(Ordering::Relaxed),
        submit_wait_us: EXECUTE_PROFILE.submit_wait_us.load(Ordering::Relaxed),
        postprocess_us: EXECUTE_PROFILE.postprocess_us.load(Ordering::Relaxed),
    })
}

#[derive(Default)]
struct ExecuteProfileSample {
    hit: bool,
    prepare_us: u64,
    try_hit_us: u64,
    hit_wait_us: u64,
    cold_prepare_us: u64,
    pin_us: u64,
    submit_wait_us: u64,
    postprocess_us: u64,
}

fn record_execute_profile(sample: ExecuteProfileSample, total_us: u64) {
    if !execute_profile_enabled() {
        return;
    }
    EXECUTE_PROFILE.calls.fetch_add(1, Ordering::Relaxed);
    if sample.hit {
        EXECUTE_PROFILE.hits.fetch_add(1, Ordering::Relaxed);
    } else {
        EXECUTE_PROFILE.misses.fetch_add(1, Ordering::Relaxed);
    }
    EXECUTE_PROFILE
        .total_us
        .fetch_add(total_us, Ordering::Relaxed);
    EXECUTE_PROFILE
        .prepare_us
        .fetch_add(sample.prepare_us, Ordering::Relaxed);
    EXECUTE_PROFILE
        .try_hit_us
        .fetch_add(sample.try_hit_us, Ordering::Relaxed);
    EXECUTE_PROFILE
        .hit_wait_us
        .fetch_add(sample.hit_wait_us, Ordering::Relaxed);
    EXECUTE_PROFILE
        .cold_prepare_us
        .fetch_add(sample.cold_prepare_us, Ordering::Relaxed);
    EXECUTE_PROFILE
        .pin_us
        .fetch_add(sample.pin_us, Ordering::Relaxed);
    EXECUTE_PROFILE
        .submit_wait_us
        .fetch_add(sample.submit_wait_us, Ordering::Relaxed);
    EXECUTE_PROFILE
        .postprocess_us
        .fetch_add(sample.postprocess_us, Ordering::Relaxed);
}

/// WASM-facing forward-pass accumulator. The WIT methods append/set
/// into `req: pie_driver_abi::ForwardRequest` directly — at `execute()`
/// we just finalize the per-request indptrs and submit. `model_id`
/// is WASM-side routing info (not on the wire) and `adapter_seed`
/// is stored separately because it doesn't have its own WIT setter
/// but is used at execute-time to populate the adapter binding.
#[derive(Debug)]
/// Host backing for the WIT `tensor` resource (`pie:core/tensor.tensor`): a
/// typed, host-resident byte buffer. Under the MVP sync interface every tensor
/// binding is *submit* — the inferlet writes the value before `execute()` and
/// the host gathers `data` into the sampling carrier at attach time.
pub struct Tensor {
    shape: Vec<u32>,
    dtype: pie::core::tensor::Dtype,
    data: Vec<u8>,
}

/// Host backing for the WIT `program` resource (`pie:core/tensor.program`): a
/// compiled sampling program. Decoded once at construction (`tensor::Program →
/// SamplingProgram`, via [`program_decode::decode_program`]) and encoded to L0
/// bytecode for the bridge carrier. Keeps the declared output kinds (for
/// response marshaling) and the input arity (for attach-time binding
/// validation).
pub struct Program {
    /// The interned, shared `#9` cache artifact: canonical bytecode, its hash
    /// (== the driver `ProgramHandle`), per-output marshaling kinds, and input
    /// arity. Identical programs across requests/inferlets share one `Arc`.
    cached: Arc<crate::api::program_cache::CachedProgram>,
}

pub struct ForwardPass {
    /// Set when the ctx is bound via `context()` (the new WIT `forward-pass`
    /// constructor takes no model — the context carries the model identity).
    pub model_id: usize,
    context_id: Option<crate::context::ContextId>,
    /// Snapshot of the bound ctx's cached speculator handle. Set in
    /// `pass.context()`; `None` until then, or when speculation is
    /// disabled for the model. Lets `execute()` call `try_hit`
    /// without taking the global REGISTRY lock.
    spec: Option<inference::StagedBatch>,
    pub adapter_seed: Option<i64>,
    allow_pass_speculation: bool,
    req: pie_driver_abi::ForwardRequest,
    /// Sampling programs attached via `sampler(...)` / `batch-sampler(...)`,
    /// each with its attach-time binding-map and gathered submit inputs.
    /// Empty for a pass that does no sampling (pure prefill); one entry for the
    /// single-sampler path; many for `batch-sampler`. Flattened into the bridge
    /// carrier at `execute()`.
    programs: Vec<AttachedProgram>,
    /// WS8 inter-pass pipelining (`next-input`): per source sampling row, the
    /// destination input position its sampled token feeds in the *next* pass on
    /// this context (`u32::MAX` = the `-1` ignore sentinel). `None` = no
    /// feed-forward (the pass's output is read via `output()` as usual).
    feed_forward: Option<Vec<u32>>,
    /// In-flight driver response + reconstruction context, stashed by
    /// `execute()` and drained by `output()`/`outputs()`. `None` until the pass
    /// is submitted (or when the pass fed its output forward via `next-input`).
    pending: Option<PendingOutput>,
}

/// A WS8 inter-pass pipeline link (`forward-pass.next-input`). Shared-resolution
/// model (a): the link **references** the producer pass (by resource rep) rather
/// than owning its output — so the producer's own `output()` still resolves
/// (consumer B: the guest reads t's token for stop/EOS detection) while the next
/// pass reads the **same** resolved per-row tokens for its input (consumer A).
/// `positions[r]` is the next pass's input position fed by source sampling row
/// `r` (`u32::MAX` = the `-1` ignore sentinel). The producer pass owns its pin
/// (released via its `output()`/drop); the link holds no resources. P2 binds the
/// device-resident `pi.sampled` `[N]` buffer + a stream event instead.
pub struct PipelineLink {
    producer_rep: u32,
    positions: Vec<u32>,
}

/// One sampling program staged on a [`ForwardPass`] before submit: its L0
/// bytecode, declared output kinds (for response marshaling), the attach-time
/// [`Binding`](pie_sampling_ir::Binding) per input slot (the binding-map ridden
/// to the carrier so the driver can wire each `Op::Input(i)`), and the
/// submit-bound tensor values gathered from the bound `tensor` resources.
#[derive(Debug)]
struct AttachedProgram {
    bytecode: Vec<u8>,
    output_kinds: Vec<pie_sampling_ir::OutputKind>,
    bindings: Vec<pie_sampling_ir::Binding>,
    submit_inputs: Vec<pie_driver_abi::SamplingInput>,
    /// Sampling positions carried by this program's `logits` binding (the WIT
    /// `input-binding::logits(positions)`); flattened into the request's
    /// `sampling_indices` at `execute()`. Empty if the program reads no logits.
    logits_positions: Vec<u32>,
}

/// Stashed on a [`ForwardPass`] by `execute()` and consumed by
/// `output()`/`outputs()`: the in-flight driver response channel plus the
/// bookkeeping needed to grow the KV lineage, release the pin, and marshal each
/// attached program's declared outputs into typed `tensor` resources. The new
/// WIT has no legacy sampler-slot path — every pass is program-driven — so the
/// sampler-count spec-fill heuristic is gone (a program owns its output shape).
struct PendingOutput {
    rx: Option<oneshot::Receiver<Result<ForwardOutput>>>,
    /// Resolved-once cache (with lineage + pin release already applied): shared
    /// by `output()` (consumer B — marshals tensors for the guest's
    /// stop-detection read) and the WS8 link (consumer A — reads per-row tokens
    /// for the next pass). Resolving is idempotent: whichever consumer runs
    /// first awaits `rx`; the other reads the cache.
    resolved: Option<Result<ForwardOutput, String>>,
    /// Declared output kinds per attached program (attach order), driving the
    /// per-program response→tensor reconstruction.
    programs_output_kinds: Vec<Vec<pie_sampling_ir::OutputKind>>,
    model_id: usize,
    context_id: crate::context::ContextId,
    was_pinned: bool,
    fill_tokens: Vec<u32>,
    fill_positions: Vec<u32>,
    fill_masks: Vec<Brle>,
    adapter_id: Option<crate::adapter::AdapterId>,
    adapter_seed: Option<i64>,
}

impl PendingOutput {
    fn release_pin(&mut self) {
        if self.was_pinned {
            context::unpin(self.model_id, self.context_id);
            self.was_pinned = false;
        }
    }

    /// Grow the bound ctx's working-page lineage with this pass's input tokens
    /// (so the next decode step sees them in the KV). Mirrors the legacy
    /// `FutureOutput` path minus the retired draft-fill bookkeeping.
    fn append_lineage(&mut self) {
        let fill_tokens = take(&mut self.fill_tokens);
        let fill_positions = take(&mut self.fill_positions);
        let fill_masks = take(&mut self.fill_masks);
        if !fill_tokens.is_empty() {
            context::append_working_page_tokens_with_repaired_spec_tail(
                self.model_id,
                self.context_id,
                fill_tokens,
                fill_positions,
                fill_masks,
                self.adapter_id,
                self.adapter_seed,
                0,
            );
        }
    }

    /// Resolve once — await the driver response, grow the ctx lineage, release
    /// the pin — and cache the result in `resolved`. Idempotent: a second call
    /// is a no-op. Shared by `output()` (consumer B) and the WS8 link (consumer
    /// A) so a fed-forward pass's output reaches both.
    async fn resolve(&mut self) {
        if self.resolved.is_some() {
            return;
        }
        let Some(rx) = self.rx.take() else {
            self.resolved = Some(Err("forward pass output already consumed".to_string()));
            return;
        };
        self.resolved = Some(match rx.await {
            Ok(Ok(output)) => {
                self.append_lineage();
                self.release_pin();
                Ok(output)
            }
            Ok(Err(e)) => {
                self.release_pin();
                tracing::warn!("forward pass failed: {e:#}");
                Err(format!("forward pass failed: {e}"))
            }
            Err(_) => {
                self.release_pin();
                Err("forward pass channel closed".to_string())
            }
        });
    }

    /// Per-row sampled tokens of the resolved output (consumer A, non-consuming);
    /// empty if not yet resolved or the pass failed.
    fn resolved_tokens(&self) -> Vec<u32> {
        match &self.resolved {
            Some(Ok(output)) => forward_output_tokens(output),
            _ => Vec::new(),
        }
    }

    /// Take the resolved output for marshaling (consumer B, consuming).
    fn take_resolved(&mut self) -> Result<ForwardOutput, String> {
        self.resolved
            .take()
            .unwrap_or_else(|| Err("forward pass output not resolved".to_string()))
    }
}

/// The per-row sampled tokens a [`ForwardOutput`] carries (for WS8 feed-forward).
fn forward_output_tokens(output: &ForwardOutput) -> Vec<u32> {
    match output {
        ForwardOutput::Token(t) => vec![*t],
        ForwardOutput::Tokens(v) => v.clone(),
        ForwardOutput::Response(resp) => resp.tokens.clone(),
    }
}

fn empty_forward_request() -> pie_driver_abi::ForwardRequest {
    pie_driver_abi::ForwardRequest {
        adapter_bindings: vec![pie_driver_abi::AdapterBinding {
            adapter_id: -1,
            seed: -1,
        }],
        output_spec_flags: vec![false],
        // Image side-channel CSR roots: the per-image pixel/mrope indptrs and
        // the per-request image_indptr all begin with a leading 0 so the
        // batch-merge in `inference::request` can offset and append cleanly.
        image_indptr: vec![0],
        image_pixel_indptr: vec![0],
        image_mrope_indptr: vec![0],
        // Audio side-channel CSR roots (leading 0, like the image roots).
        audio_feature_indptr: vec![0],
        audio_indptr: vec![0],
        ..Default::default()
    }
}


impl Tensor {
    /// An `i32` vector tensor (shape `[len]`). Token outputs use this — the
    /// dtype⇒kind convention (a) is `int ⇒ Token`, so sampled token ids are
    /// surfaced as `i32`.
    fn i32_vec(vals: Vec<i32>) -> Tensor {
        let mut data = Vec::with_capacity(vals.len() * 4);
        for v in &vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        Tensor {
            shape: vec![vals.len() as u32],
            dtype: pie::core::tensor::Dtype::I32,
            data,
        }
    }

    /// An `f32` vector tensor (shape `[len]`). Scalar / entropy / logprob /
    /// distribution-probability outputs use this.
    fn f32_vec(vals: Vec<f32>) -> Tensor {
        let mut data = Vec::with_capacity(vals.len() * 4);
        for v in &vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        Tensor {
            shape: vec![vals.len() as u32],
            dtype: pie::core::tensor::Dtype::F32,
            data,
        }
    }

    /// An `f32` tensor wrapping an already-`f32`-little-endian byte blob (raw
    /// logits), shape `[len/4]`.
    fn f32_from_le_bytes(bytes: Vec<u8>) -> Tensor {
        let n = (bytes.len() / 4) as u32;
        Tensor {
            shape: vec![n],
            dtype: pie::core::tensor::Dtype::F32,
            data: bytes,
        }
    }
}

/// Number of bytes in one element of a `tensor` dtype.
fn dtype_byte_size(dtype: pie::core::tensor::Dtype) -> usize {
    use pie::core::tensor::Dtype;
    match dtype {
        Dtype::F32 | Dtype::I32 | Dtype::U32 => 4,
        Dtype::Bool => 1,
    }
}

/// Reconstruct each attached program's declared outputs as typed `tensor`
/// payloads from a driver [`ForwardOutput`], walking the programs in attach
/// order and, within each, its declared output kinds in slot order — the
/// new-WIT analog of the retired `slot-output` walk. Token outputs become `i32`
/// tensors; every other (float) kind an `f32` tensor (the dtype⇒kind
/// convention (a), matching the decoder's inference). The outer vec mirrors the
/// program list; each inner vec follows that program's output order.
fn build_program_tensors(
    output: ForwardOutput,
    programs_output_kinds: &[Vec<pie_sampling_ir::OutputKind>],
) -> Vec<Vec<Tensor>> {
    use pie_sampling_ir::OutputKind;
    match output {
        ForwardOutput::Token(token) => single_token_outputs(vec![token], programs_output_kinds),
        ForwardOutput::Tokens(tokens) => {
            // A single program declaring a single Token output owns the whole
            // token vector as one `[N]` tensor (the common decode shape);
            // otherwise distribute one token per declared Token slot.
            if programs_output_kinds.len() == 1
                && programs_output_kinds[0].len() == 1
                && programs_output_kinds[0][0] == OutputKind::Token
            {
                vec![vec![Tensor::i32_vec(
                    tokens.into_iter().map(|t| t as i32).collect(),
                )]]
            } else {
                single_token_outputs(tokens, programs_output_kinds)
            }
        }
        ForwardOutput::Response(resp) => {
            build_program_tensors_from_response(resp, programs_output_kinds)
        }
    }
}

/// Distribute a flat token list across each program's declared `Token` outputs
/// (one token per Token slot, in order); non-token kinds yield no tensor on this
/// fast path (used only when no rich `ForwardResponse` was produced).
fn single_token_outputs(
    tokens: Vec<u32>,
    programs_output_kinds: &[Vec<pie_sampling_ir::OutputKind>],
) -> Vec<Vec<Tensor>> {
    use pie_sampling_ir::OutputKind;
    let mut tok_iter = tokens.into_iter();
    programs_output_kinds
        .iter()
        .map(|kinds| {
            kinds
                .iter()
                .filter_map(|k| match k {
                    OutputKind::Token => {
                        tok_iter.next().map(|t| Tensor::i32_vec(vec![t as i32]))
                    }
                    _ => None,
                })
                .collect()
        })
        .collect()
}

/// Per-program tensor reconstruction over a rich [`ForwardResponse`]: the
/// program analog of the retired `build_wit_output_from_response` walk. Program
/// outputs reuse the existing response channels (token / distribution / logits /
/// logprobs / entropy), pulled in attach order so each program consumes its
/// share.
fn build_program_tensors_from_response(
    resp: pie_driver_abi::ForwardResponse,
    programs_output_kinds: &[Vec<pie_sampling_ir::OutputKind>],
) -> Vec<Vec<Tensor>> {
    use pie_sampling_ir::OutputKind;

    let mut tok_iter = resp.tokens.into_iter();
    let mut dist_iter: Box<dyn Iterator<Item = Vec<f32>>> = if resp.dists_kv_indptr.len() >= 2 {
        let probs: Vec<_> = (0..resp.dists_kv_indptr.len() - 1)
            .map(|k| {
                let lo = resp.dists_kv_indptr[k] as usize;
                let hi = resp.dists_kv_indptr[k + 1] as usize;
                resp.dists_probs[lo..hi].to_vec()
            })
            .collect();
        Box::new(probs.into_iter())
    } else {
        Box::new(std::iter::empty())
    };
    let mut logit_iter: Box<dyn Iterator<Item = Vec<u8>>> = if resp.logits_byte_indptr.len() >= 2 {
        let blobs: Vec<_> = (0..resp.logits_byte_indptr.len() - 1)
            .map(|b| {
                let lo = resp.logits_byte_indptr[b] as usize;
                let hi = resp.logits_byte_indptr[b + 1] as usize;
                resp.logits_bytes[lo..hi].to_vec()
            })
            .collect();
        Box::new(blobs.into_iter())
    } else {
        Box::new(std::iter::empty())
    };
    let mut lp_iter: Box<dyn Iterator<Item = Vec<f32>>> = if resp.logprobs_val_indptr.len() >= 2 {
        let slots: Vec<_> = (0..resp.logprobs_val_indptr.len() - 1)
            .map(|s| {
                let lo = resp.logprobs_val_indptr[s] as usize;
                let hi = resp.logprobs_val_indptr[s + 1] as usize;
                resp.logprobs_values[lo..hi].to_vec()
            })
            .collect();
        Box::new(slots.into_iter())
    } else {
        Box::new(std::iter::empty())
    };
    let mut ent_iter = resp.entropies.into_iter();

    programs_output_kinds
        .iter()
        .map(|kinds| {
            kinds
                .iter()
                .filter_map(|kind| match kind {
                    OutputKind::Token => {
                        tok_iter.next().map(|t| Tensor::i32_vec(vec![t as i32]))
                    }
                    // Distribution exposes its probability vector as f32; the
                    // companion ids ride the same channel for the SDK's typed
                    // accessor (MVP tensor channel carries probs).
                    OutputKind::Distribution => dist_iter.next().map(Tensor::f32_vec),
                    OutputKind::Logits => logit_iter.next().map(Tensor::f32_from_le_bytes),
                    OutputKind::Logprobs => lp_iter.next().map(Tensor::f32_vec),
                    // Entropy and Scalar (kld / mirostat μ) share the per-slot
                    // scalar f32 entropies channel; both surface as a 1-elem f32.
                    OutputKind::Entropy | OutputKind::Scalar => {
                        ent_iter.next().map(|e| Tensor::f32_vec(vec![e]))
                    }
                    // Embedding is reserved but not currently produced.
                    OutputKind::Embedding => None,
                })
                .collect()
        })
        .collect()
}

impl pie::core::inference::Host for InstanceState {}

impl pie::core::inference::HostForwardPass for InstanceState {
    async fn new(&mut self) -> Result<Resource<ForwardPass>> {
        // Initialize the accumulator with the per-request invariants: single
        // adapter binding (-1 sentinels = unbound). The new WIT `forward-pass`
        // constructor takes no model — `model_id` is bound when the ctx is set
        // via `context()` (the context carries the model identity).
        let pass = ForwardPass {
            model_id: 0,
            context_id: None,
            spec: None,
            adapter_seed: None,
            allow_pass_speculation: true,
            req: empty_forward_request(),
            programs: Vec::new(),
            feed_forward: None,
            pending: None,
        };
        Ok(self.ctx().table.push(pass)?)
    }

    async fn context(
        &mut self,
        this: Resource<ForwardPass>,
        context: Resource<Context>,
    ) -> Result<()> {
        let ctx = self.ctx().table.get(&context)?;
        let context_id = ctx.context_id;
        let model_id = ctx.model_id;
        // Initialize the ctx's speculator cache on the first call
        // for this ctx. The OnceLock makes this lock-free on every
        // subsequent `pass.context()`, eliminating REGISTRY lookups
        // from the per-iteration hot path.
        let spec = ctx
            .spec
            .get_or_init(|| {
                let device_idx = context::get_device(model_id, context_id);
                inference::lookup_for_ctx(model_id, device_idx)
            })
            .clone();
        let pass = self.ctx().table.get_mut(&this)?;
        pass.context_id = Some(context_id);
        pass.model_id = model_id;
        pass.spec = spec;
        Ok(())
    }

    async fn input_tokens(
        &mut self,
        this: Resource<ForwardPass>,
        tokens: Vec<u32>,
        positions: Vec<u32>,
    ) -> Result<()> {
        let pass = self.ctx().table.get_mut(&this)?;
        pass.req.token_ids.extend(tokens);
        pass.req.position_ids.extend(positions);
        Ok(())
    }

    /// Splice an encoded visual span into the pending request. Appends
    /// `token_count` placeholder rows to `token_ids`/`position_ids` (so the
    /// forward has KV slots for the soft tokens; the driver overwrites their
    /// embeddings with the vision-encoder output) and stages the pixel tensor +
    /// per-patch positions + the batch row offset on the image side-channel.
    /// See MULTIMODAL.md §2.5.
    async fn input_image(
        &mut self,
        this: Resource<ForwardPass>,
        image: Resource<crate::api::media::Image>,
        anchor: u32,
    ) -> Result<()> {
        let (token_count, grid, patch_grid, uses_mrope, pixels, positions) = {
            let img = self.ctx().table.get(&image)?;
            (
                img.span.token_count,
                img.span.grid,    // merged LLM grid (for M-RoPE positions)
                img.patch_grid,   // pre-merge patch grid (for the driver encoder)
                img.uses_mrope,
                img.pixels.clone(),
                img.positions.clone(),
            )
        };

        let pass = self.ctx().table.get_mut(&this)?;
        let req = &mut pass.req;

        // Row offset (within this request) where the soft-token rows begin.
        let anchor_row = req.token_ids.len() as u32;
        req.image_anchor_rows.push(anchor_row);

        // Placeholder rows: valid token id 0 (overwritten by the encoder
        // scatter), positions sequential from `anchor` (Gemma 1-D RoPE).
        for i in 0..token_count {
            req.token_ids.push(0);
            req.position_ids.push(anchor + i);
        }

        // Pixel tensor (f32 → little-endian bytes) + per-patch positions.
        req.image_pixels
            .extend_from_slice(bytemuck::cast_slice(&pixels));
        req.image_pixel_indptr.push(req.image_pixels.len() as u32);
        req.image_patch_positions.extend_from_slice(&positions);

        // Wire `image_grids` carries the PRE-MERGE patch grid: the driver's
        // vision encoder needs patch units (t*h*w == n_patch). The merged grid
        // is used below only for the M-RoPE position side-channel.
        req.image_grids
            .extend_from_slice(&[patch_grid.t, patch_grid.h, patch_grid.w]);
        req.image_anchor_positions.push(anchor);
        if uses_mrope {
            for p in crate::multimodal::qwen_mrope_positions(grid, anchor) {
                req.image_mrope_positions.extend_from_slice(&p);
            }
        }
        req.image_mrope_indptr
            .push(req.image_mrope_positions.len() as u32);

        Ok(())
    }

    /// Splice an encoded audio clip into the pending request. Direct analog of
    /// `input_image`: appends `token_count` placeholder rows (KV slots for the
    /// audio soft tokens; the driver overwrites their embeddings with the
    /// audio-encoder output) and stages the log-mel features + the batch row
    /// offset on the audio side-channel. See audio_frontend.md.
    async fn input_audio(
        &mut self,
        this: Resource<ForwardPass>,
        audio: Resource<crate::api::media::Audio>,
        anchor: u32,
    ) -> Result<()> {
        let (token_count, mel) = {
            let a = self.ctx().table.get(&audio)?;
            (a.token_count, a.mel.clone())
        };

        let pass = self.ctx().table.get_mut(&this)?;
        let req = &mut pass.req;

        // Row offset (within this request) where the soft-token rows begin.
        let anchor_row = req.token_ids.len() as u32;
        req.audio_anchor_rows.push(anchor_row);

        // Placeholder rows: valid token id 0 (overwritten by the encoder
        // scatter), positions sequential from `anchor` (Gemma 1-D RoPE).
        for i in 0..token_count {
            req.token_ids.push(0);
            req.position_ids.push(anchor + i);
        }

        // Log-mel features (f32 → little-endian bytes).
        req.audio_features
            .extend_from_slice(bytemuck::cast_slice(&mel));
        req.audio_feature_indptr.push(req.audio_features.len() as u32);

        Ok(())
    }

    async fn attention_mask(
        &mut self,
        this: Resource<ForwardPass>,
        mask: Vec<Vec<u32>>,
    ) -> Result<()> {
        let brle_masks: Vec<Brle> = mask.into_iter().map(Brle::from_vec).collect();

        let pass = self.ctx().table.get_mut(&this)?;
        pass.req.masks = brle_masks;
        Ok(())
    }

    // ── Programmable sampler (Sampling IR) — L1/bravo host carrier ──────
    //
    // The binding-free front door: the inferlet builds a `program` resource
    // (decoded + compiled at construction, see `HostProgram::new`) and attaches
    // it with one `input-binding` per program input slot. `sampler`/`batch-sampler`
    // walk the bindings positionally (`inputs[i]` ↔ `Op::Input(i)`): a `logits`
    // binding marks the LM-head intrinsic and carries the sampling positions; a
    // `tensor` binding is a submit-bound device value the host gathers now. The
    // per-slot binding-map + gathered submit values ride the bridge carrier at
    // `execute()`; the driver wires each `Op::Input(i)` from the binding-map.
    async fn sampler(
        &mut self,
        this: Resource<ForwardPass>,
        program: Resource<Program>,
        inputs: Vec<pie::core::inference::InputBinding>,
    ) -> Result<()> {
        let attached = self.attach_program(&program, inputs)?;
        let pass = self.ctx().table.get_mut(&this)?;
        pass.programs.push(attached);
        Ok(())
    }

    async fn batch_sampler(
        &mut self,
        this: Resource<ForwardPass>,
        programs: Vec<Resource<Program>>,
        inputs: Vec<Vec<pie::core::inference::InputBinding>>,
    ) -> Result<()> {
        if programs.len() != inputs.len() {
            return Err(anyhow::anyhow!(
                "batch-sampler: {} programs but {} binding lists",
                programs.len(),
                inputs.len()
            ));
        }
        let mut attached = Vec::with_capacity(programs.len());
        for (program, binds) in programs.into_iter().zip(inputs) {
            attached.push(self.attach_program(&program, binds)?);
        }
        let pass = self.ctx().table.get_mut(&this)?;
        pass.programs.extend(attached);
        Ok(())
    }

    // ── WS8 pipelining primitives ──────────────────────────────────────
    // `next-input` declares that THIS pass's per-row sampled tokens feed the
    // *next* pass on this context (host-resolved in P1, device-resident in P2),
    // removing the guest output()+copy round-trip between decode steps.
    async fn next_input(&mut self, this: Resource<ForwardPass>, positions: Vec<u32>) -> Result<()> {
        let pass = self.ctx().table.get_mut(&this)?;
        pass.feed_forward = Some(positions);
        Ok(())
    }

    async fn next_attention_mask(
        &mut self,
        this: Resource<ForwardPass>,
        mask: Vec<Vec<u32>>,
    ) -> Result<()> {
        let _ = (this, mask);
        Ok(())
    }

    async fn next_adapter(
        &mut self,
        this: Resource<ForwardPass>,
        adapter: Resource<Adapter>,
    ) -> Result<()> {
        let _ = (this, adapter);
        Ok(())
    }

    async fn adapter(
        &mut self,
        this: Resource<ForwardPass>,
        adapter: Resource<Adapter>,
    ) -> Result<()> {
        let adapter_id = self.ctx().table.get(&adapter)?.adapter_id;
        let pass = self.ctx().table.get_mut(&this)?;
        pass.req.adapter_bindings[0].adapter_id = adapter_id as i64;
        Ok(())
    }

    async fn execute(&mut self, this: Resource<ForwardPass>) -> Result<()> {
        let profiling = execute_profile_enabled();
        let profile_start = profiling.then(Instant::now);
        let mut profile_sample = ExecuteProfileSample::default();
        let prepare_start = profiling.then(Instant::now);

        // WS8: drain any inter-pass pipeline link feeding THIS pass — the prior
        // pass on this context fed its per-row sampled tokens forward via
        // `next-input`. Resolve the producer (shared, non-consuming: its own
        // `output()` still reads it) and inject its tokens at the mapped input
        // positions before this pass's request is finalized. Depth-1 (no
        // `next-input`) skips this entirely.
        let incoming_ctx = self.ctx().table.get(&this)?.context_id;
        if let Some(ctx_id) = incoming_ctx
            && let Some(link) = self.pipeline_links.remove(&ctx_id)
        {
            let producer = Resource::<ForwardPass>::new_borrow(link.producer_rep);
            // Drain-before-producer-drop invariant: consumer A borrows the
            // producer by rep, so the producer pass must still be alive at drain
            // time. golf's one-ahead loop guarantees it (t+1 drains the link
            // before the guest reads+drops t); fail loud + diagnosable if a
            // future loop shape violates it, rather than silently feeding no
            // tokens.
            if self.ctx().table.get(&producer).is_err() {
                return Err(anyhow::anyhow!(
                    "WS8 next-input: producer pass was dropped before its successor \
                     drained the link (drain-before-producer-drop invariant violated)"
                ));
            }
            // Resolve the producer's (shared) output without consuming it.
            self.resolve_pending(&producer).await?;
            let toks = self
                .ctx()
                .table
                .get(&producer)
                .ok()
                .and_then(|p| p.pending.as_ref().map(|pe| pe.resolved_tokens()))
                .unwrap_or_default();
            let pass = self.ctx().table.get_mut(&this)?;
            for (row, &dest) in link.positions.iter().enumerate() {
                if dest != u32::MAX
                    && let Some(&tok) = toks.get(row)
                {
                    pass.req.token_ids.push(tok);
                    pass.req.position_ids.push(dest);
                }
            }
        }

        let pass = self.ctx().table.get_mut(&this)?;

        let model_id = pass.model_id;
        let context_id = pass
            .context_id
            .ok_or_else(|| anyhow::anyhow!("ForwardPass requires a context"))?;
        let adapter_seed = pass.adapter_seed;
        let spec_handle = pass.spec.clone();
        let allow_pass_speculation = pass.allow_pass_speculation;
        pass.allow_pass_speculation = true;
        // Whether this pass feeds its sampled tokens forward to the next pass.
        let feed_forward = pass.feed_forward.take();
        // Drain the accumulator. The remaining work is to synthesize
        // masks if absent and stamp the per-request indptrs onto the
        // ForwardRequest, then submit.
        let mut req = std::mem::replace(&mut pass.req, empty_forward_request());

        // Flatten every attached sampling program into the bridge carrier and
        // collect their declared output kinds (attach order) for response
        // reconstruction. Each program's `logits` binding positions become the
        // pass's sampling positions; its submit-bound tensor values ride the
        // carrier (the per-slot binding-map likewise, so the driver can wire
        // each `Op::Input(i)`).
        let attached = std::mem::take(&mut pass.programs);
        let has_programs = !attached.is_empty();
        let mut programs_output_kinds: Vec<Vec<pie_sampling_ir::OutputKind>> =
            Vec::with_capacity(attached.len());
        let mut logits_positions: Vec<u32> = Vec::new();
        for program in attached {
            programs_output_kinds.push(program.output_kinds);
            logits_positions.extend(program.logits_positions);
            // Carry the per-slot binding-map so the driver can wire each
            // `Op::Input(i)` from the binding-free bytecode (Logits intrinsic vs
            // keyed submit tensor).
            let bindings = program
                .bindings
                .iter()
                .map(|b| match b {
                    pie_sampling_ir::Binding::Logits => pie_driver_abi::SamplingBinding::Logits,
                    pie_sampling_ir::Binding::Tensor { key, .. } => {
                        pie_driver_abi::SamplingBinding::Tensor { key: *key }
                    }
                })
                .collect();
            req.push_sampling_program(&pie_driver_abi::SamplingProgramSubmission {
                bytecode: program.bytecode,
                inputs: program.submit_inputs,
                bindings,
                late_keys: Vec::new(),
                late_inputs: Vec::new(),
            });
        }
        if has_programs {
            // The `logits` binding(s) carry the sampling positions; fall back to
            // the single M=1 decode row (last token) when none were supplied so
            // the LM head still runs and the program samples there.
            if !logits_positions.is_empty() {
                req.sampling_indices = logits_positions;
            } else if req.sampling_indices.is_empty() && !req.token_ids.is_empty() {
                req.sampling_indices = vec![req.token_ids.len() as u32 - 1];
            }
        }

        // Track whether the user actually supplied masks; the kernel-dispatch
        // hint downstream needs to distinguish user masks from the runtime's
        // synthesized causal default.
        let has_user_mask = !req.masks.is_empty();

        // Save data needed for context::append_working_page_tokens() before
        // moving into request.
        let num_input_tokens = req.token_ids.len();
        let num_spec_tokens = req.spec_token_ids.len();
        let fill_tokens = req.token_ids.clone();
        let fill_positions = req.position_ids.clone();
        let fill_masks = if has_user_mask {
            req.masks.clone()
        } else {
            Vec::new()
        };
        // Adapter id for context::append_working_page_tokens.
        let adapter_id: Option<crate::adapter::AdapterId> = {
            let bound = req.adapter_bindings[0].adapter_id;
            if bound < 0 { None } else { Some(bound as u64) }
        };
        req.has_user_mask = has_user_mask;
        req.single_token_mode = !has_user_mask && req.token_ids.len() <= 1;
        // adapter_bindings[0] already has the adapter_id set by `adapter()`;
        // stamp the seed picked up out-of-band.
        req.adapter_bindings[0].seed = adapter_seed.unwrap_or(-1);
        if let Some(start) = prepare_start {
            profile_sample.prepare_us = elapsed_us(start.elapsed());
        }

        // Try the staged hit before synthesizing default masks or pinning. On
        // hit we skip pin/unpin entirely — the staged fire runs on pages from
        // the prior cycle.
        let driver_idx_hint = context::get_device(model_id, context_id);
        let use_pass_speculation = inference::should_use_pass_speculation(driver_idx_hint);
        let try_hit_start = profiling.then(Instant::now);
        let staged_rx = spec_handle
            .as_ref()
            .filter(|_| use_pass_speculation)
            // A program-bearing pass must take the cold path: it finalizes the
            // program carrier's per-request CSR and submits this request (the
            // staged path would replay a prior, program-less batch).
            .filter(|_| !has_programs)
            .and_then(|s| inference::try_hit(s, context_id, &req, allow_pass_speculation));
        if let Some(start) = try_hit_start {
            profile_sample.try_hit_us = elapsed_us(start.elapsed());
        }
        let (was_pinned, submit_result) = if let Some(rx) = staged_rx {
            profile_sample.hit = true;
            (false, Ok(rx))
        } else {
            let cold_prepare_start = profiling.then(Instant::now);
            // WIT spec: "if not provided, fallback to causal mask".
            if req.masks.is_empty() && !req.position_ids.is_empty() {
                req.masks = req
                    .position_ids
                    .iter()
                    .map(|&pos| Brle::all_true((pos + 1) as usize))
                    .collect();
            }
            // Finalize per-request indptr shape ([0, N]).
            let n_tokens = req.token_ids.len() as u32;
            let n_masks = req.masks.len() as u32;
            let n_logit = req.logit_masks.len() as u32;
            let n_sampling = req.sampling_indices.len() as u32;
            let n_samplers = req.n_samplers() as u32;
            let n_spec = req.spec_token_ids.len() as u32;
            req.qo_indptr = vec![0, n_tokens];
            req.mask_indptr = vec![0, n_masks];
            req.logit_mask_indptr = vec![0, n_logit];
            req.sampling_indptr = vec![0, n_sampling];
            req.sampler_indptr = vec![0, n_samplers];
            // Sampling-program carrier: per-request count CSR, mirroring
            // `sampler_indptr`. The nested per-program CSRs are already rooted
            // by `push_sampling_program`. `n_sampling_programs()` is 0 for the
            // legacy sampler path, giving `[0, 0]`.
            req.sampling_program_indptr = vec![0, req.n_sampling_programs() as u32];
            req.spec_indptr = vec![0, n_spec];
            req.kv_page_indptr = vec![0];
            req.context_ids = vec![context_id];
            if let Some(start) = cold_prepare_start {
                profile_sample.cold_prepare_us = elapsed_us(start.elapsed());
            }

            // Cold path: pin, validate page capacity, submit.
            let pin_start = profiling.then(Instant::now);
            let writable_tokens = num_input_tokens.saturating_add(num_spec_tokens);
            let pinned = match context::pin(model_id, context_id, writable_tokens as u32).await {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!("pin failed for ctx {context_id}: {e:#}");
                    return Err(anyhow::anyhow!("pin failed for ctx {context_id}: {e}"));
                }
            };
            if let Some(start) = pin_start {
                profile_sample.pin_us = elapsed_us(start.elapsed());
            }
            let kv_len = pinned.kv_len;
            let driver_id = pinned.driver;
            let physical_page_ids = pinned.pages;
            let extra_pages = pinned.extra_pages;
            if let Some(rs_slot) = pinned.rs_slot {
                // Speculation policy for rs_cache (hybrid GDN) models. Two
                // independent signals, both owned here by the runtime (the
                // driver stays pure mechanism):
                //   * `supported` — the driver wired a system drafter and the
                //     executor can verify drafts and advance the committed
                //     recurrent slot by exactly the accepted prefix. Without
                //     it, an externally-supplied draft cannot be verified, so
                //     reject it; the side channel would otherwise only force
                //     dense-logit scheduling and fragment prompt batching.
                //   * `enabled` — operator opt-in (`enable_system_speculation`,
                //     default off). Gates only the *auto*-drafter: when off we
                //     don't ask the driver to draft (`output_spec_flags`), but
                //     we still honor any manual/user-supplied draft tokens.
                let (system_spec_supported, system_spec_enabled) =
                    crate::model::get_model(model_id)
                        .map(|m| {
                            (m.system_speculation_supported(), m.enable_system_speculation())
                        })
                        .unwrap_or((false, false));
                if !system_spec_supported {
                    if !req.spec_token_ids.is_empty() {
                        context::unpin(model_id, context_id);
                        return Err(anyhow::anyhow!(
                            "rs_cache models do not support speculative draft tokens yet"
                        ));
                    }
                    req.output_spec_flags = vec![false];
                } else if !system_spec_enabled {
                    // Supported, but not opted in: no auto-drafting. Manual
                    // drafts in `req.spec_token_ids` are still verified.
                    req.output_spec_flags = vec![false];
                }
                req.rs_slot_ids = vec![rs_slot];
                req.rs_slot_flags = vec![pinned.rs_flags];
            }

            let num_pages = physical_page_ids.len() as u32;
            let page_size = context::tokens_per_page(model_id);
            let post_input_total_kv = kv_len + num_input_tokens as u32;
            let writable_total_kv = kv_len + writable_tokens as u32;
            let last_page_len = if num_pages == 0 {
                0
            } else {
                let r = post_input_total_kv % page_size;
                if r == 0 { page_size } else { r }
            };
            let active_page_idx = post_input_total_kv
                .saturating_add(page_size.saturating_sub(1))
                .checked_div(page_size)
                .and_then(|pages| pages.checked_sub(1))
                .map(|idx| idx as usize);

            // INVARIANT: total_kv must fit within the allocated pages.
            let page_capacity = num_pages * page_size;
            if writable_total_kv > page_capacity || num_pages == 0 {
                let msg = format!(
                    "KV_INVARIANT_VIOLATION ctx={context_id} total_kv={writable_total_kv} \
                     page_capacity={page_capacity} num_pages={num_pages} \
                     kv_len={kv_len} num_input={num_input_tokens} num_spec={num_spec_tokens} page_size={page_size} \
                     phys_ids={physical_page_ids:?}"
                );
                eprintln!("{msg}");
                context::unpin(model_id, context_id);
                return Err(anyhow::anyhow!(msg));
            }

            let driver_idx = driver_id as usize;
            let submit_start = profiling.then(Instant::now);
            let result = inference::submit_async(
                model_id,
                req,
                driver_idx,
                physical_page_ids,
                extra_pages,
                last_page_len,
                active_page_idx,
                allow_pass_speculation,
            );
            if let Some(start) = submit_start {
                profile_sample.submit_wait_us = elapsed_us(start.elapsed());
            }
            (true, result)
        };

        // On submit failure, unpin (if we pinned) and return early.
        let rx = match submit_result {
            Ok(rx) => rx,
            Err(e) => {
                if was_pinned {
                    context::unpin(model_id, context_id);
                }
                tracing::warn!("inference::submit failed for ctx {context_id}: {e:#}");
                return Err(anyhow::anyhow!("submit failed for ctx {context_id}: {e}"));
            }
        };

        let pending = PendingOutput {
            rx: Some(rx),
            resolved: None,
            programs_output_kinds,
            model_id,
            context_id,
            was_pinned,
            fill_tokens,
            fill_positions,
            fill_masks,
            adapter_id,
            adapter_seed,
        };
        let postprocess_start = profiling.then(Instant::now);
        // The pending always lives on the pass so `output()` resolves on every
        // pass (consumer B — the guest reads t's token for stop detection).
        self.ctx().table.get_mut(&this)?.pending = Some(pending);
        if let Some(positions) = feed_forward {
            // WS8: register a link REFERENCING this pass (by rep) so the next
            // pass on this context can read its per-row sampled tokens (consumer
            // A) without consuming the pass's output — shared resolution (a).
            self.pipeline_links.insert(
                context_id,
                PipelineLink {
                    producer_rep: this.rep(),
                    positions,
                },
            );
        }
        if let Some(start) = postprocess_start {
            profile_sample.postprocess_us = elapsed_us(start.elapsed());
        }
        if let Some(start) = profile_start {
            record_execute_profile(profile_sample, elapsed_us(start.elapsed()));
        }
        Ok(())
    }

    async fn output(
        &mut self,
        this: Resource<ForwardPass>,
    ) -> Result<Result<Vec<Resource<Tensor>>, String>> {
        match self.complete_pass(&this).await? {
            Ok(per_program) => {
                // Single-sampler `output()`: flatten the attached program's
                // declared output tensors in order. A pass with no sampler
                // yields an empty list; a batch pass should use `outputs()`.
                let mut handles = Vec::new();
                for program_tensors in per_program {
                    for t in program_tensors {
                        handles.push(self.ctx().table.push(t)?);
                    }
                }
                Ok(Ok(handles))
            }
            Err(e) => Ok(Err(e)),
        }
    }

    async fn outputs(
        &mut self,
        this: Resource<ForwardPass>,
    ) -> Result<Result<Vec<Vec<Resource<Tensor>>>, String>> {
        match self.complete_pass(&this).await? {
            Ok(per_program) => {
                let mut out = Vec::with_capacity(per_program.len());
                for program_tensors in per_program {
                    let mut handles = Vec::with_capacity(program_tensors.len());
                    for t in program_tensors {
                        handles.push(self.ctx().table.push(t)?);
                    }
                    out.push(handles);
                }
                Ok(Ok(out))
            }
            Err(e) => Ok(Err(e)),
        }
    }

    async fn drop(&mut self, this: Resource<ForwardPass>) -> Result<()> {
        // Release any still-held pin if the pass is dropped without `output()`.
        if let Ok(pass) = self.ctx().table.get_mut(&this)
            && let Some(p) = pass.pending.as_mut()
        {
            p.release_pin();
        }
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

impl InstanceState {
    /// Walk one `sampler(...)`/`batch-sampler(...)` attachment: validate the
    /// binding arity against the program's input slots, build the per-slot
    /// [`Binding`](pie_sampling_ir::Binding) map, and gather each submit-bound
    /// `tensor` value into the carrier (keyed by its slot index so the driver
    /// can wire `Op::Input(i)`). The `logits` binding's positions are captured
    /// as the program's sampling positions.
    fn attach_program(
        &mut self,
        program: &Resource<Program>,
        inputs: Vec<pie::core::inference::InputBinding>,
    ) -> Result<AttachedProgram> {
        use pie::core::inference::InputBinding;
        let (bytecode, output_kinds, num_inputs) = {
            let p = self.ctx().table.get(program)?;
            (p.cached.bytecode.clone(), p.cached.output_kinds.clone(), p.cached.num_inputs)
        };
        if inputs.len() != num_inputs {
            return Err(anyhow::anyhow!(
                "sampler: program declares {num_inputs} input slot(s) but {} binding(s) supplied",
                inputs.len()
            ));
        }
        let mut bindings = Vec::with_capacity(inputs.len());
        let mut submit_inputs = Vec::new();
        let mut logits_positions: Vec<u32> = Vec::new();
        let mut saw_logits = false;
        for (i, binding) in inputs.into_iter().enumerate() {
            match binding {
                InputBinding::Logits(positions) => {
                    if saw_logits {
                        return Err(anyhow::anyhow!(
                            "sampler: multiple logits bindings are not supported (MVP)"
                        ));
                    }
                    saw_logits = true;
                    logits_positions = positions;
                    bindings.push(pie_sampling_ir::Binding::Logits);
                }
                InputBinding::Tensor(tensor) => {
                    // Submit binding: the value is final once the inferlet has
                    // written it before attach, so gather its bytes now. The
                    // slot index is the TensorKey wired to `Op::Input(i)`.
                    let key = i as u32;
                    let data = self.ctx().table.get(&tensor)?.data.clone();
                    // The owned tensor handle is consumed by the binding.
                    self.ctx().table.delete(tensor)?;
                    bindings.push(pie_sampling_ir::Binding::Tensor {
                        key,
                        ready: pie_sampling_ir::Readiness::Submit,
                    });
                    submit_inputs.push(pie_driver_abi::SamplingInput { key, bytes: data });
                }
            }
        }
        Ok(AttachedProgram {
            bytecode,
            output_kinds,
            bindings,
            submit_inputs,
            logits_positions,
        })
    }

    /// Resolve a submitted pass's pending output **in place** (idempotent): take
    /// it out to await without holding the table borrow, then restore it with the
    /// result cached — so both `output()` (consumer B) and the WS8 link (consumer
    /// A, via the producer rep) can read the same resolution. No-op if the pass
    /// has no pending (never executed, or already consumed).
    async fn resolve_pending(&mut self, pass: &Resource<ForwardPass>) -> Result<()> {
        let Some(mut pending) = self.ctx().table.get_mut(pass)?.pending.take() else {
            return Ok(());
        };
        pending.resolve().await;
        self.ctx().table.get_mut(pass)?.pending = Some(pending);
        Ok(())
    }

    /// Drain a submitted pass for `output()`/`outputs()`: resolve it (shared with
    /// any WS8 link), then take its cached output and reconstruct each attached
    /// program's declared outputs as typed tensors. `Err` if the pass was never
    /// executed or the driver pass failed.
    async fn complete_pass(
        &mut self,
        this: &Resource<ForwardPass>,
    ) -> Result<Result<Vec<Vec<Tensor>>, String>> {
        self.resolve_pending(this).await?;
        let pass = self.ctx().table.get_mut(this)?;
        let Some(pending) = pass.pending.as_mut() else {
            return Ok(Err("output() called before execute()".to_string()));
        };
        match pending.take_resolved() {
            Ok(output) => Ok(Ok(build_program_tensors(
                output,
                &pending.programs_output_kinds,
            ))),
            Err(e) => Ok(Err(e)),
        }
    }
}

// =============================================================================
// tensor resources (pie:core/tensor) — Tensor (typed buffer) + Program
// =============================================================================

/// Expected packed byte length of a tensor with the given shape + dtype.
fn expected_byte_len(shape: &[u32], dtype: pie::core::tensor::Dtype) -> usize {
    let elems: usize = shape.iter().map(|&d| d as usize).product();
    elems * dtype_byte_size(dtype)
}

impl pie::core::tensor::Host for InstanceState {}

impl pie::core::tensor::HostTensor for InstanceState {
    async fn new(
        &mut self,
        shape: Vec<u32>,
        dtype: pie::core::tensor::Dtype,
    ) -> Result<Resource<Tensor>> {
        // Allocate a zero-filled buffer sized to shape×dtype; `write` /
        // `from-data` populate it before `execute()`.
        let data = vec![0u8; expected_byte_len(&shape, dtype)];
        Ok(self.ctx().table.push(Tensor { shape, dtype, data })?)
    }

    async fn from_data(
        &mut self,
        shape: Vec<u32>,
        dtype: pie::core::tensor::Dtype,
        data: Vec<u8>,
    ) -> Result<Result<Resource<Tensor>, String>> {
        let expected = expected_byte_len(&shape, dtype);
        if data.len() != expected {
            return Ok(Err(format!(
                "tensor data is {} bytes but shape {shape:?}/{dtype:?} needs {expected}",
                data.len()
            )));
        }
        Ok(Ok(self.ctx().table.push(Tensor { shape, dtype, data })?))
    }

    async fn shape(&mut self, this: Resource<Tensor>) -> Result<Vec<u32>> {
        Ok(self.ctx().table.get(&this)?.shape.clone())
    }

    async fn dtype(&mut self, this: Resource<Tensor>) -> Result<pie::core::tensor::Dtype> {
        Ok(self.ctx().table.get(&this)?.dtype)
    }

    async fn write(
        &mut self,
        this: Resource<Tensor>,
        data: Vec<u8>,
    ) -> Result<Result<(), String>> {
        let t = self.ctx().table.get_mut(&this)?;
        let expected = expected_byte_len(&t.shape, t.dtype);
        if data.len() != expected {
            return Ok(Err(format!(
                "tensor write is {} bytes but shape needs {expected}",
                data.len()
            )));
        }
        t.data = data;
        Ok(Ok(()))
    }

    async fn read(&mut self, this: Resource<Tensor>) -> Result<Result<Vec<u8>, String>> {
        Ok(Ok(self.ctx().table.get(&this)?.data.clone()))
    }

    async fn drop(&mut self, this: Resource<Tensor>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

impl pie::core::tensor::HostProgram for InstanceState {
    async fn new(
        &mut self,
        inputs: Vec<pie::core::tensor::Input>,
        ops: Vec<pie::core::tensor::Op>,
        outputs: Vec<u32>,
    ) -> Result<Resource<Program>> {
        // Decode + validate at construction so a malformed program is rejected
        // here (the constructor traps) rather than at attach/submit. Untrusted
        // inferlet input — `decode_program` is fully fallible (never panics).
        let program = crate::api::program_decode::decode_program(inputs, ops, outputs)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let bytecode = pie_sampling_ir::encode(&program);
        // #9: intern by the canonical bytecode hash (== the driver `ProgramHandle`).
        // Identical programs across requests/inferlets share one cached artifact —
        // `output_kinds` is derived once (skipped on a cache hit) — and the cache
        // is the #10 group / #8 hash-match fast-path registry. Host-side, no GPU.
        let cached = crate::api::program_cache::intern(&program, bytecode)
            .map_err(|e| anyhow::anyhow!("sampling program output kinds: {e:?}"))?;
        Ok(self.ctx().table.push(Program { cached })?)
    }

    async fn drop(&mut self, this: Resource<Program>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

// =============================================================================
// Grammar resource
// =============================================================================

/// A compiled grammar that describes valid output structure.
#[derive(Debug)]
pub struct Grammar {
    /// The original source string (for compiled grammar cache keying).
    pub source: String,
    /// The parsed grammar AST.
    pub inner: Arc<InternalGrammar>,
}

impl pie::core::inference::HostGrammar for InstanceState {
    async fn from_json_schema(
        &mut self,
        schema: String,
    ) -> Result<Result<Resource<Grammar>, String>> {
        match json_schema_to_grammar(&schema, &JsonSchemaOptions::default()) {
            Ok(g) => {
                let grammar = Grammar {
                    source: schema,
                    inner: Arc::new(g),
                };
                Ok(Ok(self.ctx().table.push(grammar)?))
            }
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn json(&mut self) -> Result<Resource<Grammar>> {
        let g = builtin_json_grammar()?;
        let grammar = Grammar {
            source: "__builtin_json__".into(),
            inner: Arc::new(g),
        };
        Ok(self.ctx().table.push(grammar)?)
    }

    async fn from_regex(&mut self, pattern: String) -> Result<Result<Resource<Grammar>, String>> {
        match regex_to_grammar(&pattern) {
            Ok(g) => {
                let grammar = Grammar {
                    source: pattern,
                    inner: Arc::new(g),
                };
                Ok(Ok(self.ctx().table.push(grammar)?))
            }
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn from_ebnf(&mut self, ebnf: String) -> Result<Result<Resource<Grammar>, String>> {
        match InternalGrammar::from_ebnf(&ebnf, "root") {
            Ok(g) => {
                let grammar = Grammar {
                    source: ebnf,
                    inner: Arc::new(g),
                };
                Ok(Ok(self.ctx().table.push(grammar)?))
            }
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn to_string(&mut self, this: Resource<Grammar>) -> Result<String> {
        let g = self.ctx().table.get(&this)?;
        Ok(g.inner.to_string())
    }

    async fn drop(&mut self, this: Resource<Grammar>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

// =============================================================================
// Matcher resource
// =============================================================================

/// Stateful matcher that walks the grammar automaton, producing token masks.
pub struct Matcher {
    pub(crate) inner: GrammarMatcher,
}

impl std::fmt::Debug for Matcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Matcher").finish()
    }
}

impl pie::core::inference::HostMatcher for InstanceState {
    async fn new(
        &mut self,
        grammar: Resource<Grammar>,
        tokenizer: Resource<crate::api::model::Tokenizer>,
    ) -> Result<Resource<Matcher>> {
        let grammar_res = self.ctx().table.get(&grammar)?;
        let source = grammar_res.source.clone();
        let grammar_inner = grammar_res.inner.clone();

        let tokenizer_res = self.ctx().table.get(&tokenizer)?;
        let tok = tokenizer_res.model.tokenizer().clone();
        let stop_tokens = tokenizer_res.model.instruct().seal();

        let compiled = CompiledGrammar::get_or_compile(&source, &grammar_inner, &tok);
        let inner = GrammarMatcher::with_compiled(compiled, tok, stop_tokens, 10);

        let matcher = Matcher { inner };
        Ok(self.ctx().table.push(matcher)?)
    }

    async fn accept_tokens(
        &mut self,
        this: Resource<Matcher>,
        token_ids: Vec<u32>,
    ) -> Result<Result<(), String>> {
        let matcher = self.ctx().table.get_mut(&this)?;
        for &id in &token_ids {
            if !matcher.inner.accept_token(id) {
                return Ok(Err(format!("token {} rejected by grammar", id)));
            }
        }
        Ok(Ok(()))
    }

    async fn next_token_logit_mask(&mut self, this: Resource<Matcher>) -> Result<Vec<u32>> {
        let matcher = self.ctx().table.get_mut(&this)?;
        let brle = matcher.inner.fill_next_token_brle();
        Ok(brle.buffer)
    }

    async fn is_terminated(&mut self, this: Resource<Matcher>) -> Result<bool> {
        let matcher = self.ctx().table.get(&this)?;
        Ok(matcher.inner.is_terminated())
    }

    async fn reset(&mut self, this: Resource<Matcher>) -> Result<()> {
        let matcher = self.ctx().table.get_mut(&this)?;
        matcher.inner.reset();
        Ok(())
    }

    async fn drop(&mut self, this: Resource<Matcher>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}


#[cfg(test)]
mod sampling_program_tests {
    use super::*;
    use crate::api::pie::core::tensor as wit;
    use crate::api::program_decode::decode_program;

    /// WIT-shaped parts for a greedy-argmax program over a `[vocab]` logits
    /// input: `Input(0)` then `ReduceArgmax` → one `i32` (Token) output.
    fn argmax_parts(vocab: u32) -> (Vec<wit::Input>, Vec<wit::Op>, Vec<u32>) {
        let inputs = vec![wit::Input {
            shape: vec![vocab],
            dtype: wit::Dtype::F32,
        }];
        let ops = vec![
            wit::Op {
                outputs: vec![wit::Value {
                    id: 0,
                    shape: vec![vocab],
                    dtype: wit::Dtype::F32,
                }],
                kind: wit::OpKind::Input(0),
            },
            wit::Op {
                outputs: vec![wit::Value {
                    id: 1,
                    shape: vec![],
                    dtype: wit::Dtype::I32,
                }],
                kind: wit::OpKind::ReduceArgmax(0),
            },
        ];
        (inputs, ops, vec![1])
    }

    #[test]
    fn decode_argmax_program_infers_token_output() {
        let (inputs, ops, outs) = argmax_parts(32);
        let program = decode_program(inputs, ops, outs).expect("valid argmax program");
        assert_eq!(program.inputs.len(), 1);
        let kinds = pie_sampling_ir::output_kinds(&program).expect("kinds");
        // i32 argmax output ⇒ Token (the dtype⇒kind convention (a)).
        assert_eq!(kinds, vec![pie_sampling_ir::OutputKind::Token]);
        // Bytecode round-trips through the L0 codec the carrier ships.
        let bytecode = pie_sampling_ir::encode(&program);
        let back = pie_sampling_ir::decode(&bytecode).expect("re-decode");
        assert_eq!(back, program);
    }

    #[test]
    fn decode_rejects_out_of_range_output() {
        // Output id 9 is never defined → fail loud (not a silent miscompile).
        let (inputs, ops, _) = argmax_parts(8);
        assert!(decode_program(inputs, ops, vec![9]).is_err());
    }

    #[test]
    fn decode_rejects_oversized_shape() {
        // A rank-5 input shape exceeds MAX_RANK(4) → fallible decode, never panic.
        let inputs = vec![wit::Input {
            shape: vec![1, 1, 1, 1, 1],
            dtype: wit::Dtype::F32,
        }];
        assert!(decode_program(inputs, Vec::new(), Vec::new()).is_err());
    }

    #[test]
    fn tensor_helpers_pack_little_endian() {
        let t = Tensor::i32_vec(vec![7]);
        assert_eq!(t.shape, vec![1]);
        assert!(matches!(t.dtype, wit::Dtype::I32));
        assert_eq!(t.data, 7i32.to_le_bytes());

        let f = Tensor::f32_vec(vec![2.5]);
        assert!(matches!(f.dtype, wit::Dtype::F32));
        assert_eq!(f.data, 2.5f32.to_le_bytes());
    }

    #[test]
    fn build_tensors_token_fast_path() {
        // ForwardOutput::Token + a single Token-output program ⇒ one i32 tensor.
        let per = build_program_tensors(
            ForwardOutput::Token(42),
            &[vec![pie_sampling_ir::OutputKind::Token]],
        );
        assert_eq!(per.len(), 1);
        assert_eq!(per[0].len(), 1);
        assert_eq!(per[0][0].data, 42i32.to_le_bytes());
    }

    #[test]
    fn build_tensors_mirostat_token_scalar_shape() {
        // The 4090 mirostat shape: a `[Token, Scalar]` program. The token rides
        // `resp.tokens`, the scalar surprise S the `entropies` channel; the walk
        // pulls them in declared order ⇒ [i32 token, f32 scalar].
        use pie_sampling_ir::OutputKind;
        let resp = pie_driver_abi::ForwardResponse {
            tokens: vec![137],
            entropies: vec![2.7],
            ..Default::default()
        };
        let per = build_program_tensors(
            ForwardOutput::Response(resp),
            &[vec![OutputKind::Token, OutputKind::Scalar]],
        );
        assert_eq!(per.len(), 1);
        assert_eq!(per[0].len(), 2);
        assert!(matches!(per[0][0].dtype, wit::Dtype::I32));
        assert_eq!(per[0][0].data, 137i32.to_le_bytes());
        assert!(matches!(per[0][1].dtype, wit::Dtype::F32));
        assert_eq!(per[0][1].data, 2.7f32.to_le_bytes());
    }

    #[test]
    fn program_bytecode_roundtrips_through_carrier() {
        // The host accumulator → bridge carrier → readback path: a program's
        // opaque bytecode + a submit-bound input survive untouched.
        let (inputs, ops, outs) = argmax_parts(32);
        let bytecode = pie_sampling_ir::encode(&decode_program(inputs, ops, outs).unwrap());
        let submission = pie_driver_abi::SamplingProgramSubmission {
            bytecode: bytecode.clone(),
            inputs: vec![pie_driver_abi::SamplingInput {
                key: 0,
                bytes: vec![1, 2, 3, 4],
            }],
            bindings: vec![
                pie_driver_abi::SamplingBinding::Logits,
                pie_driver_abi::SamplingBinding::Tensor { key: 0 },
            ],
            late_keys: vec![],
            late_inputs: vec![],
        };
        let mut req = pie_driver_abi::ForwardRequest::default();
        req.push_sampling_program(&submission);
        assert_eq!(req.n_sampling_programs(), 1);
        let back = req.sampling_program_at(0).unwrap();
        assert_eq!(back.bytecode, bytecode);
        // The per-slot binding-map survives the carrier round-trip.
        assert_eq!(
            back.bindings,
            vec![
                pie_driver_abi::SamplingBinding::Logits,
                pie_driver_abi::SamplingBinding::Tensor { key: 0 },
            ]
        );
    }

    #[test]
    fn feed_forward_extracts_per_row_tokens() {
        // WS8 next-input source: a pass's per-row sampled tokens (one per
        // sequence/row) are what feed the next pass — the N-token vector the
        // PipelineLink scatters by `positions`.
        assert_eq!(forward_output_tokens(&ForwardOutput::Token(7)), vec![7]);
        assert_eq!(
            forward_output_tokens(&ForwardOutput::Tokens(vec![3, 1, 4])),
            vec![3, 1, 4]
        );
        let resp = pie_driver_abi::ForwardResponse {
            tokens: vec![9, 8],
            ..Default::default()
        };
        assert_eq!(
            forward_output_tokens(&ForwardOutput::Response(resp)),
            vec![9, 8]
        );
    }
}
