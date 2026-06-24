//! pie:core/inference - ForwardPass, FutureOutput, Sampler, Output

use crate::api::adapter::Adapter;
use crate::api::context::Context;
use crate::api::model::Model;
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
use tokio::sync::oneshot::error::TryRecvError;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;
use wasmtime_wasi::async_trait;
use wasmtime_wasi::p2::{DynPollable, Pollable, subscribe};

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
pub struct ForwardPass {
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
}

#[derive(Debug)]
pub struct FutureOutput {
    result: Option<pie::core::inference::Output>,
    rx: Option<oneshot::Receiver<Result<ForwardOutput>>>,
    /// Samplers from the originating request — cloned before draining
    /// `pass.req` at execute() time so we can reconstruct the WIT
    /// per-slot output list against this slot order.
    samplers: Vec<pie_driver_abi::Sampler>,
    done: bool,
    model_id: usize,
    context_id: Option<crate::context::ContextId>,
    was_pinned: bool,
    fill_tokens: Vec<u32>,
    fill_positions: Vec<u32>,
    fill_masks: Vec<Brle>,
    spec_tokens_for_fill: Vec<u32>,
    spec_positions_for_fill: Vec<u32>,
    adapter_id: Option<crate::adapter::AdapterId>,
    adapter_seed: Option<i64>,
}

impl FutureOutput {
    fn release_pin(&mut self) {
        if self.was_pinned {
            if let Some(context_id) = self.context_id {
                context::unpin(self.model_id, context_id);
            }
            self.was_pinned = false;
        }
    }

    fn append_lineage(&mut self) {
        let Some(context_id) = self.context_id else {
            return;
        };
        let mut all_fill_tokens = take(&mut self.fill_tokens);
        let mut all_fill_positions = take(&mut self.fill_positions);
        let mut all_fill_masks = take(&mut self.fill_masks);
        if !self.spec_tokens_for_fill.is_empty() {
            all_fill_tokens.extend_from_slice(&self.spec_tokens_for_fill);
            all_fill_positions.extend_from_slice(&self.spec_positions_for_fill);
            if !all_fill_masks.is_empty() {
                for &pos in &self.spec_positions_for_fill {
                    all_fill_masks.push(Brle::all_true((pos + 1) as usize));
                }
            }
        }
        if !all_fill_tokens.is_empty() {
            let driver_repaired_spec_tail = self.spec_tokens_for_fill.len() as u32;
            context::append_working_page_tokens_with_repaired_spec_tail(
                self.model_id,
                context_id,
                all_fill_tokens,
                all_fill_positions,
                all_fill_masks,
                self.adapter_id,
                self.adapter_seed,
                driver_repaired_spec_tail,
            );
        }
    }

    fn finish_ok(&mut self, output: ForwardOutput) {
        if self.spec_tokens_for_fill.is_empty() {
            if let ForwardOutput::Tokens(tokens) = &output {
                // System speculation rides accepted draft tokens in the
                // `tokens` array on top of the requested sampler slots, so
                // anything *beyond* the sampler count is speculative fill
                // that must be written into the KV. When the inferlet
                // simply samples at several positions (e.g. cacheback
                // verification), tokens.len() == samplers.len() and there
                // is no fill — treating those sampler results as spec fill
                // would over-grow the working tail past its reserved pages.
                let n_samplers = self.samplers.len();
                if tokens.len() > n_samplers {
                    let start = self
                        .fill_positions
                        .last()
                        .copied()
                        .map(|pos| pos + 1)
                        .unwrap_or(0);
                    let extra = tokens.len() - n_samplers;
                    self.spec_tokens_for_fill
                        .extend_from_slice(&tokens[..extra]);
                    self.spec_positions_for_fill
                        .extend((0..extra).map(|i| start + i as u32));
                }
            }
        }
        self.append_lineage();
        self.release_pin();
        self.result = Some(build_wit_output(output, &self.samplers));
        self.done = true;
    }

    fn finish_empty(&mut self) {
        self.release_pin();
        self.result = Some(pie::core::inference::Output {
            slots: Vec::new(),
            spec_tokens: Vec::new(),
            spec_positions: Vec::new(),
        });
        self.done = true;
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

#[async_trait]
impl Pollable for FutureOutput {
    async fn ready(&mut self) {
        if self.done {
            return;
        }
        if let Some(rx) = self.rx.as_mut() {
            let output = rx.await;
            self.rx = None;
            match output {
                Ok(Ok(resp)) => self.finish_ok(resp),
                Ok(Err(e)) => {
                    tracing::warn!("future output failed: {e:#}");
                    self.finish_empty();
                }
                Err(_) => {
                    self.finish_empty();
                }
            }
        } else {
            // Another readiness probe may have already drained rx and
            // be finishing lineage/result population. Treat rx=None while
            // done=false as in-progress rather than completing empty.
        }
    }
}

/// Build the WIT-shaped per-slot output from a per-request
/// [`pie_driver_abi::ForwardResponse`] (single-request shape: `num_requests = 1`,
/// indptrs `[0, N]`) plus the original sampler list. Walks samplers in
/// slot order, pulling one item from the matching response field per
/// slot — preserving the 1:1 mapping between `pass.sampler(...)` calls
/// and returned slots.
///
/// Spec-mode requests are detected via a token-count mismatch (the
/// verifier produces a token sequence whose length is unrelated to the
/// inferlet's sampler count); in that case all slots collapse to
/// `Token` entries.
fn build_wit_output(
    output: ForwardOutput,
    samplers: &[pie_driver_abi::Sampler],
) -> pie::core::inference::Output {
    use pie::core::inference::SlotOutput as WitSlot;

    match output {
        ForwardOutput::Token(token) => {
            return pie::core::inference::Output {
                slots: vec![WitSlot::Token(token)],
                spec_tokens: Vec::new(),
                spec_positions: Vec::new(),
            };
        }
        ForwardOutput::Tokens(tokens) => {
            let slots = tokens.into_iter().map(WitSlot::Token).collect();
            return pie::core::inference::Output {
                slots,
                spec_tokens: Vec::new(),
                spec_positions: Vec::new(),
            };
        }
        ForwardOutput::Response(resp) => build_wit_output_from_response(resp, samplers),
    }
}

fn build_wit_output_from_response(
    resp: pie_driver_abi::ForwardResponse,
    samplers: &[pie_driver_abi::Sampler],
) -> pie::core::inference::Output {
    use pie::core::inference::SlotOutput as WitSlot;
    use pie_driver_abi::Sampler;

    let (spec_tokens, spec_positions): (Vec<u32>, Vec<u32>) = if resp.spec_indptr.len() >= 2 {
        let lo = resp.spec_indptr[0] as usize;
        let hi = resp.spec_indptr[1] as usize;
        (
            resp.spec_tokens.get(lo..hi).unwrap_or(&[]).to_vec(),
            resp.spec_positions.get(lo..hi).unwrap_or(&[]).to_vec(),
        )
    } else {
        (resp.spec_tokens.clone(), resp.spec_positions.clone())
    };

    let token_payload_only = resp.dists_ids.is_empty()
        && resp.dists_probs.is_empty()
        && resp.logits_bytes.is_empty()
        && resp.logprobs_values.is_empty()
        && resp.entropies.is_empty();

    let mut expected_token_slots = 0usize;
    let mut all_samplers_token = true;
    for sampler in samplers {
        let is_token = matches!(
            sampler,
            Sampler::Multinomial { .. }
                | Sampler::TopK { .. }
                | Sampler::TopP { .. }
                | Sampler::MinP { .. }
                | Sampler::TopKTopP { .. }
        );
        if is_token {
            expected_token_slots += 1;
        } else {
            all_samplers_token = false;
        }
    }

    let tokens = resp.tokens;
    let is_spec_walk =
        token_payload_only && !tokens.is_empty() && tokens.len() != expected_token_slots;

    if token_payload_only && (all_samplers_token || is_spec_walk) {
        let slots = tokens.into_iter().map(WitSlot::Token).collect();
        return pie::core::inference::Output {
            slots,
            spec_tokens,
            spec_positions,
        };
    }

    if is_spec_walk {
        let slots = tokens.into_iter().map(WitSlot::Token).collect();
        return pie::core::inference::Output {
            slots,
            spec_tokens,
            spec_positions,
        };
    }

    let mut tok_iter = tokens.into_iter();
    // Dists: walk the kv_indptr ranges for request 0.
    let mut dist_iter: Box<dyn Iterator<Item = (Vec<u32>, Vec<f32>)>> =
        if resp.dists_kv_indptr.len() >= 2 {
            let kvs: Vec<_> = (0..resp.dists_kv_indptr.len() - 1)
                .map(|k| {
                    let lo = resp.dists_kv_indptr[k] as usize;
                    let hi = resp.dists_kv_indptr[k + 1] as usize;
                    (
                        resp.dists_ids[lo..hi].to_vec(),
                        resp.dists_probs[lo..hi].to_vec(),
                    )
                })
                .collect();
            Box::new(kvs.into_iter())
        } else {
            Box::new(std::iter::empty())
        };
    // Logits: walk the byte_indptr ranges for request 0.
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
    // Logprobs: walk the val_indptr ranges.
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
    let mut ent_iter = resp.entropies.iter().copied();

    let slots: Vec<WitSlot> = samplers
        .iter()
        .filter_map(|s| match s {
            Sampler::Multinomial { .. }
            | Sampler::TopK { .. }
            | Sampler::TopP { .. }
            | Sampler::MinP { .. }
            | Sampler::TopKTopP { .. } => tok_iter.next().map(WitSlot::Token),
            Sampler::Dist { .. } => dist_iter.next().map(WitSlot::Distribution),
            Sampler::RawLogits => logit_iter.next().map(WitSlot::Logits),
            Sampler::Logprob { .. } | Sampler::Logprobs { .. } => {
                lp_iter.next().map(WitSlot::Logprobs)
            }
            Sampler::Entropy => ent_iter.next().map(WitSlot::Entropy),
            // Embedding is reserved but not currently produced by the worker.
            Sampler::Embedding => None,
        })
        .collect();

    pie::core::inference::Output {
        slots,
        spec_tokens,
        spec_positions,
    }
}

/// Translate the WIT-defined [`pie::core::inference::Sampler`] (which
/// uses anonymous tuple variants — see `interface/inferlet/core/wit/
/// inference.wit`) to the canonical [`pie_driver_abi::Sampler`] enum
/// (re-export of [`pie_driver_abi::Sampler`]). The variant set matches
/// 1:1; this is just rearranging field names.
fn wit_to_bridge_sampler(s: pie::core::inference::Sampler) -> pie_driver_abi::Sampler {
    use pie::core::inference::Sampler as Wit;
    match s {
        Wit::Multinomial((temperature, seed)) => {
            pie_driver_abi::Sampler::Multinomial { temperature, seed }
        }
        Wit::TopK((temperature, k)) => pie_driver_abi::Sampler::TopK { temperature, k },
        Wit::TopP((temperature, p)) => pie_driver_abi::Sampler::TopP { temperature, p },
        Wit::MinP((temperature, p)) => pie_driver_abi::Sampler::MinP { temperature, p },
        Wit::TopKTopP((temperature, k, p)) => pie_driver_abi::Sampler::TopKTopP { temperature, k, p },
        Wit::Embedding => pie_driver_abi::Sampler::Embedding,
        Wit::Dist((temperature, num_tokens)) => pie_driver_abi::Sampler::Dist {
            temperature,
            num_tokens,
        },
        Wit::RawLogits => pie_driver_abi::Sampler::RawLogits,
        Wit::Logprob(token_id) => pie_driver_abi::Sampler::Logprob { token_id },
        Wit::Logprobs(token_ids) => pie_driver_abi::Sampler::Logprobs { token_ids },
        Wit::Entropy => pie_driver_abi::Sampler::Entropy,
    }
}

impl pie::core::inference::Host for InstanceState {}

impl pie::core::inference::HostForwardPass for InstanceState {
    async fn new(&mut self, model: Resource<Model>) -> Result<Resource<ForwardPass>> {
        let model = self.ctx().table.get(&model)?;
        // Initialize the accumulator with the per-request invariants:
        // single adapter binding (-1 sentinels = unbound), and no
        // speculative side-channel output unless the caller explicitly
        // enables it via `output_speculative_tokens(true)`.
        let pass = ForwardPass {
            model_id: model.model_id,
            context_id: None,
            spec: None,
            adapter_seed: None,
            allow_pass_speculation: true,
            req: empty_forward_request(),
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

    async fn input_speculative_tokens(
        &mut self,
        this: Resource<ForwardPass>,
        tokens: Vec<u32>,
        positions: Vec<u32>,
    ) -> Result<()> {
        let pass = self.ctx().table.get_mut(&this)?;
        pass.req.spec_token_ids.extend(tokens);
        pass.req.spec_position_ids.extend(positions);
        Ok(())
    }

    async fn output_speculative_tokens(
        &mut self,
        this: Resource<ForwardPass>,
        flag: bool,
    ) -> Result<()> {
        let pass = self.ctx().table.get_mut(&this)?;
        pass.req.output_spec_flags = vec![flag];
        Ok(())
    }

    async fn pass_speculation(&mut self, this: Resource<ForwardPass>, flag: bool) -> Result<()> {
        let pass = self.ctx().table.get_mut(&this)?;
        pass.allow_pass_speculation = flag;
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

    async fn logit_mask(&mut self, this: Resource<ForwardPass>, mask: Vec<u32>) -> Result<()> {
        let brle = Brle::from_vec(mask);

        let pass = self.ctx().table.get_mut(&this)?;
        pass.req.logit_masks = vec![brle];
        Ok(())
    }

    async fn sampler(
        &mut self,
        this: Resource<ForwardPass>,
        indices: Vec<u32>,
        sampler: pie::core::inference::Sampler,
    ) -> Result<()> {
        // Convert directly to the canonical bridge enum and replicate
        // once per index — no stringly-typed intermediate.
        let bridge = wit_to_bridge_sampler(sampler);
        let n = indices.len();
        let pass = self.ctx().table.get_mut(&this)?;
        for _ in 0..n {
            pass.req.push_sampler(&bridge);
        }
        pass.req.sampling_indices.extend(indices);
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

    async fn execute(
        &mut self,
        this: Resource<ForwardPass>,
    ) -> Result<Result<Resource<FutureOutput>, String>> {
        let profiling = execute_profile_enabled();
        let profile_start = profiling.then(Instant::now);
        let mut profile_sample = ExecuteProfileSample::default();
        let prepare_start = profiling.then(Instant::now);
        let pass = self.ctx().table.get_mut(&this)?;

        let model_id = pass.model_id;
        let context_id = pass
            .context_id
            .ok_or_else(|| anyhow::anyhow!("ForwardPass requires a context"))?;
        let adapter_seed = pass.adapter_seed;
        let spec_handle = pass.spec.clone();
        let allow_pass_speculation = pass.allow_pass_speculation;
        pass.allow_pass_speculation = true;
        // Drain the accumulator. The remaining work is to synthesize
        // masks if absent and stamp the per-request indptrs onto the
        // ForwardRequest, then submit.
        let mut req = std::mem::replace(&mut pass.req, empty_forward_request());
        // Clone samplers BEFORE finalizing so we can reconstruct the
        // per-slot WIT output against the original slot order.
        let samplers_for_output = req.samplers();

        // Track whether the user actually supplied masks; the kernel-dispatch
        // hint downstream needs to distinguish user masks from the runtime's
        // synthesized causal default.
        let has_user_mask = !req.masks.is_empty();

        // Save data needed for context::append_working_page_tokens() before
        // moving into request. We also clone the speculative arrays so we
        // can append the verified-prefix to the working-page lineage once
        // the response tells us how many drafts were accepted.
        let num_input_tokens = req.token_ids.len();
        let num_spec_tokens = req.spec_token_ids.len();
        let fill_tokens = req.token_ids.clone();
        let fill_positions = req.position_ids.clone();
        let fill_masks = if has_user_mask {
            req.masks.clone()
        } else {
            Vec::new()
        };
        let spec_tokens_for_fill = req.spec_token_ids.clone();
        let spec_positions_for_fill = req.spec_position_ids.clone();
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
                    return Ok(Err(e.to_string()));
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
                        return Ok(Err(
                            "rs_cache models do not support speculative draft tokens yet"
                                .to_string(),
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
                return Ok(Err(msg));
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
                return Ok(Err(e.to_string()));
            }
        };

        let future_output = FutureOutput {
            result: None,
            rx: Some(rx),
            samplers: samplers_for_output,
            done: false,
            model_id,
            context_id: Some(context_id),
            was_pinned,
            fill_tokens,
            fill_positions,
            fill_masks,
            spec_tokens_for_fill,
            spec_positions_for_fill,
            adapter_id,
            adapter_seed,
        };
        let postprocess_start = profiling.then(Instant::now);
        let pushed = self.ctx().table.push(future_output)?;
        if let Some(start) = postprocess_start {
            profile_sample.postprocess_us = elapsed_us(start.elapsed());
        }
        if let Some(start) = profile_start {
            record_execute_profile(profile_sample, elapsed_us(start.elapsed()));
        }
        Ok(Ok(pushed))
    }

    async fn drop(&mut self, this: Resource<ForwardPass>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

impl pie::core::inference::HostFutureOutput for InstanceState {
    async fn pollable(&mut self, this: Resource<FutureOutput>) -> Result<Resource<DynPollable>> {
        subscribe(self.ctx().table, this)
    }

    async fn get(
        &mut self,
        this: Resource<FutureOutput>,
    ) -> Result<Option<pie::core::inference::Output>> {
        let result = self.ctx().table.get_mut(&this)?;
        if !result.done {
            if let Some(rx) = result.rx.as_mut() {
                match rx.try_recv() {
                    Ok(Ok(resp)) => {
                        result.rx = None;
                        result.finish_ok(resp);
                    }
                    Ok(Err(e)) => {
                        result.rx = None;
                        tracing::warn!("future output failed: {e:#}");
                        result.finish_empty();
                    }
                    Err(TryRecvError::Closed) => {
                        result.rx = None;
                        result.finish_empty();
                    }
                    Err(TryRecvError::Empty) => {}
                }
            }
        }
        if result.done {
            Ok(take(&mut result.result))
        } else {
            Ok(None)
        }
    }

    async fn drop(&mut self, this: Resource<FutureOutput>) -> Result<()> {
        if let Ok(future) = self.ctx().table.get_mut(&this) {
            future.release_pin();
        }
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
