//! pie:core/inference - ForwardPass, Sampler, Output

use crate::api::adapter::Adapter;
use crate::api::pie;
use crate::inference::ForwardOutput;
use crate::inference::forward_prepare;
use crate::inference::structured::compiled_grammar::CompiledGrammar;
use crate::inference::structured::grammar::Grammar as InternalGrammar;
use crate::inference::structured::json_schema::{
    JsonSchemaOptions, builtin_json_grammar, json_schema_to_grammar,
};
use crate::inference::structured::matcher::GrammarMatcher;
use crate::inference::structured::regex::regex_to_grammar;
use crate::instance::InstanceState;
use crate::inference;
use anyhow::Result;
use pie_driver_abi::Brle;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use wasmtime::component::Resource;
use wasmtime::component::{Accessor, HasSelf};
use wasmtime_wasi::WasiView;

#[derive(Debug, Clone, serde::Serialize)]
pub struct ExecuteProfileSnapshot {
    pub calls: u64,
    pub hits: u64,
    pub misses: u64,
    pub total_us: u64,
    pub prepare_us: u64,
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
    /// Explicit forward-pass memory descriptors (W5). Captured by the
    /// kv-context / kv-output / rs-context / rs-output setters and resolved to
    /// physical pages inside the atomic arena transaction at `execute()`.
    /// There is no ambient context handle and no implicit append.
    kv_context: Option<pie::core::inference::KvContext>,
    kv_output: Option<pie::core::inference::KvOutput>,
    rs_context: Option<pie::core::inference::RsBufferContext>,
    rs_output: Option<pie::core::inference::RsBufferOutput>,
    /// `fold-buffered(n)` (W9 piggyback): fold the first `n` buffered RS tokens
    /// of this pass's RS working set into its folded state as part of this
    /// forward. Lowered to `rs_fold_lens` + `RS_FLAG_FOLD` over the buffered
    /// slabs (`rs_buffer_slot_ids`); the driver gathers + replays them.
    fold_buffered_tokens: Option<u32>,
    pub adapter_seed: Option<i64>,
    req: pie_driver_abi::ForwardRequest,
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

/// Aggregate interface-level `Host` for `pie:core/working-set`, required by
/// the generated `HostKvWorkingSet` (charlie) + `HostRsWorkingSet` (delta)
/// resource impls. echo owns this (central bindgen) since it spans both lanes.
impl pie::core::working_set::Host for InstanceState {}

impl pie::core::inference::HostForwardPass for InstanceState {
    async fn new(&mut self) -> Result<Resource<ForwardPass>> {
        // Initialize the accumulator with the per-request invariants:
        // single adapter binding (-1 sentinels = unbound), and no
        // speculative side-channel output unless the caller explicitly
        // enables it via `output_speculative_tokens(true)`. Single-model:
        // the bound model is index 0.
        let pass = ForwardPass {
            model_id: 0,
            kv_context: None,
            kv_output: None,
            rs_context: None,
            rs_output: None,
            fold_buffered_tokens: None,
            adapter_seed: None,
            req: empty_forward_request(),
        };
        Ok(self.ctx().table.push(pass)?)
    }

    /// KV pages this pass reads as attention context. Replaces the old opaque
    /// `context` handle (W5). Resolved to physical pages in the txn prepare.
    async fn kv_context(
        &mut self,
        this: Resource<ForwardPass>,
        ctx: pie::core::inference::KvContext,
    ) -> Result<()> {
        self.ctx().table.get_mut(&this)?.kv_context = Some(ctx);
        Ok(())
    }

    /// KV pages this pass writes (with per-page valid lengths + captured
    /// generation for stale-mutation rejection).
    async fn kv_output(
        &mut self,
        this: Resource<ForwardPass>,
        out: pie::core::inference::KvOutput,
    ) -> Result<()> {
        self.ctx().table.get_mut(&this)?.kv_output = Some(out);
        Ok(())
    }

    /// Buffered recurrent state this pass reads (hybrid / linear-attention).
    async fn rs_context(
        &mut self,
        this: Resource<ForwardPass>,
        ctx: pie::core::inference::RsBufferContext,
    ) -> Result<()> {
        self.ctx().table.get_mut(&this)?.rs_context = Some(ctx);
        Ok(())
    }

    /// Buffered recurrent state this pass writes (without folding; W10).
    async fn rs_output(
        &mut self,
        this: Resource<ForwardPass>,
        out: pie::core::inference::RsBufferOutput,
    ) -> Result<()> {
        self.ctx().table.get_mut(&this)?.rs_output = Some(out);
        Ok(())
    }

    /// Fold the first `tokens` buffered RS tokens of this pass's RS working set
    /// into its folded recurrent state as part of this forward (W9 piggyback).
    /// Recorded here; `execute()` lowers it to `rs_fold_lens` + `RS_FLAG_FOLD`
    /// over the buffered slabs so the driver gathers + replays them in-forward.
    async fn fold_buffered(&mut self, this: Resource<ForwardPass>, tokens: u32) -> Result<()> {
        self.ctx().table.get_mut(&this)?.fold_buffered_tokens = Some(tokens);
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

    async fn pass_speculation(&mut self, _this: Resource<ForwardPass>, _flag: bool) -> Result<()> {
        // Runtime pass-level speculation chains are removed (W15); the runtime
        // keeps no hidden speculative state in working-set semantics. Manual
        // draft tokens still flow via `input-speculative-tokens`. Inert now.
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

    async fn drop(&mut self, this: Resource<ForwardPass>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

impl pie::core::inference::HostForwardPassWithStore<InstanceState> for HasSelf<InstanceState> {
    /// Native async (WASI P3 component-model-async). There is no guest-visible
    /// `future-output` resource anymore: `execute` runs prepare → submit →
    /// await → finalize inline and returns the WIT `output` directly. Because
    /// the commit/abort runs in THIS async fn (where the store is reachable via
    /// `accessor.with`), the forward txn is finalized atomically with the pass —
    /// the lost-KV-commit failure mode of the old pollable/get split (commit
    /// unreachable from `&mut FutureOutput`) is structurally impossible.
    async fn execute(
        accessor: &Accessor<InstanceState, Self>,
        this: Resource<ForwardPass>,
    ) -> Result<Result<pie::core::inference::Output, String>> {
        // Drain the accumulator: the explicit memory descriptors + the staged
        // ForwardRequest. There is no ambient context handle (W5). Every store /
        // resource-table touch in this async func goes through `accessor.with`
        // (P3: the host async fn has no `&mut self`).
        let (
            model_id,
            adapter_seed,
            kv_context,
            kv_output,
            _rs_context,
            rs_output,
            fold_buffered_tokens,
            mut req,
        ) = accessor.with(|mut access| -> Result<_> {
            let pass = access.get().ctx().table.get_mut(&this)?;
            Ok((
                pass.model_id,
                pass.adapter_seed,
                pass.kv_context.take(),
                pass.kv_output.take(),
                pass.rs_context.take(),
                pass.rs_output.take(),
                pass.fold_buffered_tokens.take(),
                std::mem::replace(&mut pass.req, empty_forward_request()),
            ))
        })?;
        // v1: single-driver. Multi-driver binds the working set's device on
        // first materialization (`bind_driver`), wired at consolidation.
        let driver_idx = 0usize;
        let samplers_for_output = req.samplers();

        // Empty-input guard: a forward must compute at least one query row.
        // Without input rows `qo_indptr` collapses to `[0, 0]` and the pass is a
        // no-op submit; the old context API rejected this. Image/audio spans
        // push placeholder rows into `token_ids`, so this covers all input kinds.
        if let Err(e) = forward_prepare::check_input_nonempty(req.token_ids.len()) {
            return Ok(Err(format!("{e:?}")));
        }

        // WIT spec: "if not provided, fallback to causal mask". Then stamp the
        // per-request indptr shape ([0, N]).
        let has_user_mask = !req.masks.is_empty();
        if req.masks.is_empty() && !req.position_ids.is_empty() {
            req.masks = req
                .position_ids
                .iter()
                .map(|&pos| Brle::all_true((pos + 1) as usize))
                .collect();
        }
        req.has_user_mask = has_user_mask;
        req.single_token_mode = !has_user_mask && req.token_ids.len() <= 1;
        req.adapter_bindings[0].seed = adapter_seed.unwrap_or(-1);
        req.qo_indptr = vec![0, req.token_ids.len() as u32];
        req.mask_indptr = vec![0, req.masks.len() as u32];
        req.logit_mask_indptr = vec![0, req.logit_masks.len() as u32];
        req.sampling_indptr = vec![0, req.sampling_indices.len() as u32];
        req.sampler_indptr = vec![0, req.n_samplers() as u32];
        req.spec_indptr = vec![0, req.spec_token_ids.len() as u32];
        req.kv_page_indptr = vec![0];
        // Batch-affinity id: the KV working set's resource handle replaces the
        // old context id (used by the scheduler for request grouping).
        let affinity = kv_output
            .as_ref()
            .map(|o| o.set.rep())
            .or_else(|| kv_context.as_ref().map(|c| c.set.rep()))
            .unwrap_or(0);
        req.context_ids = vec![affinity as u64];

        let page_size = crate::page_size::tokens_per_page(model_id);

        // ── prepare: validate descriptors; alloc/CoW + pin write targets; pin
        //    read pages; resolve to driver physical ids — all under one txn.
        //    The whole prepare is synchronous, so it runs inside one
        //    `accessor.with` closure (store + arena both reachable); the owned
        //    `txn` + projection cross back out for the async submit. ──
        type PrepOut = (
            forward_prepare::KvProjection,
            crate::arena::ArenaTxn,
            Vec<crate::arena::MovePlan>,
        );
        let prepared: std::result::Result<PrepOut, String> = accessor.with(|mut access| {
            let state = access.get();
            let arena_arc = crate::arena::get(model_id, driver_idx);
            let mut arena = arena_arc.lock().unwrap();
            let mut txn = arena.txn_begin();
            let mut move_plans: Vec<crate::arena::MovePlan> = Vec::new();

            type InnerOut = (
                forward_prepare::KvProjection,
                Vec<u32>,
                Vec<u8>,
                Vec<u32>,
                Vec<u32>,
                Vec<u32>,
            );
            let inner: std::result::Result<InnerOut, String> = 'prep: {
                // KV read context → pinned physical pages. Read only the written
                // valid-token prefix: kv-context `len` may include trailing
                // reserved slots (the WIT permits "trailing reserved slots may be
                // empty") that are not part of attention and may be unwritten;
                // resolving the full `len` would reject them. `valid_pages` is the
                // ceil of valid-tokens. Pure prefill (valid_tokens==0) reads none.
                let (context_pages, valid_tokens) = if let Some(kc) = &kv_context {
                    let valid_pages = kc.valid_tokens.div_ceil(page_size);
                    let objs = if valid_pages == 0 {
                        Vec::new()
                    } else {
                        match state.ctx().table.get(&kc.set) {
                            Ok(ws) => match ws.resolve_read(kc.start, valid_pages) {
                                Ok(o) => o,
                                Err(e) => break 'prep Err(e.to_string()),
                            },
                            Err(e) => break 'prep Err(e.to_string()),
                        }
                    };
                    let mut pages = Vec::with_capacity(objs.len());
                    for obj in &objs {
                        if let Err(e) = arena.txn_pin(&mut txn, *obj) {
                            break 'prep Err(e.to_string());
                        }
                        match arena.blocks(*obj) {
                            Ok(b) => pages.push(b[0]),
                            Err(e) => break 'prep Err(e.to_string()),
                        }
                    }
                    (pages, kc.valid_tokens)
                } else {
                    (Vec::new(), 0)
                };

                // KV write outputs → CoW'd + pinned physical pages.
                let mut writes: Vec<forward_prepare::KvWrite> = Vec::new();
                if let Some(ko) = &kv_output {
                    if ko.indices.len() != ko.per_page_valid_lens.len() {
                        break 'prep Err(format!(
                            "kv-output indices ({}) and per-page-valid-lens ({}) length mismatch",
                            ko.indices.len(),
                            ko.per_page_valid_lens.len()
                        ));
                    }
                    // Validate generation / range / uniqueness BEFORE any mutation.
                    match state.ctx().table.get(&ko.set) {
                        Ok(ws) => {
                            if let Err(e) = ws.resolve_write(&ko.indices, ko.generation) {
                                break 'prep Err(e.to_string());
                            }
                        }
                        Err(e) => break 'prep Err(e.to_string()),
                    }
                    for (i, &idx) in ko.indices.iter().enumerate() {
                        let cow = {
                            let ws = match state.ctx().table.get_mut(&ko.set) {
                                Ok(w) => w,
                                Err(e) => break 'prep Err(e.to_string()),
                            };
                            ws.cow_write_slot(idx, &mut txn, &mut arena)
                        };
                        let (obj, move_plan) = match cow {
                            Ok(v) => v,
                            Err(e) => break 'prep Err(e.to_string()),
                        };
                        if let Some(mp) = move_plan {
                            move_plans.push(mp);
                        }
                        if let Err(e) = arena.txn_pin(&mut txn, obj) {
                            break 'prep Err(e.to_string());
                        }
                        let page = match arena.blocks(obj) {
                            Ok(b) => b[0],
                            Err(e) => break 'prep Err(e.to_string()),
                        };
                        writes.push(forward_prepare::KvWrite {
                            slot_index: idx,
                            page,
                            valid_len: ko.per_page_valid_lens[i],
                        });
                    }
                }

                // Project onto the contiguous driver page run + last-page length.
                let proj = match forward_prepare::project_kv(
                    &context_pages,
                    valid_tokens,
                    &writes,
                    page_size,
                ) {
                    Ok(p) => p,
                    Err(e) => break 'prep Err(format!("{e:?}")),
                };

                // RS v1 (write + in-forward fold only; `rs-context` read deferred
                // to v2 — the hybrid read+write case needs per-request token-range
                // ABI). Buffered slabs ride `rs_buffer_slot_ids` (page-major CSR:
                // slab s = tokens [s·page, (s+1)·page)); the FOLD bit disambiguates
                // — clear = rs-output write-target (W10, write_state=false), set =
                // fold read-source. `rs_slot_ids[r]` is the folded state slot.
                let mut rs_slot_ids: Vec<u32> = Vec::new();
                let mut rs_slot_flags: Vec<u8> = Vec::new();
                let mut rs_buffer_slot_ids: Vec<u32> = Vec::new();
                let mut rs_buffer_slot_indptr: Vec<u32> = vec![0];
                let mut rs_fold_lens: Vec<u32> = Vec::new();
                if let Some(ro) = &rs_output {
                    // Materialise + pin the buffered write slabs, page-major.
                    let cow = {
                        let ws = match state.ctx().table.get_mut(&ro.set) {
                            Ok(w) => w,
                            Err(e) => break 'prep Err(e.to_string()),
                        };
                        ws.cow_write_buffer(ro.start_token, ro.len_tokens, &mut txn, &mut arena)
                    };
                    let (objs, move_plan) = match cow {
                        Ok(v) => v,
                        Err(e) => break 'prep Err(e.to_string()),
                    };
                    if let Some(mp) = move_plan {
                        move_plans.push(mp);
                    }
                    for obj in &objs {
                        if let Err(e) = arena.txn_pin(&mut txn, *obj) {
                            break 'prep Err(e.to_string());
                        }
                        match arena.blocks(*obj) {
                            Ok(b) => rs_buffer_slot_ids.push(b[0]),
                            Err(e) => break 'prep Err(e.to_string()),
                        }
                    }
                    rs_buffer_slot_indptr.push(rs_buffer_slot_ids.len() as u32);

                    // The folded recurrent_state slot (driver reads/writes it).
                    let folded = match state.ctx().table.get(&ro.set) {
                        Ok(ws) => ws.folded_object(),
                        Err(e) => break 'prep Err(e.to_string()),
                    };
                    let folded_block = match folded {
                        Some(obj) => match arena.blocks(obj) {
                            Ok(b) => b[0],
                            Err(e) => break 'prep Err(e.to_string()),
                        },
                        None => 0,
                    };
                    rs_slot_ids.push(folded_block);

                    // FOLD bit + `rs_fold_lens` iff this pass folds the buffered
                    // suffix into the folded state in-forward (W9 piggyback).
                    let mut flag = 0u8;
                    if let Some(n) = fold_buffered_tokens {
                        flag |= pie_driver_abi::RS_FLAG_FOLD;
                        rs_fold_lens.push(n);
                    } else {
                        rs_fold_lens.push(0);
                    }
                    rs_slot_flags.push(flag);
                } else if fold_buffered_tokens.is_some() {
                    // v1: the fold rides the rs-output set; a fold-only pass with
                    // no rs-output to carry the RS set isn't expressible yet (v2: a
                    // fold descriptor names its own set).
                    break 'prep Err(
                        "fold-buffered without rs-output is unsupported in v1 (rs-output carries the RS set)"
                            .to_string(),
                    );
                }

                Ok((
                    proj,
                    rs_slot_ids,
                    rs_slot_flags,
                    rs_buffer_slot_ids,
                    rs_buffer_slot_indptr,
                    rs_fold_lens,
                ))
            };

            // On any prepare failure: abort the txn (discard staged allocs/CoW
            // copies, release pins) and revert any repointed KV slots; the prior
            // mappings stay visible (W13). No driver submission happened.
            let (proj, rs_slot_ids, rs_slot_flags, rs_buffer_slot_ids, rs_buffer_slot_indptr, rs_fold_lens) =
                match inner {
                    Ok(v) => v,
                    Err(e) => {
                        arena.txn_abort(txn);
                        drop(arena);
                        if let Some(ko) = &kv_output {
                            if let Ok(ws) = state.ctx().table.get_mut(&ko.set) {
                                ws.abort_writes();
                            }
                        }
                        return Err(e);
                    }
                };

            if !rs_slot_ids.is_empty() {
                req.rs_slot_ids = rs_slot_ids;
                req.rs_slot_flags = rs_slot_flags;
                req.rs_fold_lens = rs_fold_lens;
            }
            if !rs_buffer_slot_ids.is_empty() {
                req.rs_buffer_slot_ids = rs_buffer_slot_ids;
                req.rs_buffer_slot_indptr = rs_buffer_slot_indptr;
            }

            // Release the arena lock BEFORE the async submit; the owned `txn`
            // keeps the pins/CoW copies alive until commit/abort.
            drop(arena);
            Ok((proj, txn, move_plans))
        });

        let (proj, txn, move_plans) = match prepared {
            Ok(v) => v,
            Err(e) => return Ok(Err(e)),
        };

        // Issue the device d2d for every CoW'd write target: copy the original
        // page content into the private copy before the driver writes into it.
        for mp in &move_plans {
            if let Err(e) = crate::driver::copy_d2d(driver_idx, &mp.from, &mp.to) {
                tracing::warn!("forward CoW d2d copy failed: {e:#}");
            }
        }

        // CAS-seal eligibility (W6/W7): host-hash the full pages this forward
        // fills from an EMPTY context (`valid_tokens == 0`), so the forward's
        // tokens start at a page boundary and each page's whole content is known
        // here. Chained from prev_hash 0. Sealing onto a non-empty context needs
        // the context tip's sealed hash (follow-up); until then those pages stay
        // private-dirty (W7) — correct, just no dedup.
        let seal_eligible = !proj.full_page_writes.is_empty()
            && kv_context.as_ref().map(|c| c.valid_tokens).unwrap_or(0) == 0
            && req.position_ids.len() == req.token_ids.len()
            && req.masks.len() == req.token_ids.len();
        let seal_hashes: Vec<(u32, u64)> = if seal_eligible {
            let page_hashes = crate::page_hash::compute_page_hashes(
                page_size as usize,
                &req.token_ids,
                &req.position_ids,
                &req.masks,
                0,
                adapter_seed,
            );
            proj
                .full_page_writes
                .iter()
                .filter_map(|&slot| page_hashes.get(slot as usize).map(|&h| (slot, h)))
                .collect()
        } else {
            Vec::new()
        };

        // Carry the KV working-set handle + the txn across the async boundary;
        // finalize (after the driver round-trip) commits (seal full pages) /
        // aborts on them.
        let kv_set: Option<Resource<crate::working_set::kv::KvWorkingSet>> = kv_output
            .as_ref()
            .map(|ko| Resource::new_borrow(ko.set.rep()));

        // The RS working set whose folded boundary advances on a committed
        // fold-buffered (W9) — v1 rides the rs-output set.
        let rs_fold_set: Option<Resource<crate::working_set::rs::RsWorkingSet>> =
            if fold_buffered_tokens.is_some() {
                rs_output.as_ref().map(|ro| Resource::new_borrow(ro.set.rep()))
            } else {
                None
            };

        // Single-model: the SERVICE routes to the bound model; no model_id arg.
        let submit_result =
            inference::submit_async(req, driver_idx, proj.physical_page_ids, proj.last_page_len);

        let rx = match submit_result {
            Ok(rx) => rx,
            Err(e) => {
                // Submit never reached the driver — abort the txn + revert the
                // repointed KV slots (W13).
                accessor.with(|mut access| {
                    let state = access.get();
                    let arena_arc = crate::arena::get(model_id, driver_idx);
                    arena_arc.lock().unwrap().txn_abort(txn);
                    if let Some(ko) = &kv_set {
                        if let Ok(ws) = state.ctx().table.get_mut(ko) {
                            ws.abort_writes();
                        }
                    }
                });
                tracing::warn!("inference::submit failed: {e:#}");
                return Ok(Err(e.to_string()));
            }
        };

        // Await the driver result INLINE (P3 native async), then finalize the
        // forward txn in the same async fn (store reachable via `accessor.with`).
        let forward_result: Option<ForwardOutput> = match rx.await {
            Ok(Ok(resp)) => Some(resp),
            Ok(Err(e)) => {
                tracing::warn!("future output failed: {e:#}");
                None
            }
            Err(_) => None,
        };
        let success = forward_result.is_some();

        accessor.with(|mut access| {
            access.get().finalize_forward_txn(
                success,
                txn,
                kv_set,
                seal_hashes,
                model_id,
                driver_idx,
                rs_fold_set,
                fold_buffered_tokens,
            )
        })?;

        let output = match forward_result {
            Some(resp) => build_wit_output(resp, &samplers_for_output),
            None => pie::core::inference::Output {
                slots: Vec::new(),
                spec_tokens: Vec::new(),
                spec_positions: Vec::new(),
            },
        };
        Ok(Ok(output))
    }
}

impl InstanceState {
    /// Commit (on driver success) or abort (on failure) the forward transaction
    /// from `execute()`. Commit releases pins, publishes the CoW'd write targets
    /// (`commit_writes`), and CAS-seals eligible full pages; abort discards
    /// staged objects and reverts repointed slots.
    #[allow(clippy::too_many_arguments)]
    fn finalize_forward_txn(
        &mut self,
        success: bool,
        txn: crate::arena::ArenaTxn,
        kv_set: Option<Resource<crate::working_set::kv::KvWorkingSet>>,
        seal_hashes: Vec<(u32, u64)>,
        model_id: usize,
        driver_idx: usize,
        rs_fold_set: Option<Resource<crate::working_set::rs::RsWorkingSet>>,
        fold_tokens: Option<u32>,
    ) -> Result<()> {
        let arena_arc = crate::arena::get(model_id, driver_idx);
        if success {
            // Commit, publish the repointed slots, then CAS-seal eligible full
            // pages. Lock order is arena → kv_cas, both held sync (no await).
            let mut arena = arena_arc.lock().unwrap();
            arena
                .txn_commit(txn)
                .map_err(|e| anyhow::anyhow!("forward txn_commit failed: {e}"))?;
            if let Some(kv_set) = &kv_set {
                let cas_arc = crate::working_set::kv_cas::get(model_id, driver_idx);
                let mut cas = cas_arc.lock().unwrap();
                if let Ok(ws) = self.ctx().table.get_mut(kv_set) {
                    ws.commit_writes();
                    for (slot, hash) in &seal_hashes {
                        if let Err(e) = ws.seal(*slot, *hash, &mut arena, &mut cas) {
                            tracing::warn!("CAS seal of slot {slot} failed: {e}");
                        }
                    }
                }
            }
            // Advance the RS folded boundary on a committed in-forward fold (W9):
            // consume the first `n` buffered tokens into the folded state. Only
            // on success — a fold never advances across an aborted forward.
            if let (Some(n), Some(rs_set)) = (fold_tokens, &rs_fold_set) {
                if let Ok(ws) = self.ctx().table.get_mut(rs_set) {
                    if let Err(e) = ws.advance_fold(n, &mut arena) {
                        tracing::warn!("advance_fold({n}) failed: {e:?}");
                    }
                }
            }
        } else {
            {
                let mut arena = arena_arc.lock().unwrap();
                arena.txn_abort(txn);
            }
            if let Some(kv_set) = &kv_set {
                if let Ok(ws) = self.ctx().table.get_mut(kv_set) {
                    ws.abort_writes();
                }
            }
        }
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
    async fn new(&mut self, grammar: Resource<Grammar>) -> Result<Resource<Matcher>> {
        let grammar_res = self.ctx().table.get(&grammar)?;
        let source = grammar_res.source.clone();
        let grammar_inner = grammar_res.inner.clone();

        // Single-model: the tokenizer comes from the global bound model.
        let model = crate::model::model();
        let tok = model.tokenizer().clone();
        let stop_tokens = model.instruct().seal();

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
