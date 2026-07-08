//! pie:core/inference - ForwardPass + sampler programs; pie:core/tensor -
//! Tensor + Program resources.

use crate::api::pie;
use crate::grammar::compiled_grammar::CompiledGrammar;
use crate::grammar::grammar::Grammar as InternalGrammar;
use crate::grammar::json_schema::{
    JsonSchemaOptions, builtin_json_grammar, json_schema_to_grammar,
};
use crate::grammar::matcher::GrammarMatcher;
use crate::grammar::regex::regex_to_grammar;
use crate::instance::InstanceState;
use anyhow::Result;
use pie_driver_abi::Brle;
use std::sync::Arc;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

// Engine execute pipeline moved to `crate::inference::execute` (mechanical
// split). Re-exported at crate visibility so the WIT Host impls below call the
// moved free fns and the external `api::inference::execute_profile_snapshot`
// path keeps resolving.
pub(crate) use crate::inference::execute::*;

pub struct ForwardPass {
    /// Set when the ctx is bound via `context()` (the new WIT `forward-pass`
    /// constructor takes no model — the context carries the model identity).
    pub model_id: usize,
    /// Merged kv-working-set descriptor (#21 WIT `kv-working-set`): read context
    /// `[inp_start, inp_start+inp_len)` with `valid_tokens` prefix, write the
    /// CONTIGUOUS slot range `[output_start, output_start+output_len)` whose first
    /// written row sits at in-page token `offset`. Resolved to physical pages in
    /// the atomic arena transaction at `execute()`. Replaces the split
    /// kv-context/kv-output records (no generation/indices — the inferlet owns
    /// working-set correctness).
    pub(crate) kv_ws: Option<KvWorkingSetDesc>,
    /// Merged rs-working-set descriptor (#21 WIT `rs-working-set`): the buffered
    /// recurrent-state token range `[start_token, start_token+len_tokens)`.
    pub(crate) rs_ws: Option<RsWorkingSetDesc>,
    /// `rs-fold-buffered(n)` (W9 piggyback): fold the first `n` buffered RS tokens
    /// of this pass's RS working set into its folded state as part of this
    /// forward. Lowered to `rs_fold_lens` + `RS_FLAG_FOLD` over the buffered
    /// slabs (`rs_buffer_slot_ids`); the driver gathers + replays them.
    pub(crate) fold_buffered_tokens: Option<u32>,
    pub adapter_seed: Option<i64>,
    pub(crate) req: pie_driver_abi::ForwardRequest,
    /// `next-inputs(positions)` (#21 run-ahead carrier, delta's L-map): the NEXT
    /// pass's input slots the carrier fills with THIS pass's sampled token. The
    /// host assigns the link L (`runahead::apply_next_input_carrier`); the
    /// guest threads no link-ids.
    pub(crate) next_input_positions: Vec<u32>,
    /// `next-attention-mask(mask)` (#21 run-ahead): the attention-mask carrier
    /// for the next pass (parallel to `next-inputs`). Recorded here; the carrier
    /// wiring rides delta's next-input L-map. Empty on the greedy-decode path.
    pub(crate) next_attention_mask: Vec<Brle>,
    /// #21 1c run-ahead: the in-flight forward state stored by the sync
    /// `execute()` (eager-submit) and consumed by the async `output()`/`outputs()`
    /// (await→finalize→tensor). `None` until `execute()`; the forward-pass IS the
    /// in-flight handle (Option A).
    pub(crate) pending: Option<PendingForward>,
    /// A prepare/submit error deferred from the SYNC `execute()` (the WIT
    /// `execute: func()` returns nothing, so a recoverable failure can't surface
    /// there). The async `output()`/`outputs()` reports it as the WIT `error`.
    pub(crate) exec_error: Option<String>,
    /// `fresh-generate()` (#26): this pass is the FIRST forward of a new
    /// `generate()` on its context. The run-ahead next-input carry
    /// (`pending_next_input`) lives per-instance, so a prior generate's terminal
    /// producer can leave a dangling carry; this flag tells `execute()` to drop
    /// (and free) any dangling carry for THIS context before the carrier's
    /// consumer-inject, so the new generate's prime never injects a stale token.
    /// The guest's `Generator` sets it once per generate.
    pub(crate) fresh_generate: bool,
}

/// Merged kv-working-set descriptor — see [`ForwardPass::kv_ws`].
pub(crate) struct KvWorkingSetDesc {
    pub(crate) set: Resource<crate::working_set::kv::KvWorkingSet>,
    pub(crate) inp_start: u32,
    /// Read-context page count `[inp_start, inp_start+inp_len)`. Retained for WIT
    /// descriptor completeness; the host read derives the pinned page count from
    /// `valid_tokens` (the valid attention prefix) — `inp_len`'s trailing
    /// reserved slots may be unwritten and must not be resolved. In the disjoint
    /// (Option-B) convention `inp_len == valid_tokens.div_ceil(page_size)`.
    #[allow(dead_code)]
    pub(crate) inp_len: u32,
    pub(crate) valid_tokens: u32,
    pub(crate) output_start: u32,
    pub(crate) output_len: u32,
    pub(crate) offset: u32,
}

/// Merged rs-working-set descriptor — see [`ForwardPass::rs_ws`].
pub(crate) struct RsWorkingSetDesc {
    pub(crate) set: Resource<crate::working_set::rs::RsWorkingSet>,
    // The `rs-working-set(start-token, len-tokens)` buffer token-range. Unused by
    // the in-forward folded write (basic GDN decode reads/writes the folded slot,
    // not a buffered suffix); RESERVED for the parked Ph7 buffered fold-from-slabs
    // path, which resolves the buffered write range from these.
    #[allow(dead_code)]
    pub(crate) start_token: u32,
    #[allow(dead_code)]
    pub(crate) len_tokens: u32,
}

impl pie::core::inference::Host for InstanceState {}

/// Aggregate interface-level `Host` for `pie:core/working-set`, required by
/// the generated `HostKvWorkingSet` (charlie) + `HostRsWorkingSet` (delta)
/// resource impls. echo owns this (central bindgen) since it spans both lanes.
impl pie::core::working_set::Host for InstanceState {}

impl pie::core::inference::HostForwardPass for InstanceState {
    async fn new(&mut self) -> Result<Resource<ForwardPass>> {
        // Initialize the accumulator with the per-request invariants:
        // single adapter binding (-1 sentinels = unbound). Single-model: the
        // bound model is index 0. P3 explicit working-set descriptors
        // (kv/rs-context, kv/rs-output) are bound by their setters; there is no
        // ambient context handle.
        let pass = ForwardPass {
            model_id: 0,
            kv_ws: None,
            rs_ws: None,
            fold_buffered_tokens: None,
            adapter_seed: None,
            req: empty_forward_request(),
            next_input_positions: Vec::new(),
            next_attention_mask: Vec::new(),
            pending: None,
            exec_error: None,
            fresh_generate: false,
        };
        Ok(self.ctx().table.push(pass)?)
    }

    /// Merged kv-working-set descriptor (#21 WIT `kv-working-set`): read context
    /// = prior FULL pages `[inp_start, inp_start+inp_len)` with `valid_tokens`
    /// prefix; write = the contiguous new-KV range `[output_start,
    /// output_start+output_len)`; `offset` = in-page token offset of the first
    /// written row (the first write page's partial-prior prefix → last-page
    /// valid_len). Read ⊎ write disjoint (Option B); resolved to physical pages
    /// in the txn prepare. The inferlet owns working-set correctness — no
    /// generation/indices guard.
    #[allow(clippy::too_many_arguments)]
    async fn kv_working_set(
        &mut self,
        this: Resource<ForwardPass>,
        set: Resource<crate::working_set::kv::KvWorkingSet>,
        inp_start: u32,
        inp_len: u32,
        valid_tokens: u32,
        output_start: u32,
        output_len: u32,
        offset: u32,
    ) -> Result<()> {
        let set = Resource::new_borrow(set.rep());
        self.ctx().table.get_mut(&this)?.kv_ws = Some(KvWorkingSetDesc {
            set,
            inp_start,
            inp_len,
            valid_tokens,
            output_start,
            output_len,
            offset,
        });
        Ok(())
    }

    // ── #21 run-ahead next-input carrier (delta's host L-map) ───────────
    /// `next-inputs(positions)`: the NEXT pass's input slots the carrier fills
    /// with THIS pass's sampled token. Host owns the link L (assigned at
    /// `execute()` via [`crate::inference::runahead::apply_next_input_carrier`]);
    /// the guest threads no link-ids (replaces the 3 granular link setters).
    async fn next_inputs(
        &mut self,
        this: Resource<ForwardPass>,
        positions: Vec<u32>,
    ) -> Result<()> {
        self.ctx().table.get_mut(&this)?.next_input_positions = positions;
        Ok(())
    }

    /// `set-pipeline-source-kind(kind)`: the run-ahead carrier retain-SOURCE tag
    /// for this pass's `next-inputs` window (ratified drafts-channel carrier-kind).
    /// `0 = PrevSample` (default, `pi.sampled` — byte-identical to the shipped
    /// single-token carry), `1 = PrevDrafts` (the MTP `[k+1]` `[seed, drafts]`
    /// per-link COPY buffer). Recorded on the pass's `ForwardRequest`;
    /// `apply_next_input_carrier` reads `req.pipeline_source_kind` at `execute()`.
    async fn set_pipeline_source_kind(
        &mut self,
        this: Resource<ForwardPass>,
        kind: u8,
    ) -> Result<()> {
        self.ctx()
            .table
            .get_mut(&this)?
            .req
            .set_pipeline_source_kind(kind);
        Ok(())
    }

    /// `next-attention-mask(mask)`: the attention-mask carrier for the next pass
    /// (run-ahead mask carryover, parallel to `next-inputs`). Recorded; the
    /// carrier wiring rides delta's L-map. Inert on the greedy-decode path.
    async fn next_attention_mask(
        &mut self,
        this: Resource<ForwardPass>,
        mask: Vec<Vec<u32>>,
    ) -> Result<()> {
        let brle_masks: Vec<Brle> = mask.into_iter().map(Brle::from_vec).collect();
        self.ctx().table.get_mut(&this)?.next_attention_mask = brle_masks;
        Ok(())
    }

    /// #26 `fresh-generate()`: mark this pass as the first forward of a new
    /// `generate()` so `execute()` drops any dangling next-input carry left on
    /// this context by a prior generate's terminal producer (the stop-terminal /
    /// explicit-restart path; golf's loop omits the carry on the predictable
    /// max-boundary terminal). No-arg flag; the context is the pass's
    /// kv-working-set.
    async fn fresh_generate(&mut self, this: Resource<ForwardPass>) -> Result<()> {
        self.ctx().table.get_mut(&this)?.fresh_generate = true;
        Ok(())
    }

    /// Merged rs-working-set descriptor (#21 WIT `rs-working-set`): the buffered
    /// recurrent-state token range `[start_token, start_token+len_tokens)`
    /// (read+write). Materialised + pinned in the txn prepare.
    async fn rs_working_set(
        &mut self,
        this: Resource<ForwardPass>,
        set: Resource<crate::working_set::rs::RsWorkingSet>,
        start_token: u32,
        len_tokens: u32,
    ) -> Result<()> {
        let set = Resource::new_borrow(set.rep());
        self.ctx().table.get_mut(&this)?.rs_ws = Some(RsWorkingSetDesc {
            set,
            start_token,
            len_tokens,
        });
        Ok(())
    }

    /// Fold the first `tokens` buffered RS tokens of this pass's RS working set
    /// into its folded recurrent state as part of this forward (W9 piggyback).
    /// Recorded here; `execute()` lowers it to `rs_fold_lens` + `RS_FLAG_FOLD`
    /// over the buffered slabs so the driver gathers + replays them in-forward.
    async fn rs_fold_buffered(&mut self, this: Resource<ForwardPass>, tokens: u32) -> Result<()> {
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
            for p in crate::model::multimodal::qwen_mrope_positions(grid, anchor) {
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

    async fn output_speculative_tokens(
        &mut self,
        this: Resource<ForwardPass>,
        flag: bool,
    ) -> Result<()> {
        let pass = self.ctx().table.get_mut(&this)?;
        pass.req.output_spec_flags = vec![flag];
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
    async fn drop(&mut self, this: Resource<ForwardPass>) -> Result<()> {
        // P3 `execute()` is inline (prepare→submit→await→finalize in one async
        // fn), so a dropped pass holds no pending pin to release — just free the
        // resource-table entry.
        self.ctx().table.delete(this)?;
        Ok(())
    }

    /// #21: SYNC eager-submit (`execute: func()` — no return). Prepares + submits
    /// the forward and stores the in-flight [`PendingForward`] on the forward-pass
    /// (Option A — the pass IS the in-flight handle); a recoverable prepare/submit
    /// failure is deferred to `output()`/`outputs()` (stored as `exec_error`),
    /// since `execute` has no `error` channel. Delegates to the free
    /// [`execute_impl`] (the sync `HostForwardPass` trait gives `&mut self`, so no
    /// `accessor` is threaded).
    async fn execute(&mut self, this: Resource<ForwardPass>) -> Result<()> {
        execute_impl(self, this).await
    }
}
/// v2 active self-suspend cycle (shared by the victim prologue and the
/// `SelfSuspendFirst` requester-yield path). Saves `set`'s working set — D2H
/// offloads its uniquely-owned pages + releases shared refs — reports the freed
/// blocks to `orch`, parks until the restore phase releases it, then
/// re-materialises (H2D). Returns the freed block count: **0** means nothing was
/// suspended (no reclaimable page, or a grace-blocked set) — the caller decides
/// (the victim prologue `decline_park`s; the requester path just retries). Every
/// arena/WS lock is dropped before the `.await` park (guru's invariant: hold NO
/// lock across a park). A restore-race `OutOfBlocks` re-reports the SAME
/// `freed_now` and re-parks (bounded — fail loud rather than hang).
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

    async fn mask(&mut self, this: Resource<Matcher>) -> Result<Vec<u32>> {
        let matcher = self.ctx().table.get_mut(&this)?;
        // The packed allowed-token bitmask (`[ceil(vocab/32)]` u32, bit 1 =
        // allowed) — the `mask-apply` (0x65) mask operand. Returned directly,
        // no BRLE round-trip.
        Ok(matcher.inner.fill_next_token_mask())
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


// The sampling-IR program tests (`is_recognized_standard`, `decode_program`,
// `build_output_tensors`, bytecode round-trips) were removed with the
// sampling-IR sampler subsystem (ptir succeeds it: outputs are channels, not
// declared OutputKinds). Only the transport-neutral producer-abort predicate
// test survives.
#[cfg(test)]
mod producer_abort_tests {
    use super::*;

    // #23 verify seam targeting predicate (env-free core of `test_force_producer_abort`).
    #[test]
    fn abort_target_matches_only_configured_producer() {
        // Unset target ⇒ never matches (ZERO production behavior when env unset).
        assert!(!abort_target_matches(Some(2), None));
        assert!(!abort_target_matches(None, None));
        // Configured target ⇒ matches ONLY the producer for that exact link.
        assert!(abort_target_matches(Some(2), Some(2)));
        assert!(!abort_target_matches(Some(3), Some(2))); // a different producer
        assert!(!abort_target_matches(None, Some(2))); // a non-producer pass
    }
}
