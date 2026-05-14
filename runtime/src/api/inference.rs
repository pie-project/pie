//! pie:core/inference - ForwardPass, FutureOutput, Sampler, Output

use crate::api::adapter::Adapter;
use crate::api::context::Context;
use crate::api::model::Model;
use crate::api::pie;
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
use pie_bridge::Brle;
use std::iter;
use std::mem::take;
use std::sync::Arc;
use tokio::sync::oneshot;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;
use wasmtime_wasi::async_trait;
use wasmtime_wasi::p2::{DynPollable, Pollable, subscribe};

/// WASM-facing forward-pass accumulator. The WIT methods append/set
/// into `req: pie_bridge::ForwardRequest` directly — at `execute()`
/// we just finalize the per-request indptrs and submit. `model_id`
/// is WASM-side routing info (not on the wire) and `adapter_seed`
/// is stored separately because it doesn't have its own WIT setter
/// but is used at execute-time to populate the adapter binding.
#[derive(Debug)]
pub struct ForwardPass {
    pub model_id: usize,
    context_id: Option<crate::context::ContextId>,
    pub adapter_seed: Option<i64>,
    req: pie_bridge::ForwardRequest,
}

#[derive(Debug)]
pub struct FutureOutput {
    result: Option<pie::core::inference::Output>,
    rx: Option<oneshot::Receiver<pie_bridge::ForwardResponse>>,
    /// Samplers from the originating request — cloned before draining
    /// `pass.req` at execute() time so we can reconstruct the WIT
    /// per-slot output list against this slot order.
    samplers: Vec<pie_bridge::Sampler>,
    done: bool,
}

#[async_trait]
impl Pollable for FutureOutput {
    async fn ready(&mut self) {
        if self.done {
            return;
        }
        if let Some(rx) = self.rx.take() {
            match rx.await {
                Ok(resp) => {
                    self.result = Some(build_wit_output(&resp, &self.samplers));
                    self.done = true;
                }
                Err(_) => {
                    self.result = Some(pie::core::inference::Output {
                        slots: Vec::new(),
                        spec_tokens: Vec::new(),
                        spec_positions: Vec::new(),
                    });
                    self.done = true;
                }
            }
        } else {
            self.done = true;
        }
    }
}

/// Build the WIT-shaped per-slot output from a per-request
/// [`pie_bridge::ForwardResponse`] (single-request shape: `num_requests = 1`,
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
    resp: &pie_bridge::ForwardResponse,
    samplers: &[pie_bridge::Sampler],
) -> pie::core::inference::Output {
    use pie::core::inference::SlotOutput as WitSlot;
    use pie_bridge::Sampler;

    // Spec channel: pie historically returned `spec_tokens` /
    // `spec_positions` inline; the schema's ForwardResponse doesn't
    // surface them as separate fields, so they default to empty.
    let spec_tokens: Vec<u32> = Vec::new();
    let spec_positions: Vec<u32> = Vec::new();

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

    let tokens: Vec<u32> = resp.tokens.clone();
    let is_spec_walk = !tokens.is_empty()
        && tokens.len() != expected_token_slots
        && resp.dists_ids.is_empty()
        && resp.logits_bytes.is_empty()
        && resp.logprobs_values.is_empty()
        && resp.entropies.is_empty();

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
/// uses anonymous tuple variants — see `runtime/wit/core/wit/
/// inference.wit`) to the canonical [`pie_bridge::Sampler`] enum
/// (re-export of [`pie_bridge::Sampler`]). The variant set matches
/// 1:1; this is just rearranging field names.
fn wit_to_bridge_sampler(s: pie::core::inference::Sampler) -> pie_bridge::Sampler {
    use pie::core::inference::Sampler as Wit;
    match s {
        Wit::Multinomial((temperature, seed)) => {
            pie_bridge::Sampler::Multinomial { temperature, seed }
        }
        Wit::TopK((temperature, k)) => pie_bridge::Sampler::TopK { temperature, k },
        Wit::TopP((temperature, p)) => pie_bridge::Sampler::TopP { temperature, p },
        Wit::MinP((temperature, p)) => pie_bridge::Sampler::MinP { temperature, p },
        Wit::TopKTopP((temperature, k, p)) => pie_bridge::Sampler::TopKTopP { temperature, k, p },
        Wit::Embedding => pie_bridge::Sampler::Embedding,
        Wit::Dist((temperature, num_tokens)) => pie_bridge::Sampler::Dist {
            temperature,
            num_tokens,
        },
        Wit::RawLogits => pie_bridge::Sampler::RawLogits,
        Wit::Logprob(token_id) => pie_bridge::Sampler::Logprob { token_id },
        Wit::Logprobs(token_ids) => pie_bridge::Sampler::Logprobs { token_ids },
        Wit::Entropy => pie_bridge::Sampler::Entropy,
    }
}

impl pie::core::inference::Host for InstanceState {}

impl pie::core::inference::HostForwardPass for InstanceState {
    async fn new(&mut self, model: Resource<Model>) -> Result<Resource<ForwardPass>> {
        let model = self.ctx().table.get(&model)?;
        // Initialize the accumulator with the per-request invariants:
        // single adapter binding (-1 sentinels = unbound), and a default
        // `output_speculative_tokens = true` written into the single
        // entry of `output_spec_flags`.
        let pass = ForwardPass {
            model_id: model.model_id,
            context_id: None,
            adapter_seed: None,
            req: pie_bridge::ForwardRequest {
                adapter_bindings: vec![pie_bridge::AdapterBinding {
                    adapter_id: -1,
                    seed: -1,
                }],
                output_spec_flags: vec![true],
                ..Default::default()
            },
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
        let pass = self.ctx().table.get_mut(&this)?;
        pass.context_id = Some(context_id);
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
        pass.req.samplers.extend(iter::repeat_n(bridge, n));
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
        let pass = self.ctx().table.get_mut(&this)?;

        let model_id = pass.model_id;
        let context_id = pass
            .context_id
            .ok_or_else(|| anyhow::anyhow!("ForwardPass requires a context"))?;
        let adapter_seed = pass.adapter_seed;
        // Drain the accumulator. The remaining work is to synthesize
        // masks if absent and stamp the per-request indptrs onto the
        // ForwardRequest, then submit.
        let mut req = take(&mut pass.req);
        // Clone samplers BEFORE finalizing so we can reconstruct the
        // per-slot WIT output against the original slot order. (Calling
        // `req.samplers.clone()` after the indptr stamping is fine, but
        // moving it before keeps the dependency chain explicit.)
        let samplers_for_output = req.samplers.clone();

        // Track whether the user actually supplied masks; the kernel-dispatch
        // hint downstream needs to distinguish user masks from the runtime's
        // synthesized causal default.
        let has_user_mask = !req.masks.is_empty();

        // WIT spec: "if not provided, fallback to causal mask".
        // Each token at position `pos` must attend to all (pos + 1) preceding
        // positions including itself — i.e., the row is all-True over its
        // valid prefix. Under the starts-with-False BRLE convention, that's
        // a zero-length false-run prefix followed by a true run of pos+1.
        if req.masks.is_empty() && !req.position_ids.is_empty() {
            req.masks = req
                .position_ids
                .iter()
                .map(|&pos| Brle::all_true((pos + 1) as usize))
                .collect();
        }
        req.has_user_mask = has_user_mask;
        req.single_token_mode = !has_user_mask && req.token_ids.len() <= 1;
        // Finalize per-request indptr shape ([0, N]).
        let n_tokens = req.token_ids.len() as u32;
        let n_masks = req.masks.len() as u32;
        let n_logit = req.logit_masks.len() as u32;
        let n_sampling = req.sampling_indices.len() as u32;
        let n_samplers = req.samplers.len() as u32;
        let n_spec = req.spec_token_ids.len() as u32;
        req.qo_indptr = vec![0, n_tokens];
        req.mask_indptr = vec![0, n_masks];
        req.logit_mask_indptr = vec![0, n_logit];
        req.sampling_indptr = vec![0, n_sampling];
        req.sampler_indptr = vec![0, n_samplers];
        req.spec_indptr = vec![0, n_spec];
        req.kv_page_indptr = vec![0];
        req.context_ids = vec![context_id];
        // adapter_bindings[0] already has the adapter_id set by `adapter()`;
        // stamp the seed picked up out-of-band.
        req.adapter_bindings[0].seed = adapter_seed.unwrap_or(-1);

        // Save data needed for context::append_working_page_tokens() before
        // moving into request. We also clone the speculative arrays so we
        // can append the verified-prefix to the working-page lineage once
        // the response tells us how many drafts were accepted.
        let num_input_tokens = req.token_ids.len();
        let fill_tokens = req.token_ids.clone();
        let fill_positions = req.position_ids.clone();
        let fill_masks = req.masks.clone();
        let spec_tokens_for_fill = req.spec_token_ids.clone();
        let spec_positions_for_fill = req.spec_position_ids.clone();
        // Adapter id for context::append_working_page_tokens (which keeps an
        // `Option<AdapterId>`-shaped lineage); -1 sentinel means unbound.
        let adapter_id: Option<crate::adapter::AdapterId> = {
            let bound = req.adapter_bindings[0].adapter_id;
            if bound < 0 { None } else { Some(bound as u64) }
        };

        // =====================================================================
        // Context preparation — runs in THIS process's tokio task, not the
        // inference actor. This is critical: blocking here only stalls this
        // one process, not the entire inference pipeline.
        //
        // STATE MACHINE (new 3-state context module):
        //
        //   reserve_working_pages: blocks if process is suspended — actor handles
        //     restoration via drain_queues, process resumes automatically.
        //   pin: Active → Pinned (non-evictable).
        //   unpin: Pinned → Active (evictable again).
        //
        // The process calls unpin after fill, which is the ONLY
        // transition Pinned → Active. Eviction of Pinned contexts is
        // deferred via pending_suspend flag.
        // =====================================================================

        // Step 1: Resolve physical page IDs. Atomically pins the context
        // (Active → Pinned) so pages cannot be evicted during the forward pass.
        let pinned = match context::pin(model_id, context_id, num_input_tokens as u32).await {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!("pin failed for ctx {context_id}: {e:#}");
                return Ok(Err(e.to_string()));
            }
        };
        let kv_len = pinned.kv_len;
        let last_page_len = pinned.last_page_len;
        let driver_id = pinned.driver;
        let physical_page_ids = pinned.pages;

        let num_pages = physical_page_ids.len() as u32;
        let page_size = context::tokens_per_page(model_id);
        let total_kv = kv_len + num_input_tokens as u32;

        // INVARIANT: total_kv must fit within the allocated pages.
        // Violation means working pages were lost between reserve_working_pages
        // and execute — see swap lifecycle diagnostics.
        let page_capacity = num_pages * page_size;
        if total_kv > page_capacity || num_pages == 0 {
            let msg = format!(
                "KV_INVARIANT_VIOLATION ctx={context_id} total_kv={total_kv} \
                 page_capacity={page_capacity} num_pages={num_pages} \
                 kv_len={kv_len} num_input={num_input_tokens} page_size={page_size} \
                 phys_ids={physical_page_ids:?}"
            );
            eprintln!("{msg}");
            context::unpin(model_id, context_id);
            return Ok(Err(msg));
        }

        // Step 5: Submit to inference service (inference actor just dispatches — never blocks)
        let driver_idx = driver_id as usize;
        match inference::submit(
            model_id,
            req,
            driver_idx,
            physical_page_ids.clone(),
            last_page_len,
        )
        .await
        {
            Ok(output) => {
                // Diagnostic: log prefill metadata to trace corruption
                // if num_input_tokens > 1 {
                //     eprintln!(
                //         "PREFILL_RESULT ctx={context_id} kv={kv_len} np={num_pages} \
                //          inp={num_input_tokens} lpl={last_page_len} pages={physical_page_ids:?}"
                //     );
                // }
                // Diagnostic: log first decode step metadata
                // if num_input_tokens == 1 && kv_len < 45 {
                //     eprintln!(
                //         "DECODE_STEP ctx={context_id} kv={kv_len} np={num_pages} \
                //          lpl={last_page_len} pos={:?} pages={physical_page_ids:?}",
                //         fill_positions,
                //     );
                // }
                // Step 3: Mark input tokens as forwarded WHILE still Pinned
                // (non-evictable). This ensures working_page_tokens + lineage are consistent
                // before the context becomes Active (evictable).
                //
                // For speculative-decoding flows, the forward pass also
                // wrote KV for every speculative token (accepted or not).
                // The SDK's stream code accounts for this exact pattern:
                //   * lineage += pending + ALL_drafts
                //   * truncate(n_rejected) to drop the tail of rejected drafts
                //   * commit_working_pages(...) using the post-truncate state
                // So we append the FULL speculative chain here and rely on
                // the SDK to call `truncate_working_page_tokens(n_rejected)`
                // afterwards. Skipping that truncate would leave stale
                // entries in the lineage; in practice the SDK's
                // `Stream::generate_step` always issues it.
                let mut all_fill_tokens = fill_tokens;
                let mut all_fill_positions = fill_positions;
                let mut all_fill_masks = fill_masks;
                if !spec_tokens_for_fill.is_empty() {
                    all_fill_tokens.extend_from_slice(&spec_tokens_for_fill);
                    all_fill_positions.extend_from_slice(&spec_positions_for_fill);
                    // Synthesize causal masks for each spec token, matching
                    // the convention applied above for un-masked inputs.
                    for &pos in &spec_positions_for_fill {
                        all_fill_masks.push(Brle::all_true((pos + 1) as usize));
                    }
                }
                if !all_fill_tokens.is_empty() {
                    if let Err(e) = context::append_working_page_tokens(
                        model_id,
                        context_id,
                        all_fill_tokens,
                        all_fill_positions,
                        all_fill_masks,
                        adapter_id,
                        adapter_seed,
                    )
                    .await
                    {
                        context::unpin(model_id, context_id);
                        tracing::warn!("context::fill failed for ctx {context_id}: {e:#}");
                        return Ok(Err(e.to_string()));
                    }
                }

                // Unpin — forward pass completed and tokens recorded in lineage.
                // Context is now safe to evict: lineage is consistent with working_page_tokens.
                context::unpin(model_id, context_id);

                let future_output = FutureOutput {
                    result: Some(build_wit_output(&output, &samplers_for_output)),
                    rx: None,
                    samplers: samplers_for_output,
                    done: true,
                };
                Ok(Ok(self.ctx().table.push(future_output)?))
            }
            Err(e) => {
                context::unpin(model_id, context_id);
                tracing::warn!("inference::submit failed for ctx {context_id}: {e:#}");
                return Ok(Err(e.to_string()));
            }
        }
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
        if result.done {
            Ok(take(&mut result.result))
        } else {
            Ok(None)
        }
    }

    async fn drop(&mut self, this: Resource<FutureOutput>) -> Result<()> {
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
