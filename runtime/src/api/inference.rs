//! pie:core/inference - ForwardPass, FutureOutput, Sampler, Output

use crate::api::pie;
use crate::api::context::Context;
use crate::api::model::Model;
use crate::api::adapter::Adapter;
use crate::instance::InstanceState;
use crate::inference::brle::Brle;
use crate::inference::request::{ForwardPassRequest, ForwardPassOutput, SlotOutput};
use crate::inference::structured::grammar::Grammar as InternalGrammar;
use crate::inference::structured::json_schema::{builtin_json_grammar, json_schema_to_grammar, JsonSchemaOptions};
use crate::inference::structured::regex::regex_to_grammar;
use crate::inference::structured::compiled_grammar::CompiledGrammar;
use crate::inference::structured::matcher::GrammarMatcher;
use crate::{context, inference};
use anyhow::Result;
use std::collections::HashMap;
use std::iter;
use std::mem::take;
use std::sync::Arc;
use tokio::sync::oneshot;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;
use wasmtime_wasi::async_trait;
use wasmtime_wasi::p2::{DynPollable, Pollable, subscribe};

#[derive(Debug)]
pub struct ForwardPass {
    pub model_id: usize,
    context_id: Option<crate::context::ContextId>,
    /// Snapshot of the bound ctx's cached speculator handle. Set in
    /// `pass.context()`; `None` until then, or when speculation is
    /// disabled for the model. Lets `execute()` call `try_hit`
    /// without taking the global REGISTRY lock.
    spec: Option<inference::StagedBatch>,
    input_tokens: Vec<u32>,
    input_token_positions: Vec<u32>,
    speculative_tokens: Vec<u32>,
    speculative_positions: Vec<u32>,
    mask: Vec<Brle>,
    logit_mask: Option<Brle>,
    output_token_indices: Vec<u32>,
    output_token_samplers: Vec<HashMap<String, rmpv::Value>>,
    output_speculative_tokens: bool,
    adapter: Option<u32>,
    pub adapter_seed: Option<i64>,
}

#[derive(Debug)]
pub struct FutureOutput {
    result: Option<pie::core::inference::Output>,
    rx: Option<oneshot::Receiver<ForwardPassOutput>>,
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
                Ok(output) => {
                    self.result = Some(convert_output(output));
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

/// Convert internal ForwardPassOutput (per-slot results + spec channel) to
/// the WIT `Output` record.
fn convert_output(output: ForwardPassOutput) -> pie::core::inference::Output {
    let slots = output
        .slots
        .into_iter()
        .map(|s| match s {
            SlotOutput::Token(t) => pie::core::inference::SlotOutput::Token(t),
            SlotOutput::Distribution(ids, ps) => {
                pie::core::inference::SlotOutput::Distribution((ids, ps))
            }
            SlotOutput::Logits(b) => pie::core::inference::SlotOutput::Logits(b),
            SlotOutput::Logprobs(v) => pie::core::inference::SlotOutput::Logprobs(v),
            SlotOutput::Entropy(h) => pie::core::inference::SlotOutput::Entropy(h),
            SlotOutput::Embedding(b) => pie::core::inference::SlotOutput::Embedding(b),
        })
        .collect();
    pie::core::inference::Output {
        slots,
        spec_tokens: output.spec_tokens,
        spec_positions: output.spec_positions,
    }
}

/// Convert a sampler HashMap to a request::Sampler enum.
fn convert_sampler(map: &HashMap<String, rmpv::Value>) -> inference::Sampler {
    let sampler_type = map.get("sampler")
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as u32;
    let temperature = map.get("temperature")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as f32;

    match sampler_type {
        0 => {
            let num_tokens = map.get("top_k").and_then(|v| v.as_u64()).unwrap_or(1) as u32;
            inference::Sampler::Dist { temperature, num_tokens }
        }
        1 => {
            let seed = map.get("seed").and_then(|v| v.as_u64()).map(|s| s as u32);
            inference::Sampler::Multinomial { temperature, seed }
        }
        2 => {
            let k = map.get("top_k").and_then(|v| v.as_u64()).unwrap_or(50) as u32;
            inference::Sampler::TopK { temperature, k }
        }
        3 => {
            let p = map.get("top_p").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
            inference::Sampler::TopP { temperature, p }
        }
        4 => {
            let p = map.get("min_p").and_then(|v| v.as_f64()).unwrap_or(0.05) as f32;
            inference::Sampler::MinP { temperature, p }
        }
        5 => {
            let k = map.get("top_k").and_then(|v| v.as_u64()).unwrap_or(50) as u32;
            let p = map.get("top_p").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
            inference::Sampler::TopKTopP { temperature, k, p }
        }
        6 => inference::Sampler::Embedding,
        7 => inference::Sampler::RawLogits,
        8 => {
            let token_id = map.get("token_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            inference::Sampler::Logprob { token_id }
        }
        9 => {
            let token_ids = map
                .get("token_ids")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_u64()).map(|x| x as u32).collect())
                .unwrap_or_default();
            inference::Sampler::Logprobs { token_ids }
        }
        10 => inference::Sampler::Entropy,
        _ => inference::Sampler::Multinomial { temperature, seed: None },
    }
}

enum SamplerType {
    Distribution = 0,
    Multinomial = 1,
    TopK = 2,
    TopP = 3,
    MinP = 4,
    TopKTopP = 5,
    Embedding = 6,
    RawLogits = 7,
    Logprob = 8,
    Logprobs = 9,
    Entropy = 10,
}

impl pie::core::inference::Host for InstanceState {}

impl pie::core::inference::HostForwardPass for InstanceState {
    async fn new(&mut self, model: Resource<Model>) -> Result<Resource<ForwardPass>> {
        let model = self.ctx().table.get(&model)?;
        let pass = ForwardPass {
            model_id: model.model_id,
            context_id: None,
            spec: None,
            input_tokens: vec![],
            input_token_positions: vec![],
            speculative_tokens: vec![],
            speculative_positions: vec![],
            mask: vec![],
            logit_mask: None,
            output_token_indices: vec![],
            output_token_samplers: vec![],
            output_speculative_tokens: true, // enabled by default
            adapter: None,
            adapter_seed: None,
        };
        Ok(self.ctx().table.push(pass)?)
    }

    async fn context(&mut self, this: Resource<ForwardPass>, context: Resource<Context>) -> Result<()> {
        let ctx = self.ctx().table.get(&context)?;
        let context_id = ctx.context_id;
        let model_id = ctx.model_id;
        // Initialize the ctx's speculator cache on the first call
        // for this ctx. The OnceLock makes this lock-free on every
        // subsequent `pass.context()`, eliminating REGISTRY lookups
        // from the per-iteration hot path.
        let spec = ctx.spec.get_or_init(|| {
            let device_idx = context::get_device(model_id, context_id);
            inference::lookup_for_ctx(model_id, device_idx)
        }).clone();
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
        pass.input_tokens.extend(tokens);
        pass.input_token_positions.extend(positions);
        Ok(())
    }

    async fn input_speculative_tokens(
        &mut self,
        this: Resource<ForwardPass>,
        tokens: Vec<u32>,
        positions: Vec<u32>,
    ) -> Result<()> {
        let pass = self.ctx().table.get_mut(&this)?;
        pass.speculative_tokens.extend(tokens);
        pass.speculative_positions.extend(positions);
        Ok(())
    }

    async fn output_speculative_tokens(
        &mut self,
        this: Resource<ForwardPass>,
        flag: bool,
    ) -> Result<()> {
        let pass = self.ctx().table.get_mut(&this)?;
        pass.output_speculative_tokens = flag;
        Ok(())
    }

    async fn attention_mask(&mut self, this: Resource<ForwardPass>, mask: Vec<Vec<u32>>) -> Result<()> {
        let brle_masks: Vec<Brle> = mask.into_iter().map(Brle::from_vec).collect();

        let pass = self.ctx().table.get_mut(&this)?;
        pass.mask = brle_masks;
        Ok(())
    }

    async fn logit_mask(&mut self, this: Resource<ForwardPass>, mask: Vec<u32>) -> Result<()> {
        let brle = Brle::from_vec(mask);

        let pass = self.ctx().table.get_mut(&this)?;
        pass.logit_mask = Some(brle);
        Ok(())
    }

    async fn sampler(
        &mut self,
        this: Resource<ForwardPass>,
        indices: Vec<u32>,
        sampler: pie::core::inference::Sampler,
    ) -> Result<()> {
        let mut sampler_map = HashMap::new();
        
        match sampler {
            pie::core::inference::Sampler::Multinomial((temp, seed)) => {
                sampler_map.insert("sampler".to_string(), rmpv::Value::from(SamplerType::Multinomial as u32));
                sampler_map.insert("temperature".to_string(), rmpv::Value::from(temp));
                sampler_map.insert("seed".to_string(), rmpv::Value::from(seed));
            }
            pie::core::inference::Sampler::TopK((temp, k)) => {
                sampler_map.insert("sampler".to_string(), rmpv::Value::from(SamplerType::TopK as u32));
                sampler_map.insert("temperature".to_string(), rmpv::Value::from(temp));
                sampler_map.insert("top_k".to_string(), rmpv::Value::from(k));
            }
            pie::core::inference::Sampler::TopP((temp, p)) => {
                sampler_map.insert("sampler".to_string(), rmpv::Value::from(SamplerType::TopP as u32));
                sampler_map.insert("temperature".to_string(), rmpv::Value::from(temp));
                sampler_map.insert("top_p".to_string(), rmpv::Value::from(p));
            }
            pie::core::inference::Sampler::MinP((temp, p)) => {
                sampler_map.insert("sampler".to_string(), rmpv::Value::from(SamplerType::MinP as u32));
                sampler_map.insert("temperature".to_string(), rmpv::Value::from(temp));
                sampler_map.insert("min_p".to_string(), rmpv::Value::from(p));
            }
            pie::core::inference::Sampler::TopKTopP((temp, k, p)) => {
                sampler_map.insert("sampler".to_string(), rmpv::Value::from(SamplerType::TopKTopP as u32));
                sampler_map.insert("temperature".to_string(), rmpv::Value::from(temp));
                sampler_map.insert("top_k".to_string(), rmpv::Value::from(k));
                sampler_map.insert("top_p".to_string(), rmpv::Value::from(p));
            }
            pie::core::inference::Sampler::Dist((temp, k)) => {
                sampler_map.insert("sampler".to_string(), rmpv::Value::from(SamplerType::Distribution as u32));
                sampler_map.insert("temperature".to_string(), rmpv::Value::from(temp));
                sampler_map.insert("top_k".to_string(), rmpv::Value::from(k));
            }
            pie::core::inference::Sampler::Embedding => {
                sampler_map.insert("sampler".to_string(), rmpv::Value::from(SamplerType::Embedding as u32));
            }
            pie::core::inference::Sampler::RawLogits => {
                sampler_map.insert("sampler".to_string(), rmpv::Value::from(SamplerType::RawLogits as u32));
            }
            pie::core::inference::Sampler::Logprob(token_id) => {
                sampler_map.insert("sampler".to_string(), rmpv::Value::from(SamplerType::Logprob as u32));
                sampler_map.insert("token_id".to_string(), rmpv::Value::from(token_id));
            }
            pie::core::inference::Sampler::Logprobs(token_ids) => {
                sampler_map.insert("sampler".to_string(), rmpv::Value::from(SamplerType::Logprobs as u32));
                let arr: Vec<rmpv::Value> = token_ids.iter().map(|&x| rmpv::Value::from(x)).collect();
                sampler_map.insert("token_ids".to_string(), rmpv::Value::Array(arr));
            }
            pie::core::inference::Sampler::Entropy => {
                sampler_map.insert("sampler".to_string(), rmpv::Value::from(SamplerType::Entropy as u32));
            }
        }

        let pass = self.ctx().table.get_mut(&this)?;
        pass.output_token_samplers.extend(iter::repeat(sampler_map).take(indices.len()));
        pass.output_token_indices.extend(indices);
        Ok(())
    }

    async fn adapter(&mut self, this: Resource<ForwardPass>, adapter: Resource<Adapter>) -> Result<()> {
        let adapter_id = self.ctx().table.get(&adapter)?.adapter_id;
        let pass = self.ctx().table.get_mut(&this)?;
        pass.adapter = Some(adapter_id as u32);
        Ok(())
    }

    async fn execute(
        &mut self,
        this: Resource<ForwardPass>,
    ) -> Result<Result<Resource<FutureOutput>, String>> {
        // 1. Drain the accumulated WIT state into owned values.
        let pass = self.ctx().table.get_mut(&this)?;
        let model_id = pass.model_id;
        let context_id = pass.context_id
            .ok_or_else(|| anyhow::anyhow!("ForwardPass requires a context"))?;
        let tokens = take(&mut pass.input_tokens);
        let positions = take(&mut pass.input_token_positions);
        let speculative_tokens = take(&mut pass.speculative_tokens);
        let speculative_positions = take(&mut pass.speculative_positions);
        let output_speculative_tokens = pass.output_speculative_tokens;
        let user_masks = take(&mut pass.mask);
        let has_user_mask = !user_masks.is_empty();
        let logit_mask = pass.logit_mask.take();
        let sampling_indices = take(&mut pass.output_token_indices);
        let samplers: Vec<inference::Sampler> = take(&mut pass.output_token_samplers)
            .iter().map(convert_sampler).collect();
        let adapter_id = pass.adapter.map(|id| id as u64);
        let adapter_seed = pass.adapter_seed;
        let spec_handle = pass.spec.take();
        let num_input_tokens = tokens.len() as u32;

        // 2. WIT default: when the inferlet didn't supply attention
        //    masks, synthesize a causal mask per input token. Each row
        //    is all-True over its (pos+1)-long valid prefix; under the
        //    starts-with-False BRLE convention that's a zero-length
        //    false run followed by a true run of (pos+1).
        let masks = if has_user_mask {
            user_masks
        } else {
            positions.iter().map(|&pos| Brle::all_true((pos + 1) as usize)).collect()
        };

        // 3. Pre-compute the fill payload for
        //    `append_working_page_tokens` — the lineage append that
        //    runs on the success path. Real inputs + the full
        //    speculative-draft chain; the SDK truncates rejected
        //    drafts afterwards via `truncate_working_page_tokens`.
        //    Speculative tokens get synthesized causal masks.
        let mut fill_tokens = tokens.clone();
        let mut fill_positions = positions.clone();
        let mut fill_masks = masks.clone();
        fill_tokens.extend_from_slice(&speculative_tokens);
        fill_positions.extend_from_slice(&speculative_positions);
        for &pos in &speculative_positions {
            fill_masks.push(Brle::all_true((pos + 1) as usize));
        }

        let request = ForwardPassRequest {
            context_id,
            tokens, positions, masks,
            speculative_tokens, speculative_positions, output_speculative_tokens,
            has_user_mask,
            logit_mask,
            sampling_indices, samplers,
            adapter_id, adapter_seed,
        };

        // 4. Try the lock-free staged hit before pinning. On hit we
        //    skip pin/unpin entirely (the staged fire is running on
        //    pages from the prior cycle). On miss we pin + submit.
        //    The ctx-cached `spec` handle lets us skip the REGISTRY
        //    lookup that used to live inside `try_hit`.
        let (was_pinned, submit_result) = if let Some(rx) = spec_handle
            .as_ref()
            .and_then(|s| inference::try_hit(s, context_id, &request))
        {
            (false, rx.await.map_err(|_| anyhow::anyhow!("staged rx dropped")))
        } else {
                let p = match context::pin(model_id, context_id, num_input_tokens).await {
                    Ok(p) => p,
                    Err(e) => {
                        tracing::warn!("pin failed for ctx {context_id}: {e:#}");
                        return Ok(Err(e.to_string()));
                    }
                };
                let result = inference::submit(
                    model_id, request,
                    p.device as usize, p.pages, p.extra_pages, p.last_page_len,
                ).await;
                (true, result)
            };

        // 5. On submit failure, unpin (if we pinned) and return early.
        let output = match submit_result {
            Ok(o) => o,
            Err(e) => {
                if was_pinned { context::unpin(model_id, context_id); }
                tracing::warn!("inference::submit failed for ctx {context_id}: {e:#}");
                return Ok(Err(e.to_string()));
            }
        };

        // 6. Queue the lineage append on the context actor. Fire-and-
        //    forget: errors get logged at the handler. Subsequent
        //    ops on this ctx (`unpin`, the next `pin`, `commit`,
        //    `truncate`, eviction) all go through the same mpsc and
        //    are naturally ordered behind this message, so every
        //    state-reading consumer sees a consistent post-append
        //    view — just slightly later in wall time.
        if !fill_tokens.is_empty() {
            context::append_working_page_tokens(
                model_id, context_id,
                fill_tokens, fill_positions, fill_masks,
                adapter_id, adapter_seed,
            );
        }

        // 7. Unpin (no-op on staged hit) and hand the output back.
        if was_pinned { context::unpin(model_id, context_id); }
        let future_output = FutureOutput {
            result: Some(convert_output(output)),
            rx: None,
            done: true,
        };
        Ok(Ok(self.ctx().table.push(future_output)?))
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

    async fn get(&mut self, this: Resource<FutureOutput>) -> Result<Option<pie::core::inference::Output>> {
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

    async fn from_regex(
        &mut self,
        pattern: String,
    ) -> Result<Result<Resource<Grammar>, String>> {
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

    async fn from_ebnf(
        &mut self,
        ebnf: String,
    ) -> Result<Result<Resource<Grammar>, String>> {
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

    async fn next_token_logit_mask(
        &mut self,
        this: Resource<Matcher>,
    ) -> Result<Vec<u32>> {
        let matcher = self.ctx().table.get_mut(&this)?;
        let brle = matcher.inner.fill_next_token_brle();
        Ok(brle.buffer)
    }

    async fn is_terminated(
        &mut self,
        this: Resource<Matcher>,
    ) -> Result<bool> {
        let matcher = self.ctx().table.get(&this)?;
        Ok(matcher.inner.is_terminated())
    }

    async fn reset(
        &mut self,
        this: Resource<Matcher>,
    ) -> Result<()> {
        let matcher = self.ctx().table.get_mut(&this)?;
        matcher.inner.reset();
        Ok(())
    }

    async fn drop(&mut self, this: Resource<Matcher>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}
