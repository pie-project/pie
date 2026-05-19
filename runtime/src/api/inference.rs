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

/// Mirrors `types::FutureString` / `types::FutureBlob`: a oneshot
/// receiver wrapped in a WASI Pollable. The owning host task sends
/// the final `Output` through the channel; the SDK awaits the
/// pollable and reads via `get`.
#[derive(Debug)]
pub struct FutureOutput {
    receiver: oneshot::Receiver<pie::core::inference::Output>,
    result: Option<pie::core::inference::Output>,
    done: bool,
}

impl FutureOutput {
    pub fn new(receiver: oneshot::Receiver<pie::core::inference::Output>) -> Self {
        Self { receiver, result: None, done: false }
    }
}

#[async_trait]
impl Pollable for FutureOutput {
    async fn ready(&mut self) {
        if self.done {
            return;
        }
        if let Ok(res) = (&mut self.receiver).await {
            self.result = Some(res);
        }
        self.done = true;
    }
}

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
        let pass = self.ctx().table.get_mut(&this)?;

        // Extract accumulated state
        let model_id = pass.model_id;
        let tokens = take(&mut pass.input_tokens);
        let positions = take(&mut pass.input_token_positions);
        let speculative_tokens = take(&mut pass.speculative_tokens);
        let speculative_positions = take(&mut pass.speculative_positions);
        let output_speculative_tokens = pass.output_speculative_tokens;
        let masks = take(&mut pass.mask);
        // Track whether the user actually supplied masks; the kernel-dispatch
        // hint downstream needs to distinguish user masks from the runtime's
        // synthesized causal default.
        let has_user_mask = !masks.is_empty();

        // WIT spec: "if not provided, fallback to causal mask".
        // Each token at position `pos` must attend to all (pos + 1) preceding
        // positions including itself — i.e., the row is all-True over its
        // valid prefix. Under the starts-with-False BRLE convention, that's
        // a zero-length false-run prefix followed by a true run of pos+1.
        let masks = if masks.is_empty() && !positions.is_empty() {
            positions.iter().map(|&pos| Brle::all_true((pos + 1) as usize)).collect()
        } else {
            masks
        };
        let logit_mask = pass.logit_mask.take();
        let sampling_indices = take(&mut pass.output_token_indices);
        let sampler_maps = take(&mut pass.output_token_samplers);
        let adapter_id = pass.adapter.map(|id| id as u64);
        let adapter_seed = pass.adapter_seed;

        // Convert sampler maps to request::Sampler enums
        let samplers: Vec<inference::Sampler> = sampler_maps.iter()
            .map(convert_sampler)
            .collect();

        let num_input_tokens = tokens.len();
        let context_id = pass.context_id
            .ok_or_else(|| anyhow::anyhow!("ForwardPass requires a context"))?;

        // Pre-compose the lineage append set (input + speculative drafts)
        // before the request consumes `tokens` / `positions` / `masks`.
        // Replayed into working pages after the device responds. Spec
        // drafts are appended unconditionally; the SDK calls
        // `truncate_working_page_tokens(n_rejected)` after seeing the
        // verifier verdict.
        let mut all_fill_tokens = tokens.clone();
        let mut all_fill_positions = positions.clone();
        let mut all_fill_masks = masks.clone();
        if !speculative_tokens.is_empty() {
            all_fill_tokens.extend_from_slice(&speculative_tokens);
            all_fill_positions.extend_from_slice(&speculative_positions);
            for &pos in &speculative_positions {
                all_fill_masks.push(Brle::all_true((pos + 1) as usize));
            }
        }

        // Mirrors `messaging::pull` / `session::receive_file`: oneshot
        // channel + spawned task. The task drives the whole pipeline
        // (pin → submit → await → post-process) so this host fn returns
        // without awaiting anything. That's load-bearing: a host await
        // inside a sync-shaped WIT export suspends the WASM module,
        // which would serialize sibling `execute` calls from the same
        // inferlet (defeating `future::join_all` over forked branches).
        //
        // Error propagation: on any failure the task drops `tx` without
        // sending. The SDK then sees `FutureOutput.get() -> None` and
        // turns that into `Err("No output available")` — matching the
        // pre-refactor `Ok(Err(...))` shape from this fn. Detailed cause
        // is logged via `tracing::warn!`.
        let (tx, rx) = oneshot::channel();
        tokio::spawn(async move {
            let pinned = match context::pin(model_id, context_id, num_input_tokens as u32).await {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!("pin failed for ctx {context_id}: {e:#}");
                    return;
                }
            };
            let page_size = context::tokens_per_page(model_id);
            let total_kv = pinned.kv_len + num_input_tokens as u32;
            let page_capacity = pinned.pages.len() as u32 * page_size;
            if pinned.pages.is_empty() || total_kv > page_capacity {
                tracing::warn!(
                    "KV_INVARIANT_VIOLATION ctx={context_id} total_kv={total_kv} \
                     page_capacity={page_capacity} num_pages={} \
                     kv_len={} num_input={num_input_tokens} page_size={page_size} \
                     phys_ids={:?}",
                    pinned.pages.len(), pinned.kv_len, pinned.pages,
                );
                context::unpin(model_id, context_id);
                return;
            }

            let request = ForwardPassRequest {
                context_id,
                tokens,
                positions,
                speculative_tokens,
                speculative_positions,
                output_speculative_tokens,
                masks,
                has_user_mask,
                logit_mask,
                sampling_indices,
                samplers,
                adapter_id,
                adapter_seed,
            };

            let inner_rx = match inference::submit_nowait(
                model_id, request, pinned.device as usize,
                pinned.pages, pinned.last_page_len,
            ) {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!("inference::submit_nowait failed for ctx {context_id}: {e:#}");
                    context::unpin(model_id, context_id);
                    return;
                }
            };

            let raw = match inner_rx.await {
                Ok(raw) => raw,
                Err(_) => {
                    context::unpin(model_id, context_id);
                    return;
                }
            };

            // Append lineage BEFORE unpin to keep working_page_tokens
            // consistent with KV while the context is still non-evictable.
            // If append fails the response is discarded — the alternative
            // is silently leaving the next forward pass on this context
            // running against an inconsistent lineage.
            if !all_fill_tokens.is_empty() {
                if let Err(e) = context::append_working_page_tokens(
                    model_id, context_id,
                    all_fill_tokens, all_fill_positions, all_fill_masks,
                    adapter_id, adapter_seed,
                ).await {
                    tracing::warn!(
                        "context::append_working_page_tokens failed for ctx {context_id}: {e:#}",
                    );
                    context::unpin(model_id, context_id);
                    return;
                }
            }
            context::unpin(model_id, context_id);

            let _ = tx.send(convert_output(raw));
        });

        Ok(Ok(self.ctx().table.push(FutureOutput::new(rx))?))
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
        let fo = self.ctx().table.get_mut(&this)?;
        if fo.done {
            Ok(take(&mut fo.result))
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
