//! pie:core/inference - ForwardPass, FutureOutput, Sampler, Output

use crate::api::pie;
use crate::api::context::Context;
use crate::api::model::Model;
use crate::api::adapter::Adapter;
use crate::linker::InstanceState;
use crate::inference::brle::Brle;
use crate::inference::request::{ForwardPassRequest, ForwardPassOutput};
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
use std::time::Instant;
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
                    self.result = Some(pie::core::inference::Output::None);
                    self.done = true;
                }
            }
        } else {
            self.done = true;
        }
    }
}

/// Convert internal ForwardPassOutput to WIT Output variant.
fn convert_output(output: ForwardPassOutput) -> pie::core::inference::Output {
    match output {
        ForwardPassOutput::None => pie::core::inference::Output::None,
        ForwardPassOutput::Tokens(tokens) => pie::core::inference::Output::Tokens(tokens),
        ForwardPassOutput::TokensWithSpeculation(accepted, spec_tokens, spec_positions) => {
            pie::core::inference::Output::TokensWithSpeculation((accepted, spec_tokens, spec_positions))
        }
        ForwardPassOutput::Embeddings(embeddings) => pie::core::inference::Output::Embeddings(embeddings),
        ForwardPassOutput::Distributions(dists) => pie::core::inference::Output::Distributions(dists),
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
        let context_id = pass.context_id;
        let tokens = take(&mut pass.input_tokens);
        let positions = take(&mut pass.input_token_positions);
        let speculative_tokens = take(&mut pass.speculative_tokens);
        let speculative_positions = take(&mut pass.speculative_positions);
        let output_speculative_tokens = pass.output_speculative_tokens;
        let masks = take(&mut pass.mask);

        // WIT spec: "if not provided, fallback to causal mask"
        let masks = if masks.is_empty() && !positions.is_empty() {
            positions.iter().map(|&pos| Brle::new((pos + 1) as usize)).collect()
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

        // Save data needed for context::fill() before moving into request
        let num_input_tokens = tokens.len();
        let fill_positions = positions.clone();
        let fill_masks = masks.clone();

        // Build the forward pass request
        let request = ForwardPassRequest {
            context_id,
            tokens,
            positions,
            speculative_tokens,
            speculative_positions,
            output_speculative_tokens,
            masks,
            logit_mask,
            sampling_indices,
            samplers,
            adapter_id,
            adapter_seed,
            arrival_time: Some(Instant::now()),
        };

        // Submit to inference service
    match inference::forward_pass(model_id, request).await {
        Ok(output) => {
            // Mark input tokens as forwarded in the context
            if let Some(ctx_id) = context_id {
                if num_input_tokens > 0 {
                    context::fill(
                        model_id, ctx_id, num_input_tokens,
                        fill_positions, fill_masks, adapter_id,
                    )?;
                }
            }

            let future_output = FutureOutput {
                result: Some(convert_output(output)),
                rx: None,
                done: true,
            };
            Ok(Ok(self.ctx().table.push(future_output)?))
        }
        Err(e) => {
            Ok(Err(e.to_string()))
        },
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
