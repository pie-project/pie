use crate::api::core::Queue;
use crate::api::inferlet;
use crate::instance::InstanceState;
use crate::model::request::{ForwardPassRequest, ForwardPassResponse, Request};
use crate::model::resource::{EMBED_TYPE_ID, KV_PAGE_TYPE_ID, ResourceId};
use crate::model::submit_request;
use anyhow::{Result, bail};
use std::collections::HashMap;
use std::iter;
use std::mem::take;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use tokio::sync::oneshot;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;
use wasmtime_wasi::async_trait;
use wasmtime_wasi::p2::{DynPollable, Pollable, subscribe};

/// Accumulator for per-step WIT overhead timing (PIE_WIT_TIMING=1).
static WIT_TIMING_COUNT: AtomicU64 = AtomicU64::new(0);
static WIT_TIMING_SUM_NS: AtomicU64 = AtomicU64::new(0);
static WIT_EXECUTE_SUM_NS: AtomicU64 = AtomicU64::new(0);
static WIT_READY_SUM_NS: AtomicU64 = AtomicU64::new(0);
static WIT_RESULT_SUM_NS: AtomicU64 = AtomicU64::new(0);

fn wit_timing_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("PIE_WIT_TIMING").is_ok())
}

fn mono_ns() -> u64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    (ts.tv_sec as u64) * 1_000_000_000 + (ts.tv_nsec as u64)
}

/// Print and reset WIT timing accumulators every N steps.
fn wit_timing_maybe_report() {
    let count = WIT_TIMING_COUNT.load(AtomicOrdering::Relaxed);
    if count > 0 && count % 100 == 0 {
        let total = WIT_TIMING_SUM_NS.swap(0, AtomicOrdering::Relaxed);
        let execute = WIT_EXECUTE_SUM_NS.swap(0, AtomicOrdering::Relaxed);
        let ready = WIT_READY_SUM_NS.swap(0, AtomicOrdering::Relaxed);
        let result = WIT_RESULT_SUM_NS.swap(0, AtomicOrdering::Relaxed);
        let n = WIT_TIMING_COUNT.swap(0, AtomicOrdering::Relaxed);
        if n > 0 {
            eprintln!(
                "[WIT-TIMING] n={} avg: execute_build={:.0}us ready_wait={:.0}us result_extract={:.0}us pre+post={:.0}us",
                n,
                execute as f64 / n as f64 / 1000.0,
                ready as f64 / n as f64 / 1000.0,
                result as f64 / n as f64 / 1000.0,
                (total - execute - ready - result) as f64 / n as f64 / 1000.0,
            );
        }
    }
}

#[derive(Debug)]
pub struct ForwardPass {
    pub queue: Queue,
    input_tokens: Vec<u32>,
    input_token_positions: Vec<u32>,
    input_embed_ptrs: Vec<u32>,
    input_embed_positions: Vec<u32>,
    pub adapter: Option<u32>,
    pub adapter_seed: Option<i64>,
    mask: Vec<Vec<u32>>,
    kv_page_ptrs: Vec<u32>,
    kv_page_last_len: u32,
    output_token_indices: Vec<u32>,
    output_token_samplers: Vec<HashMap<String, rmpv::Value>>,
    output_embed_ptrs: Vec<u32>,
    output_embed_indices: Vec<u32>,
    /// Return distributions alongside tokens for each step.
    return_distributions: bool,
}

#[derive(Debug)]
pub struct ForwardPassResult {
    pub receiver: oneshot::Receiver<ForwardPassResponse>,
    pub distributions: Vec<(Vec<u32>, Vec<f32>)>,
    pub tokens: Vec<u32>,
    pub done: bool,
}

enum Sampler {
    Distribution = 0,
    Multinomial = 1,
    TopP = 2,
    TopK = 3,
    MinP = 4,
    TopKTopP = 5,
}

#[async_trait]
impl Pollable for ForwardPassResult {
    async fn ready(&mut self) {
        if self.done {
            return;
        }

        let t0 = if wit_timing_enabled() { Some(mono_ns()) } else { None };

        // Wait for oneshot response.
        if let Ok(res) = (&mut self.receiver).await {
            self.distributions = res.dists;
            self.tokens = res.tokens;
        }
        self.done = true;

        if let Some(t0) = t0 {
            WIT_READY_SUM_NS.fetch_add(mono_ns() - t0, AtomicOrdering::Relaxed);
        }
    }
}

impl inferlet::core::forward::Host for InstanceState {
    async fn create_forward_pass(
        &mut self,
        queue: Resource<Queue>,
    ) -> Result<Resource<ForwardPass>> {
        let queue = self.ctx().table.get(&queue)?.clone();

        let pass = ForwardPass {
            queue,
            input_tokens: vec![],
            input_token_positions: vec![],
            input_embed_ptrs: vec![],
            input_embed_positions: vec![],
            adapter: None,
            adapter_seed: None,
            mask: vec![],
            kv_page_ptrs: vec![],
            kv_page_last_len: 0,
            output_token_indices: vec![],
            output_token_samplers: vec![],
            output_embed_ptrs: vec![],
            output_embed_indices: vec![],
            return_distributions: false,
        };
        Ok(self.ctx().table.push(pass)?)
    }

    async fn attention_mask(
        &mut self,
        pass: Resource<ForwardPass>,
        mask: Vec<Vec<u32>>,
    ) -> Result<()> {
        let pass = self.ctx().table.get_mut(&pass)?;
        pass.mask = mask;
        Ok(())
    }

    async fn kv_cache(
        &mut self,
        pass: Resource<ForwardPass>,
        kv_page_ptrs: Vec<ResourceId>,
        kv_page_last_len: u32,
    ) -> Result<()> {
        let svc_id = self.ctx().table.get(&pass)?.queue.service_id;
        let translated = self.translate_kv_pages_cached(svc_id, &kv_page_ptrs)?;
        let pass = self.ctx().table.get_mut(&pass)?;
        pass.kv_page_ptrs = translated;
        pass.kv_page_last_len = kv_page_last_len;
        Ok(())
    }

    async fn input_embeddings(
        &mut self,
        pass: Resource<ForwardPass>,
        mut emb_ptrs: Vec<ResourceId>,
        positions: Vec<u32>,
    ) -> Result<()> {
        let svc_id = self.ctx().table.get(&pass)?.queue.service_id;

        emb_ptrs.iter_mut().try_for_each(|emb_ptr| {
            *emb_ptr = self.translate_resource_ptr(svc_id, EMBED_TYPE_ID, *emb_ptr)?;
            Ok::<_, anyhow::Error>(())
        })?;

        let pass = self.ctx().table.get_mut(&pass)?;

        if pass.input_tokens.len() + emb_ptrs.len() > pass.queue.info.max_batch_tokens {
            bail!(
                "max batch tokens exceeded, input tokens: {}, max tokens: {}",
                emb_ptrs.len(),
                pass.queue.info.max_batch_tokens
            );
        }

        pass.input_embed_ptrs.extend(emb_ptrs);
        pass.input_embed_positions.extend(positions);
        Ok(())
    }

    async fn input_tokens(
        &mut self,
        pass: Resource<ForwardPass>,
        input_tokens: Vec<u32>,
        positions: Vec<u32>,
    ) -> Result<()> {
        let pass = self.ctx().table.get_mut(&pass)?;

        if pass.input_tokens.len() + input_tokens.len() > pass.queue.info.max_batch_tokens {
            println!(
                "max batch tokens exceeded, input tokens: {}, max tokens: {}",
                input_tokens.len(),
                pass.queue.info.max_batch_tokens
            );
            bail!(
                "max batch tokens exceeded, input tokens: {}, max tokens: {}",
                input_tokens.len(),
                pass.queue.info.max_batch_tokens
            );
        }

        // check if token ids are in the vocab range
        let num_vocabs = pass.queue.info.tokenizer.num_vocab() as u32;
        for &token in input_tokens.iter() {
            if token >= num_vocabs {
                println!("token id {} is out of range", token);
                bail!("token id {} is out of range", token);
            }
        }

        pass.input_tokens.extend(input_tokens);
        pass.input_token_positions.extend(positions);
        Ok(())
    }

    async fn output_embeddings(
        &mut self,
        pass: Resource<ForwardPass>,
        mut emb_ptrs: Vec<ResourceId>,
        indices: Vec<u32>,
    ) -> Result<()> {
        let svc_id = self.ctx().table.get(&pass)?.queue.service_id;
        emb_ptrs.iter_mut().try_for_each(|emb_ptr| {
            *emb_ptr = self.translate_resource_ptr(svc_id, EMBED_TYPE_ID, *emb_ptr)?;
            Ok::<_, anyhow::Error>(())
        })?;

        let pass = self.ctx().table.get_mut(&pass)?;
        pass.output_embed_ptrs.extend(emb_ptrs);
        pass.output_embed_indices.extend(indices);
        Ok(())
    }

    async fn output_distributions(
        &mut self,
        pass: Resource<ForwardPass>,
        indices: Vec<u32>,
        temperature: f32,
        top_k: Option<u32>,
    ) -> Result<()> {
        let mut sampler = HashMap::new();
        sampler.insert(
            "sampler".to_string(),
            rmpv::Value::from(Sampler::Distribution as u32),
        );
        sampler.insert("temperature".to_string(), rmpv::Value::from(temperature));
        sampler.insert("top_k".to_string(), rmpv::Value::from(top_k.unwrap_or(32)));

        let pass = self.ctx().table.get_mut(&pass)?;
        pass.output_token_samplers
            .extend(iter::repeat(sampler.clone()).take(indices.len()));
        pass.output_token_indices.extend(indices);
        Ok(())
    }

    async fn output_tokens(
        &mut self,
        pass: Resource<ForwardPass>,
        indices: Vec<u32>,
        temperature: f32,
    ) -> Result<()> {
        let mut sampler = HashMap::new();

        sampler.insert(
            "sampler".to_string(),
            rmpv::Value::from(Sampler::Multinomial as u32),
        );
        sampler.insert("temperature".to_string(), rmpv::Value::from(temperature));

        let samplers = iter::repeat(sampler.clone())
            .take(indices.len())
            .collect::<Vec<_>>();

        let pass = self.ctx().table.get_mut(&pass)?;
        pass.output_token_indices.extend(indices);
        pass.output_token_samplers.extend(samplers);
        Ok(())
    }

    async fn output_tokens_top_k(
        &mut self,
        pass: Resource<ForwardPass>,
        indices: Vec<u32>,
        temperature: f32,
        top_k: u32,
    ) -> Result<()> {
        let mut sampler = HashMap::new();

        sampler.insert(
            "sampler".to_string(),
            rmpv::Value::from(Sampler::TopK as u32),
        );
        sampler.insert("temperature".to_string(), rmpv::Value::from(temperature));
        sampler.insert("top_k".to_string(), rmpv::Value::from(top_k));

        let pass = self.ctx().table.get_mut(&pass)?;
        pass.output_token_samplers
            .extend(iter::repeat(sampler.clone()).take(indices.len()));
        pass.output_token_indices.extend(indices);

        Ok(())
    }

    async fn output_tokens_top_p(
        &mut self,
        pass: Resource<ForwardPass>,
        indices: Vec<u32>,
        temperature: f32,
        top_p: f32,
    ) -> Result<()> {
        let mut sampler = HashMap::new();

        sampler.insert(
            "sampler".to_string(),
            rmpv::Value::from(Sampler::TopP as u32),
        );
        sampler.insert("temperature".to_string(), rmpv::Value::from(temperature));
        sampler.insert("top_p".to_string(), rmpv::Value::from(top_p));

        let pass = self.ctx().table.get_mut(&pass)?;
        pass.output_token_samplers
            .extend(iter::repeat(sampler.clone()).take(indices.len()));
        pass.output_token_indices.extend(indices);

        Ok(())
    }

    async fn output_tokens_min_p(
        &mut self,
        pass: Resource<ForwardPass>,
        indices: Vec<u32>,
        temperature: f32,
        min_p: f32,
    ) -> Result<()> {
        let mut sampler = HashMap::new();

        sampler.insert(
            "sampler".to_string(),
            rmpv::Value::from(Sampler::MinP as u32),
        );
        sampler.insert("temperature".to_string(), rmpv::Value::from(temperature));
        sampler.insert("min_p".to_string(), rmpv::Value::from(min_p));

        let pass = self.ctx().table.get_mut(&pass)?;
        pass.output_token_samplers
            .extend(iter::repeat(sampler.clone()).take(indices.len()));
        pass.output_token_indices.extend(indices);

        Ok(())
    }

    async fn output_tokens_top_k_top_p(
        &mut self,
        pass: Resource<ForwardPass>,
        indices: Vec<u32>,
        temperature: f32,
        top_k: u32,
        top_p: f32,
    ) -> Result<()> {
        let mut sampler = HashMap::new();

        sampler.insert(
            "sampler".to_string(),
            rmpv::Value::from(Sampler::TopKTopP as u32),
        );
        sampler.insert("temperature".to_string(), rmpv::Value::from(temperature));
        sampler.insert("top_k".to_string(), rmpv::Value::from(top_k));
        sampler.insert("top_p".to_string(), rmpv::Value::from(top_p));

        let pass = self.ctx().table.get_mut(&pass)?;
        pass.output_token_samplers
            .extend(iter::repeat(sampler.clone()).take(indices.len()));
        pass.output_token_indices.extend(indices);

        Ok(())
    }

}

impl inferlet::core::forward::HostForwardPass for InstanceState {
    async fn execute(
        &mut self,
        this: Resource<ForwardPass>,
    ) -> Result<Option<Resource<ForwardPassResult>>> {
        let t_execute_start = if wit_timing_enabled() { Some(mono_ns()) } else { None };

        // 1) Check whether we need output (immutable borrow)
        let returns_output = {
            let pass = self.ctx().table.get(&this)?;
            !pass.output_token_indices.is_empty()
        };

        // 2) Build the request by MOVING data out of the pass (mutable borrow)
        let (mut request, svc_id, queue_id, priority) = {
            let pass = self.ctx().table.get_mut(&this)?;

            let svc_id = pass.queue.service_id;
            let queue_id = pass.queue.uid;
            let priority = pass.queue.priority;

            let request = ForwardPassRequest {
                input_tokens: take(&mut pass.input_tokens),
                input_token_positions: take(&mut pass.input_token_positions),
                input_embed_ptrs: take(&mut pass.input_embed_ptrs),
                input_embed_positions: take(&mut pass.input_embed_positions),
                adapter: pass.adapter,
                adapter_seed: pass.adapter_seed,
                mask: take(&mut pass.mask),
                kv_page_ptrs: take(&mut pass.kv_page_ptrs),
                kv_page_last_len: pass.kv_page_last_len,
                output_token_indices: take(&mut pass.output_token_indices),
                output_token_samplers: take(&mut pass.output_token_samplers),
                output_embed_ptrs: take(&mut pass.output_embed_ptrs),
                output_embed_indices: take(&mut pass.output_embed_indices),
                return_distributions: pass.return_distributions,
                arrival_time: None, // Set in Model::submit() before queuing
                inst_id: Some(self.id()),
            };

            (request, svc_id, queue_id, priority)
        };

        // Always create a response channel so every request participates in
        // the batch response protocol.
        let (tx, rx) = oneshot::channel();
        let req = Request::ForwardPass(request, Some(tx));
        submit_request(svc_id, queue_id, priority, req)?;

        if let Some(t0) = t_execute_start {
            WIT_EXECUTE_SUM_NS.fetch_add(mono_ns() - t0, AtomicOrdering::Relaxed);
        }

        let res = ForwardPassResult {
            receiver: rx,
            distributions: vec![],
            tokens: vec![],
            done: false,
        };

        Ok(Some(self.ctx().table.push(res)?))
    }
    async fn drop(&mut self, this: Resource<ForwardPass>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

impl inferlet::core::forward::HostForwardPassResult for InstanceState {
    async fn pollable(
        &mut self,
        this: Resource<ForwardPassResult>,
    ) -> Result<Resource<DynPollable>> {
        subscribe(self.ctx().table, this)
    }

    async fn get_distributions(
        &mut self,
        this: Resource<ForwardPassResult>,
    ) -> Result<Option<Vec<(Vec<u32>, Vec<f32>)>>> {
        let result = self.ctx().table.get_mut(&this)?;

        if result.done {
            Ok(Some(take(&mut result.distributions)))
        } else {
            Ok(None)
        }
    }

    async fn get_tokens(&mut self, this: Resource<ForwardPassResult>) -> Result<Option<Vec<u32>>> {
        let t0 = if wit_timing_enabled() { Some(mono_ns()) } else { None };
        let result = self.ctx().table.get_mut(&this)?;

        let ret = if !result.tokens.is_empty() {
            // Streaming or final: return buffered tokens, clear buffer.
            Ok(Some(take(&mut result.tokens)))
        } else if result.done {
            // Done and no more tokens — signal completion.
            Ok(None)
        } else {
            // Not done but no tokens yet (shouldn't happen after ready()).
            // Return empty vec to distinguish from None (done).
            Ok(Some(vec![]))
        };
        if let Some(t0) = t0 {
            let elapsed = mono_ns() - t0;
            WIT_RESULT_SUM_NS.fetch_add(elapsed, AtomicOrdering::Relaxed);
            WIT_TIMING_COUNT.fetch_add(1, AtomicOrdering::Relaxed);
            wit_timing_maybe_report();
        }
        ret
    }

    async fn drop(&mut self, this: Resource<ForwardPassResult>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}
