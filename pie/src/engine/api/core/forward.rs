use crate::engine::api::core::Queue;
use crate::engine::api::inferlet;
use crate::engine::instance::InstanceState;
use crate::engine::model::request::{ForwardPassRequest, ForwardPassResponse, Request};
use crate::engine::model::resource::{EMBED_TYPE_ID, KV_PAGE_TYPE_ID, ResourceId};
use crate::engine::model::submit_request;
use anyhow::{Result, bail};
use std::collections::HashMap;
use std::iter;
use std::mem::take;
use tokio::sync::oneshot;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;
use wasmtime_wasi::async_trait;
use wasmtime_wasi::p2::{DynPollable, Pollable, subscribe};

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

        if let Ok(res) = (&mut self.receiver).await {
            self.distributions = res.dists;
            self.tokens = res.tokens;
        }

        self.done = true;
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
        mut kv_page_ptrs: Vec<ResourceId>,
        kv_page_last_len: u32,
    ) -> Result<()> {
        let svc_id = self.ctx().table.get(&pass)?.queue.service_id;

        kv_page_ptrs.iter_mut().try_for_each(|kv_page_ptr| {
            *kv_page_ptr = self.translate_resource_ptr(svc_id, KV_PAGE_TYPE_ID, *kv_page_ptr)?;
            Ok::<_, anyhow::Error>(())
        })?;

        let pass = self.ctx().table.get_mut(&pass)?;
        pass.kv_page_ptrs = kv_page_ptrs;
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
        // 1) Check whether we need output (immutable borrow)
        let returns_output = {
            let pass = self.ctx().table.get(&this)?;
            !pass.output_token_indices.is_empty()
        };

        // 2) Build the request by MOVING data out of the pass (mutable borrow)
        let (request, svc_id, queue_id, priority) = {
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
            };

            (request, svc_id, queue_id, priority)
        };

        if returns_output {
            let (tx, rx) = oneshot::channel();

            let req = Request::ForwardPass(request, Some(tx));

            submit_request(svc_id, queue_id, priority, req)?;

            let res = ForwardPassResult {
                receiver: rx,
                distributions: vec![],
                tokens: vec![],
                done: false,
            };

            Ok(Some(self.ctx().table.push(res)?))
        } else {
            let req = Request::ForwardPass(request, None);

            submit_request(svc_id, queue_id, priority, req)?;

            Ok(None)
        }
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
        let result = self.ctx().table.get_mut(&this)?;

        if result.done {
            Ok(Some(take(&mut result.tokens)))
        } else {
            Ok(None)
        }
    }

    async fn drop(&mut self, this: Resource<ForwardPassResult>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}
