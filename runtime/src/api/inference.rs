//! pie:core/inference - ForwardPass, FutureOutput, Sampler, Output

use crate::api::pie;
use crate::api::types::Queue;
use crate::api::context::Context;
use crate::api::model::Model;
use crate::api::adapter::Adapter;
use crate::instance::InstanceState;
use crate::model::request::{ForwardPassRequest, ForwardPassResponse, Request};
use crate::model::submit_request;
use anyhow::Result;
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
    pub model_service_id: usize,
    pub queue_uid: u32,
    input_tokens: Vec<u32>,
    input_token_positions: Vec<u32>,
    speculative_tokens: Vec<u32>,
    speculative_positions: Vec<u32>,
    mask: Vec<Vec<u32>>,
    sampling_mask: Option<Vec<u32>>,
    output_token_indices: Vec<u32>,
    output_token_samplers: Vec<HashMap<String, rmpv::Value>>,
    adapter: Option<u32>,
}

#[derive(Debug)]
pub struct FutureOutput {
    receiver: oneshot::Receiver<ForwardPassResponse>,
    result: Option<pie::core::inference::Output>,
    done: bool,
}

#[async_trait]
impl Pollable for FutureOutput {
    async fn ready(&mut self) {
        if self.done {
            return;
        }
        if let Ok(res) = (&mut self.receiver).await {
            if !res.dists.is_empty() {
                self.result = Some(pie::core::inference::Output::Distributions(res.dists));
            } else if !res.tokens.is_empty() {
                self.result = Some(pie::core::inference::Output::Tokens(res.tokens));
            } else {
                self.result = Some(pie::core::inference::Output::None);
            }
        }
        self.done = true;
    }
}

enum SamplerType {
    Distribution = 0,
    Multinomial = 1,
    TopP = 2,
    TopK = 3,
    MinP = 4,
    TopKTopP = 5,
}

impl pie::core::inference::Host for InstanceState {}

impl pie::core::inference::HostForwardPass for InstanceState {
    async fn new(&mut self, model: Resource<Model>) -> Result<Resource<ForwardPass>> {
        let model = self.ctx().table.get(&model)?;
        let pass = ForwardPass {
            model_service_id: model.service_id,
            queue_uid: 0, // Will be set when execute is called
            input_tokens: vec![],
            input_token_positions: vec![],
            speculative_tokens: vec![],
            speculative_positions: vec![],
            mask: vec![],
            sampling_mask: None,
            output_token_indices: vec![],
            output_token_samplers: vec![],
            adapter: None,
        };
        Ok(self.ctx().table.push(pass)?)
    }

    async fn kv_pages(&mut self, this: Resource<ForwardPass>, page_ids: Vec<u32>) -> Result<()> {
        let pass = self.ctx().table.get_mut(&this)?;
        // TODO: Store page IDs for KV cache lookup during forward pass
        let _ = page_ids; // Store in pass when implementing
        let _ = pass;
        Ok(())
    }

    async fn context(&mut self, _this: Resource<ForwardPass>, _context: Resource<Context>) -> Result<()> {
        // TODO: Set context for KV cache
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

    async fn attention_mask(&mut self, this: Resource<ForwardPass>, mask: Vec<Vec<u32>>) -> Result<()> {
        let pass = self.ctx().table.get_mut(&this)?;
        pass.mask = mask;
        Ok(())
    }

    async fn sampling_mask(&mut self, this: Resource<ForwardPass>, mask: Vec<u32>) -> Result<()> {
        let pass = self.ctx().table.get_mut(&this)?;
        pass.sampling_mask = Some(mask);
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
            pie::core::inference::Sampler::Multinomial((temp, _seed)) => {
                sampler_map.insert("sampler".to_string(), rmpv::Value::from(SamplerType::Multinomial as u32));
                sampler_map.insert("temperature".to_string(), rmpv::Value::from(temp));
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
        }

        let pass = self.ctx().table.get_mut(&this)?;
        pass.output_token_samplers.extend(iter::repeat(sampler_map).take(indices.len()));
        pass.output_token_indices.extend(indices);
        Ok(())
    }

    async fn adapter(&mut self, this: Resource<ForwardPass>, adapter: Resource<Adapter>) -> Result<()> {
        let adapter_ptr = self.ctx().table.get(&adapter)?.ptr;
        let pass = self.ctx().table.get_mut(&this)?;
        pass.adapter = Some(adapter_ptr);
        Ok(())
    }

    async fn execute(
        &mut self,
        this: Resource<ForwardPass>,
        queue: Resource<Queue>,
    ) -> Result<Result<Resource<FutureOutput>, String>> {
        let queue_data = self.ctx().table.get(&queue)?;
        let svc_id = queue_data.service_id;
        let queue_id = queue_data.uid;

        let pass = self.ctx().table.get_mut(&this)?;

        let request = ForwardPassRequest {
            input_tokens: take(&mut pass.input_tokens),
            input_token_positions: take(&mut pass.input_token_positions),
            input_embed_ptrs: vec![],
            input_embed_positions: vec![],
            adapter: pass.adapter,
            adapter_seed: None,
            mask: take(&mut pass.mask),
            sampling_mask: take(&mut pass.sampling_mask),
            kv_page_ptrs: vec![],
            kv_page_last_len: 0,
            output_token_indices: take(&mut pass.output_token_indices),
            output_token_samplers: take(&mut pass.output_token_samplers),
            output_embed_ptrs: vec![],
            output_embed_indices: vec![],
            arrival_time: None,
            inst_id: Some(self.id()),
        };

        let (tx, rx) = oneshot::channel();
        let req = Request::ForwardPass(request, Some(tx));
        submit_request(svc_id, queue_id, 0, req)?;

        let future_output = FutureOutput {
            receiver: rx,
            result: None,
            done: false,
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
