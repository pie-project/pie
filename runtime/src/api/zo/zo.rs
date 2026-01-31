//! pie:zo/zo - Zero-Order Optimization functions

use crate::api::pie;
use crate::api::inference::ForwardPass;
use crate::api::adapter::Adapter;
use crate::instance::InstanceState;
use crate::legacy_model::request::{InitializeAdapterRequest, Request, UpdateAdapterRequest};
use crate::legacy_model::submit_request;
use anyhow::Result;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

impl pie::zo::zo::Host for InstanceState {
    async fn adapter_seed(&mut self, pass: Resource<ForwardPass>, seed: i64) -> Result<()> {
        let pass = self.ctx().table.get_mut(&pass)?;
        // Store seed for adapter perturbation
        // In the new API, we'll need to extend ForwardPass to store this
        let _ = (pass, seed);
        Ok(())
    }

    async fn initialize(
        &mut self,
        adapter: Resource<Adapter>,
        rank: u32,
        alpha: f32,
        population_size: u32,
        mu_fraction: f32,
        initial_sigma: f32,
    ) -> Result<Result<(), String>> {
        let adapter = self.ctx().table.get(&adapter)?;
        let model_idx = adapter.model_idx;

        let req = Request::InitializeAdapter(InitializeAdapterRequest {
            adapter_ptr: adapter.adapter_id as u32,
            rank,
            alpha,
            population_size,
            mu_fraction,
            initial_sigma,
        });

        submit_request(model_idx, 0, 0, req)?;
        Ok(Ok(()))
    }

    async fn update(
        &mut self,
        adapter: Resource<Adapter>,
        scores: Vec<f32>,
        seeds: Vec<i64>,
        max_sigma: f32,
    ) -> Result<Result<(), String>> {
        let adapter = self.ctx().table.get(&adapter)?;
        let model_idx = adapter.model_idx;

        let req = Request::UpdateAdapter(UpdateAdapterRequest {
            adapter_ptr: adapter.adapter_id as u32,
            scores,
            seeds,
            max_sigma,
        });

        submit_request(model_idx, 0, 0, req)?;
        Ok(Ok(()))
    }
}
