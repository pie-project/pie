//! pie:zo/zo - Zero-Order Optimization functions

use crate::adapter;
use crate::api::pie;
use crate::api::inference::ForwardPass;
use crate::api::adapter::Adapter;
use crate::linker::InstanceState;
use anyhow::Result;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

impl pie::zo::zo::Host for InstanceState {
    async fn adapter_seed(&mut self, pass: Resource<ForwardPass>, seed: i64) -> Result<()> {
        // TODO: Add `seed: Option<i64>` field to ForwardPass and plumb through inference
        let _ = (pass, seed);
        Ok(())
    }

    async fn initialize(
        &mut self,
        adapter_res: Resource<Adapter>,
        rank: u32,
        alpha: f32,
        population_size: u32,
        mu_fraction: f32,
        initial_sigma: f32,
    ) -> Result<Result<(), String>> {
        let adapter = self.ctx().table.get(&adapter_res)?;
        let adapter_id = adapter.adapter_id;
        let model_idx = adapter.model_idx;

        match adapter::zo_initialize(
            model_idx, adapter_id, rank, alpha, population_size, mu_fraction, initial_sigma,
        ).await {
            Ok(()) => Ok(Ok(())),
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn update(
        &mut self,
        adapter_res: Resource<Adapter>,
        scores: Vec<f32>,
        seeds: Vec<i64>,
        max_sigma: f32,
    ) -> Result<Result<(), String>> {
        let adapter = self.ctx().table.get(&adapter_res)?;
        let adapter_id = adapter.adapter_id;
        let model_idx = adapter.model_idx;

        match adapter::zo_update(model_idx, adapter_id, scores, seeds, max_sigma).await {
            Ok(()) => Ok(Ok(())),
            Err(e) => Ok(Err(e.to_string())),
        }
    }
}
