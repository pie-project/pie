use crate::api::core::Queue;
use crate::api::core::forward::ForwardPass;
use crate::api::inferlet;
use crate::instance::InstanceState;
use crate::model::request::{InitializeAdapterRequest, Request, UpdateAdapterRequest};
use crate::model::resource::{ADAPTER_TYPE_ID, ResourceId};
use crate::model::submit_request;
use anyhow::{Result, bail};
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

impl inferlet::zo::evolve::Host for InstanceState {
    async fn set_adapter_seed(&mut self, pass: Resource<ForwardPass>, seed: i64) -> Result<()> {
        let pass = self.ctx().table.get_mut(&pass)?;

        if pass.adapter.is_some() {
            pass.adapter_seed = Some(seed);
        } else {
            bail!("Adapter not set");
        }

        Ok(())
    }
    async fn initialize_adapter(
        &mut self,
        queue: Resource<Queue>,
        mut adapter_ptr: ResourceId,
        rank: u32,
        alpha: f32,
        population_size: u32,
        mu_fraction: f32,
        initial_sigma: f32,
    ) -> Result<()> {
        let (svc_id, queue_id, priority) = self.read_queue(&queue)?;
        adapter_ptr = self.translate_resource_ptr(svc_id, ADAPTER_TYPE_ID, adapter_ptr)?;

        let req = Request::InitializeAdapter(InitializeAdapterRequest {
            adapter_ptr,
            rank,
            alpha,
            population_size,
            mu_fraction,
            initial_sigma,
        });

        submit_request(svc_id, queue_id, priority, req)?;

        Ok(())
    }

    async fn update_adapter(
        &mut self,
        queue: Resource<Queue>,
        mut adapter_ptr: ResourceId,
        scores: Vec<f32>,
        seeds: Vec<i64>,
        max_sigma: f32,
    ) -> Result<()> {
        let (svc_id, queue_id, priority) = self.read_queue(&queue)?;
        adapter_ptr = self.translate_resource_ptr(svc_id, ADAPTER_TYPE_ID, adapter_ptr)?;

        let req = Request::UpdateAdapter(UpdateAdapterRequest {
            adapter_ptr,
            scores,
            seeds,
            max_sigma,
        });

        submit_request(svc_id, queue_id, priority, req)?;

        Ok(())
    }
}
