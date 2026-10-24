use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

use crate::inferlet::ProcessCtx;
pub use crate::pipeline::Pipeline;

use super::pie;

impl pie::inferlet::pipeline::Host for ProcessCtx {
    async fn live(&mut self) -> anyhow::Result<u32> {
        Ok(u32::try_from(Pipeline::live()).unwrap_or(u32::MAX))
    }

    async fn verify_budgeted(&mut self) -> anyhow::Result<bool> {
        Ok(crate::pipeline::verify::budget() != crate::pipeline::verify::Budget::Off)
    }
}

impl pie::inferlet::pipeline::HostPipeline for ProcessCtx {
    async fn new(&mut self) -> anyhow::Result<Resource<Pipeline>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let pipeline = Pipeline::new();
        self.register_pipeline(&pipeline.scope, &pipeline.fires);
        Ok(self.ctx().table.push(pipeline)?)
    }

    async fn allot_verify(
        &mut self,
        this: Resource<Pipeline>,
        survival: Vec<f32>,
    ) -> anyhow::Result<u32> {
        let id = self.ctx().table.get(&this)?.id;
        // Every open pipeline fires at least one row a wave.
        let base = Pipeline::live().saturating_mul(crate::scheduler::configured_frame_size());
        Ok(crate::pipeline::verify::allot(
            id,
            &survival,
            u32::try_from(base).unwrap_or(u32::MAX),
        ))
    }

    async fn close(&mut self, this: Resource<Pipeline>) -> anyhow::Result<()> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        crate::pipeline::fire::pipeline_close(self, this).await
    }

    async fn drop(&mut self, this: Resource<Pipeline>) -> anyhow::Result<()> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        crate::pipeline::fire::pipeline_drop(self, this).await
    }
}
