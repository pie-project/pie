use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

use crate::inferlet::ProcessCtx;
pub use crate::pipeline::Pipeline;

use super::pie;

impl pie::inferlet::pipeline::Host for ProcessCtx {
    async fn live(&mut self) -> anyhow::Result<u32> {
        Ok(u32::try_from(Pipeline::live()).unwrap_or(u32::MAX))
    }
}

impl pie::inferlet::pipeline::HostPipeline for ProcessCtx {
    async fn new(&mut self) -> anyhow::Result<Resource<Pipeline>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let pipeline = Pipeline::new();
        self.register_pipeline(&pipeline.scope, &pipeline.fires);
        crate::scheduler::worker::notify_lane_open(pipeline.scope.scheduler_id(), self.id());
        Ok(self.ctx().table.push(pipeline)?)
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
