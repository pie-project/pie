use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

use crate::inferlet::ProcessCtx;
pub use crate::pipeline::Pipeline;

use super::pie;

impl pie::inferlet::pipeline::Host for ProcessCtx {}

impl pie::inferlet::pipeline::HostPipeline for ProcessCtx {
    async fn new(&mut self) -> anyhow::Result<Resource<Pipeline>> {
        // A lane is expected at every frame boundary from its opening, so
        // it opens only once its process is admitted to execute: opened
        // while its process waits at the bind or execution gate, a frame
        // waited on a lane that could not submit, and the admitted
        // processes holding the permits waited on that frame (gemma-4-E2B
        // decode c256: 37 lanes missing, 37 waiters at the execution gate).
        crate::inferlet::process::ensure_execution_admitted(self).await;
        crate::inferlet::process::gate::residency_gate(self).await?;
        let pipeline = Pipeline::new();
        self.register_pipeline(&pipeline.scope, &pipeline.fires);
        crate::scheduler::worker::notify_lane_open(pipeline.scope.scheduler_id(), self.id());
        Ok(self.ctx().table.push(pipeline)?)
    }

    async fn speculation_advised(&mut self, this: Resource<Pipeline>) -> anyhow::Result<bool> {
        Ok(self.ctx().table.get(&this)?.speculation_advised())
    }

    async fn close(&mut self, this: Resource<Pipeline>) -> anyhow::Result<()> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        crate::pipeline::fire::pipeline_close(self, this).await
    }

    async fn retract(
        &mut self,
        this: Resource<Pipeline>,
        fires: u32,
    ) -> anyhow::Result<Result<(), String>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        crate::pipeline::fire::pipeline_retract(self, this, fires).await
    }

    async fn drop(&mut self, this: Resource<Pipeline>) -> anyhow::Result<()> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        crate::pipeline::fire::pipeline_drop(self, this).await
    }
}
