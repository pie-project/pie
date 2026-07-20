//! WIT host glue for `pie:inferlet/pipeline` — thin `Host`/`HostPipeline`
//! impls over the pipeline-owned [`crate::pipeline::Pipeline`] resource type.
//! The ordering-domain algorithm (the in-flight fire FIFO, close/drop
//! draining) lives in [`crate::pipeline::fire`]; these impls only push/get/
//! delete the WASM resource and delegate through
//! [`crate::pipeline::fire::FireContext`] (implemented for `ProcessCtx`
//! below).

use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

use crate::inferlet::ProcessCtx;
pub use crate::pipeline::Pipeline;

use super::pie;

impl pie::inferlet::pipeline::Host for ProcessCtx {}

impl pie::inferlet::pipeline::HostPipeline for ProcessCtx {
    async fn new(&mut self) -> anyhow::Result<Resource<Pipeline>> {
        crate::inferlet::process::preemption::honor(self).await?;
        let pipeline = Pipeline::new();
        self.register_pipeline(&pipeline.fires);
        Ok(self.ctx().table.push(pipeline)?)
    }

    async fn finish(&mut self, this: Resource<Pipeline>) -> anyhow::Result<()> {
        crate::inferlet::process::preemption::honor(self).await?;
        crate::pipeline::fire::pipeline_finish(self, this).await
    }

    async fn close(&mut self, this: Resource<Pipeline>) -> anyhow::Result<()> {
        crate::inferlet::process::preemption::honor(self).await?;
        crate::pipeline::fire::pipeline_close(self, this).await
    }

    async fn drop(&mut self, this: Resource<Pipeline>) -> anyhow::Result<()> {
        crate::inferlet::process::preemption::honor(self).await?;
        crate::pipeline::fire::pipeline_drop(self, this).await
    }
}
