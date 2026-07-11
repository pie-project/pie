//! pie:instruct/reasoning — Reasoning/thinking block detection
//!
//! Imported by inferlets that support reasoning capabilities.
//! Delegates to the model's `Instruct` implementation.

use crate::inferlet::ProcessCtx;
use crate::inferlet::host::pie;
use anyhow::Result;
use pie_model::instruct::{ReasoningDecoder, ReasoningEvent};
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

/// Reasoning decoder resource — wraps a model-specific ReasoningDecoder trait object.
pub struct Decoder {
    inner: Box<dyn ReasoningDecoder>,
}

impl std::fmt::Debug for Decoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("reasoning::Decoder").finish()
    }
}

impl pie::inferlet::reasoning::Host for ProcessCtx {
    async fn create_decoder(&mut self) -> Result<Resource<Decoder>> {
        let inner = pie_model::model().instruct().reasoning_decoder();
        let decoder = Decoder { inner };
        Ok(self.ctx().table.push(decoder)?)
    }
}

impl pie::inferlet::reasoning::HostDecoder for ProcessCtx {
    async fn feed(
        &mut self,
        this: Resource<Decoder>,
        tokens: Vec<u32>,
    ) -> Result<Result<pie::inferlet::reasoning::Event, pie::inferlet::types::Error>> {
        let decoder = self.ctx().table.get_mut(&this)?;
        let event = decoder.inner.feed(&tokens);
        Ok(Ok(match event {
            ReasoningEvent::Start => pie::inferlet::reasoning::Event::Start,
            ReasoningEvent::Delta(s) => pie::inferlet::reasoning::Event::Delta(s),
            ReasoningEvent::Complete(s) => pie::inferlet::reasoning::Event::Complete(s),
        }))
    }

    async fn reset(&mut self, this: Resource<Decoder>) -> Result<()> {
        let decoder = self.ctx().table.get_mut(&this)?;
        decoder.inner.reset();
        Ok(())
    }

    async fn drop(&mut self, this: Resource<Decoder>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}
