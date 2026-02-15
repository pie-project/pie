//! pie:instruct/reasoning — Reasoning/thinking block detection
//!
//! Imported by inferlets that support reasoning capabilities.
//! Delegates to the model's `Instruct` implementation.

use crate::api::pie;
use crate::linker::InstanceState;
use crate::model::instruct::{ReasoningDecoder, ReasoningEvent};
use anyhow::Result;
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

impl pie::instruct::reasoning::Host for InstanceState {
    async fn create_decoder(
        &mut self,
        model: Resource<crate::api::model::Model>,
    ) -> Result<Resource<Decoder>> {
        let model = self.ctx().table.get(&model)?;
        let inner = model.model.instruct().reasoning_decoder();
        let decoder = Decoder { inner };
        Ok(self.ctx().table.push(decoder)?)
    }
}

impl pie::instruct::reasoning::HostDecoder for InstanceState {
    async fn feed(
        &mut self,
        this: Resource<Decoder>,
        tokens: Vec<u32>,
    ) -> Result<Result<pie::instruct::reasoning::Event, pie::core::types::Error>> {
        let decoder = self.ctx().table.get_mut(&this)?;
        let event = decoder.inner.feed(&tokens);
        Ok(Ok(match event {
            ReasoningEvent::Start => pie::instruct::reasoning::Event::Start,
            ReasoningEvent::Delta(s) => pie::instruct::reasoning::Event::Delta(s),
            ReasoningEvent::Complete(s) => pie::instruct::reasoning::Event::Complete(s),
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
