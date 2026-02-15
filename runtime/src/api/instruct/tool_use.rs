//! pie:instruct/tool-use — Tool calling support
//!
//! Imported by inferlets that support tool-use capabilities.
//! Delegates to the model's `Instruct` implementation.

use crate::api::pie;
use crate::api::context::Context;
use crate::context;
use crate::model;
use crate::linker::InstanceState;
use crate::model::instruct::{ToolDecoder, ToolEvent};
use anyhow::Result;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

/// Tool-use decoder resource — wraps a model-specific ToolDecoder trait object.
pub struct Decoder {
    inner: Box<dyn ToolDecoder>,
}

impl std::fmt::Debug for Decoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("tool_use::Decoder").finish()
    }
}

impl pie::instruct::tool_use::Host for InstanceState {
    async fn equip(
        &mut self,
        ctx: Resource<Context>,
        tools: Vec<String>,
    ) -> Result<Result<(), pie::core::types::Error>> {
        let ctx = self.ctx().table.get(&ctx)?;
        let model = model::get_model(ctx.model_id).ok_or_else(|| anyhow::anyhow!("model not found"))?;
        let tokens = model.instruct().equip(&tools);
        context::append_buffered_tokens(ctx.model_id, ctx.context_id, ctx.lock_id.unwrap_or(0), tokens)?;
        Ok(Ok(()))
    }

    async fn answer(
        &mut self,
        ctx: Resource<Context>,
        name: String,
        value: String,
    ) -> Result<()> {
        let ctx = self.ctx().table.get(&ctx)?;
        let model = model::get_model(ctx.model_id).ok_or_else(|| anyhow::anyhow!("model not found"))?;
        let tokens = model.instruct().answer(&name, &value);
        context::append_buffered_tokens(ctx.model_id, ctx.context_id, ctx.lock_id.unwrap_or(0), tokens)?;
        Ok(())
    }

    async fn create_decoder(
        &mut self,
        model: Resource<crate::api::model::Model>,
    ) -> Result<Resource<Decoder>> {
        let model = self.ctx().table.get(&model)?;
        let inner = model.model.instruct().tool_decoder();
        let decoder = Decoder { inner };
        Ok(self.ctx().table.push(decoder)?)
    }

    async fn create_matcher(
        &mut self,
        _model: Resource<crate::api::model::Model>,
        _tools: Vec<String>,
    ) -> Result<Resource<crate::api::inference::Matcher>> {
        todo!("tool_use::create_matcher — requires grammar compilation")
    }
}

impl pie::instruct::tool_use::HostDecoder for InstanceState {
    async fn feed(
        &mut self,
        this: Resource<Decoder>,
        tokens: Vec<u32>,
    ) -> Result<Result<pie::instruct::tool_use::Event, pie::core::types::Error>> {
        let decoder = self.ctx().table.get_mut(&this)?;
        let event = decoder.inner.feed(&tokens);
        Ok(Ok(match event {
            ToolEvent::Start => pie::instruct::tool_use::Event::Start,
            ToolEvent::Call(name, args) => pie::instruct::tool_use::Event::Call((name, args)),
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
