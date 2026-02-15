//! pie:instruct/chat — Conversation management
//!
//! Imported by inferlets that support chat-style interaction.
//! Delegates to the model's `Instruct` implementation.

use crate::api::pie;
use crate::api::context::Context;
use crate::context;
use crate::model;
use crate::linker::InstanceState;
use crate::model::instruct::{ChatDecoder, ChatEvent};
use anyhow::Result;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

/// Chat decoder resource — wraps a model-specific ChatDecoder trait object.
pub struct Decoder {
    inner: Box<dyn ChatDecoder>,
}

impl std::fmt::Debug for Decoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("chat::Decoder").finish()
    }
}

impl pie::instruct::chat::Host for InstanceState {
    async fn system(&mut self, ctx: Resource<Context>, message: String) -> Result<()> {
        let ctx = self.ctx().table.get(&ctx)?;
        let model = model::get_model(ctx.model_id).ok_or_else(|| anyhow::anyhow!("model not found"))?;
        let tokens = model.instruct().system(&message);
        context::append_buffered_tokens(ctx.model_id, ctx.context_id, ctx.lock_id.unwrap_or(0), tokens)?;
        Ok(())
    }

    async fn user(&mut self, ctx: Resource<Context>, message: String) -> Result<()> {
        let ctx = self.ctx().table.get(&ctx)?;
        let model = model::get_model(ctx.model_id).ok_or_else(|| anyhow::anyhow!("model not found"))?;
        let tokens = model.instruct().user(&message);
        context::append_buffered_tokens(ctx.model_id, ctx.context_id, ctx.lock_id.unwrap_or(0), tokens)?;
        Ok(())
    }

    async fn assistant(&mut self, ctx: Resource<Context>, message: String) -> Result<()> {
        let ctx = self.ctx().table.get(&ctx)?;
        let model = model::get_model(ctx.model_id).ok_or_else(|| anyhow::anyhow!("model not found"))?;
        let tokens = model.instruct().assistant(&message);
        context::append_buffered_tokens(ctx.model_id, ctx.context_id, ctx.lock_id.unwrap_or(0), tokens)?;
        Ok(())
    }

    async fn cue(&mut self, ctx: Resource<Context>) -> Result<()> {
        let ctx = self.ctx().table.get(&ctx)?;
        let model = model::get_model(ctx.model_id).ok_or_else(|| anyhow::anyhow!("model not found"))?;
        let tokens = model.instruct().cue();
        context::append_buffered_tokens(ctx.model_id, ctx.context_id, ctx.lock_id.unwrap_or(0), tokens)?;
        Ok(())
    }

    async fn seal(&mut self, ctx: Resource<Context>) -> Result<()> {
        let ctx = self.ctx().table.get(&ctx)?;
        let model = model::get_model(ctx.model_id).ok_or_else(|| anyhow::anyhow!("model not found"))?;
        let tokens = model.instruct().seal();
        context::append_buffered_tokens(ctx.model_id, ctx.context_id, ctx.lock_id.unwrap_or(0), tokens)?;
        Ok(())
    }

    async fn stop_tokens(&mut self, model_res: Resource<crate::api::model::Model>) -> Result<Vec<u32>> {
        let model = self.ctx().table.get(&model_res)?;
        Ok(model.model.instruct().seal())
    }

    async fn create_decoder(&mut self, model: Resource<crate::api::model::Model>) -> Result<Resource<Decoder>> {
        let model = self.ctx().table.get(&model)?;
        let inner = model.model.instruct().chat_decoder();
        let decoder = Decoder { inner };
        Ok(self.ctx().table.push(decoder)?)
    }
}

impl pie::instruct::chat::HostDecoder for InstanceState {
    async fn feed(&mut self, this: Resource<Decoder>, tokens: Vec<u32>) -> Result<Result<pie::instruct::chat::Event, pie::core::types::Error>> {
        let decoder = self.ctx().table.get_mut(&this)?;
        let event = decoder.inner.feed(&tokens);
        Ok(Ok(match event {
            ChatEvent::Delta(s) => pie::instruct::chat::Event::Delta(s),
            ChatEvent::Interrupt(id) => pie::instruct::chat::Event::Interrupt(id),
            ChatEvent::Done(s) => pie::instruct::chat::Event::Done(s),
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
