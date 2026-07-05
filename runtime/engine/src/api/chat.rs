//! pie:instruct/chat — Conversation management
//!
//! Imported by inferlets that support chat-style interaction.
//! Delegates to the model's `Instruct` implementation.

use crate::api::pie;
use crate::inferlet::ProcessCtx;
use anyhow::Result;
use pie_model::instruct::{ChatDecoder, ChatEvent};
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

impl pie::inferlet::chat::Host for ProcessCtx {
    async fn system(&mut self, message: String) -> Result<Vec<u32>> {
        Ok(pie_model::model().instruct().system(&message))
    }

    async fn user(&mut self, message: String) -> Result<Vec<u32>> {
        Ok(pie_model::model().instruct().user(&message))
    }

    async fn first_user(&mut self, message: String) -> Result<Vec<u32>> {
        Ok(pie_model::model().instruct().first_user(&message))
    }

    async fn system_user(&mut self, system: String, user: String) -> Result<Vec<u32>> {
        Ok(pie_model::model().instruct().system_user(&system, &user))
    }

    async fn assistant(&mut self, message: String) -> Result<Vec<u32>> {
        Ok(pie_model::model().instruct().assistant(&message))
    }

    async fn cue(&mut self) -> Result<Vec<u32>> {
        Ok(pie_model::model().instruct().cue())
    }

    async fn seal(&mut self) -> Result<Vec<u32>> {
        Ok(pie_model::model().instruct().seal())
    }

    async fn stop_tokens(&mut self) -> Result<Vec<u32>> {
        Ok(pie_model::model().instruct().seal())
    }

    async fn create_decoder(&mut self) -> Result<Resource<Decoder>> {
        let inner = pie_model::model().instruct().chat_decoder();
        let decoder = Decoder { inner };
        Ok(self.ctx().table.push(decoder)?)
    }
}

impl pie::inferlet::chat::HostDecoder for ProcessCtx {
    async fn feed(
        &mut self,
        this: Resource<Decoder>,
        tokens: Vec<u32>,
    ) -> Result<Result<pie::inferlet::chat::Event, pie::inferlet::types::Error>> {
        let decoder = self.ctx().table.get_mut(&this)?;
        let event = decoder.inner.feed(&tokens);
        Ok(Ok(match event {
            ChatEvent::Delta(s) => pie::inferlet::chat::Event::Delta(s),
            ChatEvent::Interrupt(id) => pie::inferlet::chat::Event::Interrupt(id),
            ChatEvent::Done(s) => pie::inferlet::chat::Event::Done(s),
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
