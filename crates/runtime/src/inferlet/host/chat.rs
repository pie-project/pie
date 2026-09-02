//! pie:instruct/chat — Conversation management
//!
//! Imported by inferlets that support chat-style interaction.
//! Delegates to the model's `Instruct` implementation.

use crate::inferlet::ProcessCtx;
use crate::inferlet::host::pie;
use anyhow::Result;
use models::template::{ChatDecoder, ChatEvent};
use std::collections::VecDeque;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

/// Chat decoder resource — wraps a model-specific ChatDecoder trait object.
///
/// A decoder answers a batch with everything that batch contained, which can
/// be more than one event: a stop token in the middle of a batch closes the
/// turn and the tokens after it are the next one. The WIT `feed` hands back a
/// single event, so the surplus queues here and drains on the following calls
/// rather than being dropped.
pub struct Decoder {
    inner: Box<dyn ChatDecoder>,
    pending: VecDeque<ChatEvent>,
}

impl std::fmt::Debug for Decoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("chat::Decoder").finish()
    }
}

impl pie::inferlet::chat::Host for ProcessCtx {
    async fn prefix(&mut self) -> Result<Vec<u32>> {
        Ok(crate::model::model().instruct().prefix())
    }

    async fn system(&mut self, message: String) -> Result<Vec<u32>> {
        Ok(crate::model::model().instruct().system(&message))
    }

    async fn user(&mut self, message: String) -> Result<Vec<u32>> {
        Ok(crate::model::model().instruct().user(&message))
    }

    async fn first_user(&mut self, message: String) -> Result<Vec<u32>> {
        Ok(crate::model::model().instruct().first_user(&message))
    }

    async fn system_user(&mut self, system: String, user: String) -> Result<Vec<u32>> {
        Ok(crate::model::model().instruct().system_user(&system, &user))
    }

    async fn assistant(&mut self, message: String) -> Result<Vec<u32>> {
        Ok(crate::model::model().instruct().assistant(&message))
    }

    async fn cue(&mut self) -> Result<Vec<u32>> {
        Ok(crate::model::model().instruct().cue())
    }

    async fn seal(&mut self) -> Result<Vec<u32>> {
        Ok(crate::model::model().instruct().seal())
    }

    async fn stop_tokens(&mut self) -> Result<Vec<u32>> {
        Ok(crate::model::model().instruct().seal())
    }
}

impl pie::inferlet::chat::HostDecoder for ProcessCtx {
    async fn new(&mut self) -> Result<Resource<Decoder>> {
        let inner = crate::model::model().instruct().chat_decoder();
        let decoder = Decoder {
            inner,
            pending: VecDeque::new(),
        };
        Ok(self.ctx().table.push(decoder)?)
    }

    async fn feed(
        &mut self,
        this: Resource<Decoder>,
        tokens: Vec<u32>,
    ) -> Result<Result<pie::inferlet::chat::Event, pie::inferlet::types::Error>> {
        let decoder = self.ctx().table.get_mut(&this)?;
        let events = decoder.inner.feed(&tokens);
        decoder.pending.extend(events);
        let event = decoder
            .pending
            .pop_front()
            .unwrap_or(ChatEvent::Delta(String::new()));
        Ok(Ok(match event {
            ChatEvent::Delta(s) => pie::inferlet::chat::Event::Delta(s),
            ChatEvent::Interrupt(id) => pie::inferlet::chat::Event::Interrupt(id),
            ChatEvent::Done(s) => pie::inferlet::chat::Event::Done(s),
        }))
    }

    async fn reset(&mut self, this: Resource<Decoder>) -> Result<()> {
        let decoder = self.ctx().table.get_mut(&this)?;
        decoder.inner.reset();
        decoder.pending.clear();
        Ok(())
    }

    async fn drop(&mut self, this: Resource<Decoder>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}
