//! pie:instruct/chat — Conversation management
//!
//! Imported by inferlets that support chat-style interaction.

use crate::api::pie;
use crate::linker::InstanceState;
use anyhow::Result;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

/// Chat decoder resource — classifies generated tokens into text deltas,
/// interrupts, and done signals.
#[derive(Debug)]
pub struct Decoder {
    // TODO: model-specific state machine
}

impl pie::instruct::chat::Host for InstanceState {
    async fn system(&mut self, _ctx: Resource<crate::api::context::Context>, _message: String) -> Result<()> {
        todo!("chat::system")
    }

    async fn user(&mut self, _ctx: Resource<crate::api::context::Context>, _message: String) -> Result<()> {
        todo!("chat::user")
    }

    async fn assistant(&mut self, _ctx: Resource<crate::api::context::Context>, _message: String) -> Result<()> {
        todo!("chat::assistant")
    }

    async fn cue(&mut self, _ctx: Resource<crate::api::context::Context>) -> Result<()> {
        todo!("chat::cue")
    }

    async fn seal(&mut self, _ctx: Resource<crate::api::context::Context>) -> Result<()> {
        todo!("chat::seal")
    }

    async fn create_decoder(&mut self, _model: Resource<crate::api::model::Model>) -> Result<Resource<Decoder>> {
        todo!("chat::create_decoder")
    }
}

impl pie::instruct::chat::HostDecoder for InstanceState {
    async fn feed(&mut self, _this: Resource<Decoder>, _tokens: Vec<u32>) -> Result<Result<pie::instruct::chat::Event, pie::core::types::Error>> {
        todo!("chat::decoder::feed")
    }

    async fn reset(&mut self, _this: Resource<Decoder>) -> Result<()> {
        todo!("chat::decoder::reset")
    }

    async fn drop(&mut self, this: Resource<Decoder>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}
