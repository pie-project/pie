//! pie:instruct/tool-use — Tool calling support
//!
//! Exported by inferlets that support tool-use capabilities.

use crate::api::pie;
use crate::linker::InstanceState;
use anyhow::Result;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

/// Tool-use decoder resource — detects tool call patterns in generated tokens.
#[derive(Debug)]
pub struct Decoder {
    // TODO: model-specific tool call detection state machine
}

impl pie::instruct::tool_use::Host for InstanceState {
    async fn equip(
        &mut self,
        _ctx: Resource<crate::api::context::Context>,
        _tools: Vec<String>,
    ) -> Result<Result<(), pie::core::types::Error>> {
        todo!("tool_use::equip")
    }

    async fn answer(
        &mut self,
        _ctx: Resource<crate::api::context::Context>,
        _name: String,
        _value: String,
    ) -> Result<()> {
        todo!("tool_use::answer")
    }

    async fn create_decoder(
        &mut self,
        _model: Resource<crate::api::model::Model>,
    ) -> Result<Resource<Decoder>> {
        todo!("tool_use::create_decoder")
    }

    async fn create_matcher(
        &mut self,
        _model: Resource<crate::api::model::Model>,
        _tools: Vec<String>,
    ) -> Result<Resource<crate::api::inference::Matcher>> {
        todo!("tool_use::matcher")
    }
}

impl pie::instruct::tool_use::HostDecoder for InstanceState {
    async fn feed(
        &mut self,
        _this: Resource<Decoder>,
        _tokens: Vec<u32>,
    ) -> Result<Result<pie::instruct::tool_use::Event, pie::core::types::Error>> {
        todo!("tool_use::decoder::feed")
    }

    async fn reset(&mut self, _this: Resource<Decoder>) -> Result<()> {
        todo!("tool_use::decoder::reset")
    }

    async fn drop(&mut self, this: Resource<Decoder>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}
