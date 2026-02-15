//! pie:instruct/reasoning — Reasoning/thinking block detection
//!
//! Exported by inferlets that support reasoning capabilities.

use crate::api::pie;
use crate::linker::InstanceState;
use anyhow::Result;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

/// Reasoning decoder resource — detects thinking blocks in generated tokens.
#[derive(Debug)]
pub struct Decoder {
    // TODO: model-specific thinking block detection state machine
}

impl pie::instruct::reasoning::Host for InstanceState {
    async fn create_decoder(
        &mut self,
        _model: Resource<crate::api::model::Model>,
    ) -> Result<Resource<Decoder>> {
        todo!("reasoning::create_decoder")
    }
}

impl pie::instruct::reasoning::HostDecoder for InstanceState {
    async fn feed(
        &mut self,
        _this: Resource<Decoder>,
        _tokens: Vec<u32>,
    ) -> Result<Result<pie::instruct::reasoning::Event, pie::core::types::Error>> {
        todo!("reasoning::decoder::feed")
    }

    async fn reset(&mut self, _this: Resource<Decoder>) -> Result<()> {
        todo!("reasoning::decoder::reset")
    }

    async fn drop(&mut self, this: Resource<Decoder>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}
