//! pie:structured/grammar - Grammar compilation for structured output
//!
//! Stub implementation — will be backed by XGrammar.

use crate::api::pie;
use crate::linker::InstanceState;
use anyhow::Result;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

/// A compiled grammar that describes valid output structure.
#[derive(Debug)]
pub struct Grammar {
    // TODO: hold compiled XGrammar handle
}

impl pie::structured::grammar::Host for InstanceState {}

impl pie::structured::grammar::HostGrammar for InstanceState {
    async fn from_json_schema(
        &mut self,
        schema: String,
    ) -> Result<Result<Resource<Grammar>, String>> {
        // TODO: compile via XGrammar
        let _ = schema;
        Ok(Err("Grammar compilation not yet implemented".into()))
    }

    async fn json(&mut self) -> Result<Resource<Grammar>> {
        // TODO: return built-in free-form JSON grammar
        let grammar = Grammar {};
        Ok(self.ctx().table.push(grammar)?)
    }

    async fn from_regex(
        &mut self,
        pattern: String,
    ) -> Result<Result<Resource<Grammar>, String>> {
        let _ = pattern;
        Ok(Err("Grammar compilation not yet implemented".into()))
    }

    async fn from_ebnf(
        &mut self,
        ebnf: String,
    ) -> Result<Result<Resource<Grammar>, String>> {
        let _ = ebnf;
        Ok(Err("Grammar compilation not yet implemented".into()))
    }

    async fn drop(&mut self, this: Resource<Grammar>) -> Result<()> {
        let _ = self.ctx().table.delete(this);
        Ok(())
    }
}
