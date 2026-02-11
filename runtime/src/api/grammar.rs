//! pie:structured/grammar - Structured output grammar interface
//!
//! Provides grammar compilation and token-level constrained decoding
//! for JSON Schema, regex, and EBNF grammars.

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

/// Stateful matcher that walks the grammar automaton, producing token masks.
#[derive(Debug)]
pub struct Matcher {
    // TODO: hold XGrammar matcher state
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

impl pie::structured::matcher::Host for InstanceState {}

impl pie::structured::matcher::HostMatcher for InstanceState {
    async fn new(
        &mut self,
        _grammar: Resource<Grammar>,
        _tokenizer: Resource<crate::api::model::Tokenizer>,
    ) -> Result<Resource<Matcher>> {
        // TODO: compile matcher from grammar + tokenizer vocab
        let matcher = Matcher {};
        Ok(self.ctx().table.push(matcher)?)
    }

    async fn accept_tokens(
        &mut self,
        _this: Resource<Matcher>,
        _token_ids: Vec<u32>,
    ) -> Result<Result<(), String>> {
        // TODO: advance matcher state
        Ok(Ok(()))
    }

    async fn next_token_logit_mask(
        &mut self,
        _this: Resource<Matcher>,
    ) -> Result<Vec<u32>> {
        // TODO: compute BRLE bitmask
        Ok(vec![])
    }

    async fn is_terminated(
        &mut self,
        _this: Resource<Matcher>,
    ) -> Result<bool> {
        Ok(false)
    }

    async fn reset(
        &mut self,
        _this: Resource<Matcher>,
    ) -> Result<()> {
        Ok(())
    }

    async fn drop(&mut self, this: Resource<Matcher>) -> Result<()> {
        let _ = self.ctx().table.delete(this);
        Ok(())
    }
}
