//! pie:structured/matcher - Token-level constrained decoding
//!
//! Stub implementation — will be backed by XGrammar.

use crate::api::pie;
use crate::api::structured::grammar::Grammar;
use crate::linker::InstanceState;
use anyhow::Result;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

/// Stateful matcher that walks the grammar automaton, producing token masks.
#[derive(Debug)]
pub struct Matcher {
    // TODO: hold XGrammar matcher state
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
