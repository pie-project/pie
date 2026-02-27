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
use crate::inference::structured::grammar::Grammar as InternalGrammar;
use crate::inference::structured::compiled_grammar::CompiledGrammar;
use crate::inference::structured::matcher::GrammarMatcher;
use anyhow::Result;
use std::sync::Arc;
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
        context::append_buffered_tokens(ctx.model_id, ctx.context_id, tokens)?;
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
        context::append_buffered_tokens(ctx.model_id, ctx.context_id, tokens)?;
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
        model: Resource<crate::api::model::Model>,
        tools: Vec<String>,
    ) -> Result<Resource<crate::api::inference::Matcher>> {
        let model_res = self.ctx().table.get(&model)?;
        let instruct = model_res.model.instruct();
        let tok = model_res.model.tokenizer().clone();
        let stop_tokens = instruct.seal();

        let ebnf = instruct.tool_call_grammar(&tools)
            .ok_or_else(|| anyhow::anyhow!("model does not support constrained tool-call generation"))?;

        let grammar = InternalGrammar::from_ebnf(&ebnf, "root")
            .map_err(|e| anyhow::anyhow!("failed to compile tool-call grammar: {}", e))?;
        let grammar_arc = Arc::new(grammar);

        let compiled = CompiledGrammar::get_or_compile(&ebnf, &grammar_arc, &tok);
        let inner = GrammarMatcher::with_compiled(compiled, tok, stop_tokens, 10);

        let matcher = crate::api::inference::Matcher { inner };
        Ok(self.ctx().table.push(matcher)?)
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
