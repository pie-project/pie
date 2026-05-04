//! pie:instruct/tool-use — Tool calling support
//!
//! Imported by inferlets that support tool-use capabilities.
//! Delegates to the model's `Instruct` implementation.

use crate::api::pie;
use crate::instance::InstanceState;
use crate::model::instruct::{ToolDecoder, ToolEvent};
use crate::inference::structured::compiled_grammar::CompiledGrammar;
use crate::inference::structured::matcher::GrammarMatcher;
use anyhow::Result;
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
        model: Resource<crate::api::model::Model>,
        tools: Vec<String>,
    ) -> Result<Result<Vec<u32>, pie::core::types::Error>> {
        let model = self.ctx().table.get(&model)?;
        let tokens = model.model.instruct().equip(&tools);
        Ok(Ok(tokens))
    }

    async fn answer(
        &mut self,
        model: Resource<crate::api::model::Model>,
        name: String,
        value: String,
    ) -> Result<Vec<u32>> {
        let model = self.ctx().table.get(&model)?;
        Ok(model.model.instruct().answer(&name, &value))
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

    async fn format(
        &mut self,
        model: Resource<crate::api::model::Model>,
        tools: Vec<String>,
    ) -> Result<Option<Resource<crate::api::inference::Grammar>>> {
        let model_res = self.ctx().table.get(&model)?;
        let Some(tg) = model_res.model.instruct().tool_call_grammar(&tools) else {
            return Ok(None);
        };
        let grammar = crate::api::inference::Grammar {
            source: tg.source,
            inner: tg.grammar,
        };
        Ok(Some(self.ctx().table.push(grammar)?))
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

        let tg = instruct.tool_call_grammar(&tools)
            .ok_or_else(|| anyhow::anyhow!("model does not support constrained tool-call generation"))?;

        let compiled = CompiledGrammar::get_or_compile(&tg.source, &tg.grammar, &tok);
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
