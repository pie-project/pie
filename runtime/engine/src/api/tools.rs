//! pie:instruct/tool-use — Tool calling support
//!
//! Imported by inferlets that support tool-use capabilities.
//! Delegates to the model's `Instruct` implementation.

use crate::api::pie;
use crate::inferlet::ProcessCtx;
use anyhow::Result;
use pie_grammar::compiled_grammar::CompiledGrammar;
use pie_grammar::matcher::GrammarMatcher;
use pie_model::instruct::{ToolDecoder, ToolEvent};
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

impl pie::inferlet::tools::Host for ProcessCtx {
    async fn equip(
        &mut self,
        tools: Vec<String>,
    ) -> Result<Result<Vec<u32>, pie::inferlet::types::Error>> {
        let tokens = pie_model::model().instruct().equip(&tools);
        Ok(Ok(tokens))
    }

    async fn answer(&mut self, name: String, value: String) -> Result<Vec<u32>> {
        Ok(pie_model::model().instruct().answer(&name, &value))
    }

    async fn create_decoder(&mut self) -> Result<Resource<Decoder>> {
        let inner = pie_model::model().instruct().tool_decoder();
        let decoder = Decoder { inner };
        Ok(self.ctx().table.push(decoder)?)
    }

    async fn format(
        &mut self,
        tools: Vec<String>,
    ) -> Result<Option<Resource<crate::api::grammar::Grammar>>> {
        let Some(tg) = pie_model::model().instruct().tool_call_grammar(&tools) else {
            return Ok(None);
        };
        let grammar = crate::api::grammar::Grammar {
            source: tg.source,
            inner: tg.grammar,
        };
        Ok(Some(self.ctx().table.push(grammar)?))
    }

    async fn create_matcher(
        &mut self,
        tools: Vec<String>,
    ) -> Result<Resource<crate::api::grammar::Matcher>> {
        let model = pie_model::model();
        let instruct = model.instruct();
        let tok = model.tokenizer().clone();
        let stop_tokens = instruct.seal();

        let tg = instruct.tool_call_grammar(&tools).ok_or_else(|| {
            anyhow::anyhow!("model does not support constrained tool-call generation")
        })?;

        let compiled = CompiledGrammar::get_or_compile(&tg.source, &tg.grammar, &tok);
        let inner = GrammarMatcher::with_compiled(compiled, tok, stop_tokens, 10);

        let matcher = crate::api::grammar::Matcher { inner };
        Ok(self.ctx().table.push(matcher)?)
    }
}

impl pie::inferlet::tools::HostDecoder for ProcessCtx {
    async fn feed(
        &mut self,
        this: Resource<Decoder>,
        tokens: Vec<u32>,
    ) -> Result<Result<pie::inferlet::tools::Event, pie::inferlet::types::Error>> {
        let decoder = self.ctx().table.get_mut(&this)?;
        let event = decoder.inner.feed(&tokens);
        Ok(Ok(match event {
            ToolEvent::Start => pie::inferlet::tools::Event::Start,
            ToolEvent::Call(name, args) => pie::inferlet::tools::Event::Call((name, args)),
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
