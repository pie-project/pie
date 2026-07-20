//! pie:instruct/tool-use — Tool calling support
//!
//! Imported by inferlets that support tool-use capabilities.
//! Delegates to the model's `Instruct` implementation.

use crate::api::pie;
use crate::inference::structured::compiled_grammar::CompiledGrammar;
use crate::inference::structured::matcher::GrammarMatcher;
use crate::instance::InstanceState;
use crate::model::instruct::{ToolDecoder, ToolEvent};
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
        let instruct = model.model.instruct();
        // Fail loudly rather than returning zero declarations: the caller is
        // about to build a matcher from the same toolset, and an unconstrained
        // prompt paired with a constrained decode is the one combination that
        // makes the model call tools it was never shown.
        if instruct.tool_declarations_require_system_turn() && !tools.is_empty() {
            return Ok(Err(format!(
                "model `{}` declares tools inside the system turn; use system-equip instead of equip",
                model.model.name()
            )));
        }
        let tokens = instruct.equip(&tools);
        Ok(Ok(tokens))
    }

    async fn system_equip(
        &mut self,
        model: Resource<crate::api::model::Model>,
        system: String,
        tools: Vec<String>,
    ) -> Result<Result<Vec<u32>, pie::core::types::Error>> {
        let model = self.ctx().table.get(&model)?;
        let Some(tokens) = model.model.instruct().system_equip(&system, &tools) else {
            return Ok(Err(format!(
                "model `{}` cannot declare this toolset; no tools were declared",
                model.model.name()
            )));
        };
        Ok(Ok(tokens))
    }

    /// Render a tool result into tokens, trapping when it cannot be represented.
    ///
    /// The WIT signature has no way to report an unrepresentable result, so an
    /// architecture that refuses (Gemma 4, via [`try_answer`]) surfaces as a
    /// trap in the calling component. That is abrupt but loud; returning an
    /// empty or truncated token list would hand the model a turn that means
    /// something other than what the tool returned, which is the failure this
    /// whole path exists to prevent.
    ///
    /// [`try_answer`]: crate::model::instruct::Instruct::try_answer
    async fn answer(
        &mut self,
        model: Resource<crate::api::model::Model>,
        name: String,
        value: String,
    ) -> Result<Vec<u32>> {
        let model = self.ctx().table.get(&model)?;
        model
            .model
            .instruct()
            .try_answer(&name, &value)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "model `{}` cannot represent this tool result; it carries a delimiter the wire format has no escape for",
                    model.model.name()
                )
            })
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

    async fn create_decoder_for_tools(
        &mut self,
        model: Resource<crate::api::model::Model>,
        tools: Vec<String>,
    ) -> Result<Resource<Decoder>> {
        let model = self.ctx().table.get(&model)?;
        let inner = model.model.instruct().tool_decoder_for_tools(&tools);
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

        let tg = instruct.tool_call_grammar(&tools).ok_or_else(|| {
            anyhow::anyhow!("model does not support constrained tool-call generation")
        })?;

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
