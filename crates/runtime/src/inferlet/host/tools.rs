//! pie:instruct/tool-use — Tool calling support
//!
//! Imported by inferlets that support tool-use capabilities.
//! Delegates to the model's `Instruct` implementation.

use crate::inferlet::ProcessCtx;
use crate::inferlet::host::pie;
use anyhow::Result;
use grammar::matcher::GrammarMatcher;
use models::template::{ToolDecoder, ToolEvent};
use std::collections::VecDeque;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

/// Tool-use decoder resource — wraps a model-specific ToolDecoder trait object.
///
/// A decoder answers a batch with every call that batch completed, and a batch
/// can complete more than one. The WIT `feed` hands back a single event, so
/// the surplus queues here and drains on the following calls; an idle batch
/// answers with `ToolEvent::None`, which is the WIT's `start` because the
/// variant has no idle arm and `start` is what an idle feed has always said.
pub struct Decoder {
    inner: Box<dyn ToolDecoder>,
    pending: VecDeque<ToolEvent>,
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
        let tokens = crate::model::model().instruct().equip(&tools);
        Ok(Ok(tokens))
    }

    async fn answer(&mut self, name: String, value: String) -> Result<Vec<u32>> {
        Ok(crate::model::model().instruct().answer(&name, &value))
    }

    async fn format(
        &mut self,
        tools: Vec<String>,
    ) -> Result<Option<Resource<crate::inferlet::host::grammar::Grammar>>> {
        let Some(tg) = crate::model::model().instruct().tool_call_grammar(&tools) else {
            return Ok(None);
        };
        let compiled =
            crate::inferlet::host::grammar::grammar_compiler().compile_ebnf(&tg.source, "root")?;
        let grammar = crate::inferlet::host::grammar::Grammar { compiled };
        Ok(Some(self.ctx().table.push(grammar)?))
    }

    async fn create_matcher(
        &mut self,
        tools: Vec<String>,
    ) -> Result<Resource<crate::inferlet::host::grammar::Matcher>> {
        let model = crate::model::model();
        let instruct = model.instruct();
        let stop_tokens = instruct.seal();

        let tg = instruct.tool_call_grammar(&tools).ok_or_else(|| {
            anyhow::anyhow!("model does not support constrained tool-call generation")
        })?;

        let compiled =
            crate::inferlet::host::grammar::grammar_compiler().compile_ebnf(&tg.source, "root")?;
        let inner = GrammarMatcher::with_compiled(compiled, stop_tokens, 10);

        let matcher = crate::inferlet::host::grammar::Matcher { inner };
        Ok(self.ctx().table.push(matcher)?)
    }
}

impl pie::inferlet::tools::HostDecoder for ProcessCtx {
    async fn new(&mut self) -> Result<Resource<Decoder>> {
        let inner = crate::model::model().instruct().tool_decoder();
        let decoder = Decoder {
            inner,
            pending: VecDeque::new(),
        };
        Ok(self.ctx().table.push(decoder)?)
    }

    async fn feed(
        &mut self,
        this: Resource<Decoder>,
        tokens: Vec<u32>,
    ) -> Result<Result<pie::inferlet::tools::Event, pie::inferlet::types::Error>> {
        let decoder = self.ctx().table.get_mut(&this)?;
        let events = decoder.inner.feed(&tokens);
        decoder.pending.extend(events);
        let event = decoder.pending.pop_front().unwrap_or(ToolEvent::None);
        Ok(Ok(match event {
            ToolEvent::None | ToolEvent::Start => pie::inferlet::tools::Event::Start,
            ToolEvent::Call(name, args) => {
                pie::inferlet::tools::Event::Call(pie::inferlet::tools::ToolCall {
                    name,
                    arguments_json: args,
                })
            }
        }))
    }

    async fn reset(&mut self, this: Resource<Decoder>) -> Result<()> {
        let decoder = self.ctx().table.get_mut(&this)?;
        decoder.inner.reset();
        decoder.pending.clear();
        Ok(())
    }

    async fn drop(&mut self, this: Resource<Decoder>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}
