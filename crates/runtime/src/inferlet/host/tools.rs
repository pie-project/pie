use crate::inferlet::ProcessCtx;
use crate::inferlet::host::pie;
use anyhow::Result;
use chat_template::{ToolDecoder, ToolEvent};
use std::collections::VecDeque;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

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
    async fn answers(&mut self, results: Vec<(String, String)>) -> Result<Vec<u32>> {
        Ok(crate::model::model().instruct().answers(&results))
    }

    async fn assistant_calls(
        &mut self,
        message: String,
        calls: Vec<pie::inferlet::tools::ToolCall>,
        current: bool,
    ) -> Result<Vec<u32>> {
        let calls: Vec<(String, String)> = calls
            .into_iter()
            .map(|call| (call.name, call.arguments_json))
            .collect();
        Ok(crate::model::model()
            .instruct()
            .assistant_calls(&message, &calls, current))
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
