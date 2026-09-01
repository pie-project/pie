//! pie:instruct/reasoning — Reasoning/thinking block detection
//!
//! Imported by inferlets that support reasoning capabilities.
//! Delegates to the model's `Instruct` implementation.

use crate::inferlet::ProcessCtx;
use crate::inferlet::host::pie;
use anyhow::Result;
use models::template::{ReasoningDecoder, ReasoningEvent};
use std::collections::VecDeque;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

/// Reasoning decoder resource — wraps a model-specific ReasoningDecoder trait object.
///
/// A batch can both close a thinking block and carry the reply that follows
/// it. The WIT `feed` hands back one event, so the surplus queues here and
/// drains on the following calls rather than being dropped.
pub struct Decoder {
    inner: Box<dyn ReasoningDecoder>,
    pending: VecDeque<ReasoningEvent>,
}

impl std::fmt::Debug for Decoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("reasoning::Decoder").finish()
    }
}

impl pie::inferlet::reasoning::Host for ProcessCtx {}

impl pie::inferlet::reasoning::HostDecoder for ProcessCtx {
    async fn new(&mut self) -> Result<Resource<Decoder>> {
        let inner = crate::model::model().instruct().reasoning_decoder();
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
    ) -> Result<Result<pie::inferlet::reasoning::Event, pie::inferlet::types::Error>> {
        let decoder = self.ctx().table.get_mut(&this)?;
        let events = decoder.inner.feed(&tokens);
        decoder.pending.extend(events);
        let event = decoder
            .pending
            .pop_front()
            .unwrap_or(ReasoningEvent::Delta(String::new()));
        Ok(Ok(match event {
            ReasoningEvent::Start => pie::inferlet::reasoning::Event::Start,
            ReasoningEvent::Delta(s) => pie::inferlet::reasoning::Event::Delta(s),
            ReasoningEvent::Complete(s) => pie::inferlet::reasoning::Event::Complete(s),
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
