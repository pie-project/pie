use serde::{Deserialize, Serialize};

use crate::adapter::AdapterRegistration;
use crate::caps::DeviceFacts;
use crate::channel::{ChannelId, ChannelRegistration, RegisteredChannel};
use crate::error::{Error, Result};
use crate::fire::{FrameId, FrameSubmission, FrameTicket, MediaEncode, Step};
use crate::load::{LoadRequest, Loaded};
use crate::program::{BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration};
use crate::transfer::{KvCopy, KvHandle, StateCopy};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StepDone {
    pub frame: FrameId,
    pub step: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepOutcome {
    Committed,
    Wrote(Vec<KvWrites>),
    Faulted(String),
}

/// What one lane wrote to the KV cache in a step, row by row: the token it
/// embedded at its position, into the slot at `offsets[i]` of pool page
/// `pages[i]`. Rows are empty where the engine cannot place them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KvWrites {
    pub tag: u64,
    pub tokens: Vec<u32>,
    pub positions: Vec<u32>,
    pub pages: Vec<u32>,
    pub offsets: Vec<u32>,
    pub fold: Option<u32>,
}

pub type CompletionSink = std::sync::Arc<dyn Fn(StepDone, StepOutcome) + Send + Sync>;

/// A group of a tapped lane's pages, read out: each page's spans of group
/// `group` (of [`Engine::tap_groups`]), page after page; or why not. A tap
/// that fails says so once, for its first group not told.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tapped {
    pub tag: u64,
    pub group: u32,
    pub bytes: std::result::Result<Vec<u8>, String>,
}

pub type TapSink = std::sync::Arc<dyn Fn(Tapped) + Send + Sync>;

/// The groups a tap reads a page's planes out in, each its spans `(offset,
/// bytes)` of a host kv slot.
pub type TapGroups = Vec<Vec<(u64, u64)>>;

pub trait Engine: Send {
    fn kind(&self) -> &'static str;

    fn device_facts(&self) -> Option<&DeviceFacts> {
        None
    }

    fn export_kv_handle(&self) -> Option<KvHandle> {
        None
    }

    fn bind_thread(&mut self) -> Result<()> {
        Ok(())
    }

    fn load(&mut self, request: LoadRequest) -> Result<Loaded>;

    fn register_program(&mut self, registration: &ProgramRegistration) -> Result<ProgramId> {
        let _ = registration;
        Err(self.unsupported("register_program"))
    }

    fn register_channel(
        &mut self,
        registration: &ChannelRegistration,
    ) -> Result<RegisteredChannel> {
        let _ = registration;
        Err(self.unsupported("register_channel"))
    }

    fn bind_instance(&mut self, binding: &InstanceBinding) -> Result<BoundInstance> {
        let _ = binding;
        Err(self.unsupported("bind_instance"))
    }

    fn close_instance(&mut self, id: InstanceId) -> Result<()> {
        let _ = id;
        Err(self.unsupported("close_instance"))
    }

    fn close_channel(&mut self, id: ChannelId) -> Result<()> {
        let _ = id;
        Err(self.unsupported("close_channel"))
    }

    fn publish_channel(&mut self, instance: InstanceId, channel: u32, cell: &[u8]) -> Result<bool> {
        let _ = (instance, channel, cell);
        Err(self.unsupported("publish_channel"))
    }

    fn take_channel(&mut self, instance: InstanceId, channel: u32) -> Result<Option<Vec<u8>>> {
        let _ = (instance, channel);
        Err(self.unsupported("take_channel"))
    }

    fn register_adapter(&mut self, registration: &AdapterRegistration) -> Result<()> {
        let _ = registration;
        Err(self.unsupported("register_adapter"))
    }

    fn submit(&mut self, frame: &FrameSubmission) -> Result<FrameTicket>;

    fn settles_asynchronously(&self) -> bool {
        false
    }

    fn on_complete(&mut self, sink: CompletionSink) {
        let _ = sink;
    }

    /// The groups a tapped page's planes are read out in, each as spans
    /// `(offset, bytes)` of a host kv slot, in the order they land; `None`:
    /// this engine taps nothing.
    fn tap_groups(&self) -> Option<TapGroups> {
        None
    }

    fn on_tapped(&mut self, sink: TapSink) {
        let _ = sink;
    }

    fn settle_frame(&mut self, ticket: &mut FrameTicket) -> Result<()> {
        let _ = ticket;
        Ok(())
    }

    /// Publishes what the guests' programs of the steps already landed put
    /// on their channels, without waiting on a step still in flight. An
    /// engine that publishes as each step lands needs nothing here.
    fn publish_landed(&mut self) -> Result<()> {
        Ok(())
    }

    fn expect_fire(&mut self, submission: &Step) {
        let _ = submission;
    }

    fn copy_kv(&mut self, copy: &KvCopy) -> Result<()> {
        let _ = copy;
        Err(self.unsupported("copy_kv"))
    }

    fn copy_state(&mut self, copy: &StateCopy) -> Result<()> {
        let _ = copy;
        Err(self.unsupported("copy_state"))
    }

    fn encode(&mut self, plan: &mut MediaEncode) -> Result<()> {
        let _ = plan;
        Err(self.unsupported("encode"))
    }

    /// How [`Engine::read_host_kv`] lays out pages, when another engine of
    /// another layout should be able to take them; `None` when this one
    /// cannot say.
    fn host_kv_layout(&self) -> Option<state::PageLayout> {
        None
    }

    /// The bytes of host kv slots `slots`, one slot after another, laid out
    /// as this engine lays a slot out, once every copy into them has landed.
    /// What one worker's slots hold, another worker of the same layout takes
    /// with [`Engine::write_host_kv`].
    fn read_host_kv(&mut self, slots: &[u32]) -> Result<Vec<u8>> {
        let _ = slots;
        Err(self.unsupported("read_host_kv"))
    }

    /// Writes `bytes`, laid out as [`Engine::read_host_kv`] returns them, into
    /// host kv slots `slots`.
    fn write_host_kv(&mut self, slots: &[u32], bytes: &[u8]) -> Result<()> {
        let _ = (slots, bytes);
        Err(self.unsupported("write_host_kv"))
    }

    fn disconnect(&self, message: &str) {
        let _ = message;
    }

    fn unsupported(&self, verb: &'static str) -> Error {
        Error::unsupported(self.kind(), verb)
    }
}

const _: () = {
    #[allow(dead_code)]
    fn object_safe(engine: &dyn Engine) -> &'static str {
        engine.kind()
    }
};
