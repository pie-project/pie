use serde::{Deserialize, Serialize};

use crate::adapter::AdapterRegistration;
use crate::caps::DeviceFacts;
use crate::channel::{ChannelId, ChannelRegistration, RegisteredChannel};
use crate::chunk::{Evacuation, SlotCopy, WeightResidency, WeightUses};
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

    /// Puts memory under every kv page below `kv_pages` and every state
    /// slot, ahead of the runtime handing any of them out, so no frame finds
    /// its pages unbacked. A refusal leaves what was committed standing. A
    /// pool that is whole from its load has nothing to do.
    fn commit_pool(&mut self, kv_pages: u32) -> Result<()> {
        let _ = kv_pages;
        Ok(())
    }

    /// Gives back the memory under every kv page at and past `kv_pages`. No
    /// frame reads those pages: the runtime has moved what they held and
    /// waited out every frame that named them, so nothing here drains.
    fn release_pool(&mut self, kv_pages: u32) -> Result<()> {
        let _ = kv_pages;
        Ok(())
    }

    /// Puts memory under every state slot and buffer page below `slots`,
    /// where `PoolFacts::elastic_state_slot_bytes` says it does not stand
    /// from the load.
    fn commit_state(&mut self, slots: u32) -> Result<()> {
        let _ = slots;
        Ok(())
    }

    /// Gives back the memory under every state slot and buffer page at and
    /// past `slots`; as [`Engine::release_pool`], no frame reads them.
    fn release_state(&mut self, slots: u32) -> Result<()> {
        let _ = slots;
        Ok(())
    }

    /// Backs the pool's first `segments` segments
    /// ([`crate::chunk::ChunkFacts::segments`]), ahead of the runtime
    /// handing out any of their chunks; never on a fire's path. A refusal
    /// leaves what was committed standing.
    fn commit_segments(&mut self, segments: u32) -> Result<()> {
        let _ = segments;
        Err(self.unsupported("commit_segments"))
    }

    /// Gives back every segment at and past `segments`. The runtime has
    /// evacuated their chunks and waited out every frame that named them.
    fn release_segments(&mut self, segments: u32) -> Result<()> {
        let _ = segments;
        Err(self.unsupported("release_segments"))
    }

    /// Copies each moved chunk's bytes onto its new physical chunk and
    /// updates the engine's [`crate::chunk::ChunkTable`], stream-ordered
    /// before the next fire; the slot refs the runtime holds stand.
    fn evacuate(&mut self, evacuation: &Evacuation) -> Result<()> {
        let _ = evacuation;
        Err(self.unsupported("evacuate"))
    }

    /// Copies slots of one shape between tiers or within one; replaces
    /// [`Engine::copy_kv`] and [`Engine::copy_state`] once every backend
    /// serves it.
    ///
    /// Tiers (CONTRACT-BC 22): an engine that states
    /// [`crate::chunk::ChunkFacts::host_chunks`] moves whole slots of every
    /// state shape between its device domain and
    /// [`crate::transfer::MemoryDomain::HostPinned`], a host ref `r` of
    /// shape `s` at [`crate::chunk::SlotShape::tier_offset`] of its host
    /// chunks, laid out as the device slot (one span a slot); one that
    /// states [`crate::chunk::ChunkFacts::disk_chunks`] moves kv list slots
    /// between the device (or the host tier) and
    /// [`crate::transfer::MemoryDomain::LocalDisk`], disk ref `r` slot `r`
    /// of its kv slot file. Token runs are device to device only. Each copy
    /// is ordered after every fire submitted before it and before every one
    /// after (a stream-ordered copy, or a drain), and copies in submission
    /// order: the runtime reuses a source slot and names a destination
    /// again as soon as the call returns.
    fn copy_slots(&mut self, copy: &SlotCopy) -> Result<()> {
        let _ = copy;
        Err(self.unsupported("copy_slots"))
    }

    /// Which atoms of the streamable weights
    /// ([`crate::chunk::ChunkFacts::weights`]) are resident from the next
    /// fire on, and in which pool slots; the whole set each time. The
    /// runtime calls it once a governor tick at most, ordered before the
    /// next fire; the engine fills slots newly named (from its host copy)
    /// before a fire reads them. A use of an atom not named is the
    /// engine's: it streams through a staging slot and is not kept.
    fn weight_residency(&mut self, residency: &WeightResidency) -> Result<()> {
        let _ = residency;
        Err(self.unsupported("weight_residency"))
    }

    /// How often fires read each streamable atom since the last call,
    /// resident or not: the record the runtime orders the weight cache by.
    fn weight_uses(&mut self) -> Result<WeightUses> {
        Err(self.unsupported("weight_uses"))
    }

    /// Gives back what fires have touched of the scratch, waiting out the
    /// frames in flight; the next fires pay for what they touch again.
    fn release_scratch(&mut self) -> Result<()> {
        Ok(())
    }

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

    /// The bytes of host rs rows: `slots`' state, then `pages`' buffers, one
    /// after another, once every copy into them has landed. For a model whose
    /// state is a window, `pages` are host windowed pages and `slots` empty.
    fn read_host_state(&mut self, slots: &[u32], pages: &[u32]) -> Result<Vec<u8>> {
        let _ = (slots, pages);
        Err(self.unsupported("read_host_state"))
    }

    /// Writes `bytes`, laid out as [`Engine::read_host_state`] returns them.
    fn write_host_state(&mut self, slots: &[u32], pages: &[u32], bytes: &[u8]) -> Result<()> {
        let _ = (slots, pages, bytes);
        Err(self.unsupported("write_host_state"))
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
