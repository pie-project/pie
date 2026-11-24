use serde::{Deserialize, Serialize};

use crate::adapter::AdapterRegistration;
use crate::caps::DeviceFacts;
use crate::channel::{ChannelId, ChannelRegistration, RegisteredChannel};
use crate::chunk::{Evacuation, SlotCopy, WeightResidency, WeightUses};
use crate::error::{Error, Result};
use crate::fire::{FrameId, FrameSubmission, FrameTicket, Step};
use crate::load::{LoadRequest, Loaded, Reload, Reloaded};
use crate::program::{
    BoundInstance, InstanceBinding, InstanceId, ProgramCompiler, ProgramId, ProgramRegistration,
};
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
/// `group` (of [`Taps::groups`]), page after page; or why not. A tap
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

    /// Whether the engine keeps a disk kv tier. One that keeps none is not
    /// handed the default `disk_kv_dir`; a stated one it still refuses.
    fn disk_kv_tier(&self) -> bool {
        true
    }

    fn bind_thread(&mut self) -> Result<()> {
        Ok(())
    }

    fn load(&mut self, request: LoadRequest) -> Result<Loaded>;

    fn submit(&mut self, frame: &FrameSubmission) -> Result<FrameTicket>;

    fn refused_unlaunched(&self) -> bool {
        false
    }

    fn settle_frame(&mut self, ticket: &mut FrameTicket) -> Result<()>;

    fn completion(&mut self) -> Option<&mut dyn Completion> {
        None
    }

    fn programs(&mut self) -> Option<&mut dyn Programs> {
        None
    }

    fn channels(&mut self) -> Option<&mut dyn Channels> {
        None
    }

    fn pool(&mut self) -> Option<&mut dyn Pool> {
        None
    }

    fn weights(&mut self) -> Option<&mut dyn WeightCache> {
        None
    }

    fn tiers(&mut self) -> Option<&mut dyn Tiers> {
        None
    }

    fn peers(&mut self) -> Option<&mut dyn Peers> {
        None
    }

    fn taps(&mut self) -> Option<&mut dyn Taps> {
        None
    }
}

pub trait Completion: Engine {
    fn on_complete(&mut self, sink: CompletionSink);

    /// Publishes what the guests' programs of the steps already landed put
    /// on their channels, without waiting on a step still in flight. An
    /// engine that publishes as each step lands needs nothing here.
    fn publish_landed(&mut self) -> Result<()> {
        Ok(())
    }

    fn expect_fire(&mut self, submission: &Step) {
        let _ = submission;
    }
}

pub trait Programs: Engine {
    fn register(&mut self, registration: &ProgramRegistration) -> Result<ProgramId>;

    fn compiler(&self) -> Option<ProgramCompiler> {
        None
    }

    fn bind(&mut self, binding: &InstanceBinding) -> Result<BoundInstance>;

    fn close(&mut self, id: InstanceId) -> Result<()>;

    fn adapter(&mut self, registration: &AdapterRegistration) -> Result<()>;
}

pub trait Channels: Engine {
    fn open(&mut self, registration: &ChannelRegistration) -> Result<RegisteredChannel>;

    fn close(&mut self, id: ChannelId) -> Result<()>;

    fn publish(&mut self, instance: InstanceId, channel: u32, cell: &[u8]) -> Result<bool>;

    fn take(&mut self, instance: InstanceId, channel: u32) -> Result<Option<Vec<u8>>>;
}

pub trait Pool: Engine {
    /// Backs the pool's first `segments` segments
    /// ([`crate::chunk::ChunkFacts::segments`]), ahead of the runtime
    /// handing out any of their chunks; never on a fire's path. A refusal
    /// leaves what was committed standing.
    fn commit(&mut self, segments: u32) -> Result<()>;

    /// Gives back every segment at and past `segments`. The runtime has
    /// evacuated their chunks and waited out every frame that named them.
    fn release(&mut self, segments: u32) -> Result<()>;

    /// Copies each moved chunk's bytes onto its new physical chunk and
    /// updates the engine's [`crate::chunk::ChunkTable`], stream-ordered
    /// before the next fire; the slot refs the runtime holds stand.
    fn evacuate(&mut self, evacuation: &Evacuation) -> Result<()>;
}

pub trait WeightCache: Engine {
    /// Which atoms of the streamable weights
    /// ([`crate::chunk::ChunkFacts::weights`]) are resident from the next
    /// fire on, and in which pool slots; the whole set each time. The
    /// runtime calls it once a governor tick at most, ordered before the
    /// next fire; the engine fills slots newly named (from its host copy)
    /// before a fire reads them. A use of an atom not named is the
    /// engine's: it streams through a staging slot and is not kept.
    fn set_residency(&mut self, residency: &WeightResidency) -> Result<()>;

    /// How often fires read each streamable atom, resident or not: the
    /// record the runtime orders the weight cache by. An engine may answer
    /// a call behind (the reads between the two calls before, its use
    /// counters read without waiting for the device), every read counted
    /// once.
    fn uses(&mut self) -> Result<WeightUses>;

    fn reload(&mut self, reload: &Reload) -> Result<Reloaded> {
        let _ = reload;
        Err(Error::unsupported(self.kind(), "reload_weights"))
    }
}

pub trait Tiers: Engine {
    /// Copies slots of one shape between tiers or within one; replaces
    /// [`Tiers::copy_kv`] and [`Tiers::copy_state`] once every backend
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
    fn copy(&mut self, copy: &SlotCopy) -> Result<()>;

    /// The bytes of host tier slots (CONTRACT-BC 22): for each `(shape,
    /// refs)` in order, each ref's slot of that shape at
    /// [`crate::chunk::SlotShape::tier_offset`] of the host chunks, laid out
    /// as the device slot (what [`Tiers::copy`] moves whole), one
    /// slot after another; once every copy into them has landed (the
    /// runtime waits on that copy's completion first). What a shared pool's
    /// snapshot reads of a set's state: its record's pieces and its rs list
    /// or window slots, parked on the host.
    fn read(&mut self, slots: &[(crate::chunk::ShapeId, Vec<u32>)]) -> Result<Vec<u8>>;

    /// Writes `bytes`, laid out as [`Tiers::read`] returns
    /// them, into host tier slots `slots`; a later [`Tiers::copy`]
    /// from the host tier reads them.
    fn write(&mut self, slots: &[(crate::chunk::ShapeId, Vec<u32>)], bytes: &[u8]) -> Result<()>;

    fn copy_kv(&mut self, copy: &KvCopy) -> Result<()>;

    fn copy_state(&mut self, copy: &StateCopy) -> Result<()> {
        let _ = copy;
        Err(Error::unsupported(self.kind(), "copy_state"))
    }
}

pub trait Peers: Engine {
    fn export(&self) -> Option<KvHandle>;

    /// How [`Peers::read`] lays out pages, when another engine of
    /// another layout should be able to take them; `None` when this one
    /// cannot say.
    fn layout(&self) -> Option<state::PageLayout>;

    /// The bytes of host kv slots `slots`, one slot after another, laid out
    /// as this engine lays a slot out, once every copy into them has landed.
    /// What one worker's slots hold, another worker of the same layout takes
    /// with [`Peers::write`].
    fn read(&mut self, slots: &[u32]) -> Result<Vec<u8>>;

    /// Writes `bytes`, laid out as [`Peers::read`] returns them, into
    /// host kv slots `slots`.
    fn write(&mut self, slots: &[u32], bytes: &[u8]) -> Result<()>;
}

pub trait Taps: Engine {
    /// The groups a tapped page's planes are read out in, each as spans
    /// `(offset, bytes)` of a host kv slot, in the order they land; `None`:
    /// this engine taps nothing.
    fn groups(&self) -> Option<TapGroups>;

    fn subscribe(&mut self, sink: TapSink);
}

const _: () = {
    #[allow(dead_code)]
    fn object_safe(engine: &dyn Engine) -> &'static str {
        engine.kind()
    }
};
