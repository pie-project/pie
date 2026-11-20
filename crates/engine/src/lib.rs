#![deny(
    clippy::todo,
    clippy::unimplemented,
    clippy::dbg_macro,
    clippy::mem_forget
)]
#![deny(clippy::print_stdout, clippy::print_stderr)]
#![forbid(unsafe_code)]

pub mod adapter;
pub mod caps;
pub mod channel;
pub mod chunk;
#[cfg(any(unix, windows, target_arch = "wasm32"))]
pub mod disk;
pub mod engine;
pub mod error;
pub mod fire;
pub mod fit;
pub mod frame;
pub mod load;
pub mod placement;
pub mod ports;
pub mod program;
pub mod rs;
pub mod runahead;
pub mod transfer;

pub use adapter::{AdapterPlane, AdapterRegistration};
pub use caps::{Capabilities, DeviceFacts, FireLimits, PoolFacts};
pub use channel::{
    ChannelId, ChannelRegistration, ChannelSeed, HostMirror, RegisteredChannel, Ticket,
};
pub use chunk::{
    ChunkFacts, ChunkRun, ChunkTable, Evacuation, Growth, Pinned, Reserved, ReservedRun,
    ResidentWeight, SegmentFacts, ShapeId, SlotCopy, SlotKind, SlotRole, SlotRun, SlotShape,
    WeightAtom, WeightKind, WeightResidency, WeightShape, WeightUses,
};
pub use engine::{
    Channels, Completion, CompletionSink, Engine, KvWrites, Peers, Pool, Programs, StepDone,
    StepOutcome, TapGroups, TapSink, Tapped, Taps, Tiers, WeightCache,
};
pub use error::{Error, Result};
pub use fire::{
    Attachment, FireId, FireTicket, FoldLen, FrameId, FrameSubmission, FrameTicket, KvDelta, Lane,
    LaneReadout, Mask, Masking, Readout, RsReset, RsSnapshot, RsVerb, Serves, Step,
};
pub use load::{Budgets, Checkpoint, LoadFacts, LoadRequest, Loaded, Reload, Reloaded, Residency};
pub use program::{
    BindExtents, BoundInstance, InstanceBinding, InstanceId, ProgramCompiler, ProgramId,
    ProgramRegistration,
};
pub use transfer::{
    KvCopy, KvHandle, KvLayout, KvLayoutKind, KvMove, KvRegion, MemoryDomain, StateCopy, StateMove,
};
