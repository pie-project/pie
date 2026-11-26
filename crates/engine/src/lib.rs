#![deny(
    clippy::todo,
    clippy::unimplemented,
    clippy::dbg_macro,
    clippy::mem_forget
)]
#![deny(clippy::print_stdout, clippy::print_stderr)]
// Unsafe only where an OS reading has no safe call: `pressure`'s macOS
// sysctl and `disk`'s Windows free space, allowed there by name.
#![deny(unsafe_code)]

pub mod adapter;
pub mod beam;
pub mod caps;
pub mod channel;
pub mod chunk;
#[cfg(any(unix, windows, target_arch = "wasm32"))]
pub mod disk;
pub mod engine;
pub mod error;
pub mod extent;
pub mod fire;
pub mod fit;
pub mod frame;
pub mod guest_claims;
pub mod host_memory;
pub mod live;
pub mod load;
pub mod memory;
pub mod placement;
pub mod ports;
pub mod pressure;
pub mod program;
pub mod rates;
pub mod roofline;
pub mod rs;
pub mod runahead;
pub mod shapes;
pub mod transfer;

pub use adapter::{AdapterPlane, AdapterRegistration};
pub use caps::{
    Capabilities, DeviceFacts, FireLimits, GuestCost, HostOp, HostOps, LibraryForms, MixedLanes,
    MixedRung, PoolFacts, stated_frames,
};
pub use channel::{ChannelId, ChannelRegistration, ChannelSeed, HostMirror, RegisteredChannel};
pub use chunk::{
    ChunkFacts, ChunkRun, ChunkTable, Evacuation, Growth, Pinned, Reserved, ReservedRun,
    ResidentWeight, SegmentFacts, ShapeId, SlotCopy, SlotKind, SlotRole, SlotRun, SlotShape,
    WeightAtom, WeightKind, WeightResidency, WeightShape, WeightUses, WindowedLayer,
};
pub use engine::{
    BehindNeed, BehindSeat, BehindServing, Channels, Completion, CompletionSink, Engine, HostRoom,
    KvWrites, Landing, LandingSink, LaneFault, Peers, Pool, Prepared, ProgramCompile, Programs,
    StepDone, StepOutcome, TapGroups, TapSink, Tapped, Taps, Tiers, WeightCache,
};
pub use error::{Error, Result};
pub use fire::{
    Attachment, FireId, FireTicket, FoldLen, FrameId, FrameSubmission, FrameTicket, KvDelta, Lane,
    LaneReadout, LaneRecord, LaneReplay, ListKind, Mask, Masking, Readout, ReplayBound, ReplayPast,
    ReplayRows, RsReset, RsSnapshot, RsVerb, RungReplay, ScratchBytes, Serves, Step, WindowFold,
};
pub use load::{
    Allowance, Budgets, Checkpoint, DeviceMemory, HostTier, LoadFacts, LoadRequest, Loaded, Reload,
    Reloaded, Residency,
};
pub use memory::SAFETY_FLOOR_BYTES;
pub use program::{
    BindExtents, BoundInstance, FireFacts, FirePair, Footprint, GuestState, GuestTerm,
    InstanceBinding, InstanceId, LaidLanes, ProgramFacts, ProgramId, ProgramRegistration,
    ReadoutFacts, ReadoutLanding, Rebind, RingFacts, RingSlots, ScratchPlace,
};
pub use shapes::{
    Attachments, FireShape, StatedSteps, StepComposition, StepFacts, StepGap, StepKind, StepRules,
    StepShape, StepShapes, Treatment,
};
pub use transfer::{
    DeviceReach, KvCopy, KvHandle, KvLayout, KvLayoutKind, KvMove, KvRegion, MemoryDomain,
    StateCopy, StateMove,
};
