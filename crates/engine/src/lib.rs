#![deny(
    clippy::todo,
    clippy::unimplemented,
    clippy::dbg_macro,
    clippy::mem_forget
)]
#![deny(clippy::print_stdout, clippy::print_stderr)]
#![forbid(unsafe_code)]

pub mod adapter;
pub mod ahead;
pub mod caps;
pub mod channel;
#[cfg(any(unix, windows))]
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
pub use caps::{Capabilities, DeviceFacts, FireLimits, KvCopyDomains, PoolFacts};
pub use channel::{
    ChannelId, ChannelRegistration, ChannelSeed, HostMirror, RegisteredChannel, Ticket,
};
pub use engine::{
    CompletionSink, Engine, KvWrites, StepDone, StepOutcome, TapGroups, TapSink, Tapped,
};
pub use error::{Error, Result};
pub use fire::{
    Attachment, FireId, FireTicket, FoldLen, FrameId, FrameSubmission, FrameTicket, KvDelta, Lane,
    LaneReadout, Mask, Masking, MediaEncode, Readout, RsReset, RsSnapshot, RsVerb, Serves, Step,
};
pub use load::{Budgets, Checkpoint, LoadFacts, LoadRequest, Loaded, Residency};
pub use program::{
    BindExtents, BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration,
};
pub use transfer::{
    KvCopy, KvExport, KvHandle, KvLayout, KvLayoutKind, KvMove, KvRegion, MemoryDomain, StateCopy,
    StateDirection, StateMove,
};
