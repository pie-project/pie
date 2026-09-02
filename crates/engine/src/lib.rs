//! The runtime↔engine contract: what an engine *is*, in types.

#![deny(missing_docs)]
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
pub mod engine;
pub mod error;
pub mod fire;
pub mod frame;
pub mod load;
pub mod program;
pub mod runahead;
pub mod transfer;

pub use adapter::{AdapterPlane, AdapterRegistration};
pub use caps::{Capabilities, DeviceFacts, FireLimits, KvCopyDomains, PoolFacts};
pub use channel::{
    ChannelId, ChannelRegistration, ChannelSeed, HostMirror, RegisteredChannel, Ticket,
};
pub use engine::{CompletionSink, Engine, StepDone, StepOutcome};
pub use error::{Error, Result};
pub use fire::{
    Attachment, Boundary, FireId, FireTicket, FoldLen, FrameId, FrameSubmission, FrameTicket,
    KvDelta, Lane, LaneReadout, LayerScores, Mask, Masking, MediaEncode, Readout, RsReset, RsVerb,
    Serves, Step,
};
pub use load::{Budgets, Checkpoint, LoadFacts, LoadRequest, Loaded, Residency};
pub use program::{
    BindExtents, BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration,
};
pub use transfer::{
    KvCopy, KvExport, KvHandle, KvLayout, KvLayoutKind, KvMove, KvRegion, MemoryDomain, StateCopy,
    StateMove,
};
