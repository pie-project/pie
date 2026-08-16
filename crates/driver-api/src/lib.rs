//! The runtime ↔ driver contract — what the two sides say to each other, and
//! what a driver promises.
//!
//! Both drivers are Rust and both are called directly, so there is no ABI
//! here and no longer a crate named for one.
//!
//! It is `driver-api` rather than `driver` because it is not driver-common:
//! nine crates depend on it and five of them are not drivers at all
//! (`engine`, `transport`, `controller-api`, `worker`, `tensor-compiler`).
//! A crate both sides speak is a CONTRACT; the substrate only drivers use is
//! `driver`, which is what that name is now for.
//!
//! [`Driver`] is the contract itself: fourteen verbs one execution device
//! answers. Three things had to move here for the trait to be statable, and
//! each was already contract rather than engine:
//!
//! - [`completion`] — a driver MINTS a completion (its `launch` returns one),
//!   so the broker that mints them is vocabulary, not scheduler internals.
//! - [`instance`] — the bind plan and the handle a driver answers with.
//! - [`channel`] — the seed value a bind carries and the registration a
//!   driver answers. The `ChannelEndpoint` an application holds stayed in
//!   `engine`: no verb here takes one.
//!
//! The rest of the vocabulary:
//!
//! - [`local`]: the five records a driver answers with, and the constants
//!   both sides name.
//! - [`capabilities`]: what a driver answers at create and load time.
//! - [`transfer`]: the KV transfer vocabulary shared with cross-node transport.
//! - [`plan`]: owned verb plans shared by local and remote backends.
//! - [`submission`]: the sealed frame a `launch` takes.
//! - [`remote`]: the versioned worker-to-executor protocol.

pub mod capabilities;
pub mod channel;
// The runtime half. See the `runtime` feature: the device crates want the
// plain data below and nothing here, and one of them proves it.
#[cfg(feature = "runtime")]
pub mod completion;
#[cfg(feature = "runtime")]
pub mod driver;
pub mod geometry;
#[cfg(feature = "runtime")]
pub mod instance;
pub mod local;
pub mod plan;
pub mod remote;
pub mod submission;
pub mod transfer;

pub use capabilities::{
    DeviceFacts, DriverCapabilities, ExpertSiteSummary, KV_COPY_DEVICE_TO_DEVICE,
    KV_COPY_DEVICE_TO_HOST, KV_COPY_HOST_TO_DEVICE, KV_COPY_HOST_TO_HOST, ModelFacts,
    ModelLoadDesc, ModelSiteSummary, Mxfp4MoeRequest,
};
pub use channel::{ChannelValue, RegisteredChannel};
#[cfg(feature = "runtime")]
pub use completion::{CompletionBroker, CompletionLease, SubmissionCompletion, WorkItemCompletion};
#[cfg(feature = "runtime")]
pub use driver::{Driver, FrameLaunchOutcome, Unsupported};
pub use geometry::{
    GeometryClass, PIE_DECODE_ENVELOPE_PORTS, PIE_DEVICE_GEOMETRY_PORTS, PIE_DEVICE_PORT_ATTN_MASK,
    PIE_DEVICE_PORT_EMBED_TOKENS, PIE_DEVICE_PORT_KV_LEN, PIE_DEVICE_PORT_PAGE_INDPTR,
    PIE_DEVICE_PORT_PAGES, PIE_DEVICE_PORT_POSITIONS, PIE_DEVICE_PORT_RS_BUFFER_INDPTR,
    PIE_DEVICE_PORT_RS_BUFFER_LEN, PIE_DEVICE_PORT_RS_BUFFER_PAGES, PIE_DEVICE_PORT_RS_W_OFF,
    PIE_DEVICE_PORT_RS_W_SLOT, PIE_DEVICE_PORT_W_OFF, PIE_DEVICE_PORT_W_SLOT,
    PIE_DEVICE_RS_BUFFER_PORTS,
};
#[cfg(feature = "runtime")]
pub use instance::{BoundInstance, BoundWaitSlots, InstanceBindingPlan, InstanceId, ProgramId};
pub use local::*;
pub use plan::{
    CHANNEL_TICKET_NONE, ChannelRegistrationPlan, EmittedKernel, EncodedMask, KvCopyPlan,
    LaunchPlan, MaskWords, MediaEncodePlan, PoolResizePlan, ProgramRegistration,
    RS_FLAG_BUFFER_WRITE, RS_FLAG_FOLD, RS_FLAG_FOLD_LEN_DEVICE, RS_FLAG_RESET, StateCopyPlan,
};
pub use remote::*;
pub use submission::{FrameSubmission, StepSubmission};
pub use transfer::{KvDtype, KvExport, KvHandle, KvLayout, KvLayoutKind, KvRegion, MemoryDomain};
