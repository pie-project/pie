//! The runtime ↔ driver vocabulary — what the two sides say to each other.
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
//! What is left is the vocabulary the engine and the drivers say the same
//! things in:
//!
//! - [`local`]: the fire descriptors — a step, a frame, a program, a channel.
//!   Still `#[repr(C)]` and still `{ptr, len}` where a slice would do, which
//!   is the shape they took when they crossed a C boundary. Nothing crosses
//!   one now; converting them to borrowed Rust is a mechanical sweep over
//!   ~65 construction sites in `engine` and `driver-cuda` (it was ~65
//!   across three crates before `driver-dummy` was deleted), and it is the
//!   next step rather than this one.
//! - [`adopt`]: the [`local`] → [`plan`] direction, copying a borrowed launch
//!   package into the owned one a driver keeps for the life of the program.
//!   It survives exactly as long as `local`'s borrowed shape does.
//! - [`capabilities`]: what a driver answers at create and load time.
//! - [`transfer`]: the KV transfer vocabulary shared with cross-node transport.
//! - [`plan`]: owned verb plans shared by local and remote backends.
//! - [`remote`]: the versioned worker-to-executor protocol.
//!
//! The C header this crate used to generate, the `unsafe extern "C"` block
//! that declared thirteen `pie_cuda_*` symbols, and the cbindgen binary that
//! wrote the header are all gone with the C++ drivers they served.

pub mod adopt;
pub mod capabilities;
pub mod geometry;
pub mod image;
pub mod local;
pub mod plan;
pub mod remote;
pub mod submission;
pub mod transfer;

pub use adopt::{adopt_emitted_kernels, adopt_package, adopt_region_analysis};
pub use capabilities::{
    DeviceFacts, DriverCapabilities, ExpertSiteSummary, KV_COPY_DEVICE_TO_DEVICE,
    KV_COPY_DEVICE_TO_HOST, KV_COPY_HOST_TO_DEVICE, KV_COPY_HOST_TO_HOST, ModelLoadDesc,
    ModelSiteSummary, Mxfp4MoeRequest,
};
pub use geometry::{
    GeometryClass, PIE_DECODE_ENVELOPE_PORTS, PIE_DEVICE_GEOMETRY_PORTS, PIE_DEVICE_PORT_ATTN_MASK,
    PIE_DEVICE_PORT_EMBED_TOKENS, PIE_DEVICE_PORT_KV_LEN, PIE_DEVICE_PORT_PAGE_INDPTR,
    PIE_DEVICE_PORT_PAGES, PIE_DEVICE_PORT_POSITIONS, PIE_DEVICE_PORT_RS_BUFFER_INDPTR,
    PIE_DEVICE_PORT_RS_BUFFER_LEN, PIE_DEVICE_PORT_RS_BUFFER_PAGES, PIE_DEVICE_PORT_RS_W_OFF,
    PIE_DEVICE_PORT_RS_W_SLOT, PIE_DEVICE_PORT_W_OFF, PIE_DEVICE_PORT_W_SLOT,
    PIE_DEVICE_RS_BUFFER_PORTS,
};
pub use local::*;
pub use plan::{
    CHANNEL_TICKET_NONE, ChannelRegistrationPlan, EmittedKernel, EncodedMask, KvCopyPlan,
    LaunchPlan, MediaEncodePlan, PoolResizePlan, ProgramRegistration, RS_FLAG_BUFFER_WRITE,
    RS_FLAG_FOLD, RS_FLAG_FOLD_LEN_DEVICE, RS_FLAG_RESET, StateCopyPlan,
};
pub use remote::*;
pub use submission::{FrameSubmission, StepSubmission};
pub use transfer::{KvDtype, KvExport, KvHandle, KvLayout, KvLayoutKind, KvRegion, MemoryDomain};
