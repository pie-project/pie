//! pie-driver-abi — the final local runtime ↔ driver contract.
//!
//! This crate exposes the local ABI plus process-independent driver schemas:
//!
//! - [`local`]: plain `#[repr(C)]` direct-FFI descriptors and symbol declarations.
//! - [`capabilities`]: reduced cold-path JSON facts used at create time.
//! - [`transfer`]: Rust-only KV transfer vocabulary shared with cross-node transport.
//! - [`plan`]: owned verb plans shared by local and remote backends.
//! - [`remote`]: versioned worker-to-executor protocol.
//!
//! The committed `include/pie_driver_abi.h` header is generated from [`local`]
//! via `pie-driver-abi-cbindgen`.

pub mod capabilities;
pub mod geometry;
pub mod image;
pub mod local;
pub mod plan;
pub mod remote;
pub mod transfer;

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
    LaunchPlan, MediaEncodePlan, PoolResizePlan, ProgramRegistration, RS_FLAG_BUFFER_WRITE, RS_FLAG_FOLD, RS_FLAG_FOLD_LEN_DEVICE, RS_FLAG_RESET,
    StateCopyPlan,
};
pub use remote::*;
pub use transfer::{KvDtype, KvExport, KvHandle, KvLayout, KvLayoutKind, KvRegion, MemoryDomain};
