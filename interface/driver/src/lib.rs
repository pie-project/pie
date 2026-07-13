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
pub mod local;
pub mod plan;
pub mod remote;
pub mod transfer;

pub use capabilities::{
    DeviceFacts, DriverCapabilities, KV_COPY_DEVICE_TO_DEVICE, KV_COPY_DEVICE_TO_HOST,
    KV_COPY_HOST_TO_DEVICE, KV_COPY_HOST_TO_HOST, ModelLoadDesc,
};
pub use local::*;
pub use plan::{
    CHANNEL_TICKET_NONE, ChannelRegistrationPlan, EncodedMask, KvCopyPlan, LaunchPlan,
    MediaEncodePlan, PoolResizePlan, ProgramRegistration, RS_FLAG_FOLD, RS_FLAG_RESET,
    StateCopyPlan,
};
pub use remote::*;
pub use transfer::{KvDtype, KvExport, KvHandle, KvLayout, KvLayoutKind, KvRegion, MemoryDomain};
