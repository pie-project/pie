pub use engine_gpu::adapter;
pub mod api;
pub use engine_gpu::arena;

pub use engine_gpu::blob;
pub mod boot;
pub mod chip;
pub mod chunks;

pub mod comm;
pub use engine_gpu::decoded;
pub mod device;
mod dispatch;
pub mod encode;
use engine_gpu::error;
pub mod experts;
pub mod feeds;
pub mod guest;
pub mod inputs;
pub use engine_gpu::mask;
pub mod probe;
pub mod program;

pub(crate) mod record;
pub use engine_gpu::rs;
pub mod run;

mod scratch;
pub mod serve;
pub use engine_gpu::settle;
pub mod store;
pub mod tune;
pub mod tuning;
pub use engine_gpu::weight_store;
pub mod weights;
pub use engine_gpu::window;

pub use adapter::{
    Binding as AdapterBinding, Key as AdapterKey, Role as AdapterRole, Site,
    Source as AdapterSource,
};
pub use api::{ContractFor, DeviceBoot, Vulkan};
pub use arena::Arena;
pub use blob::{Layout as AdapterLayout, Manifest as AdapterManifest, Stamp as AdapterStamp};
pub use boot::open;
pub use device::{Buffer, Context, Handles, Pipelines};
pub use encode::{
    DispatchShape, Sink, barriers, between_fires_ns, capture_shapes, fence_ns, frame_span_ns,
    host_encode_ns, kernel_profile, profile_kernels, profiling, reset_kernel_profile, submit_ns,
    take_shapes,
};
pub use error::{Fault, Result};
pub use inputs::Inputs;
pub use run::{
    CacheGeometry, CachePool, CacheTable, FireBindings, FireTables, PoolSlabs, Run, SlotTable,
    StructSlot, WeightRow, WeightTable,
};
pub use scratch::Scratch;
pub use serve::{Attached, Boot, Enqueued, Landed, Lane, Prepared, Seated, Shell, StepView};
pub use settle::{Airborne, Arms, Done};
pub use store::Pools;
pub use weights::{AdapterPlane, BankSeat, Weights};
pub use window::{Copies, Gathered, GatheredTable, Window, Windows};

/// The fusions this backend implements: none, so the compiler forms no
/// fused op here.
pub static FUSIONS: [poem_compiler::fuse::Rule; 0] = [];
