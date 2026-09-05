pub mod adapter;
pub mod api;
pub mod arena;

pub mod blob;
pub mod boot;

pub mod comm;
pub mod decoded;
pub mod device;
mod dispatch;
pub mod encode;
mod error;
pub mod experts;
pub mod guest;
pub mod inputs;
pub mod mask;
pub mod probe;
pub mod program;

pub(crate) mod record;
pub mod rs;
pub mod run;

pub mod scores;
mod scratch;
pub mod serve;
pub mod settle;
pub mod store;
pub mod weight_store;
pub mod weights;
pub mod window;

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
pub use scores::ScoreSeat;
pub use scratch::Scratch;
pub use serve::{
    Attached, Boot, Enqueued, FireCost, Landed, Lane, Prepared, Seated, Shell, StepView,
};
pub use settle::{Airborne, Arms, Done};
pub use store::Pools;
pub use weights::{AdapterPlane, BankSeat, Weights};
pub use window::{Copies, Cursor, Gathered, GatheredSpace, Window, Windows};
