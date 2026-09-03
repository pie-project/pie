//! The Metal engine's dispatch layer: the [`Run`] that resolves plan ids to
//! device handles. Encodes through `kernels_metal::Ctx`, so this crate
//! builds and tests on any OS; only the shell behind the sink is macOS-bound.

pub mod abi;
/// The `lora` sink: which channels an adapter's planes are, what a seeded
/// cell means as bank bytes, and which slot it lands in.
pub mod adapter;
pub mod api;
pub mod arena;
/// The shared-adapter store: the deployment mount, `adapter.toml` grammar,
/// host byte cache, and per-layer resolver from blob files to bank planes.
pub mod blob;
pub mod boot;
pub mod device;
mod dispatch;
pub mod encode;
pub mod weight_store;
mod error;
pub mod experts;
/// The gathered row slab: the static-demand residency class.
pub mod gather;
pub mod host_source;
/// The indirect command buffer. Metal-only; everything else in this crate
/// compiles on any target.
#[cfg(target_vendor = "apple")]
pub mod icb;
pub mod rebind;
pub mod inputs;
pub mod mapping;
pub mod mask;
pub mod program;
pub mod record;
pub mod rs;
pub mod run;
/// The observability slab the attention capture arm writes and the
/// epilogue's `attn_score` intrinsic reads.
pub mod scores;
pub mod decoded;
mod keepalive;
mod scratch;
pub mod serve;
/// The settlement plane: run-ahead counters, completion seam, and the A/B
/// seat ring the asynchronous fire path is built on.
pub mod settle;
pub mod store;
pub mod weights;
pub mod window;

pub use abi::{Armed, At as AbiAt, Axis, DescriptorAbi, Law, SlotAbi, Survey};
pub use api::{ContractFor, DeviceBoot, Metal};
pub use boot::open;
pub use arena::Arena;
pub use device::{Buffer, Context, Handles, Pipelines};
pub use encode::{Sink, kernel_profile, reset_kernel_profile};
pub use error::{Fault, Result};
pub use experts::{GroupResidency, PREDICTION_PREFIXES, Plan as ResidencyPlan, Prediction};
#[cfg(target_vendor = "apple")]
pub use icb::{Icb, Rebound};
pub use inputs::Inputs;
pub use program::{Fired, Launched, Plane as ProgramPlane, Session as ProgramSession};
pub use record::{Arg, Point, Recording, Slot, Tape};
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
pub use adapter::{
    Binding as AdapterBinding, Key as AdapterKey, Role as AdapterRole, Site, Source as AdapterSource,
};
pub use blob::{Layout as AdapterLayout, Manifest as AdapterManifest, Stamp as AdapterStamp};
pub use weights::{AdapterPlane, BankSeat, Weights};
pub use window::{Copies, Cursor, Gathered, GatheredSpace, Window, Windows};
