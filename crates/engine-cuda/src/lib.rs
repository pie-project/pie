//! Dispatch layer for the CUDA engine: resolves plan ops to device launches
//! (`Run`, `impl Dispatch*`), plus the serving shell (device, weights, store,
//! arena, inputs, mask, window, record, serve).

/// Resolves the `lora` sink from a launch package: host-side conversion of
/// guest-seeded f32 cells into bank bytes.
pub mod adapter;
pub mod api;
pub mod arena;
/// Shared-adapter store: read-only mount, refcounted host cache, device bank
/// slots keyed by blob identity (N adapter instances share one device copy).
pub mod blob;
/// Opens a device from a [`DeviceBoot`] config.
pub mod boot;
/// Routed-expert residency: pinned host copy of every expert, a device slab
/// for some of them, and the indirection table a captured graph reads through.
pub mod checkpoint_serving;
pub mod device;
mod dispatch;
mod error;
pub mod experts;
pub mod exports;
pub mod inputs;
pub mod mask;
pub mod program;
pub mod record;
pub mod rotate;
pub mod run;
pub mod scores;
pub mod serve;
/// Tracks in-flight requests and where their settlement callbacks are queued.
pub mod settle;
/// Pinned double-buffered host-to-device pump for bulk weight transfer:
/// multiple lanes overlap host memcpy with DMA in flight.
pub mod staged_h2d;
pub mod store;
pub mod weights;
pub mod window;

/// Op names that must not run on two streams at once. Empty: scratch is now
/// keyed per `(arena, name, stream)`, so nothing needs this.
pub const EXCLUSIVE: [&str; 0] = [];

/// Ops that can run over a segment list (union of row intervals plus the
/// intervals themselves) in one launch, touching no row in the gaps.
pub const GROUPED: [&str; 1] = ["linear.lora_correct"];

/// Whether `op` is addressed via the window seat: reads/writes only rows
/// `[start, start+count)` off the plane's base. Declared per entry beside
/// the wrapper that passes the seat ([`kernels_cuda::ENTRIES`]); an
/// undeclared name costs a body, a wrong declaration is silent corruption.
#[must_use]
pub fn shifted(op: &str) -> bool {
    matches!(
        kernels_cuda::seat::reads(op),
        Reads::Rows | Reads::RowsAndLanes
    )
}

/// Prepare-phase planner ops: no launch node in the captured graph, rebuilt
/// fresh per fire, so they're safe in a windowed region alongside [`shifted`] ops.
pub const PLANNED: [&str; 2] = ["attention.plan_decode", "attention.plan_prefill"];

/// Whether `op` reads every per-lane table absolutely (fire-wide) and finds
/// its own lane off the seat, rather than `lane_offset`-shifted like the
/// rest of [`shifted`]: the FA2 arms and the chunked recurrent arms.
#[must_use]
pub fn lane_shifted(op: &str) -> bool {
    kernels_cuda::seat::reads(op) == Reads::RowsAndLanes
}


pub use api::{ClassifyFor, ContractFor, Cuda, DeviceBoot};
pub use kernels_cuda::{EntryInfo, Reads};
pub use boot::{open, ordinal_of};
pub use error::{Fault, Result};
pub use mask::{LaneMask, Staged as StagedMask};
pub use program::{Fired, Plane as ProgramPlane, Session as ProgramSession};
pub use record::{
    AxisKey, BodyCensus, BodyKey, BodyStats, BodyTally, Bodies as GraphCache, LastCapture,
};
pub use run::{
    CacheGeometry, CachePlanning, CachePool, CacheTable, FireBindings, FireTables, Planning,
    PoolSlabs, Run, SlotTable, StructSlot, WeightRow, WeightTable,
};
pub use serve::{
    Armed, Boot, Recording, Seal, DEFAULT_BODIES_MEGABYTES, DEFAULT_GPU_MEM_UTILIZATION, FireCost, Golden, Graphs, Knobs,
    Lane, Media, Seated, Shell,
};

pub use blob::{
    Adapters, Binding, Site as AdapterSite, Source as AdapterSource, layer_of, role_of, site_of,
};
/// Per-layer attention scores a captured fire returns; re-exported so callers
/// of [`Shell::fire_captured`] don't need to reach into `engine` directly.
pub use engine::fire::LayerScores;
pub use weights::{AdapterPlane, BankSeat};
pub use window::{Cursor, Window, WindowShape, Windows};
