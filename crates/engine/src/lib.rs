#![cfg_attr(docsrs, feature(doc_auto_cfg))]
// `deny(missing_docs)` stood here, and the workspace lints table still says
// deny; the allow below overrides it. The program/ plane's prose was stripped
// by the owner's sweep, and a lint that contradicts the text it governs
// forced a RUSTFLAGS override onto every consumer build — the lint follows
// the text (palo ruling). New modules (fire/) document themselves by
// convention, not by threat.
#![allow(missing_docs)]
#![deny(
    clippy::todo,
    clippy::unimplemented,
    clippy::dbg_macro,
    clippy::mem_forget
)]
#![deny(clippy::print_stdout, clippy::print_stderr)]

mod error;
mod program;

// A THIRD OF THIS LIST WAS EXPORTING ITSELF (alto E, survey debt 5). Twenty
// names had zero consumers anywhere outside this crate — `resolve.rs`'s CSR
// ghost and `names.rs`'s MLX weight table whole, and beside them a scattering
// of items the shells never asked for: the six `CHANNEL_*` lane-table flag
// bits, four `#[repr(C)]` mirrors of a device lane table, the op-metadata
// pair, `bounded_mtp_row_base`, `decode_wire`. The exports are gone; where the
// ITEM had no reader inside the crate either, the item went with them and its
// line says what it was for.
//
// THE MODEL PLANE, AND IT IS NAVIGATED BY PATH. Every `pub use` below flattens
// the guest-program plane into the crate root, which is the shape 22 files of
// one subsystem grew into and not a shape worth extending: `fire` is new API
// with a small, deliberate surface — `fire::compose`, `fire::walk`,
// `fire::FireDescriptor`, `fire::Sink` — and a reader who sees `fire::walk` at
// a call site knows where to go and what it is about.
pub mod fire;
// THE EXECUTION PLANE, AND IT IS NAVIGATED BY PATH FOR THE SAME REASON `fire`
// IS. `frame` is the typed prepare/enqueue/settle seam (alto design §3) and
// `runahead` is the one place a depth is spelled (article 8); both are small,
// deliberate surfaces that a reader should reach through their module name.
pub mod frame;
pub mod law;
pub mod runahead;
pub mod store;

pub use error::{Error, Result};

pub use engine_api;

pub use tensor_ir;

pub use program::cache::{
    Bounded, Failure, MAX_NEGATIVE_ENTRIES, MAX_PROGRAM_ENTRIES, MAX_STAGE_ENTRIES,
    Stats as CacheStats,
};
pub use program::channel::{
    ChannelState, HostOp, InterpInstance, host_put, host_take, make_host_instance,
};
pub use program::emitted::{Duplicate, Emitted, Slot};
pub use program::extent::{Extents, Role, Unresolvable, ValueDesc, describe};
pub use program::group::{GroupKey, MAX_CHANNELS, used_channel_slots};
pub use program::identity::{
    Backend, COMPILER_VERSION, REGION_PLAN_VERSION, Versions, cache_identity, combined_signature,
};
pub use program::lane::{
    ABI_VERSION as LANE_ABI_VERSION, ChannelSlot as LaneChannelSlot,
    HEADER_BYTES as LANE_HEADER_BYTES, Header as LaneHeader, RECORD_BYTES as LANE_RECORD_BYTES,
    Record as LaneRecord, SLOT_BYTES as LANE_SLOT_BYTES, Shape as LaneShape,
};
pub use program::meta::{Malformed, channel_effects};
pub use program::params::{OpParams, Runtime as OpRuntime};
pub use program::plan::{Boundaries, ExecPlan, adopt_launch_package, adopt_launch_package_with};
pub use program::readiness::{NO_TICKET, Readiness, Ticket, Words, check};
pub use program::registry::{
    Channel, ChannelSpec, Direction, EmittedKernel, Endpoint, Geometry, HostRole, Instance,
    Program, Registry,
};
pub use program::scratch::{ALIGN as SCRATCH_ALIGN, Layout, layout};
pub use program::stage_cache::{Lookup, Stages};
pub use program::status::{
    Fault, FaultClass, Outcome as StatusOutcome, STATUS_BYTES, Site, State, Status, describe_fault,
    report as report_status,
};
pub use program::step::{PassInputs, StepOutcome, step};
pub use program::value::{Value, concrete_dtype, encode_wire, value_matches, wire_cell_bytes};

pub(crate) fn shape_numel(dims: &[u32]) -> u64 {
    dims.iter().map(|&d| u64::from(d)).product()
}
