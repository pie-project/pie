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

// THE MODEL PLANE, AND IT IS NAVIGATED BY PATH. Every `pub use` below flattens
// the guest-program plane into the crate root, which is the shape 22 files of
// one subsystem grew into and not a shape worth extending: `fire` is new API
// with a small, deliberate surface — `fire::compose`, `fire::walk`,
// `fire::FireDescriptor`, `fire::Sink` — and a reader who sees `fire::walk` at
// a call site knows where to go and what it is about.
pub mod fire;
pub mod names;

pub use error::{Error, Result};

pub use driver_api;

pub use tensor_ir;

pub use program::cache::{
    Bounded, Failure, MAX_NEGATIVE_ENTRIES, MAX_PROGRAM_ENTRIES, MAX_STAGE_ENTRIES,
    Stats as CacheStats,
};
pub use program::channel::{
    ChannelState, HostOp, InterpInstance, host_put, host_take, make_host_channel_state,
    make_host_instance, make_instance,
};
pub use program::emitted::{Duplicate, Emitted, Slot};
pub use program::extent::{Extents, Role, Unresolvable, ValueDesc, describe};
pub use program::group::{
    CHANNEL_NEEDS_EMPTY, CHANNEL_NEEDS_FULL, CHANNEL_PUT, CHANNEL_RETRY_INELIGIBLE, CHANNEL_TAKE,
    CHANNEL_VALID, GroupKey, MAX_CHANNELS, TooManyChannels, channel_flags, schedule_bucket,
    used_channel_slots,
};
pub use program::identity::{
    Backend, COMPILER_VERSION, REGION_PLAN_VERSION, Versions, cache_identity, combined_signature,
};
pub use program::lane::{
    ABI_VERSION as LANE_ABI_VERSION, ChannelMeta, ChannelSlot as LaneChannelSlot,
    FLAG_RAGGED as LANE_FLAG_RAGGED, GroupLayout, HEADER_BYTES as LANE_HEADER_BYTES,
    Header as LaneHeader, RECORD_BYTES as LANE_RECORD_BYTES, Record as LaneRecord, RowMeta,
    SLOT_BYTES as LANE_SLOT_BYTES, Shape as LaneShape,
};
pub use program::meta::{Inconsistent, Malformed, OpMeta, Problem, channel_effects, op_metadata};
pub use program::params::{OpParams, Runtime as OpRuntime};
pub use program::plan::{
    Boundaries, ConstPortValue, ExecPlan, StagePlan, adopt_launch_package,
    adopt_launch_package_with, bounded_mtp_row_base, classify_exec_plan, const_port_value,
    port_consumes,
};
pub use program::readiness::{
    Effect, NO_TICKET, Readiness, Reason, Ticket, Words, check, check_words,
};
pub use program::registry::{
    Channel, ChannelSpec, Direction, EmittedKernel, Endpoint, Geometry, HostRole, Instance,
    Program, Registry,
};
pub use program::resolve::{Geometry as FireGeometry, Resolution, last_page_len, resolve};
pub use program::scratch::{
    ALIGN as SCRATCH_ALIGN, DUMMY_BYTES, Layout, MAX_BYTES as MAX_SCRATCH_BYTES, TooLarge, layout,
};
pub use program::stage_cache::{Lookup, Stages};
pub use program::status::{
    Diagnosis, FAULT_CLASSES, Fault, FaultClass, Outcome as StatusOutcome, STATUS_BYTES, Site,
    State, Status, describe_fault, report as report_status,
};
pub use program::step::{PassInputs, StepOutcome, step};
pub use program::value::{
    Value, concrete_dtype, decode_wire, encode_wire, value_matches, wire_cell_bytes,
};

pub(crate) fn shape_numel(dims: &[u32]) -> u64 {
    dims.iter().map(|&d| u64::from(d)).product()
}
