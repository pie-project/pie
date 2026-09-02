//! The guest-program plane: the ETA host half. Adopts a launch package, runs
//! the channel ring, and interprets ops, with no device API call anywhere in
//! this crate. Does not name the runtime<->engine contract; the launch
//! package it adopts is `eta_compiler::codegen::launch`'s.
//!
//! Every `pub use` below lifts a module's items into the crate root.
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
// overrides the workspace's deny(missing_docs): this plane keeps no per-item docs.
#![allow(missing_docs)]
#![deny(
    clippy::todo,
    clippy::unimplemented,
    clippy::dbg_macro,
    clippy::mem_forget
)]
#![deny(clippy::print_stdout, clippy::print_stderr)]

mod cache;
mod channel;
mod emitted;
mod error;
mod extent;
mod group;
mod identity;
mod lane;
mod meta;
mod op;
mod params;
mod plan;
mod readiness;
mod registry;
mod scratch;
mod stage_cache;
mod status;
mod step;
mod value;

pub use error::{Error, Result};

pub use cache::{
    Bounded, Failure, MAX_NEGATIVE_ENTRIES, MAX_PROGRAM_ENTRIES, MAX_STAGE_ENTRIES,
    Stats as CacheStats,
};
pub use channel::{ChannelState, HostOp, InterpInstance, host_put, host_take, make_host_instance};
pub use emitted::{Duplicate, Emitted, Slot};
pub use extent::{Extents, Role, Unresolvable, ValueDesc, describe};
pub use group::{GroupKey, MAX_CHANNELS, used_channel_slots};
pub use identity::{
    Backend, COMPILER_VERSION, REGION_PLAN_VERSION, Versions, cache_identity, combined_signature,
};
pub use lane::{
    ABI_VERSION as LANE_ABI_VERSION, ChannelSlot as LaneChannelSlot,
    HEADER_BYTES as LANE_HEADER_BYTES, Header as LaneHeader, RECORD_BYTES as LANE_RECORD_BYTES,
    Record as LaneRecord, SLOT_BYTES as LANE_SLOT_BYTES, Shape as LaneShape,
};
pub use meta::{Malformed, channel_effects};
pub use params::{OpParams, Runtime as OpRuntime};
pub use plan::{Boundaries, ExecPlan, adopt_launch_package, adopt_launch_package_with};
pub use readiness::{NO_TICKET, Readiness, Ticket, Words, check};
pub use registry::{
    Channel, ChannelSpec, Direction, EmittedKernel, Endpoint, Geometry, HostRole, Instance,
    Program, Registry,
};
pub use scratch::{ALIGN as SCRATCH_ALIGN, Layout, MAX_BYTES as SCRATCH_MAX_BYTES, layout};
pub use stage_cache::{Lookup, Stages};
pub use status::{
    Fault, FaultClass, Outcome as StatusOutcome, STATUS_BYTES, Site, State, Status, describe_fault,
    report as report_status,
};
pub use step::{PassInputs, StepOutcome, step};
pub use value::{Value, concrete_dtype, encode_wire, value_matches, wire_cell_bytes};

pub(crate) fn shape_numel(dims: &[u32]) -> u64 {
    dims.iter().map(|&d| u64::from(d)).product()
}
