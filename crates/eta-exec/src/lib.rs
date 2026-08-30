//! The guest-program plane: the ETA host half.
//!
//! Everything a device shell needs to adopt a launch package, run the channel
//! ring, and derive at bind time — the launch-package adoption, the channel
//! ring, the reference pass, and the op interpreter — with no device API call
//! anywhere in this crate. The model forward substrate (`fire`, `store`,
//! `law`) sits beside it in `model-exec`, not under it: the two shared one
//! roof until this pass and the cross-references between them were zero.
//!
//! **THIS PLANE IS FLATTENED, AND THE FLATTENING IS INHERITED.** Every `pub
//! use` below lifts a module's items into the crate root. That is the shape 19
//! files of one subsystem grew into under a crate that also held `fire`, where
//! the flat names were how the guest plane distinguished itself from a plane
//! navigated by path. As a crate of its own the qualification is back in the
//! crate name — `eta_exec::step`, `eta_exec::Registry` — so the flat surface
//! now reads as what it always meant: one subsystem, one namespace.
//!
//! **It does not name the runtime↔engine contract.** Not "does not happen to";
//! does not, by rule — see the manifest. The launch package it adopts is
//! `eta_compiler::codegen::launch`'s, from the compiler that produced it.
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
// `deny(missing_docs)` stood in the crate this was carved out of, and the
// workspace lints table still says deny; the allow below overrides it. This
// plane's prose was stripped by the owner's sweep, and a lint that contradicts
// the text it governs forced a RUSTFLAGS override onto every consumer build —
// the lint follows the text (palo ruling).
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

// A THIRD OF THIS LIST WAS EXPORTING ITSELF (alto E, survey debt 5). Twenty
// names had zero consumers anywhere outside this crate — `resolve.rs`'s CSR
// ghost and `names.rs`'s MLX weight table whole, and beside them a scattering
// of items the shells never asked for: the six `CHANNEL_*` lane-table flag
// bits, four `#[repr(C)]` mirrors of a device lane table, the op-metadata
// pair, `bounded_mtp_row_base`, `decode_wire`. The exports are gone; where the
// ITEM had no reader inside the crate either, the item went with them and its
// line says what it was for.
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

// NOT SHARED, AND ONLY LOOKED IT. This sat in the old crate root beside the
// model plane's exports, which is the only reason it read as common ground:
// all fourteen of its call sites were in this plane, and the model plane never
// named it. A private helper is what it always was.
pub(crate) fn shape_numel(dims: &[u32]) -> u64 {
    dims.iter().map(|&d| u64::from(d)).product()
}
