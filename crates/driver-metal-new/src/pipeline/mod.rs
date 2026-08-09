//! The channel-plane interpreter: a CPU fallback for a launch program's
//! prologue/epilogue "shell" stages.
//!
//! # The problem this solves
//!
//! A launch program is not only GPU kernels. Around each fire sit small
//! host-visible stages — read a token off a channel, compare, select, argmax a
//! logits row, push the result back onto a channel — that move scalars and tiny
//! vectors between the model and the runtime. Compiling those to Metal would be
//! pure overhead: they run once per fire on a handful of lanes, and they must
//! be **bit-for-bit reproducible** so a replay on any machine lands on the same
//! token. This module executes them on the CPU with that reproducibility as a
//! first-class contract.
//!
//! # Why it reuses the launch ABI rather than re-porting it
//!
//! The C++ interpreter carried its own `launch::` structs that mirror Rust's
//! [`driver_abi::plan::LaunchPackage`]. Re-porting that mirror would fork the
//! source of truth. Instead this module adopts the Rust owned types directly
//! ([`plan::adopt_launch_package`]) and reuses [`tensor_ir`]'s op tags, dtype,
//! intrinsic ids, port registry, and RNG contract. Only genuine *runtime state*
//! — the [`value::Value`] cell, the [`channel::ChannelState`] ring, the
//! [`plan::ExecPlan`], and the interpreter pass — is defined here, because none
//! of it has a Rust home yet.
//!
//! # Shape of a fire
//!
//! [`plan::adopt_launch_package`] turns a package into an [`plan::ExecPlan`];
//! [`channel::make_instance`] binds rings to it; [`step::step`] runs one fire
//! pass-atomically, publishing channel effects only after every resulting ring
//! is validated.

mod cache;
mod channel;
mod emitted;
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
mod resolve;
mod scratch;
mod stage_cache;
mod status;
mod step;
mod value;

pub use cache::{
    Bounded, Failure, MAX_NEGATIVE_ENTRIES, MAX_PROGRAM_ENTRIES, MAX_STAGE_ENTRIES,
    Stats as CacheStats,
};
pub use channel::{
    ChannelState, HostOp, InterpInstance, host_put, host_take, make_host_channel_state,
    make_host_instance, make_instance,
};
pub use emitted::{Duplicate, Emitted, Slot};
pub use extent::{Extents, Role, Unresolvable, ValueDesc, describe};
pub use group::{
    CHANNEL_NEEDS_EMPTY, CHANNEL_NEEDS_FULL, CHANNEL_PUT, CHANNEL_RETRY_INELIGIBLE, CHANNEL_TAKE,
    CHANNEL_VALID, GroupKey, MAX_CHANNELS, TooManyChannels, channel_flags, schedule_bucket,
    used_channel_slots,
};
pub use identity::{Versions, cache_identity, combined_signature};
pub use lane::{
    ABI_VERSION as LANE_ABI_VERSION, ChannelMeta, ChannelSlot as LaneChannelSlot,
    FLAG_RAGGED as LANE_FLAG_RAGGED, GroupLayout, HEADER_BYTES as LANE_HEADER_BYTES,
    Header as LaneHeader, RECORD_BYTES as LANE_RECORD_BYTES, Record as LaneRecord, RowMeta,
    SLOT_BYTES as LANE_SLOT_BYTES, Shape as LaneShape,
};
pub use meta::{Inconsistent, Malformed, OpMeta, Problem, channel_effects, op_metadata};
pub use params::{OpParams, Runtime as OpRuntime};
pub use plan::{
    ConstPortValue, ExecPlan, StagePlan, adopt_launch_package, bounded_mtp_row_base,
    classify_exec_plan, const_port_value, port_consumes,
};
pub use readiness::{Effect, NO_TICKET, Readiness, Reason, Ticket, Words, check, check_words};
pub use registry::{
    Channel, ChannelSpec, Direction, EmittedKernel, Endpoint, Geometry, HostRole, Instance,
    Program, Registry, channel_dtype,
};
pub use resolve::{Geometry as FireGeometry, Resolution, last_page_len, resolve};
pub use scratch::{
    ALIGN as SCRATCH_ALIGN, DUMMY_BYTES, Layout, MAX_BYTES as MAX_SCRATCH_BYTES, TooLarge, layout,
};
pub use stage_cache::{Lookup, Stages};
pub use status::{
    Diagnosis, FAULT_CLASSES, Fault, FaultClass, Outcome as StatusOutcome, STATUS_BYTES, Site,
    State, Status, describe_fault, report as report_status,
};
pub use step::{PassInputs, StepOutcome, step};
pub use value::{Value, concrete_dtype, decode_wire, encode_wire, value_matches, wire_cell_bytes};

/// The element count of a shape, as the product of its extents.
///
/// A private helper shared by every submodule instead of each recomputing it,
/// because a cell's lane count is `numel`, and getting the widening wrong (a
/// `u32` product that overflows silently) would mis-size a ring or a wire cell.
/// The product is taken in `u64` so a large but legal shape cannot wrap; an
/// empty shape is a scalar, so the product of no extents is one.
pub(crate) fn shape_numel(dims: &[u32]) -> u64 {
    dims.iter().map(|&d| u64::from(d)).product()
}

#[cfg(test)]
mod tests {
    use super::shape_numel;

    #[test]
    fn an_empty_shape_is_a_scalar_of_one_lane_not_zero() {
        assert_eq!(
            shape_numel(&[]),
            1,
            "a scalar has no extents; the product of no extents is one, so a \
             scalar cell holds exactly one lane, not zero"
        );
    }

    #[test]
    fn shape_numel_widens_before_multiplying_so_a_large_shape_cannot_wrap() {
        // Two extents whose u32 product overflows but whose u64 product does not.
        assert_eq!(
            shape_numel(&[100_000, 100_000]),
            10_000_000_000,
            "the product must be taken in u64; a u32 multiply would wrap this \
             to a small wrong lane count and mis-size the cell"
        );
    }
}
