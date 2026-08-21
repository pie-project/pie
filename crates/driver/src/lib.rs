//! What every driver shares and none of them owns.
//!
//! The PTIR channel plane: a launch program's runtime state, and the CPU
//! fallback for its prologue/epilogue "shell" stages.
//!
//! # Why this crate has the bare name
//!
//! It was `driver-pipeline`, beside a `driver-abi` that held the runtime <->
//! driver vocabulary. The ABI is gone, and the two crates turned out to have
//! OPPOSITE shapes: this one is depended on by exactly the two drivers, while
//! the vocabulary is depended on by nine crates, five of which are not drivers
//! at all (`engine`, `transport`, `controller-api`, `worker`,
//! `tensor-compiler`).
//!
//! So this is the driver-common substrate and the vocabulary is not: it is
//! what the runtime and a driver SAY TO EACH OTHER, which is why it is
//! `driver-api` and this is `driver`. A crate both sides speak is a contract;
//! a crate only drivers use is a substrate, and only the second one deserves
//! to be named after them.
//!
//! # Why this is a crate and not a module
//!
//! It was a module of `driver-metal` until the CUDA shell reached the same
//! file. Every line of it is arithmetic over the launch ABI's records: not one
//! names a Metal type, and the compiler agrees — the whole directory built and
//! tested on Linux while the crate around it did not. A layer that two device
//! shells need and neither one owns is a crate, and the alternative was the
//! thing the C++ actually did: THREE hand-written copies of one golden model
//! (`tensor-compiler`'s interpreter, the CUDA driver's `tier0_runner.hpp`, the
//! Metal driver's `interp.hpp`), which is why `.wiki/driver/progress-metal.md` opens by
//! counting them.
//!
//! The seam is directional rather than polymorphic, and that is why there is
//! no `trait Backend` here. This crate never calls a device; a device shell
//! calls this crate. `driver-cuda` packs the lane table this crate lays
//! out and hands it to `cuLaunchKernel`; `driver-metal` packs the same
//! bytes into an argument table. The bytes are the interface.
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
//! [`driver_api::plan::LaunchPackage`]. Re-porting that mirror would fork the
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

#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![deny(missing_docs)]
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
mod resolve;
mod scratch;
mod stage_cache;
mod status;
mod step;
mod value;

// The one PUBLIC module here, where every other is private behind a flat
// re-export. It stays a module because it is the only thing in this crate a
// caller navigates by path rather than by symbol: two driver test suites
// already say `names::Naming::mlx()`, and both keep saying it after the move
// because each shell re-exports this module under its own name.
//
// Strings and a static table, so it is ungated on every shell that takes it:
// a caller that only wants to know what a checkpoint calls `layer.3.down`
// should not have to link a graphics API to ask.
pub mod names;

pub use error::{Error, Result};

/// The launch ABI, re-exported.
///
/// Not a convenience. This crate's public API is *written in* these types —
/// [`ExecPlan::package`] is a [`driver_api::plan::LaunchPackage`], every stage
/// plan is a [`driver_api::plan::LaunchStagePlan`], and the emitted-kernel
/// kinds are `driver_api::local`'s constants — so a consumer cannot call
/// `adopt_launch_package` or read what it returns without naming them. Making
/// them reachable from here is what stops two shells from each declaring their
/// own `driver-api` dependency and, one day, resolving it to two versions:
/// the types would then be nominally distinct and the mismatch would surface
/// as an inscrutable trait error rather than as a version conflict.
pub use driver_api;

/// The IR vocabulary, re-exported.
///
/// Re-exported for the same reason as [`driver_api`], one step weaker: this
/// crate's *signatures* mostly do not name `tensor_ir` types, but its
/// *contracts* are written in them — [`Value`] is a `DType`'s lanes, an
/// [`OpParams`] carries an `IntrinsicId`, and the RNG a shell must reproduce
/// on-device is `tensor_ir::rng`'s. A shell that has to restate the dependency
/// to read them is a shell that can resolve a second copy of the op table, and
/// two op tables that disagree by one tag is a wrong token rather than a build
/// error.
pub use tensor_ir;

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
pub use identity::{
    Backend, COMPILER_VERSION, REGION_PLAN_VERSION, Versions, cache_identity, combined_signature,
};
pub use lane::{
    ABI_VERSION as LANE_ABI_VERSION, ChannelMeta, ChannelSlot as LaneChannelSlot,
    FLAG_RAGGED as LANE_FLAG_RAGGED, GroupLayout, HEADER_BYTES as LANE_HEADER_BYTES,
    Header as LaneHeader, RECORD_BYTES as LANE_RECORD_BYTES, Record as LaneRecord, RowMeta,
    SLOT_BYTES as LANE_SLOT_BYTES, Shape as LaneShape,
};
pub use meta::{Inconsistent, Malformed, OpMeta, Problem, channel_effects, op_metadata};
pub use params::{OpParams, Runtime as OpRuntime};
pub use plan::{
    Boundaries, ConstPortValue, ExecPlan, StagePlan, adopt_launch_package,
    adopt_launch_package_with, bounded_mtp_row_base, classify_exec_plan, const_port_value,
    port_consumes,
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
