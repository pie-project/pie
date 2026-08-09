//! One module per kernel family, each owning the units it compiles.
//!
//! # Why the units are split by family rather than listed in one place
//!
//! A unit is one NVRTC compile: one `.cuh` of `__global__` templates, one
//! header set, many name expressions. There are as many of them as there are
//! device headers in `kernels-cuda/csrc/src`, and a single list would be a file
//! every migration touches — the shape that makes parallel work collide and a
//! diff unreadable.
//!
//! So a family owns its own units, its own rows, and the `include_str!` that
//! carries its source. Adding a family is one module and one line in [`ALL`];
//! adding a unit to a family touches one file.
//!
//! # What a family module holds
//!
//! * `UNITS` — one entry per `.cuh` the family compiles.
//! * The [`crate::device::DeviceKernel`] rows those units instantiate, and the
//!   `KernelSig`s behind them.
//!
//! The sigs are written here rather than reused from [`crate::table`] because
//! they are not the same contract. A table row describes a `pie_k_*` entry
//! point: a host function holding a `<<<>>>`, taking a stream. A row here
//! describes a template instantiation and states its geometry as a
//! `LaunchRule` — the thing the launcher used to hold. `norm_device.rs`
//! records the measurement: the same six kernels went from thirty-one operands
//! to twenty-one, and the ten that vanished were six streams and four extents
//! the rules recover.
//!
//! # The order is stable, not semantic
//!
//! [`crate::unit::UNITS`] concatenates these in [`ALL`]'s order, and a unit's
//! position there is its slot in the module cache. Nothing depends on which
//! slot a unit gets, and a reordering invalidates nothing: the cache is
//! per-process and the cubin cache keys on the unit's NAME.

use crate::unit::Unit;

pub mod adapter;
pub mod attn;
/// The FlashInfer FA2 lattice — 56 units over four axes, the last thing in
/// the tree that needed nvcc.
///
/// A family of its own and not a corner of [`attn`], for the reason [`graph`]
/// is one: it is a whole instantiation lattice with its own axes, its own
/// derivation ([`crate::fa2`]) and its own vendored header closure, and
/// [`attn`] is already 4,000 lines of unrelated launchers. It is also the only
/// family whose units are macro-generated, which reads badly interleaved with
/// hand-written rows.
pub mod fa2;
pub mod gemm;
/// The supergraph's two arming kernels — the one family named after a SHELL
/// object rather than a kind of value. See its header for why that is a
/// directory of its own and not a corner of `layout`.
pub mod graph;
pub mod layout;
pub mod marlin;
pub mod mlp;
pub mod moe;
pub mod norm;
pub mod quant;
pub mod sample;
pub mod ssm;
pub mod vision;

/// Every family's units, in a stable order.
///
/// Listed by name rather than discovered, because a family that appeared
/// silently would be compilable and unreachable — `unit_of` scans this, so a
/// unit not in it hosts no symbol and every fire of its rows is refused as
/// unknown.
///
/// # `rope` is not a module here, and that is the migration
///
/// [`crate::x::rope`] declares its own unit with `unit!`, which generates the
/// same [`Unit`] and the same `DeviceKernel` rows this module's families
/// write by hand — so `unit_of`, `cache::module`, `Args::bind` and
/// `tests/units.rs` reach it unchanged. What is gone is the hand-written
/// `KernelSig` beside each row: the operand list comes from the declared
/// parameter types through `x::Abi`, and the geometry is a `fn`.
pub static ALL: &[&[Unit]] = &[
    adapter::UNITS,
    attn::UNITS,
    fa2::UNITS,
    gemm::UNITS,
    graph::UNITS,
    layout::UNITS,
    marlin::UNITS,
    mlp::UNITS,
    moe::UNITS,
    norm::UNITS,
    quant::UNITS,
    crate::x::rope::UNITS,
    sample::UNITS,
    ssm::UNITS,
    vision::UNITS,
];
