//! The NCCL collectives — declared, and refused, because no NCCL bindings
//! are generated and nothing in this tree calls a communicator.
//!
//! Declared because a lowered TP text may NAME them, and `check_plan` refuses
//! a model whose launched symbol is undeclared — which would name the wrong
//! problem. Refused because `cudarc` is built without its `nccl` feature, so
//! no `ncclAllReduce` binding exists to call. These are the ABOVE-crossover
//! arm a sharded text picks for large messages; [`crate::comm`]'s P2P kernel
//! is the below-crossover one, and its absence is independent of this one.
//!
//! Every row is `whole` because every rank must enter the same collective the
//! same number of times: a row window that split one rank's launch and not
//! another's would DEADLOCK rather than compute a wrong answer.
//!
//! # Why these are `routine!` and the `comm::` pair is not
//!
//! A collective takes a communicator, and that was read as putting all five
//! TP symbols out of the column's reach. It does not: NCCL resolves its own
//! communicator inside the call, so what these three take is a buffer, a
//! second buffer and a count -- and the count is `rows x width`, which F6
//! gives to the launcher. `driver-cuda` held three arms doing that
//! multiplication until this row said `routine!`.
//!
//! `comm::`'s custom reduction is the one that really wants a handle: it maps
//! peer memory and takes a PLANE, which no statement carries and no `Source`
//! names.

use crate::jit::Ctx;
use crate::jit::abi::bf16;
use kernels::Refusal;
use kernels::routine::{In, Out};

/// What every body here says, in one place so three refusals cannot drift.
fn no_nccl(what: &'static str) -> Refusal {
    let _ = what;
    Refusal::Absent {
        what: "NCCL: `cudarc` is built without its `nccl` feature, so no \
               communicator binding is generated and nothing in this \
               workspace calls one. This is the ABOVE-CROSSOVER arm; below \
               the crossover `comm::all_reduce_bf16` is the one that runs",
    }
}

/// `dist::all_reduce_bf16` — the in-place sum across the group.
///
/// `in_place = &[(0, 0)]`: the buffer is read and written, which is the whole
/// difference from [`all_reduce_bf16_out`].
///
/// # Errors
/// Always. See the module header.
#[kernels_macros::routine]
pub fn all_reduce_bf16(ctx: &Ctx, buf: Out<0, bf16>) -> Result<(), Refusal> {
    let r = buf.all("out_width(0)")?;
    all_reduce_in_place(ctx, r.ptr.cast(), i64::from(r.elements()))
}

/// [`all_reduce_bf16`] over a span the caller already holds.
///
/// The routine above is the statement's door; this is the one a launcher
/// reaches, and there is exactly one body under both.
///
/// # Errors
/// Always. See the module header.
pub fn all_reduce_in_place(
    _ctx: &Ctx,
    _buf: *mut core::ffi::c_void,
    _elems: i64,
) -> Result<(), Refusal> {
    Err(no_nccl("all_reduce"))
}

/// `dist::all_reduce_bf16_out` — the same collective with a separate
/// destination and no alias pair.
///
/// # Errors
/// Always. See the module header.
#[kernels_macros::routine]
pub fn all_reduce_bf16_out(
    ctx: &Ctx,
    src: In<0, bf16>,
    dst: Out<0, bf16>,
) -> Result<(), Refusal> {
    let d = dst.all("out_width(0)")?;
    all_reduce_out_of_place(ctx, src.ptr.cast(), d.ptr.cast(), i64::from(d.elements()))
}

/// [`all_reduce_bf16_out`] over spans the caller already holds.
///
/// `comm::fall_back_out_of_place` is the caller: the P2P reduction declines
/// and reaches NCCL with raw pointers and a count, having never had a
/// statement in hand.
///
/// # Errors
/// Always. See the module header.
pub fn all_reduce_out_of_place(
    _ctx: &Ctx,
    _src: *const core::ffi::c_void,
    _dst: *mut core::ffi::c_void,
    _elems: i64,
) -> Result<(), Refusal> {
    Err(no_nccl("all_reduce_out"))
}

/// `dist::all_gather_bf16` — each rank's shard concatenated on every rank.
///
/// # Errors
/// Always. See the module header.
#[kernels_macros::routine]
pub fn all_gather_bf16(ctx: &Ctx, src: In<0, bf16>, dst: Out<0, bf16>) -> Result<(), Refusal> {
    // The INPUT's width, not the output's: the count is per rank, and the
    // destination is `world_size` times as wide. Reading the output here would
    // have every rank write past its own band.
    let s = src.all("in_width(0)")?;
    all_gather(ctx, s.ptr.cast(), dst.ptr.cast(), i64::from(s.elements()))
}

/// [`all_gather_bf16`] over spans the caller already holds.
///
/// # Errors
/// Always. See the module header.
pub fn all_gather(
    _ctx: &Ctx,
    _src: *const core::ffi::c_void,
    _dst: *mut core::ffi::c_void,
    _elems_per_rank: i64,
) -> Result<(), Refusal> {
    Err(no_nccl("all_gather"))
}

/// The three symbols, declared so a TP model text resolves rather than being
/// refused for the wrong reason.
///
/// `routine!` and not `driver_bound!`: NCCL resolves its own communicator
/// inside the collective, so every argument these take IS one a statement
/// supplies, and a `driver_bound!` row's empty column is what kept three
/// hand-written arms alive in `driver-cuda` for arithmetic a launcher owns.
pub static ROUTINES: &[crate::jit::Routine] = &[
    crate::routine!(all_reduce_bf16, whole, in_place = &[(0, 0)]),
    crate::routine!(all_reduce_bf16_out, whole),
    crate::routine!(all_gather_bf16, whole),
];

/// `dist`, as a trace names it.
pub static FAMILY: crate::jit::Family = crate::family!(ROUTINES);
