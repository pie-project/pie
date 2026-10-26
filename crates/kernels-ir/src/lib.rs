//! The contract between what forms a fused kernel and the kernels-*
//! backends that build it.
//!
//! - **A region** ([`Region`]): what a fused chain computes, a function of one
//!   row. A dataflow of primitive ops over the row's elements and the row's
//!   scalars, typed, with explicit rounding points: each value rounds to the
//!   dtype the unfused ops would have stored it in, where they would have.
//!   poem-compiler lowers its chains to one; nothing here knows poem.
//! - **A kernel** ([`Kernel`]): the region scheduled. Passes over the row,
//!   which values stay in registers across a pass, where a pass ends (at a
//!   reduction's reader), each reduction's partials and their order.
//!   Target-neutral: a backend prints it in its language.
//! - **The schedule** ([`schedule`]): region to kernel, or why the row core
//!   does not cover a region.
//! - **The skeleton** ([`skeleton`]): the primitives a kernel runs over, a row's
//!   loads and stores, a scalar parameter, validity, the reductions over the
//!   row's threads, the rounding to a narrow dtype. Each backend's skeleton
//!   provides them; the printers name them.
//! - **The interpreter** ([`interp`]): both forms on the host, the oracle
//!   every printed kernel is held to.
//! - **The builders** ([`build`]): a row's math (a norm, a head's norm, a
//!   rotation) as region ops, in the order the hand kernels compute it.
//! - **The anchored region** ([`anchor`]): a contraction (`y = a · wᵀ`)
//!   with a region pointwise over its operand tile before it and one over
//!   its product tile after it, verified to be pointwise there, and its
//!   interpreter. A backend's contraction skeletons fill their hooks with the
//!   regions' printed bodies.

pub mod anchor;
pub mod build;
pub mod interp;
pub mod kernel;
pub mod region;
pub mod schedule;
pub mod skeleton;

pub use anchor::{
    Anchored, CacheDtype, CachePlane, Contraction, Gain, Input, Layout, Paged, Pointwise, Quant, RowNorm,
    Space, Window,
};
pub use kernel::{
    Expr, HeadAssign, Kernel, Loop, Operand, Partial, Phase, RowAssign, Table, Total, Uniform,
};
pub use region::{Binary, Elem, Literal, Op, Reduce, Region, RegionError, Shape, Unary, ValueId};
pub use schedule::{Unscheduled, schedule};

/// FNV-1a over `bytes`: the hash a region's key and a printed body's unit
/// are keyed by.
#[must_use]
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}
