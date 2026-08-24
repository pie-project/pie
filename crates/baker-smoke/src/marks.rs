//! Rectangles, and the marks a point's declaration takes them as.
//!
//! `model_compiler::program::Slot` says WHERE a value lives and how wide it
//! is; `kernels::points` says what each operand of a point is MARKED as.
//! This module is the one place the two meet: a [`Rect`] is a slot resolved
//! against this fire's bases, and the four constructors below wear it as
//! `In` / `Out` / `InOut` / `Const`.
//!
//! NO NEW SURFACE WAS NEEDED. `kernels::routine::{In, Out, InOut}` are
//! plain structs with public `ptr`/`rows`/`width` fields
//! (`kernels/src/routine.rs:493-517`) and `Const` has `Const::new`
//! (`kernels/src/routine.rs:617-625`), so an executor outside the driver
//! can build every mark honestly, with no transmute and no `unsafe`.
//!
//! AND NO `column`. A mark carries no stride, so a rectangle taken `elems`
//! into a packed row and called `width` wide claims a row stride of `width`
//! for bytes whose stride is the packed one — true at one row, false at
//! two, and silent either way. This module used to carry that cut for the
//! gdn seam; the seam's points state their own operands now and do every
//! packed→compact cut in a kernel that is told the packing (W10). What an
//! executor hands a kernel is DENSE rectangles, and only those.

use core::ffi::c_void;

use kernels::points::Scalar;
use kernels::routine::{Const, In, InOut, Out};
use kernels_cuda::jit::abi::Tensor;
use model_compiler::program::Dt;

/// One value of this fire, addressed: `rows` rows of `width` elements of
/// `dt` at `ptr`.
///
/// ROWS ARE THE FIRE'S TIMES THE SLOT'S FACTOR. `program.rs` answers the
/// factor (`Rows::Fire`, or `Rows::FireTimes(top_k)` on a routed value) and
/// deliberately does not answer the fire's own count. This binary fires one
/// row of a dense text, so `rows` is 1 on every arena rectangle; both halves
/// are carried rather than assumed because the routines read them
/// (`write_kv_to_pages` takes its token count off `k_curr.rows`,
/// `qo_indptr.rows` IS the request count).
#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub ptr: *mut c_void,
    pub rows: i32,
    pub width: i32,
    pub dt: Dt,
}

impl Rect {
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.rows as usize * self.width as usize * self.dt.size() as usize
    }
}

#[must_use]
pub fn rin<T: Scalar>(r: Rect) -> In<Tensor<T>> {
    In {
        ptr: r.ptr.cast::<T>().cast_const(),
        rows: r.rows,
        width: r.width,
    }
}

#[must_use]
pub fn rout<T: Scalar>(r: Rect) -> Out<Tensor<T>> {
    Out {
        ptr: r.ptr.cast::<T>(),
        rows: r.rows,
        width: r.width,
    }
}

#[must_use]
pub fn rio<T: Scalar>(r: Rect) -> InOut<Tensor<T>> {
    InOut {
        ptr: r.ptr.cast::<T>(),
        rows: r.rows,
        width: r.width,
    }
}

/// A weight, as a `Const` slot takes it: an ADDRESS AND NO RECTANGLE. Every
/// point that reads a bank reads its dimensions off something else -- the
/// result's width, or a stated scalar -- and `program.rs` says why.
#[must_use]
pub fn wconst<T: Scalar>(p: *mut c_void) -> Const<Tensor<T>> {
    Const::new(p.cast::<T>().cast_const())
}
