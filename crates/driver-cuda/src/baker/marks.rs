//! Rectangles, and the marks a point's declaration takes them as.
//!
//! LIFTED FROM `baker-smoke/src/marks.rs`, and deliberately not adapted.
//! That file is the executable spec this module is moving into the driver;
//! the way to move it wrong would be to "improve" it on the way in, so it
//! arrives verbatim and the divergences are the ones written down here.
//! There is exactly one, at [`Rect::rows`], and it is the whole reason W1
//! is a different job from the smoke.
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
//! (`kernels/src/routine.rs:617-625`), so an executor outside the kernel
//! crate can build every mark honestly, with no transmute and no `unsafe`
//! at all.
//!
//! # There is no `column`, and that is the W10 decision
//!
//! This module used to carry `Rect::column`: the same rectangle `elems`
//! elements in, `width` wide, for a routine that took a packed operand's
//! halves as two pointers. It was the multi-row blocker. A mark carries
//! `{ptr, rows, width}` and NO STRIDE, so the cut reported the CUT's width
//! as its row stride when the bytes stride by the PACKED width — true at
//! one row, false at two, and silent either way because every address
//! stayed inside the arena.
//!
//! What replaced it is not a stride on the mark. It is the rule the
//! strideless mark already implied and nothing was enforcing: **every
//! rectangle an executor hands a kernel is DENSE — `rows` rows of `width`
//! elements, `width` apart — and a packed row is cut by a kernel that is
//! told the packing, never by an executor that offsets a pointer.**
//! [`super::fire::Fire::rect`] reads the arena value-major for the same
//! reason; the four cuts that were left lived at the gdn seam and are now
//! `GdnShape::stage` in `kernels-cuda/src/ssm.rs`, which is where the
//! chunked point had already put them.

use core::ffi::c_void;

use kernels::plane::{Const, In, InOut, Out};
use kernels::points::Scalar;
use kernels_cuda::jit::abi::Tensor;
use model_compiler::program::Dt;

/// One value of this fire, addressed: `rows` rows of `width` elements of
/// `dt` at `ptr`.
///
/// ROWS ARE THE FIRE'S, which is what `program.rs` says the width table
/// deliberately does not answer — and in the driver that sentence finally
/// has teeth. The smoke fired one row and could write `1` at every
/// construction site; a fire assembled out of a `FrameSubmission` carries
/// whatever the scheduler batched, so `rows` is threaded from
/// `Fire::rows` and is never a literal. The routines read it: the KV
/// appender takes its token count off `k_curr.rows`, and `qo_indptr.rows`
/// IS the request count.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Rect {
    pub ptr: *mut c_void,
    pub rows: i32,
    pub width: i32,
    pub dt: Dt,
}

impl Rect {
    pub(crate) fn bytes(&self) -> usize {
        self.rows as usize * self.width as usize * self.dt.size() as usize
    }
}

pub(crate) fn rin<T: Scalar>(r: Rect) -> In<Tensor<T>> {
    In {
        ptr: r.ptr.cast::<T>().cast_const(),
        rows: r.rows,
        width: r.width,
    }
}

pub(crate) fn rout<T: Scalar>(r: Rect) -> Out<Tensor<T>> {
    Out {
        ptr: r.ptr.cast::<T>(),
        rows: r.rows,
        width: r.width,
    }
}

pub(crate) fn rio<T: Scalar>(r: Rect) -> InOut<Tensor<T>> {
    InOut {
        ptr: r.ptr.cast::<T>(),
        rows: r.rows,
        width: r.width,
    }
}

/// A weight, as a `Const` slot takes it: an ADDRESS AND NO RECTANGLE. Every
/// point that reads a bank reads its dimensions off something else -- the
/// result's width, or a stated scalar -- and `program.rs` says why.
pub(crate) fn wconst<T: Scalar>(p: *mut c_void) -> Const<Tensor<T>> {
    Const::new(p.cast::<T>().cast_const())
}
