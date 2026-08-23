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

use core::ffi::c_void;

use kernels::points::Scalar;
use kernels::routine::{Const, In, InOut, Out};
use kernels_cuda::jit::abi::Tensor;
use model_compiler::program::Dt;

/// One value of this fire, addressed: `rows` rows of `width` elements of
/// `dt` at `ptr`.
///
/// ROWS ARE THE FIRE'S, which is what `program.rs` says the width table
/// deliberately does not answer. This binary fires one row, so `rows` is 1
/// on every arena rectangle; it is carried rather than assumed because the
/// routines read it (`write_kv_to_pages` takes its token count off
/// `k_curr.rows`, `qo_indptr.rows` IS the request count).
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

    /// The same rectangle `elems` elements in, `width` wide -- the cut a
    /// packed operand's half needs when the routine takes the halves as two
    /// separate pointers and the statement carries only the packed row.
    #[must_use]
    pub fn column(&self, elems: i32, width: i32) -> Rect {
        Rect {
            ptr: unsafe {
                self.ptr
                    .cast::<u8>()
                    .add(elems as usize * self.dt.size() as usize)
                    .cast()
            },
            rows: self.rows,
            width,
            dt: self.dt,
        }
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
