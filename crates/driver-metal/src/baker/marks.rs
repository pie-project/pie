//! The region this plane names, and the marks a point's declaration takes one
//! as.
//!
//! `driver-cuda/src/baker/marks.rs` is the sibling and the shape is the same
//! one: `model_compiler::program::Slot` says WHERE a value lives and how wide
//! it is, `kernels::points` says what each operand of a point is MARKED as, and
//! this module is the one place the two meet. Everything that differs is the
//! payload -- **which is why this file is what stayed here when the executor
//! left.** [`crate::walk::marks`] holds the three types that merely CONTAIN a
//! region and were identical between the two shader planes to the character;
//! what is below is the divergence itself.
//!
//! # A handle, not an address
//!
//! Cuda's mark carries `*mut c_void` and a kernel dereferences it. Metal has no
//! such thing at the shader boundary: a compute kernel reads BUFFER BINDINGS,
//! numbered, and the driver sets each one on an argument table before the
//! dispatch runs. So `Plane::Tensor<T>` on this plane is
//! `kernels_metal::plane::Handle<T>` -- an index -- and what it indexes is
//! [`Bindings`], the fire's own list of the regions it has bound.
//!
//! THE INDEX IS MINTED, NEVER STATED. A `Slot::Arena` gives an offset into the
//! fire's arena and a width; the executor turns that into a [`Slice`] and asks
//! [`Bindings::take`] for a number to call it by. Two statements naming one
//! value mint two handles onto one region, which is exactly right -- a handle
//! is a *binding*, not an identity, and the encoder sets the same address
//! twice.
//!
//! # Why the region is an address and an extent
//!
//! Which is the sentence `driver-wgpu`'s copy of this file cannot say, and the
//! reason the two files are not one. A Metal buffer is bound by a GPU VIRTUAL
//! ADDRESS, so a region is an address and a length and the driver keeps an
//! address→buffer map beside it for the replay path. A `wgpu::BufferBinding` is
//! `(buffer, offset, size)` -- an OBJECT and two numbers -- because WebGPU has
//! no address space to do arithmetic in. Both are `Plane::Slice` to the walk,
//! which asks exactly two things of one: `span` and `bytes`.

use kernels::plane::{Const, In, InOut, Out};
use kernels::points::Scalar;
use kernels_metal::points::{Handle, Planes};

/// The fire's binding list, at this plane's regions.
pub type Bindings = crate::walk::Bindings<Slice>;

/// One resolved operand, at this plane's regions.
pub type Bound = crate::walk::BoundRegion<Slice>;

/// One value of this fire, addressed.
pub type Rect = crate::walk::Rect<Slice>;

/// Where an operand is: a device address and the bytes it may address.
///
/// LIFTED VERBATIM from the deleted `lowering::executor`, which is the one
/// thing in that module that was never about the legacy walk. An extent
/// travelling with an address is the invariant the whole binder rests on: an
/// arena reused across fires can be smaller than the new fire needs, and a
/// launch that addressed past it would corrupt whatever the allocator placed
/// next.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Slice {
    /// GPU address of the first byte.
    pub address: u64,
    /// Bytes addressable from it.
    pub bytes: u64,
}

impl Slice {
    /// The sub-region `bytes` long starting `at` bytes in, or `None` when that
    /// leaves this one.
    #[must_use]
    pub const fn span(self, at: u64, bytes: u64) -> Option<Self> {
        match (at.checked_add(bytes), self.address.checked_add(at)) {
            (Some(end), Some(address)) if end <= self.bytes => Some(Self { address, bytes }),
            _ => None,
        }
    }

    /// Whether this names anything at all. A zero-length region is this
    /// backend's honest null: the encoder binds it and a shader that reads it
    /// faults loudly rather than reading a neighbour.
    #[must_use]
    pub const fn is_nothing(self) -> bool {
        self.address == 0 || self.bytes == 0
    }
}

/// A handle that addresses nothing -- what an absent pool or table answers.
///
/// A `const` HERE AND A FUNCTION UPSTAIRS. `crate::walk::BoundRegion::nothing`
/// is generic and cannot be `const`, because `Default::default` is not a
/// `const fn` for a region whose spelling is not known yet. This one's is, and
/// `super::encode` names it in a position that wants a constant.
pub const NOTHING: Bound = Bound {
    slice: Slice {
        address: 0,
        bytes: 0,
    },
    width: 0,
};

/// The operand mark, over a freshly minted handle.
pub fn rin<T: Scalar>(b: &mut Bindings, r: Rect) -> In<Handle<T>> {
    In {
        ptr: Handle::new(b.take(r.bound())),
        rows: r.rows,
        width: r.width,
    }
}

/// The result mark.
pub fn rout<T: Scalar>(b: &mut Bindings, r: Rect) -> Out<Handle<T>> {
    Out {
        ptr: Handle::new(b.take(r.bound())),
        rows: r.rows,
        width: r.width,
    }
}

/// The in-place mark: ONE handle standing in two columns.
///
/// Cuda's `tinout` copies the operand's bytes into the result's rectangle
/// before the kernel writes through it, because the walk mints a fresh
/// rectangle for every result. This one does the same -- the copy is scheduled
/// by the walk (`Fire::inout`) and staged by `fire::run::submit` -- and what
/// arrives here is the RESULT's rectangle with the operand's bytes already in
/// it. So the mark is one handle, which is what the marks
/// `kernels_metal::plane::{read_half, write_half}` cut it into expect.
pub fn rio<T: Scalar>(b: &mut Bindings, r: Rect) -> InOut<Handle<T>> {
    InOut {
        ptr: Handle::new(b.take(r.bound())),
        rows: r.rows,
        width: r.width,
    }
}

/// A weight, as a `Const` slot takes it: AN ADDRESS AND NO RECTANGLE.
///
/// Every point that reads a bank reads its dimensions off something else -- the
/// result's width, or a stated scalar -- so the width column is zero and the
/// extent is the tensor's own.
pub fn wconst<T: Scalar>(b: &mut Bindings, w: Slice) -> Const<Handle<T>> {
    Const::new(Handle::new(b.take(Bound { slice: w, width: 0 })))
}

/// A quantised bank, as the plane's own view of its byte planes.
///
/// TWO HANDLES AND ONE SLOT -- the one place a slot reads more than one column.
/// `mxfp4` stores a bank as packed codes and a per-block exponent plane,
/// indexed at different strides, which is why `kernels_metal::plane::Planes`
/// names two fields rather than holding an array.
pub fn wbank<R: kernels::points::Repr>(
    b: &mut Bindings,
    codes: Slice,
    scales: Slice,
) -> Const<Planes<R>> {
    let codes = b.take(Bound {
        slice: codes,
        width: 0,
    });
    let scales = b.take(Bound {
        slice: scales,
        width: 0,
    });
    Const::new(Planes::new(codes, scales))
}
