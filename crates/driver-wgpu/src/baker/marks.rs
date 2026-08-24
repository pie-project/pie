//! Rectangles, the regions they name, and the marks a point's declaration
//! takes them as.
//!
//! `driver-metal/src/baker/marks.rs` is the sibling and `driver-cuda`'s is the
//! reference; the shape is the same one. `model_compiler::program::Slot` says
//! WHERE a value lives and how wide it is, `kernels::points` says what each
//! operand of a point is MARKED as, and this module is the one place the two
//! meet. Everything that differs is the payload.
//!
//! # A handle, not an address — and a region that names its buffer
//!
//! Cuda's mark carries `*mut c_void` and a kernel dereferences it. Metal's
//! carries a handle into a per-fire binding list, and what that list holds is
//! `{address, bytes}`: a Metal buffer is bound by a GPU VIRTUAL ADDRESS, so a
//! region is an address and a length and the driver keeps an address→buffer
//! map beside it for the replay path.
//!
//! **WebGPU has no addresses at all.** A `wgpu::BufferBinding` is
//! `(buffer, offset, size)` — three fields, all validated, and the first is an
//! OBJECT rather than a number. So this plane's [`Slice`] names its buffer:
//!
//! ```text
//! cuda    Bank { ptr: *mut c_void }                 an address
//! metal   Slice { address: u64, bytes: u64 }        an address and an extent
//! wgpu    Slice { buffer: BufferId, at, bytes }     a buffer, an offset, an extent
//! ```
//!
//! That is the whole of the divergence and it is FORCED rather than chosen.
//! Metal's file could have been copied verbatim and an `address` invented —
//! a flat space the driver carved buffers out of — and it would have been a
//! fiction: two wgpu buffers have no ordering between them, so any arithmetic
//! across one would be meaningless and any bounds check inside one would be
//! checking a number nothing published. A [`BufferId`] is an index into what
//! the FIRE was given, which is a fact, and `bytes` is checked against that
//! buffer's own length by the half that owns it.
//!
//! The portable half stays portable: a `BufferId` is a number, so the walk,
//! the marks and `tests/the_walk_is_the_program.rs` name no `wgpu` type.
//!
//! # Why there is no `column`
//!
//! The same W10 decision cuda's file records, and it lands hardest here. A
//! mark carries `{ptr, rows, width}` and NO STRIDE, so an executor that cut a
//! packed row by offsetting would be reporting the CUT's width as the row
//! stride. On this plane it cannot even be spelled — `Payload<T>`'s
//! `advance_read` returns the handle UNMOVED, which `kernels-wgpu`'s
//! `points.rs` states as "A HANDLE DOES NOT ADVANCE" — so the rule the
//! strideless mark already meant is the only reading available: **every
//! rectangle an executor hands a kernel is DENSE, and a packed row is cut by
//! a kernel that is told the packing.**
//!
//! # The binder never deduplicates, and here that is load-bearing
//!
//! [`crate::walk::Bindings::take`] says so at length and this plane is the
//! reason. `kernels_wgpu::norm`'s `residual_add`, `scale` and `logit_softcap`
//! each bind ONE handle into TWO of a shader's bindings --
//! `&[x.ptr.arg(), s.arg(), x.arg()]` -- because every invocation reads and
//! writes the same index. A binder that deduplicated would collapse two slots
//! into one and bind the shader's second buffer to nothing; a binder that
//! REFUSED would refuse three claimed points. That the aliasing is legal is
//! measured rather than assumed: `tests/device.rs`'s
//! `two_read_write_bindings_into_one_buffer_are_legal` fires it on a real
//! adapter.
//!
//! This is exactly the seam `kernels_wgpu::mlp` names: five of the six `Mlp`
//! points declare one packed operand and this plane claims only the sixth,
//! because binding one packed handle at both the gate slot and the up slot
//! would read the gate half twice. That refusal is honest here rather than
//! papered over, and it is why `mlp.swiglu` is unclaimed.

use kernels::plane::{Const, In, InOut, Out};
use kernels::points::Scalar;
use kernels_wgpu::points::{Bank, BankHandles, Handle, Payload};

/// Which device allocation a region is part of.
///
/// AN INDEX AND NOT A POINTER, so the portable half can hold one. What it
/// indexes is the fire's own list of allocations, which the device half fills
/// with `wgpu::Buffer`s and a walk test fills with numbers. The two never meet
/// and neither has to know about the other, which is the same trick
/// [`Bindings`] plays one level up.
///
/// The ids themselves are the DRIVER's to assign and are stable for a load:
/// the weight arena, the activation arena and each pool plane get one when
/// they are allocated, and a fire's staged tables get theirs when they are
/// staged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferId(pub u32);

impl BufferId {
    /// The id no allocation has.
    ///
    /// A region carrying it addresses nothing, which is this backend's honest
    /// null — see [`Slice::is_nothing`]. `u32::MAX` and not `0` because `0` is
    /// a perfectly good buffer and the first one a driver allocates.
    pub const NONE: Self = Self(u32::MAX);
}

impl Default for BufferId {
    fn default() -> Self {
        Self::NONE
    }
}

/// Where an operand is: which allocation, how far in, and how many bytes it
/// may address from there.
///
/// THE EXTENT IS NOT OPTIONAL and never was on a shader plane. An arena reused
/// across fires can be smaller than a new fire needs, and a `wgpu` binding
/// whose `offset + size` leaves its buffer is a VALIDATION ERROR — which is
/// the good case. The bad case is the one metal's file names: a launch that
/// addressed past its region corrupting whatever the allocator placed next.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Slice {
    /// The allocation this is part of.
    pub buffer: BufferId,
    /// Bytes from the start of that allocation.
    pub at: u64,
    /// Bytes addressable from `at`.
    pub bytes: u64,
}

impl Slice {
    /// A whole allocation, from its first byte.
    #[must_use]
    pub const fn whole(buffer: BufferId, bytes: u64) -> Self {
        Self {
            buffer,
            at: 0,
            bytes,
        }
    }

    /// The sub-region `bytes` long starting `at` bytes in, or `None` when
    /// that leaves this one.
    ///
    /// Checked rather than assumed, and it is the only way to make a smaller
    /// region out of a bigger one — there is no arithmetic on a `Slice`
    /// anywhere else in this executor.
    #[must_use]
    pub const fn span(self, at: u64, bytes: u64) -> Option<Self> {
        match (at.checked_add(bytes), self.at.checked_add(at)) {
            (Some(end), Some(start)) if end <= self.bytes => Some(Self {
                buffer: self.buffer,
                at: start,
                bytes,
            }),
            _ => None,
        }
    }

    /// Whether this names anything at all.
    ///
    /// A zero-length region is this backend's honest null. The encoder binds
    /// it as a zero-size binding rather than skipping the slot, so a shader
    /// that reads it reads nothing loudly instead of reading a neighbour —
    /// and `wgpu`'s own validation is the thing that says so.
    #[must_use]
    pub const fn is_nothing(self) -> bool {
        self.bytes == 0 || self.buffer.0 == BufferId::NONE.0
    }
}

/// The fire's binding list, at this plane's regions.
pub type Bindings = crate::walk::Bindings<Slice>;

/// One resolved operand, at this plane's regions.
pub type Bound = crate::walk::BoundRegion<Slice>;

/// One value of this fire, addressed.
pub type Rect = crate::walk::Rect<Slice>;

/// A handle that addresses nothing -- what an absent pool or table answers.
///
/// A `const` HERE AND A FUNCTION UPSTAIRS. `crate::walk::BoundRegion::nothing`
/// is generic and cannot be `const`, because `Default::default` is not a
/// `const fn` for a region whose spelling is not known yet. This one's is, and
/// [`super::encode`] names it in a position that wants a constant.
pub const NOTHING: Bound = Bound {
    slice: Slice {
        buffer: BufferId::NONE,
        at: 0,
        bytes: 0,
    },
    width: 0,
};

/// The operand mark, over a freshly minted handle.
pub fn rin<T: Scalar>(b: &mut Bindings, r: Rect) -> In<Payload<T>> {
    In {
        ptr: Handle(b.take(r.bound())),
        rows: r.rows,
        width: r.width,
    }
}

/// The result mark.
pub fn rout<T: Scalar>(b: &mut Bindings, r: Rect) -> Out<Payload<T>> {
    Out {
        ptr: Handle(b.take(r.bound())),
        rows: r.rows,
        width: r.width,
    }
}

/// The in-place mark: ONE handle standing in two columns.
///
/// The walk mints a FRESH rectangle for every result, so the operand's bytes
/// have to be in the result's region before the kernel writes through it.
/// `Fire::inout` schedules that copy and the device half encodes it; what
/// arrives here is the RESULT's rectangle with the operand's bytes already on
/// their way into it.
///
/// So the mark is one handle — which is what the three `Norm` bodies that
/// alias it expect, and what [`Bindings::take`]'s no-dedupe rule exists for.
pub fn rio<T: Scalar>(b: &mut Bindings, r: Rect) -> InOut<Payload<T>> {
    InOut {
        ptr: Handle(b.take(r.bound())),
        rows: r.rows,
        width: r.width,
    }
}

/// A weight, as a `Const` slot takes it: A REGION AND NO RECTANGLE.
///
/// Every point that reads a bank reads its dimensions off something else — the
/// result's width, or a stated scalar — so the width column is zero and the
/// extent is the tensor's own.
pub fn wconst<T: Scalar>(b: &mut Bindings, w: Slice) -> Const<Payload<T>> {
    Const::new(Handle(b.take(Bound { slice: w, width: 0 })))
}

/// A quantised bank, as the plane's own view of its byte planes.
///
/// TWO HANDLES AND ONE SLOT — the one place a slot reads more than one column.
/// `mxfp4` stores a bank as packed codes and a per-block exponent plane,
/// indexed at different strides, which is why `kernels_wgpu::points::
/// BankHandles` names two fields rather than holding an array.
///
/// NOTHING ON THIS PLANE CALLS IT YET, and the reason is this driver's
/// headline gap rather than an oversight: `kernels-wgpu` claims no point that
/// takes a `Const<Self::Bank<R>>`, because the three `Gemm` points, both
/// `moe.matmul_select*` and `layout.embed` all wait on the floor's
/// `Bank<R: Repr>` payload reaching their DECLARATIONS — all six declare
/// `Const<Self::Tensor<T>>` today. The accessor is written because
/// `BoundOp::bank` is a floor method this executor owes an answer for, and
/// because the day a declaration grows the slot this is what will fill it.
pub fn wbank<R: kernels::points::Repr>(
    b: &mut Bindings,
    codes: Slice,
    scales: Slice,
) -> Const<Bank<R>> {
    let codes = Handle(b.take(Bound {
        slice: codes,
        width: 0,
    }));
    let scales = Handle(b.take(Bound {
        slice: scales,
        width: 0,
    }));
    Const::new(BankHandles { codes, scales })
}
