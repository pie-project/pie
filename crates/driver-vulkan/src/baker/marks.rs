//! Rectangles, the regions they name, and the marks a point's declaration
//! takes them as.
//!
//! `driver-wgpu/src/baker/marks.rs` is the sibling and `driver-cuda`'s is the
//! reference; the shape is the same one. `model_compiler::program::Slot` says
//! WHERE a value lives and how wide it is, `kernels::points` says what each
//! operand of a point is MARKED as, and this module is the one place the two
//! meet. Everything that differs is the payload.
//!
//! # A handle, not an address — and a region that names its allocation
//!
//! Cuda's mark carries `*mut c_void` and a kernel dereferences it. Metal's
//! carries a handle into a per-fire binding list, and what that list holds is
//! `{address, bytes}`: a Metal buffer is bound by a GPU VIRTUAL ADDRESS, so a
//! region is an address and a length.
//!
//! **Vulkan has no addresses in the binding path either.** A descriptor is
//! written from a `VkDescriptorBufferInfo`, which is `{buffer, offset, range}`
//! — three fields, all validated, and the first is an OBJECT handle rather
//! than a number. `crate::device::Bound` is already exactly those three
//! (`buffer: &Buffer, offset: u64, len: u64`), and [`Slice`] is its portable
//! half:
//!
//! ```text
//! cuda    Bank  { ptr: *mut c_void }               an address
//! metal   Slice { address: u64, bytes: u64 }       an address and an extent
//! wgpu    Slice { buffer: BufferId, at, bytes }    a buffer, an offset, an extent
//! vulkan  Slice { buffer: BufferId, at, bytes }    the same three, for the same reason
//! ```
//!
//! THAT THE LAST TWO ROWS AGREE IS A MEASUREMENT AND NOT A SHARED FILE. They
//! agree because `VkDescriptorBufferInfo` and `wgpu::BufferBinding` describe
//! the same thing — wgpu's Vulkan backend writes one from the other — and they
//! stay separate because the crates that hold them are separate: this one's
//! `BufferId` indexes what a Vulkan fire was given, that one's indexes what a
//! WebGPU fire was given, and neither number means anything in the other's
//! process. `driver-metal`'s row is the evidence that a shared `Slice` would
//! have been a fight rather than a saving.
//!
//! The portable half stays portable: a [`BufferId`] is a number, so this
//! module, the walk and `tests/the_walk_is_the_program.rs` name no `ash` type
//! and build with `default = []`.
//!
//! # Why there is no `column`
//!
//! The same W10 decision cuda's file records. A mark carries `{ptr, rows,
//! width}` and NO STRIDE, so an executor that cut a packed row by offsetting
//! would be reporting the CUT's width as the row stride. On this plane it
//! cannot even be spelled — `kernels_vulkan::points::Handle<T>`'s
//! `advance_read` returns the handle unmoved, and that crate's `Staged::window`
//! refuses a windowed binding by name: *"a descriptor names a whole allocation,
//! so a packed row's second half is not addressable here"*. So the rule the
//! strideless mark already meant is the only reading available: **every
//! rectangle an executor hands a kernel is DENSE, and a packed row is cut by a
//! kernel that is told the packing.**
//!
//! # The binder never deduplicates, and here that is load-bearing
//!
//! [`crate::walk::Bindings::take`] says so at length and this plane needs it
//! for the same reason wgpu does. `kernels_vulkan::attn`'s `logit_softcap`
//! binds ONE handle into TWO of a shader's bindings —
//! `&[x.ptr.arg(), x.arg(), cap.arg()]` — because every invocation reads and
//! writes the same index. A binder that deduplicated would collapse the two
//! into one and leave the shader's second buffer bound to nothing; a binder
//! that REFUSED would refuse a point this plane claims.

use kernels::plane::{Const, In, InOut, Out};
use kernels::points::Scalar;
use kernels_vulkan::points::{Handle, Planes};

/// Which device allocation a region is part of.
///
/// AN INDEX AND NOT A `vk::Buffer`, so the portable half can hold one. What it
/// indexes is the fire's own list of allocations, which the device half fills
/// with real `crate::device::Buffer`s and a walk test fills with numbers. The
/// two never meet and neither has to know about the other, which is the same
/// trick [`crate::walk::Bindings`] plays one level up.
///
/// The ids themselves are the DRIVER's to assign and are stable for a load: the
/// weight arena, the activation arena and each pool plane get one when they are
/// allocated, and a fire's staged tables get theirs when they are staged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferId(pub u32);

impl BufferId {
    /// The id no allocation has.
    ///
    /// A region carrying it addresses nothing, which is this backend's honest
    /// null — see [`Slice::is_nothing`]. `u32::MAX` and not `0` because `0` is a
    /// perfectly good allocation and the first one a driver makes.
    pub const NONE: Self = Self(u32::MAX);
}

impl Default for BufferId {
    fn default() -> Self {
        Self::NONE
    }
}

/// Where an operand is: which allocation, how far in, and how many bytes it may
/// address from there.
///
/// THE EXTENT IS NOT OPTIONAL and never was on a shader plane. An arena reused
/// across fires can be smaller than a new fire needs, and a descriptor whose
/// `offset + range` leaves its buffer is what `crate::device::Bound::within`
/// already refuses as `Failed::Overrun` — which is the good case. The bad case
/// is a launch that addressed past its region and corrupted whatever the
/// allocator placed next.
///
/// THE OFFSET'S ALIGNMENT IS NOT CHECKED HERE, and that is deliberate rather
/// than missing. `minStorageBufferOffsetAlignment` is a number the DEVICE
/// reports and this half has no device; `Bound::within` takes it as an argument
/// for exactly that reason, so that a plan can be checked against the
/// specification's 256 on a machine with no adapter. What this module owes is
/// that a sub-region stays inside its parent, which is [`Slice::span`].
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

    /// The sub-region `bytes` long starting `at` bytes in, or `None` when that
    /// leaves this one.
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
    /// A zero-length region is this backend's honest null, and Vulkan is the
    /// plane that is loudest about it: `Bound::within` refuses `len == 0`
    /// outright, because a zero-range descriptor is illegal and the defect
    /// behind one is always the same — a width computed from a shape that came
    /// out empty. So a region that answers `true` here is one the device half
    /// will refuse BY NAME rather than bind.
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

/// A handle that addresses nothing — what an absent pool or table answers.
///
/// A `const` HERE AND A FUNCTION UPSTAIRS. [`crate::walk::BoundRegion::nothing`]
/// is generic and cannot be `const`, because `Default::default` is not a
/// `const fn` for a region whose spelling is not known yet. This one's is,
/// which is what lets a device half or a fixture name it where a constant is
/// wanted — [`super::views`]'s `plane` reaches the same value through
/// `Option::unwrap_or_default`, which is the expression form of it.
///
/// NO CLAIM BODY ON THIS PLANE ASKS FOR ONE, which is where the sibling
/// diverges: `driver-wgpu`'s `Encode::resolve` answers `Asks::absent()` with a
/// handle onto this, because six of its sdpa arms declare a binding their point
/// does not carry. `kernels-vulkan`'s arms bind every slot they declare, so
/// [`super::encode::Encoder`]'s `resolve` refuses outright and this constant is
/// reached only through the staging.
pub const NOTHING: Bound = Bound {
    slice: Slice {
        buffer: BufferId::NONE,
        at: 0,
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
/// The walk mints a FRESH rectangle for every result, so the operand's bytes
/// have to be in the result's region before the kernel writes through it.
/// `Fire::inout` schedules that copy and the device half encodes it; what
/// arrives here is the RESULT's rectangle with the operand's bytes already on
/// their way into it.
///
/// So the mark is one handle — which is what `attn::logit_softcap` expects when
/// it writes `&[x.ptr.arg(), x.arg(), cap.arg()]`, and what
/// [`crate::walk::Bindings::take`]'s no-dedupe rule exists for.
pub fn rio<T: Scalar>(b: &mut Bindings, r: Rect) -> InOut<Handle<T>> {
    InOut {
        ptr: Handle::new(b.take(r.bound())),
        rows: r.rows,
        width: r.width,
    }
}

/// A weight, as a `Const` slot takes it: A REGION AND NO RECTANGLE.
///
/// Every point that reads a bank reads its dimensions off something else — the
/// result's width, or a stated scalar — so the width column is zero and the
/// extent is the tensor's own.
pub fn wconst<T: Scalar>(b: &mut Bindings, w: Slice) -> Const<Handle<T>> {
    Const::new(Handle::new(b.take(Bound { slice: w, width: 0 })))
}

/// A quantised bank, as the plane's own view of its byte planes.
///
/// TWO HANDLES AND ONE SLOT — the one place a slot reads more than one column.
/// `mxfp4` stores a bank as packed codes and a per-block exponent plane,
/// indexed at different strides, which is why `kernels_vulkan::points::Planes`
/// names two fields rather than holding an array.
///
/// THE CODES ARE `Handle<u32>` AND THE SCALES `Handle<u8>`, which is that
/// struct's own declaration and not this module's choice: `mxfp4_qmv_routed_bias`
/// reads the code plane as packed words and the exponent plane as bytes, and
/// the two handles are typed at what the shader reads rather than at what the
/// checkpoint stored. `crate::walk::bound::Bound::bank` has already refused any
/// plane whose stored element is not `u8`, so the retyping here is a statement
/// about the READ and loses nothing.
///
/// ONE POINT ON THIS PLANE REACHES IT: `moe.matmul_select_bias`, whose
/// declaration states `Const<Self::Bank<R>>` and whose body reads
/// `planes.codes` and `planes.scales` off exactly this pair.
pub fn wbank<R: kernels::points::Repr>(
    b: &mut Bindings,
    codes: Slice,
    scales: Slice,
) -> Const<Planes<R>> {
    let codes = Handle::new(b.take(Bound {
        slice: codes,
        width: 0,
    }));
    let scales = Handle::new(b.take(Bound {
        slice: scales,
        width: 0,
    }));
    Const::new(Planes::new(codes, scales))
}
