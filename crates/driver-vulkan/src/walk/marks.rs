//! The fire's binding list, and the rectangles it is made of.
//!
//! THREE TYPES OUT OF A DRIVER'S `marks.rs`, and the three that were identical
//! to the character in both. What stays in a driver is the [`Plane::Slice`] the
//! three are parameterised by and the five mark constructors that read one --
//! `rin`, `rout`, `rio`, `wconst`, `wbank` -- because those are where a Metal
//! address and a `wgpu::BufferBinding` actually part.
//!
//! The split is exactly where the divergence is. A `Slice` differs; a LIST of
//! them does not, and neither does a rectangle's `{rows, width, dt}`.
//!
//! # Why there is no `column`
//!
//! The W10 decision both drivers record, and it belongs with these types rather
//! than with either plane's. A mark carries `{ptr, rows, width}` and NO STRIDE,
//! so an executor that cut a packed row by offsetting would be reporting the
//! CUT's width as the row stride. Neither shader plane could even spell it -- a
//! handle has no arithmetic and `advance_read` returns it unmoved -- so the
//! rule the strideless mark already meant is the only reading available:
//! **every rectangle an executor hands a kernel is DENSE, and a packed row is
//! cut by a kernel that is told the packing.**
//!
//! [`Plane::Slice`]: crate::Plane::Slice

use model_compiler::program::Dt;

/// One resolved operand: the region it addresses, and how wide one row is.
///
/// The width is what an ENCODER never needs and a hazard set never reads; it
/// rides here because a bound region with no width would make [`Bindings`] two
/// lists instead of one, and because a diagnostic that can print the rectangle
/// a slot bound is worth the four bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bound<S> {
    /// The region the operand addresses.
    pub slice: S,
    /// Elements per row, or zero for a bank (whose extent is the tensor's).
    pub width: u32,
}

impl<S: Default> Bound<S> {
    /// A handle that addresses nothing -- what an absent pool or table answers.
    ///
    /// A FUNCTION WHERE BOTH DRIVERS HAD A `const NOTHING`, and the change is
    /// the generic and nothing else: `S::default()` is not a `const fn`, so the
    /// null cannot be computed at compile time for a region whose spelling is
    /// not known yet. Each driver keeps its `NOTHING` as a `const` of its own
    /// concrete `Bound<Slice>`, which is what its readers already name.
    #[must_use]
    pub fn nothing() -> Self {
        Self {
            slice: S::default(),
            width: 0,
        }
    }
}

/// The fire's binding list: what every handle it minted stands for.
///
/// ONE PER FIRE, and the numbering is this list's own. A point's declaration
/// says a slot is `In<Self::Tensor<T>>`; the executor answers with a handle;
/// the claim body passes that number to `ctx.fire`; the driver's own `encode`
/// looks it up here and binds the region. Nothing between those four steps
/// needs to know where the bytes are.
#[derive(Debug)]
pub struct Bindings<S> {
    bound: Vec<Bound<S>>,
}

impl<S> Default for Bindings<S> {
    fn default() -> Self {
        Self { bound: Vec::new() }
    }
}

impl<S: Copy> Bindings<S> {
    /// An empty list.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Take a handle for `bound`, whatever it is.
    ///
    /// NEVER DEDUPLICATES, and on a shader plane that is load-bearing rather
    /// than merely honest. `kernels_wgpu::norm`'s `residual_add`, `scale` and
    /// `logit_softcap` each bind ONE handle into TWO of a shader's bindings,
    /// because every invocation reads and writes the same index, and
    /// `attention.kv_append_shared` is the same shape on metal. A binder that
    /// deduplicated would collapse two slots into one and bind the shader's
    /// second buffer to nothing; a binder that REFUSED would refuse points both
    /// planes claim. It does neither: a handle is a BINDING, not an identity,
    /// and two of them may stand for one region.
    pub fn take(&mut self, bound: Bound<S>) -> u32 {
        let at = u32::try_from(self.bound.len()).unwrap_or(u32::MAX);
        self.bound.push(bound);
        at
    }

    /// What `handle` stands for, or `None` for a number this fire never minted
    /// -- which is a body reaching past its own statement.
    #[must_use]
    pub fn at(&self, handle: u32) -> Option<Bound<S>> {
        self.bound.get(handle as usize).copied()
    }

    /// How many handles this fire has minted.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bound.len()
    }

    /// Whether this fire has minted none.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bound.is_empty()
    }
}

/// One value of this fire, addressed: `rows` rows of `width` elements of `dt`
/// in `slice`.
///
/// ROWS ARE THE FIRE'S, which is the sentence `model_compiler::program` says
/// the width table deliberately does not answer. A fire assembled out of a
/// `FrameSubmission` carries whatever the scheduler batched, so `rows` is
/// threaded from the fire and is never a literal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect<S> {
    /// The region the rectangle occupies, sized `rows * width * dt`.
    pub slice: S,
    /// Rows in this fire, times the slot's own row factor.
    pub rows: i32,
    /// Elements per row.
    pub width: i32,
    /// The element the walk decided this value holds.
    pub dt: Dt,
}

impl<S: Copy> Rect<S> {
    /// The bytes this rectangle covers.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        u64::from(self.rows.unsigned_abs()) * u64::from(self.width.unsigned_abs()) * self.dt.size()
    }

    /// What this rectangle is, as the binder records it.
    #[must_use]
    pub const fn bound(&self) -> Bound<S> {
        Bound {
            slice: self.slice,
            width: self.width.unsigned_abs(),
        }
    }
}
