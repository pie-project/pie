//! What `kernels::points` IS on this plane: the payload a mark carries, the
//! elements a family method may be instantiated at, and the seam surface a
//! claim body reaches for when a declaration carries no slot for what a
//! shader entrypoint reads.
//!
//! `.wiki/baker.md`'s endpoint is that a `#[claims] impl Family for Ctx`
//! method IS the launcher — no `#[routine]` under it, no delegation through
//! one. This module is the floor that makes such a body writable here, and
//! the three ways this plane differs from `kernels-cuda` are all in it:
//!
//! 1. **A payload is a HANDLE, not an address.** Cuda's `Tensor<T>` is a
//!    phantom whose `Elem::Read` is `*const T`; a body does pointer
//!    arithmetic on it and hands the result to a kernel. Here the value the
//!    driver binds is a descriptor index — [`Handle`] — and the ONE thing a
//!    body may do with it is bind it whole. Every consequence below follows
//!    from that single fact.
//!
//! 2. **The element rides the ENTRYPOINT NAME.** Cuda spells `T::CPP` into a
//!    JIT template argument, so one body serves every element the point
//!    quantifies over. A `.slang` module is instantiated per element by a
//!    `// pie:instantiate` line, and every shipping instantiation in this
//!    tree is `_bfloat16`. So a claim body's first line is [`at_bf16`] and
//!    its refusal names the point — the shader-plane form of cuda's
//!    "at an element this plane does not instantiate".
//!
//! 3. **`Ctx` is a trait object.** `crate::routine::Ctx<'a>` is
//!    `dyn Encode + 'a`, so `impl Plane for Ctx<'_>` is an impl on a local
//!    trait object and every `Encode` implementor — `driver-vulkan`'s
//!    `Encoder`, a probe, a test double — answers the whole family surface
//!    at once. There is no struct to hang staging off, which is why the
//!    staging this plane needs is a TRAIT ([`Staged`]) and why its methods
//!    refuse rather than answer today.

use core::marker::PhantomData;

use kernels::Ty;
use kernels::bound::{Axis, Rides};
use kernels::points::Scalar;
use kernels::routine::{Claim, ConstRun, Elem, Refusal};
use kernels::shader::ShaderValue;

use crate::routine::Ctx;

/// This plane's half-precision element, and the reason it is declared here
/// rather than taken from `kernels::shader`.
///
/// `kernels::shader::bf16` already exists and is the SHADER spelling: a
/// zero-sized tag whose whole job is to pick `StructuredBuffer<uint16_t>`
/// out of [`kernels::shader::Lang`] for a `.slang` parameter. It implements
/// `shader::Element` and nothing else, and it cannot implement
/// `points::Scalar` — that bound is `Elem<Read = *const Self>`, a
/// POINTER-shaped element, and pointer arithmetic over a zero-sized tag is
/// a lie the floor should not be asked to tell.
///
/// So the two spellings are two types, exactly as they are on cuda:
/// `kernels_cuda::jit::abi::bf16` is that plane's `Scalar` and lives in that
/// plane's crate for the same orphan reason. What a `Handle<bf16>` actually
/// binds is a descriptor index; the element is a CLAIM ABOUT THE BYTES
/// behind it, checked by the dispatch (`BoundOp::tin::<bf16>` refuses a
/// rectangle that rides something else) and spent by [`at_bf16`] here.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct bf16(pub u16);

/// The IEEE half, for the `Capability::Fp16` instantiations. No entrypoint
/// in this tree is stamped at it yet — the `fp16` tier recompiles the same
/// `_bfloat16` modules with `shaderFloat16` — so nothing dispatches here.
/// It is declared beside [`bf16`] because `Rides` needs a name to refuse by.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct f16(pub u16);

macro_rules! plane_elem {
    ($t:ty, $tc:ident, $tm:ident, $axis:ident) => {
        impl Elem for $t {
            type Read = *const $t;
            type Write = *mut $t;

            unsafe fn advance_read(read: Self::Read, elems: usize) -> Self::Read {
                unsafe { read.add(elems) }
            }

            unsafe fn advance_write(write: Self::Write, elems: usize) -> Self::Write {
                unsafe { write.add(elems) }
            }

            // A shader plane has no device TEXT to spell an element into:
            // `rms_single_row_bfloat16` carries it in the entrypoint's own
            // name, stamped by a `// pie:instantiate` line. Empty is the
            // honest answer and the same one `kernels::shader::Tensor`
            // gives.
            const CPP: &'static str = "";
            const CPP_CONST: &'static str = "";
            const CPP_MUT: &'static str = "";
            const TY_CONST: Ty = Ty::$tc;
            const TY_MUT: Ty = Ty::$tm;
        }

        impl Rides for $t {
            const AXIS: Axis = Axis::$axis;
        }
    };
}

plane_elem!(bf16, Bf16s, Bf16sMut, Bf16);
plane_elem!(f16, F16s, F16sMut, F16);

/// The payload every operand mark carries on this plane: a BINDING HANDLE
/// and the element the statement claims for the bytes behind it.
///
/// `ArgValue::Buffer { handle, .. }` is what the driver resolves an operand
/// to and what `Encode::fire` writes into a descriptor set, so this is that
/// value wearing the element the point quantifies over. The rectangle is
/// NOT in here — `In`/`Out`/`InOut` carry `rows` and `width` beside the
/// payload, and every body below reads them off the mark.
///
/// # Why there is no offset
///
/// A cuda body cuts a packed row by advancing a pointer. A descriptor
/// binding has no such move: a `StructuredBuffer<T>` starts where the
/// allocation starts, and `slangc` indexes it from zero. Every point whose
/// operand is a PACKED row therefore needs either a windowed binding or a
/// base scalar in the shader's push block, and this plane has neither —
/// [`Staged::window`] is where that absence is stated once.
#[derive(Debug)]
pub struct Handle<T> {
    /// The descriptor index the driver minted for this operand.
    pub handle: u32,

    held: PhantomData<fn() -> T>,
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Handle<T> {}

impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle
    }
}
impl<T> Eq for Handle<T> {}

impl<T> Handle<T> {
    #[must_use]
    pub const fn new(handle: u32) -> Self {
        Self {
            handle,
            held: PhantomData,
        }
    }

    /// The same binding, re-elemented.
    ///
    /// NOT A CAST OF BYTES — a descriptor index names an allocation and
    /// nothing about its contents, so this changes only which
    /// `StructuredBuffer<..>` the shader will read it through. Every use
    /// below is one a `.slang` signature already forces: a `u32` word plane
    /// under a 4-bit bank, a `u8` exponent plane under mxfp4.
    #[must_use]
    pub const fn as_<U>(self) -> Handle<U> {
        Handle::new(self.handle)
    }
}

impl<T: Scalar> Elem for Handle<T> {
    // A HANDLE IS ITS OWN READ AND ITS OWN WRITE. `kernels::shader::Tensor`
    // says the same thing for the same reason: there is no address to
    // advance, so `In<Handle<T>>::ptr` is the handle itself and the two
    // `advance_*` moves are identities. That identity is exactly what makes
    // `Region::window` unusable on this plane — see [`Staged::window`].
    type Read = Self;
    type Write = Self;

    unsafe fn advance_read(read: Self::Read, _elems: usize) -> Self::Read {
        read
    }

    unsafe fn advance_write(write: Self::Write, _elems: usize) -> Self::Write {
        write
    }

    const CPP: &'static str = "";
    const CPP_CONST: &'static str = "";
    const CPP_MUT: &'static str = "";
    const TY_CONST: Ty = <T as Elem>::TY_CONST;
    const TY_MUT: Ty = <T as Elem>::TY_MUT;
}

impl<T: Scalar> ConstRun for Handle<T> {
    const RUN: Claim = Claim::Weight;
    const TY: Ty = <T as Elem>::TY_CONST;
    type Held = Self;
}

/// What a `Const<Self::Bank<R>>` slot carries on this plane: one descriptor
/// per byte plane of a quantised bank. Unlike [`Bank`] — the `Staged` view a
/// body PULLS when a declaration cannot spell the triple — nothing here is
/// optional or runtime-numbered: the group and bit width are `R::FORM`'s
/// facts, and the planes a repr stores are exactly `Repr::PLANES` fields.
pub struct Planes<R> {
    /// The packed codes, read as `StructuredBuffer<uint>`.
    pub codes: Handle<u32>,

    /// The block-scale bytes: mxfp4's shared exponents, one per 32 codes.
    pub scales: Handle<u8>,

    held: PhantomData<fn() -> R>,
}

impl<R> Clone for Planes<R> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<R> Copy for Planes<R> {}

impl<R: kernels::points::Repr> ConstRun for Planes<R> {
    const RUN: Claim = Claim::Weight;
    const TY: Ty = Ty::U8s;
    type Held = Self;
}

impl<V: ShaderValue, T> kernels::routine::Bind<V> for Handle<T> {
    fn arg(self) -> V {
        V::buffer(self.handle)
    }
}

impl<V: ShaderValue, T> kernels::routine::BindMut<V> for Handle<T> {
    fn arg_mut(self) -> V {
        V::buffer_mut(self.handle)
    }
}

/// What this plane is to a `points` declaration.
///
/// The two pool associated types are the raises [`crate::views`] declares —
/// `Struct<KvCache>` and `Struct<RecurrentState>` — so a `Cache` slot lands
/// on the exact view the encoder already builds. Nothing new is invented
/// here; what the views are MISSING for a claim body to use them is stated
/// on [`Staged`] and in each family's own doc.
impl kernels::points::Plane for Ctx<'_> {
    type Tensor<T: Scalar> = Handle<T>;

    type Bank<R: kernels::points::Repr> = Planes<R>;

    type Recurrent = kernels::raises::Struct<crate::views::RecurrentState>;

    type Pages = kernels::raises::Struct<crate::views::KvCache>;
}

/// The element gate every claim body opens with.
///
/// Cuda's equivalent compares `T::CPP` because a JIT template argument IS
/// the element; here `CPP` is empty on every plane element, so the identity
/// is compared instead. The refusal names the POINT, not the element, which
/// is the same shape `points_dispatch`'s `_ =>` arm answers with one call
/// further out.
///
/// # Errors
///
/// [`Refusal::Absent`] naming `what` when `T` is not this plane's `bf16`.
pub fn at_bf16<T: Scalar>(what: &'static str) -> Result<(), Refusal> {
    if core::any::TypeId::of::<T>() == core::any::TypeId::of::<bf16>() {
        Ok(())
    } else {
        Err(Refusal::Absent { what })
    }
}

/// A stated `u32` as the `i32` every push block on this plane spells.
///
/// # Errors
///
/// [`Refusal::Wide`] when the statement's number does not fit the shader's.
pub fn stated(what: &'static str, v: u32) -> Result<i32, Refusal> {
    i32::try_from(v).map_err(|_| Refusal::Wide {
        what,
        at: i64::from(v),
        max: i64::from(i32::MAX),
    })
}

/// How many `each`-wide slices divide `row`.
///
/// The one derivation every per-head point on this plane makes: a head
/// COUNT is never stated (the declarations say so — "an operand's rectangle
/// is the ROW, so a head count follows from the row and the stated head
/// width") and every shader here grids by it.
///
/// # Errors
///
/// [`Refusal::Empty`] for a zero divisor, [`Refusal::Narrow`] when the row
/// is not a whole number of slices.
pub fn heads(what: &'static str, row: i32, each: i32) -> Result<i32, Refusal> {
    if each <= 0 {
        return Err(Refusal::Empty { what });
    }
    if row <= 0 || row % each != 0 {
        return Err(Refusal::Narrow {
            what,
            at: i64::from(row),
        });
    }
    Ok(row / each)
}

/// The paged pool's head geometry: `(kv_heads, head_dim)`.
///
/// SEAM, AND IT IS A VIEW FIELD RATHER THAN A DRIVER METHOD. Every paged
/// attention entrypoint on this plane grids by the KEY head count and
/// addresses pages by the head width, and neither number is anywhere a
/// claim body can reach:
///
/// * `attention.decode`, `attention.masked` and `attention.kv_append`
///   state neither — only `attention.prefill` states `kv_heads`, and no
///   point states the pool's head width at all.
/// * A `q` row is `q_heads * head_dim` wide, so ONE of the two would
///   settle the other; the row alone settles neither.
/// * [`crate::views::PagedKvView`] carries `page_size`, `seq_stride` and
///   `head_stride`, and `driver-vulkan`'s `views::kv` answers the two
///   strides as ZERO on a paged fire — its own header says so — so the
///   geometry cannot be divided back out of them either.
///
/// This is the same growth W7 made on cuda, where `PagedKvView` gained
/// `qo_indptr`, `row_valid` and `requests` so that a body claiming
/// `mla.kv_append` could read the fire off the pool row. The two fields
/// this plane needs next are `heads` and `head_dim`, filled by
/// `driver-vulkan`'s `views::kv` out of the layer's pool descriptor —
/// which is where both numbers already live.
///
/// # Errors
///
/// Always, today: [`Refusal::Unstated`] naming the two missing fields.
pub fn pool_heads(view: &crate::views::PagedKvView) -> Result<(i32, i32), Refusal> {
    let _ = view;
    Err(Refusal::Unstated {
        what: "the paged pool's `(kv_heads, head_dim)`: no point states both \
               and `PagedKvView` carries neither",
    })
}

/// The affine bank triple a quantised entrypoint on this plane reads.
///
/// SEAM — `Bank<R: Repr>` on baker's ledger (`.wiki/baker-todo.md`:
/// "`moe.matmul_select_bias` — grouped mxfp4+bias gemm + `Bank<R: Repr>`
/// floor type"). A `Const<Handle<T>>` carries ONE descriptor index. Every
/// `qmm_t`, `qmv`, `embed_gather` and routed matmul in this crate reads
/// THREE planes — the packed words, the per-group scales, the per-group
/// biases — plus the `(group, bits)` pair that says how to unpack them, and
/// mxfp4 reads a `u8` exponent plane instead of the scale/bias pair. That
/// is not staging a body can derive: it is what the LOAD contract bound to
/// the weight's name, and a single `Const` slot cannot spell it.
#[derive(Debug, Clone, Copy)]
pub struct Bank<T> {
    /// The packed 4- or 8-bit words, read as `StructuredBuffer<uint>`.
    pub words: Handle<u32>,
    /// The per-group scale plane, at the activation element.
    pub scales: Handle<T>,
    /// The per-group bias plane. Absent under mxfp4, where
    /// [`Self::exponents`] carries the scale instead.
    pub biases: Handle<T>,
    /// mxfp4's shared-exponent plane, when the bank is mxfp4.
    pub exponents: Option<Handle<u8>>,
    /// Elements per quantisation group: 32, 64 or 128.
    pub group: i32,
    /// Bits per weight: 4 or 8.
    pub bits: i32,
}

/// What a claim body pulls off `self` because no declaration carries it.
///
/// `.wiki/baker.md`: "Plane staging (fa2 plan residents, host mirrors) never
/// appears in a declaration — the body pulls it from `self`." On cuda `self`
/// is a struct with a stream, a scratch arena and a `Ctx::scratch` door. On
/// this plane `self` is `dyn Encode`, whose whole surface is
/// `fire`/`resolve`/`best` — and `resolve` reaches
/// `driver_vulkan::bind::one`, which answers HANDLES AND SCALARS BY COLUMN
/// and nothing else.
///
/// So every method below is a door the driver does not have yet, stated
/// once here instead of five times in the family bodies. Each returns a
/// refusal that NAMES the missing door; the bodies that call them are whole
/// and will fire the moment the door opens. This is the P5 work list.
pub trait Staged {
    /// A tier-1 runtime stream this fire staged — `"positions"`,
    /// `"request_of_token"`, `"qo_indptr"`, `"row_valid"`
    /// (`kernels::runtime::TIER1`).
    ///
    /// SEAM. The LOWERED path splices one into the statement's input
    /// column, so `bind::one` answers it as `Source::Slot(Kind::In, n)`. A
    /// point declares no such column — `attention.decode` states `q`, the
    /// pool row and three scalars — so a claim body has to ask by NAME, and
    /// `Encode` has no method that takes one.
    ///
    /// What P5 needs: `Encode::stream(&self, name: &'static str)`, answered
    /// out of the same `Facts` the columned path already reads.
    ///
    /// # Errors
    ///
    /// Always, today: [`Refusal::Unstated`] naming the door.
    fn stream<T: Scalar>(&self, name: &'static str) -> Result<Handle<T>, Refusal>;

    /// A named, grow-on-demand device slab — cuda's `Ctx::scratch`.
    ///
    /// SEAM. Three families here need one and none can be claimed without
    /// it: `moe.matmul_select` (the permutation, the row/tile expert maps,
    /// the inverse map, the gathered rows), `moe.weighted_sum` (the inverse
    /// map the sorted combine reads), and any packed-row cut that has to
    /// materialise its halves.
    ///
    /// What P5 needs: `Encode::scratch(&self, name, bytes)` minting a
    /// handle into the driver's own arena, with `Ctx::scratch`'s per-process
    /// lifetime rule — alive between the launches of ONE body, since the
    /// launches are ordered on one queue.
    ///
    /// # Errors
    ///
    /// Always, today: [`Refusal::Unstated`] naming the door.
    fn scratch<T: Scalar>(&self, name: &'static str, elements: i64) -> Result<Handle<T>, Refusal>;

    /// A sub-range of a binding: `of`'s elements from `at`, `width` wide.
    ///
    /// SEAM, AND THE ONE THAT IS NOT A DRIVER METHOD. Cuda cuts a packed
    /// `[gate | up]` row by advancing a pointer; a descriptor binding has
    /// no base, so `mlp.swiglu`, `mlp.geglu_tanh_packed` and
    /// `mlp.swiglu_clamp_alpha` — every point whose operand is ONE packed
    /// row — cannot be launched by binding that row twice. `gated.slang`
    /// proves it: the strided arm reads `gate[m * gate_pitch + k]` and
    /// `up[m * up_pitch + k]`, two pitches over two bindings that both
    /// start at zero.
    ///
    /// Two honest answers and P5 picks one: a `VkDescriptorBufferInfo`
    /// offset behind a driver-minted sub-handle (this method), or a `base`
    /// word in each packed shader's push block (shader work, and it changes
    /// an ABI `kernels-metal` shares).
    ///
    /// # Errors
    ///
    /// Always, today: [`Refusal::Unstated`] naming the door.
    fn window<T: Scalar>(&self, of: Handle<T>, at: i64, width: i32) -> Result<Handle<T>, Refusal>;

    /// A resident view this fire staged that no `Cache` slot names.
    ///
    /// SEAM. `Cache<Self::Pages>` and `Cache<Self::Recurrent>` are bound by
    /// the dispatch, so those two arrive. `AttnMask` and `AttnSplit` do
    /// not: they are this plane's own staging, and `driver-vulkan`'s
    /// `views::raise` builds one only for a `Source::Slot(Kind::In, n)` —
    /// a POSITION in a routine's operand column. A claim body has no
    /// column, so it must ask by key.
    ///
    /// What P5 needs: `Encode::resident(&self, key: &'static str)`
    /// answering `ArgValue::Raised`, with `views::raise`'s match keyed on
    /// the NAME instead of on the slot the name was found at.
    ///
    /// # Errors
    ///
    /// Always, today: [`Refusal::Unstated`] naming the door.
    fn resident<R: kernels::raises::Raise>(&self) -> Result<*const R::Value, Refusal>;

    /// The three planes and two numbers behind a quantised weight.
    ///
    /// SEAM — see [`Bank`]. Every matmul this plane owns is quantised, so
    /// the whole `Gemm` family and both `moe.matmul_select*` points wait on
    /// this one.
    ///
    /// # Errors
    ///
    /// Always, today: [`Refusal::Unstated`] naming the missing floor type.
    fn bank<T: Scalar>(&self, of: kernels::routine::Const<Handle<T>>) -> Result<Bank<T>, Refusal>;
}

impl Staged for Ctx<'_> {
    fn stream<T: Scalar>(&self, name: &'static str) -> Result<Handle<T>, Refusal> {
        // SEAM: `Encode` has no by-name door. `Encode::resolve` reaches
        // `driver_vulkan::bind::one`, which reads `Source::Slot(Kind::In, n)`
        // — a column index a point does not have.
        let _ = name;
        Err(Refusal::Unstated {
            what: "a tier-1 runtime stream, asked for by name: `Encode` \
                   resolves an operand by COLUMN and a claim body has no column",
        })
    }

    fn scratch<T: Scalar>(&self, name: &'static str, elements: i64) -> Result<Handle<T>, Refusal> {
        // SEAM: no arena door on `Encode`. `driver-vulkan`'s `Handles` mints
        // a handle per bound operand and per `ctx.params()` block; nothing
        // mints one for a body that wants a slab of its own.
        let _ = (name, elements);
        Err(Refusal::Unstated {
            what: "a named device scratch slab: this plane's `Encode` has no \
                   arena door, where cuda's `Ctx::scratch` is one",
        })
    }

    fn window<T: Scalar>(&self, of: Handle<T>, at: i64, width: i32) -> Result<Handle<T>, Refusal> {
        // SEAM: a descriptor binding has no base. See the trait's doc for
        // the two answers P5 may pick between.
        let _ = (of, at, width);
        Err(Refusal::Unstated {
            what: "a windowed binding: a descriptor names a whole allocation, \
                   so a packed row's second half is not addressable here",
        })
    }

    fn resident<R: kernels::raises::Raise>(&self) -> Result<*const R::Value, Refusal> {
        // SEAM: `views::raise` keys on the slot a raise was found at, not on
        // the raise's own `Raise::KEY`.
        Err(Refusal::Unstated {
            what: "a resident view, asked for by key: `views::raise` answers \
                   only a raise found at a routine's own input slot",
        })
    }

    fn bank<T: Scalar>(&self, of: kernels::routine::Const<Handle<T>>) -> Result<Bank<T>, Refusal> {
        // SEAM: the floor's `Const<Tensor<T>>` carries one address. See
        // [`Bank`].
        let _ = of;
        Err(Refusal::Unstated {
            what: "a quantised weight's scale and bias planes: the floor's \
                   `Const<Tensor<T>>` carries one address and every matmul \
                   on this plane reads three",
        })
    }
}
