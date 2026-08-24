//! What this plane is to a `kernels::points` declaration.
//!
//! `.wiki/baker.md`'s endpoint shape, brought to wgpu: a family method's
//! body IS the launcher, and there is nothing else left. The `#[routine]`
//! fns that stood beside these blocks were the legacy driver's by-name
//! reach; that driver left the workspace at R3 and the 104 rows went with
//! it. The impl blocks live in the family files — `norm.rs`, `mlp.rs`,
//! `rope.rs`, `moe.rs`, `layout.rs`, `attn.rs`, `ssm.rs`, `quant.rs` — so
//! `#[claims]` puts each `*_CLAIMS` table beside the shaders it is a claim
//! about. This file is the three things every one of them needs: the payload
//! a mark carries, the `Plane` impl that names it, and the families this
//! plane has NO shader for.
//!
//! # The Ctx is a trait object, and that is the first seam
//!
//! `kernels-cuda`'s `Ctx<'a>` is a struct with a stream, a module cache and
//! an env behind it, so `impl Norm for Ctx<'_>` is an ordinary inherent-ish
//! impl and a body pulls its staging off `self`. All three shader planes
//! spell it `dyn Encode + 'a` instead (`routine.rs`, and the same line in
//! `kernels-vulkan` and `kernels-metal`), because the ENCODER is what a
//! shader plane's fire needs and the driver owns it. Implementing a family
//! for a `dyn` type is legal and this crate already does it once —
//! `impl kernels::routine::Answers<Wgpu> for Ctx<'_>` — so every block here
//! reads `for Ctx<'_>` and means `for dyn Encode + '_`.
//!
//! **SEAM (P5, floor):** `kernels::bound::BoundOp` declares
//! `type Plane: Plane`, and an associated type is implicitly `Sized`. A
//! generated `dispatch(ctx: &Ctx<'p>, op: &B) where B: BoundOp<Plane = Ctx<'p>>`
//! therefore does not typecheck on ANY shader plane: `dyn Encode` is not
//! `Sized`. The fix is one bound — `type Plane: Plane + ?Sized` — and it is
//! the floor's to make, not this crate's. Everything below is written as if
//! it were already made.
//!
//! # Two files here have no family and are not backlog
//!
//! `sample.rs` (`argmax_logits`) and `ptir.rs` (`copy_logits_bf16`) carry no
//! claim block, and their absence from the census is not a gap.
//! `kernels::points` declares families for the FORWARD PASS; sampling and the
//! delivery tail are the driver's, fired outside the lowered plan on both
//! cuda and here (cuda's `sample.rs` has no `#[claims]` block either). If
//! either ever becomes a statement, it wants a family declared for it first.

use core::marker::PhantomData;

use kernels::routine::Refusal;
use kernels::shader::ShaderValue;

use crate::routine::{ArgValue, Ctx};

/// A wgpu buffer handle: the whole of what a mark carries on this plane.
///
/// cuda's marks carry an ADDRESS, which is why `kernels::points::Scalar`
/// demands `Elem<Read = *const Self>`. Nothing on this plane has an address:
/// a `Fire`'s argument list is handles and scalars, the driver turns each
/// handle into a bind-group entry, and the offset arithmetic a cuda body does
/// with pointer casts is done here by the SHADER, told the packing. That is
/// the same rule W10 landed on cuda — an executor hands a kernel dense
/// rectangles only — arrived at from the other side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Handle(pub u32);

impl<V: ShaderValue> kernels::Bind<V> for Handle {
    fn arg(self) -> V {
        V::buffer(self.0)
    }
}

impl<V: ShaderValue> kernels::BindMut<V> for Handle {
    fn arg_mut(self) -> V {
        V::buffer_mut(self.0)
    }
}

/// `Plane::Tensor<T>` for wgpu: the handle, plus the element the point
/// quantifies over.
///
/// # Why this is not `kernels::shader::Tensor<E>`
///
/// `kernels::shader::Tensor<E>` is keyed on `shader::Element`, which is a
/// SPELLING — what a WGSL binding says it holds. A point is keyed on
/// `points::Scalar`, which is an AXIS — what a dispatch matches over. The two
/// sets overlap but neither contains the other (`Element` has no `i64`;
/// `Scalar` has no notion of a WGSL type), and a generic associated type
/// bounded by one cannot be written in terms of the other without a
/// type-level map the floor does not have. So the points layer gets its own
/// carrier and the view structs keep the shader spelling, and the two meet
/// where a body binds a view's field beside its own operands — both hand the
/// same `ArgValue::Buffer` to the same fire.
#[derive(Debug)]
pub struct Payload<T>(PhantomData<T>);

impl<T> Clone for Payload<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Payload<T> {}

impl<T: kernels::Elem> kernels::Elem for Payload<T> {
    type Read = Handle;
    type Write = Handle;

    /// A HANDLE DOES NOT ADVANCE. On cuda this is `ptr.add(elems)` and it is
    /// how a body cuts a packed row; here the cut is the shader's, told the
    /// packing — see [`Handle`].
    unsafe fn advance_read(read: Self::Read, _elems: usize) -> Self::Read {
        read
    }

    unsafe fn advance_write(write: Self::Write, _elems: usize) -> Self::Write {
        write
    }

    const CPP: &'static str = "";
    const CPP_CONST: &'static str = "";
    const CPP_MUT: &'static str = "";
    const TY_CONST: kernels::Ty = <T as kernels::Elem>::TY_CONST;
    const TY_MUT: kernels::Ty = <T as kernels::Elem>::TY_MUT;
}

impl<T: kernels::Elem> kernels::ConstRun for Payload<T> {
    const RUN: kernels::routine::Claim = kernels::routine::Claim::Weight;
    const TY: kernels::Ty = <T as kernels::Elem>::TY_CONST;
    type Held = Handle;
}

/// Two buffer bindings where [`Payload`] answers one: a quantised bank's
/// packed codes and its block scales. Named fields and not an array — the
/// two planes are indexed at different strides, so an off-by-one between
/// them is silently a wrong number rather than a fault.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BankHandles {
    /// The packed codes plane.
    pub codes: Handle,

    /// The block-scale plane.
    pub scales: Handle,
}

/// The mark a `Const<Self::Bank<R>>` slot carries on this plane.
#[derive(Debug)]
pub struct Bank<R>(PhantomData<R>);

impl<R> Clone for Bank<R> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<R> Copy for Bank<R> {}

impl<R: kernels::points::Repr> kernels::ConstRun for Bank<R> {
    const RUN: kernels::routine::Claim = kernels::routine::Claim::Weight;
    const TY: kernels::Ty = kernels::Ty::U8s;
    type Held = BankHandles;
}

/// Every shader in this crate is instantiated at bf16 and nowhere else.
///
/// The entrypoint tables say so — `rms_single_row_bfloat16`,
/// `sdpa_paged_decode_bfloat16_d_128`, `affine_qmm_t_bfloat16_gs_64_b_4_…` —
/// and the `//#include "common/bf16.inc.wgsl"` under each is a pair of
/// bf16 halves unpacked out of a `u32` word, because WGSL has no 16-bit
/// storage type at all. A point quantifies over `Scalar`, so a claim states
/// the pin as a REFUSAL BY NAME rather than widening it with a cast no
/// shader stands behind. cuda's `ssm::at_bf16` is the precedent and the
/// reasoning is the same one, at the whole-plane scale rather than one
/// family's.
///
/// A second element wants a second `pie:instantiate` line, not a cast here.
pub fn at_bf16<T: kernels::points::Scalar>(what: &'static str) -> Result<(), Refusal> {
    if matches!(<T as kernels::Elem>::TY_CONST, kernels::Ty::Bf16s) {
        Ok(())
    } else {
        Err(Refusal::Absent { what })
    }
}

/// The absent buffer, as the sdpa arms bind it.
///
/// `Asks::absent` resolves `Ty::Buf` against `Lit::Null`, which the driver
/// answers with whatever it uses for an unbound slot. Named here because six
/// attention bodies want it and `self.absent()` needs the backend inferred.
pub(crate) fn absent(ctx: &Ctx<'_>) -> Result<ArgValue, Refusal> {
    kernels::routine::Asks::<crate::routine::Wgpu>::absent(ctx)
}

/// What a mark carries on this plane, as the declaration floor asks it.
///
/// * `Tensor<T>` — a binding handle plus the axis ([`Payload`]).
/// * `Recurrent` — the GDN/mamba slab pair, [`crate::views::RecurrentView`]
///   unchanged.
/// * `Pages` — [`crate::views::AttnFire`], which is `PagedKvView` PLUS the
///   per-fire staging every sdpa arm reads. See that type for the seam.
impl kernels::points::Plane for Ctx<'_> {
    type Tensor<T: kernels::points::Scalar> = Payload<T>;

    type Bank<R: kernels::points::Repr> = Bank<R>;

    type Recurrent = kernels::raises::Struct<crate::views::RecurrentState>;

    type Pages = kernels::raises::Struct<crate::views::AttnFire>;
}

/// `Dist` — no shader, and there is no draft of one.
///
/// The family is one point, `dist.all_reduce`, and it is a MULTI-DEVICE
/// statement: cuda claims it through `comm::Plane`, an NCCL communicator the
/// driver builds per rank. WebGPU has no device-to-device path at all — a
/// wgpu shell owns one adapter and a browser tab owns exactly one — so this
/// is not a shader somebody has not written, it is a point this plane cannot
/// hold. The block is empty ON PURPOSE: it puts `DIST_CLAIMS` in the table
/// as the empty list, so a lane that states an all-reduce refuses with the
/// point named instead of resolving to nothing.
#[kernels_macros::claims]
impl kernels::points::Dist for Ctx<'_> {}

/// `Mla` — eleven points, no shader.
///
/// dsv4/kimi/glm's latent attention is cuda-only today: `attn/mla*.cuh`,
/// `attn::plan::mla`, and the fa2 schedule the W7 note describes. There is no
/// `.wgsl` under `kernels/attn/` that reads a latent page, so all eleven are
/// measured backlog rows.
#[kernels_macros::claims]
impl kernels::points::Mla for Ctx<'_> {}

/// `Index` — glm's DSA indexer, no shader.
///
/// `index.topk` has no ground truth on ANY plane yet (baker-todo names it as
/// design-needed), and the other three ride the same latent pages `Mla`
/// does.
#[kernels_macros::claims]
impl kernels::points::Index for Ctx<'_> {}

/// `Pool` — the compressed-entry pool, no shader.
///
/// cuda's block is empty too (`kernels-cuda/src/attn/mod.rs`), so this is
/// the one family where wgpu is not behind.
#[kernels_macros::claims]
impl kernels::points::Pool for Ctx<'_> {}

/// `Hc` — gemma's hyper-connections, no shader.
///
/// Five points, all claimed on cuda out of `norm.rs`'s `Hc` block, none of
/// which has a WGSL twin: the expand/gates/fold trio is a per-stream mixer
/// with an f32 sinkhorn, and nothing in this tree's shaders reads a stream
/// axis at all.
#[kernels_macros::claims]
impl kernels::points::Hc for Ctx<'_> {}

/// The bf16 axis, re-exported so a generated dispatch can name it without
/// reaching into `kernels::shader`.
pub use kernels::shader::bf16;
