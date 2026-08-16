//! The MLP family: the gated activations between the two dense projections.
//!
//! [`swiglu`], [`swiglu_clamp`], [`situ`], [`geglu_tanh`] and [`gpt_oss_glu`]
//! take `gate` and `up` as two regions; the `chunked_*` forms take one packed
//! `rows * 2 * width` region and halve it themselves. Extents come off the
//! bound `In`/`Out` regions' `rows`/`width`, never a separate parameter.

#![allow(clippy::too_many_arguments)]

use crate::jit::{Ctx, Family, Launch, Routine};
use crate::driver_internal::sigmoid_gate_inplace_bf16;
use crate::{routine};
use crate::jit::Abi;
use crate::jit::abi::Inst;
use crate::jit::abi::{bf16, f16};
use kernels::Env;
use kernels::In;
use kernels::Out;
use kernels::Refusal;
use kernels::Weight;
use kernels::keys;

/// Threads per block, everywhere in this family.
const BLOCK: u32 = 256;

/// Threads per warp — the unit the reductions' shared scratch is counted in.
const WARP: u32 = 32;

// Widths reaching only `Ctx::launch`'s grid need no guard, since it refuses a
// zero grid itself. `sigmoid_dot_scalar_gate_add`'s `h` and `gaussian_topk`'s
// `dim` reach a kernel argument instead, so those two guard by hand.

/// `LaunchRule::Elementwise`, as the expression it evaluates to.
#[must_use]
const fn elementwise(n: i32) -> Launch {
    Launch::flat(n.unsigned_abs(), BLOCK)
}

/// `LaunchRule::ElementwiseRows`, as the expression it evaluates to.
#[must_use]
const fn elementwise_rows(rows: i32, width: i32) -> Launch {
    Launch::grid([rows.unsigned_abs(), width.unsigned_abs().div_ceil(BLOCK), 1], [BLOCK, 1, 1])
}

/// `LaunchRule::Rms`, as the expression it evaluates to.
#[must_use]
const fn rms(rows: i32) -> Launch {
    /// The dynamic shared memory the two reducing kernels fold through.
    const RMS_SMEM: u32 = (BLOCK / WARP) * 4;

    Launch::per_row(rows.unsigned_abs(), BLOCK).smem(RMS_SMEM)
}

/// gpt-oss's GLU alpha, used as a fixed constant since no deployment fact
/// carries it yet.
pub const GPT_OSS_GLU_ALPHA: f32 = 1.702;

/// `y[i] = silu(gate[i]) * up[i]` over the result — `mlp::swiglu_bf16`.
///
/// # Safety
///
/// `gate` and `up` must address `y.rows * y.width` live bf16 elements and
/// `y` that many writable ones.
#[kernels_macros::routine]
pub fn swiglu<T>(
    ctx: &Ctx,
    gate: In<0, T>,
    up: In<1, T>,
    y: Out<0, T>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    let n = y.rows.saturating_mul(y.width);
    // SAFETY: `call()`'s contract — every pointer addresses live device
    // memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "mlp/swiglu.cuh",
            &format!("::pie::mlp::swiglu<{}>", T::CPP),
            elementwise(n),
            &[gate.ptr.arg(), up.ptr.arg(), y.ptr.arg(), n.arg()],
        )
    }
}

/// The same with the gate clamped to `±limit` — `mlp::swiglu_clamp_bf16`.
///
/// # Safety
///
/// [`swiglu`]'s.
#[kernels_macros::routine]
pub fn swiglu_clamp<T>(
    ctx: &Ctx,
    gate: In<0, T>,
    up: In<1, T>,
    y: Out<0, T>,
    limit: Env<keys::GluLimit>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    let n = y.rows.saturating_mul(y.width);
    unsafe {
        ctx.launch(
            "mlp/swiglu.cuh",
            &format!("::pie::mlp::swiglu_clamp<{}>", T::CPP),
            elementwise(n),
            &[gate.ptr.arg(), up.ptr.arg(), y.ptr.arg(), n.arg(), limit.arg()],
        )
    }
}

/// SiTU — `mlp::situ_bf16`.
///
/// # Safety
///
/// [`swiglu`]'s.
#[kernels_macros::routine]
pub fn situ<T>(
    ctx: &Ctx,
    gate: In<0, T>,
    up: In<1, T>,
    y: Out<0, T>,
    // Unbound: `Deployment` states neither field yet, so both are always 0.0.
    beta: f32,
    linear_beta: f32,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    let n = y.rows.saturating_mul(y.width);
    unsafe {
        ctx.launch(
            "mlp/swiglu.cuh",
            &format!("::pie::mlp::situ<{}>", T::CPP),
            elementwise(n),
            &[gate.ptr.arg(), up.ptr.arg(), y.ptr.arg(), n.arg(), beta.arg(), linear_beta.arg()],
        )
    }
}

/// GeGLU-tanh — `mlp::geglu_tanh_bf16`.
///
/// # Safety
///
/// [`swiglu`]'s. `y` may alias `gate`.
#[kernels_macros::routine]
pub fn geglu_tanh<T>(
    ctx: &Ctx,
    // Stated explicitly, not derived: `y` aliases `gate` (`in_place =
    // &[(0, 0)]`), so counting operands would miscount which is which.
    gate: In<0, T>,
    up: In<1, T>,
    y: Out<0, T>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    let n = y.rows.saturating_mul(y.width);
    unsafe {
        ctx.launch(
            "mlp/swiglu.cuh",
            &format!("::pie::mlp::geglu_tanh<{}>", T::CPP),
            elementwise(n),
            &[gate.ptr.arg(), up.ptr.arg(), y.ptr.arg(), n.arg()],
        )
    }
}

/// `y = max(x, 0)^2` — `mlp::relu2_bf16`.
///
/// # Safety
///
/// `x` must address `y.rows * y.width` live bf16 elements and `y` that many
/// writable ones.
#[kernels_macros::routine]
pub fn relu2<T>(ctx: &Ctx, x: In<0, T>, y: Out<0, T>) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    let n = y.rows.saturating_mul(y.width);
    unsafe {
        ctx.launch(
            "mlp/swiglu.cuh",
            &format!("::pie::mlp::relu2<{}>", T::CPP),
            elementwise(n),
            &[x.ptr.arg(), y.ptr.arg(), n.arg()],
        )
    }
}

/// gpt-oss's clamped GLU — `mlp::gpt_oss_glu_bf16`.
///
/// # Safety
///
/// [`swiglu`]'s, plus: when `y_fp16` is `Some`, it must address
/// `y.rows * y.width` writable fp16 elements. `y` may alias `gate`.
#[kernels_macros::routine]
pub fn gpt_oss_glu<T>(
    ctx: &Ctx,
    gate: In<0, T>,
    up: In<1, T>,
    y: Out<0, T>,
    limit: Env<keys::GluLimit>,
    // `quant`'s fused MXFP4 gate/up computes the same activation from this
    // fact independently; nothing keeps the two numerically in sync.
    alpha: Env<keys::GluAlpha>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    let n = y.rows.saturating_mul(y.width);
    unsafe {
        ctx.launch(
            "mlp/swiglu.cuh",
            &format!("::pie::mlp::gpt_oss_glu<{}>", T::CPP),
            elementwise(n),
            &[
                gate.ptr.arg(),
                up.ptr.arg(),
                y.ptr.arg(),
                // Always null: this fp16 side-output slot is never filled.
                ::core::ptr::null_mut::<f16>().arg(),
                n.arg(),
                limit.arg(),
                alpha.arg(),
            ],
        )
    }
}

/// SwiGLU over a packed gate‖up bank — `mlp::chunked_swiglu_bf16`.
///
/// # Safety
///
/// `packed` must address `y.rows * 2 * y.width` live bf16 elements and `y`
/// `y.rows * y.width` writable ones. `y` may alias the second half of
/// `packed`, which is what `in_place = &[(0, 1)]` declares.
#[kernels_macros::routine]
pub fn chunked_swiglu<T>(
    ctx: &Ctx,
    packed: In<0, T>,
    // Extent comes off `y` (`rows * width`), not `packed` (`rows * 2 * width`).
    y: Out<0, T>,
    // The C++'s `chunked_swiglu_gate_second` twin has no `ROUTINES` row, so it
    // never fires; the checkpoint-side `gate_second` flag shares the name only.
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    unsafe {
        ctx.launch(
            "mlp/swiglu.cuh",
            &format!("::pie::mlp::chunked_swiglu<{}>", T::CPP),
            elementwise_rows(y.rows, y.width),
            &[packed.ptr.arg(), y.ptr.arg(), y.width.arg()],
        )
    }
}

/// The packed form with the gate clamped —
///
/// # Safety
///
/// [`chunked_swiglu`]'s.
#[kernels_macros::routine]
pub fn chunked_swiglu_clamp<T>(
    ctx: &Ctx,
    packed: In<0, T>,
    y: Out<0, T>,
    limit: Env<keys::GluLimit>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    unsafe {
        ctx.launch(
            "mlp/swiglu.cuh",
            &format!("::pie::mlp::chunked_swiglu_clamp<{}>", T::CPP),
            elementwise_rows(y.rows, y.width),
            &[packed.ptr.arg(), y.ptr.arg(), y.width.arg(), limit.arg()],
        )
    }
}

/// SiTU over a packed bank — `mlp::chunked_situ_bf16`.
///
/// # Safety
///
/// [`chunked_swiglu`]'s.
#[kernels_macros::routine]
pub fn chunked_situ<T>(
    ctx: &Ctx,
    packed: In<0, T>,
    y: Out<0, T>,
    // [`situ`]'s `beta`/`linear_beta`, for the same reason (unbound).
    beta: f32,
    linear_beta: f32,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    unsafe {
        ctx.launch(
            "mlp/swiglu.cuh",
            &format!("::pie::mlp::chunked_situ<{}>", T::CPP),
            elementwise_rows(y.rows, y.width),
            &[packed.ptr.arg(), y.ptr.arg(), y.width.arg(), beta.arg(), linear_beta.arg()],
        )
    }
}

/// GeGLU-tanh over a packed bank — `mlp::chunked_geglu_tanh_bf16`.
///
/// # Safety
///
/// [`chunked_swiglu`]'s.
#[kernels_macros::routine]
pub fn chunked_geglu_tanh<T>(
    ctx: &Ctx,
    packed: In<0, T>,
    y: Out<0, T>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    unsafe {
        ctx.launch(
            "mlp/swiglu.cuh",
            &format!("::pie::mlp::chunked_geglu_tanh<{}>", T::CPP),
            elementwise_rows(y.rows, y.width),
            &[packed.ptr.arg(), y.ptr.arg(), y.width.arg()],
        )
    }
}

/// `out += y * sigmoid(x · gate_w)` — `mlp::sigmoid_dot_scalar_gate_add_bf16`.
///
/// # Safety
///
/// `x`, `y` and `out` must each address `out.rows * out.width` live bf16
/// elements — `out` writable, and it is the residual stream the statement
/// takes as its second operand, which is what `in_place = &[(0, 1)]`
/// declares. `gate_w` must address `out.width` live bf16 elements.
#[kernels_macros::routine]
pub fn sigmoid_dot_scalar_gate_add<T>(
    ctx: &Ctx,
    x: In<0, T>,
    // The statement's only weight, so `Weight<0, *const T>` and a name lookup agree.
    gate_w: Weight<0, *const T>,
    out: Out<0, T>,
    // Stated as `In<2, _>`, not derived: `in_place = &[(0, 1)]` claims operand
    // 1 for the aliased `out`, so the true next statement operand is 2.
    y: In<2, T>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    // `h` is a pitch, not an extent: `swiglu.cuh` strides `x`, `out` and `y`
    // by the same number. `layout::stated` turns `all()`'s `Absent` into this
    // family's own `Empty { what: "the row width" }`.
    let row = crate::layout::stated(out.all("the row width"))?;
    let h = row.stride;
    unsafe {
        ctx.launch(
            "mlp/swiglu.cuh",
            &format!("::pie::mlp::sigmoid_dot_scalar_gate_add<{}>", T::CPP),
            rms(row.rows),
            &[x.ptr.arg(), gate_w.ptr.arg(), out.ptr.arg(), y.ptr.arg(), h.arg()],
        )
    }
}

/// AltUp's activation sparsity, in place — `mlp::gaussian_topk_bf16`.
///
/// # Safety
///
/// `x` must address `x.rows * x.width` live and writable bf16 elements.
#[kernels_macros::routine]
pub fn gaussian_topk<T>(
    ctx: &Ctx,
    // In place (`in_place = &[(0, 0)]`): also `In(0)`, but `Out<0, T>` matches
    // the launcher's own `*mut`.
    x: Out<0, T>,
    // Off the statement's `params`, not a driver table (`ParamF32(0)`, as
    // `swiglu_clamp`'s `limit` does). Hand-armed: `Facts` is `Copy` and can't
    // derive a variable-length `ParamF32` run.
    #[source(ParamF32(0))]
    std_multiplier: f32,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *mut T: Abi,
{
    // Same reason as `sigmoid_dot_scalar_gate_add`'s `h`: `dim` is both the
    // loop bound and the row stride in `gaussian_topk.cuh`, so one number
    // serves as both extent and pitch.
    let row = crate::layout::stated(x.all("the row width"))?;
    let dim = row.stride;
    unsafe {
        ctx.launch(
            "mlp/gaussian_topk.cuh",
            &format!("::pie::mlp::gaussian_topk<{}>", T::CPP),
            rms(row.rows),
            &[x.ptr.arg(), dim.arg(), std_multiplier.arg()],
        )
    }
}

// Pins each launcher's derived operand-binding metadata, so a change to
// `#[source(...)]` that alters a slot's binding is caught here, not at first use.
const _: () = {
    assert!(<sigmoid_dot_scalar_gate_add as ::kernels::Derivation>::DERIVED.len() == 4);
    assert!(matches!(<sigmoid_dot_scalar_gate_add as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(kernels::source_is_named(&<sigmoid_dot_scalar_gate_add as ::kernels::Derivation>::DERIVED[1].source, <kernels::keys::NamedWeight as kernels::keys::Fact>::KEY));
    assert!(matches!(<sigmoid_dot_scalar_gate_add as ::kernels::Derivation>::DERIVED[2].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(<sigmoid_dot_scalar_gate_add as ::kernels::Derivation>::DERIVED[3].source, Some(kernels::Source::Slot(kernels::Kind::In, 2))));

    assert!(<gpt_oss_glu as ::kernels::Derivation>::DERIVED.len() == 5);
    assert!(kernels::source_is_named(&<gpt_oss_glu as ::kernels::Derivation>::DERIVED[3].source, <kernels::keys::GluLimit as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&<gpt_oss_glu as ::kernels::Derivation>::DERIVED[4].source, <kernels::keys::GluAlpha as kernels::keys::Fact>::KEY));

    assert!(<chunked_swiglu as ::kernels::Derivation>::DERIVED.len() == 2);
    assert!(matches!(<chunked_swiglu as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(<chunked_swiglu as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));

    assert!(<chunked_geglu_tanh as ::kernels::Derivation>::DERIVED.len() == 2);
    assert!(matches!(<chunked_geglu_tanh as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(<chunked_geglu_tanh as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));

    // Index 2 (the beta pair's first half) is deliberately `None`: this
    // row stays refused rather than bound to a guessed source.
    assert!(<chunked_situ as ::kernels::Derivation>::DERIVED[2].source.is_none());

    // The counterpart: this one is bound, and off the statement rather
    // than off a fact, which is why it needs a hand arm.
    assert!(<gaussian_topk as ::kernels::Derivation>::DERIVED.len() == 2);
    assert!(matches!(<gaussian_topk as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(<gaussian_topk as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::ParamF32, 0))));

    assert!(<swiglu as ::kernels::Derivation>::DERIVED.len() == 3);
    assert!(matches!(<swiglu as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(<swiglu as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(<swiglu as ::kernels::Derivation>::DERIVED[2].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));

    assert!(<relu2 as ::kernels::Derivation>::DERIVED.len() == 2);
    assert!(matches!(<relu2 as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(<relu2 as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));

    assert!(<geglu_tanh as ::kernels::Derivation>::DERIVED.len() == 3);
    assert!(matches!(<geglu_tanh as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(<geglu_tanh as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(<geglu_tanh as ::kernels::Derivation>::DERIVED[2].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(<geglu_tanh as ::kernels::Derivation>::DERIVED[0].stated);
};

/// This family's routines, and what a trace may say about each.
///
/// `in_place = &[..]` states which operands share an address — the one
/// thing no signature captures.
pub static ROUTINES: &[Routine] = &[
    routine!(swiglu_bf16 = swiglu::<bf16>, ),
    routine!(swiglu_clamp_bf16 = swiglu_clamp::<bf16>, ),
    routine!(situ_bf16 = situ::<bf16>, ),
    routine!(geglu_tanh_bf16 = geglu_tanh::<bf16>, in_place = &[(0, 0)], ),
    routine!(relu2_bf16 = relu2::<bf16>, ),
    routine!(gpt_oss_glu_bf16 = gpt_oss_glu::<bf16>, in_place = &[(0, 0)], ),
    routine!(chunked_swiglu_bf16 = chunked_swiglu::<bf16>, in_place = &[(0, 1)], ),
    routine!(chunked_swiglu_clamp_bf16 = chunked_swiglu_clamp::<bf16>, ),
    routine!(chunked_situ_bf16 = chunked_situ::<bf16>, ),
    routine!(chunked_geglu_tanh_bf16 = chunked_geglu_tanh::<bf16>, ),
    routine!(sigmoid_dot_scalar_gate_add_bf16 = sigmoid_dot_scalar_gate_add::<bf16>, in_place = &[(0, 1)], ),
    routine!(gaussian_topk_bf16 = gaussian_topk::<bf16>, in_place = &[(0, 0)], ),
    // Bare identifier, imported above: the row lives here (not in
    // `driver_internal`) because `Family::symbol` needs the `mlp::` prefix.
    routine!(sigmoid_gate_inplace_bf16, in_place = &[(0, 0)]),
];

/// `mlp`, as a trace names it.
pub static FAMILY: Family = crate::family!(ROUTINES);

// Pins `std_multiplier`'s `#[source(ParamF32(0))]` binding, so "simplifying"
// the attribute away and silently returning the row to unbound is caught here.
const _: () = {
    let d = <gaussian_topk as kernels::Derivation>::DERIVED;
    assert!(d.len() == 2);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(d[1].source, Some(kernels::Source::Slot(kernels::Kind::ParamF32, 0))));
};
