#![allow(clippy::too_many_arguments)]

use crate::jit::{Ctx, Family, Launch, Routine};
// `driver_bound!` names its `fn` by IDENTIFIER, exactly as `routine!` does, so
// the one host program declared here that does not live in this file has to be
// nameable without its path.
use crate::driver_internal::sigmoid_gate_inplace_bf16;
use crate::{driver_bound, routine};
use crate::jit::Abi;
use crate::jit::abi::Elem;
use crate::jit::abi::{bf16, f16};
use kernels::Refusal;

use core::ptr::NonNull;

/// Threads per block, everywhere in this family.
const BLOCK: u32 = 256;

/// Threads per warp — the unit the reductions' shared scratch is counted in.
const WARP: u32 = 32;

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

/// gpt-oss's `alpha`, which was a defaulted argument of a header that no
pub const GPT_OSS_GLU_ALPHA: f32 = 1.702;

/// `y[i] = silu(gate[i]) * up[i]` over `n` elements — `mlp::swiglu_bf16`.
///
/// # Safety
///
/// `gate` and `up` must address `n` live bf16 elements and `y` `n` writable
/// ones.
pub fn swiglu<T>(
    ctx: &Ctx,
    gate: *const T,
    up: *const T,
    y: *mut T,
    n: i32,
) -> Result<(), Refusal>
where
    T: Elem,
    *const T: Abi,
    *mut T: Abi,
{
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "mlp/swiglu.cuh",
            &format!("::pie::mlp::swiglu<{}>", T::CPP),
            elementwise(n),
            &[gate.arg(), up.arg(), y.arg(), n.arg()],
        )
    }
}

/// The same with the gate clamped to `±limit` — `mlp::swiglu_clamp_bf16`.
///
/// # Safety
///
/// [`swiglu`]'s.
pub fn swiglu_clamp<T>(
    ctx: &Ctx,
    gate: *const T,
    up: *const T,
    y: *mut T,
    n: i32,
    limit: f32,
) -> Result<(), Refusal>
where
    T: Elem,
    *const T: Abi,
    *mut T: Abi,
{
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "mlp/swiglu.cuh",
            &format!("::pie::mlp::swiglu_clamp<{}>", T::CPP),
            elementwise(n),
            &[gate.arg(), up.arg(), y.arg(), n.arg(), limit.arg()],
        )
    }
}

/// SiTU — `mlp::situ_bf16`.
///
/// # Safety
///
/// [`swiglu`]'s.
pub fn situ<T>(
    ctx: &Ctx,
    gate: *const T,
    up: *const T,
    y: *mut T,
    n: i32,
    beta: f32,
    linear_beta: f32,
) -> Result<(), Refusal>
where
    T: Elem,
    *const T: Abi,
    *mut T: Abi,
{
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "mlp/swiglu.cuh",
            &format!("::pie::mlp::situ<{}>", T::CPP),
            elementwise(n),
            &[gate.arg(), up.arg(), y.arg(), n.arg(), beta.arg(), linear_beta.arg()],
        )
    }
}

/// GeGLU-tanh — `mlp::geglu_tanh_bf16`.
///
/// # Safety
///
/// [`swiglu`]'s. `y` may alias `gate`.
pub fn geglu_tanh<T>(
    ctx: &Ctx,
    gate: *const T,
    up: *const T,
    y: *mut T,
    n: i32,
) -> Result<(), Refusal>
where
    T: Elem,
    *const T: Abi,
    *mut T: Abi,
{
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "mlp/swiglu.cuh",
            &format!("::pie::mlp::geglu_tanh<{}>", T::CPP),
            elementwise(n),
            &[gate.arg(), up.arg(), y.arg(), n.arg()],
        )
    }
}

/// `y = max(x, 0)^2` — `mlp::relu2_bf16`.
///
/// # Safety
///
/// `x` must address `n` live bf16 elements and `y` `n` writable ones.
pub fn relu2<T>(ctx: &Ctx, x: *const T, y: *mut T, n: i32) -> Result<(), Refusal>
where
    T: Elem,
    *const T: Abi,
    *mut T: Abi,
{
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch("mlp/swiglu.cuh", &format!("::pie::mlp::relu2<{}>", T::CPP), elementwise(n), &[x.arg(), y.arg(), n.arg()])
    }
}

/// gpt-oss's clamped GLU — `mlp::gpt_oss_glu_bf16`.
///
/// # Safety
///
/// [`swiglu`]'s, plus: when `y_fp16` is `Some`, it must address `n`
/// writable **fp16** elements. `y` may alias `gate`.
pub fn gpt_oss_glu<T>(
    ctx: &Ctx,
    gate: *const T,
    up: *const T,
    y: *mut T,
    y_fp16: Option<NonNull<f16>>,
    n: i32,
    limit: f32,
    alpha: f32,
) -> Result<(), Refusal>
where
    T: Elem,
    *const T: Abi,
    *mut T: Abi,
{
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "mlp/swiglu.cuh",
            &format!("::pie::mlp::gpt_oss_glu<{}>", T::CPP),
            elementwise(n),
            &[gate.arg(), up.arg(), y.arg(), y_fp16.arg(), n.arg(), limit.arg(), alpha.arg()],
        )
    }
}

/// SwiGLU over a packed gate‖up bank — `mlp::chunked_swiglu_bf16`.
///
/// `gate_second` picks the INSTANTIATION and not an argument: which half of
/// the bank is the gate is a template parameter, so the two spellings are two
/// symbols and the branch is exclusive — one launch either way.
///
/// # Safety
///
/// `packed` must address `rows * 2 * i` live bf16 elements and `y`
/// `rows * i` writable ones. `y` may alias the second half of `packed`,
/// which is what `in_place = &[(0, 1)]` declares.
pub fn chunked_swiglu<T>(
    ctx: &Ctx,
    packed: *const T,
    y: *mut T,
    rows: i32,
    i: i32,
    gate_second: bool,
) -> Result<(), Refusal>
where
    T: Elem,
    *const T: Abi,
    *mut T: Abi,
{
    let instantiation = if gate_second {
        &format!("::pie::mlp::chunked_swiglu_gate_second<{}>", T::CPP)
    } else {
        &format!("::pie::mlp::chunked_swiglu<{}>", T::CPP)
    };
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "mlp/swiglu.cuh",
            instantiation,
            elementwise_rows(rows, i),
            &[packed.arg(), y.arg(), i.arg()],
        )
    }
}

/// The packed form with the gate clamped —
///
/// # Safety
///
/// [`chunked_swiglu`]'s.
pub fn chunked_swiglu_clamp<T>(
    ctx: &Ctx,
    packed: *const T,
    y: *mut T,
    rows: i32,
    i: i32,
    limit: f32,
) -> Result<(), Refusal>
where
    T: Elem,
    *const T: Abi,
    *mut T: Abi,
{
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "mlp/swiglu.cuh",
            &format!("::pie::mlp::chunked_swiglu_clamp<{}>", T::CPP),
            elementwise_rows(rows, i),
            &[packed.arg(), y.arg(), i.arg(), limit.arg()],
        )
    }
}

/// SiTU over a packed bank — `mlp::chunked_situ_bf16`.
///
/// # Safety
///
/// [`chunked_swiglu`]'s.
pub fn chunked_situ<T>(
    ctx: &Ctx,
    packed: *const T,
    y: *mut T,
    rows: i32,
    i: i32,
    beta: f32,
    linear_beta: f32,
    gate_second: bool,
) -> Result<(), Refusal>
where
    T: Elem,
    *const T: Abi,
    *mut T: Abi,
{
    let instantiation = if gate_second {
        &format!("::pie::mlp::chunked_situ_gate_second<{}>", T::CPP)
    } else {
        &format!("::pie::mlp::chunked_situ<{}>", T::CPP)
    };
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "mlp/swiglu.cuh",
            instantiation,
            elementwise_rows(rows, i),
            &[packed.arg(), y.arg(), i.arg(), beta.arg(), linear_beta.arg()],
        )
    }
}

/// GeGLU-tanh over a packed bank — `mlp::chunked_geglu_tanh_bf16`.
///
/// # Safety
///
/// [`chunked_swiglu`]'s.
pub fn chunked_geglu_tanh<T>(
    ctx: &Ctx,
    packed: *const T,
    y: *mut T,
    rows: i32,
    i: i32,
    gate_second: bool,
) -> Result<(), Refusal>
where
    T: Elem,
    *const T: Abi,
    *mut T: Abi,
{
    let instantiation = if gate_second {
        &format!("::pie::mlp::chunked_geglu_tanh_gate_second<{}>", T::CPP)
    } else {
        &format!("::pie::mlp::chunked_geglu_tanh<{}>", T::CPP)
    };
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "mlp/swiglu.cuh",
            instantiation,
            elementwise_rows(rows, i),
            &[packed.arg(), y.arg(), i.arg()],
        )
    }
}

/// `out += y * sigmoid(x · gate_w)` — `mlp::sigmoid_dot_scalar_gate_add_bf16`.
///
/// # Safety
///
/// `x`, `y` and `out` must each address `rows * h` live bf16 elements —
/// `out` writable, and it IS the residual stream the statement takes as its
/// second operand, which is what `in_place = &[(0, 1)]` declares. `gate_w`
/// must address `h` live bf16 elements.
pub fn sigmoid_dot_scalar_gate_add<T>(
    ctx: &Ctx,
    x: *const T,
    gate_w: *const T,
    out: *mut T,
    y: *const T,
    rows: i32,
    h: i32,
) -> Result<(), Refusal>
where
    T: Elem,
    *const T: Abi,
    *mut T: Abi,
{
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "mlp/swiglu.cuh",
            &format!("::pie::mlp::sigmoid_dot_scalar_gate_add<{}>", T::CPP),
            rms(rows),
            &[x.arg(), gate_w.arg(), out.arg(), y.arg(), h.arg()],
        )
    }
}

/// AltUp's activation sparsity, in place — `mlp::gaussian_topk_bf16`.
///
/// # Safety
///
/// `x` must address `rows * dim` live and writable bf16 elements.
pub fn gaussian_topk<T>(
    ctx: &Ctx,
    x: *mut T,
    rows: i32,
    dim: i32,
    std_multiplier: f32,
) -> Result<(), Refusal>
where
    T: Elem,
    *mut T: Abi,
{
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "mlp/gaussian_topk.cuh",
            &format!("::pie::mlp::gaussian_topk<{}>", T::CPP),
            rms(rows),
            &[x.arg(), dim.arg(), std_multiplier.arg()],
        )
    }
}

/// This family's routines, and what a trace may say about each.
///
/// The argument lists are DERIVED from the `fn`s above -- `routine!` sees only
/// the identifier. What is stated here is what no signature carries: which
/// operands must be given the same address. Nothing in `mlp` is `whole` and
/// nothing takes part in the depth-prefix plan.
///
/// The last line is `driver_bound!` and its host program is not above it —
/// see [`crate::driver_internal`]'s header, which is where that `fn` lives
/// and why it stays there.
pub static ROUTINES: &[Routine] = &[
    routine!(swiglu_bf16 = swiglu::<bf16>),
    routine!(swiglu_clamp_bf16 = swiglu_clamp::<bf16>),
    routine!(situ_bf16 = situ::<bf16>),
    routine!(geglu_tanh_bf16 = geglu_tanh::<bf16>, in_place = &[(0, 0)]),
    routine!(relu2_bf16 = relu2::<bf16>),
    routine!(gpt_oss_glu_bf16 = gpt_oss_glu::<bf16>, in_place = &[(0, 0)]),
    routine!(chunked_swiglu_bf16 = chunked_swiglu::<bf16>, in_place = &[(0, 1)]),
    routine!(chunked_swiglu_clamp_bf16 = chunked_swiglu_clamp::<bf16>),
    routine!(chunked_situ_bf16 = chunked_situ::<bf16>),
    routine!(chunked_geglu_tanh_bf16 = chunked_geglu_tanh::<bf16>),
    routine!(sigmoid_dot_scalar_gate_add_bf16 = sigmoid_dot_scalar_gate_add::<bf16>, in_place = &[(0, 1)]),
    routine!(gaussian_topk_bf16 = gaussian_topk::<bf16>, in_place = &[(0, 0)]),
    // gemma3n and the qwen3.5 hybrid lower `OpKind::SigmoidGateMul` to this
    // symbol, and nothing declared it. The host program is
    // `driver_internal::sigmoid_gate_inplace_bf16` and stays there; the
    // declaration has to be here, because `Family::symbol` is the module
    // path's first segment plus the routine's name and no `Family` in
    // `driver_internal` could offer an `mlp::` symbol at all.
    //
    // The alias pair is stated for the reason `KernelSig::in_place`'s own doc
    // gives -- it is a fact about the KERNEL and not about a statement, and
    // `attn_out *= sigmoid(gate)` rewrites the gated value where it lies. It
    // is also what `model-ir`'s `semantic_in_place` already answers for the
    // op that lowers to this, and that is the reader today: `in_place_pairs`
    // consults this row only for an `OpKind::Launch` naming the symbol, which
    // no text writes. Stating the same pair on both sides costs nothing and
    // removes the way they could disagree.
    //
    // **This declares the symbol and does not arm it.** A fire naming it
    // still refuses with `NoArm` -- `bind/arms/mlp.rs` has no entry for it.
    driver_bound!(sigmoid_gate_inplace_bf16, in_place = &[(0, 0)]),
];

/// `mlp`, as a trace names it.
pub static FAMILY: Family = crate::family!(ROUTINES);
