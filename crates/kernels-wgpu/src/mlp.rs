//! The dense FFN activations.
//!
//! `gptoss_swiglu` is one of the three kernels in this crate that earn a model
//! name: it bakes gpt-oss's asymmetric clamp, its `alpha` and its `(up + 1)`
//! term, and its own first line says so.

use kernels::{KernelSig, kernel};

use crate::axes::*;

pub static KERNELS: &[KernelSig] = &[
    // 1 in geglu_tanh.wgsl
    // gemma's activation: `gelu_tanh(gate) * up`, where the gelu is the tanh
    // approximation and not the erf one. A third symbol beside `silu_mul` and
    // `gptoss_swiglu`, and a text names which.
    kernel!(geglu_tanh "geglu_tanh", file = Some("mlp/gated.wgsl"),
        launch = kernels::LaunchRule::Elementwise,
        operands = kernels::operands![
            gate: Buf <- kernels::Source::In(0),
            up: Buf <- kernels::Source::In(1),
            out: BufMut <- kernels::Source::Out(0),
            // `GegluParams`: the element count, packed.
            params: Buf <- kernels::Source::Param(0),
        ],
        axes = &[BF16]),
    // 1 in geglu_tanh.wgsl
    // The same activation over rows that are not contiguous: gemma's PLE
    // reads a narrow gate out of a wide buffer, so each of the three operands
    // states its own pitch.
    kernel!(geglu_tanh_strided "geglu_tanh_strided", file = Some("mlp/gated.wgsl"),
        launch = kernels::LaunchRule::Elementwise,
        operands = kernels::operands![
            gate: Buf <- kernels::Source::In(0),
            up: Buf <- kernels::Source::In(1),
            out: BufMut <- kernels::Source::Out(0),
            // `GegluStridedParams`: width, rows and the three pitches.
            params: Buf <- kernels::Source::Param(0),
        ],
        axes = &[BF16]),
    // 1 in gptoss.wgsl
    // gpt-oss's activation, which is not anyone else's: the gate is clamped
    // ABOVE only, the linear branch is clamped both ways and carries a `+1`.
    // `silu_mul` cannot serve it -- dropping either produces a model that runs
    // and is wrong -- so it is a symbol a text names, not a flag.
    kernel!(gptoss_swiglu "gptoss_swiglu", file = Some("mlp/gated.wgsl"),
        launch = kernels::LaunchRule::Elementwise,
        operands = kernels::operands![
            gate: Buf <- kernels::Source::In(0),
            up: Buf <- kernels::Source::In(1),
            out: BufMut <- kernels::Source::Out(0),
            // `GptOssSwiGluParams`: n, limit, alpha -- packed.
            params: Buf <- kernels::Source::Param(0),
        ],
        axes = &[BF16]),
    // 1 in silu_mul.wgsl
    kernel!(silu_mul "silu_mul", file = Some("mlp/gated.wgsl"), launch = kernels::LaunchRule::Elementwise,
        operands = kernels::operands![
            gate: Buf <- kernels::Source::In(0),
            up: Buf <- kernels::Source::In(1),
            out: BufMut <- kernels::Source::Out(0),
        ],
        axes = &[BF16]),
    // 1 in silu_mul.wgsl
    kernel!(silu_mul_strided "silu_mul_strided", axes = &[BF16]),
];

use crate::routine::{Bind, Buf, BufMut, Ctx, Env, Fire, Routine};
use kernels::routine::Refusal;
use kernels::shader::elementwise;

/// `out = silu(gate) * up`, elementwise over the FFN intermediate.
///
/// Three buffers and no params: the element count is the grid's, and a body
/// that needed it would be asking the shader to recompute what the launch
/// already said.
///
/// # Errors
///
/// [`kernels::shader::elementwise`]'s, for a zero width or row count.
pub fn silu_mul(
    ctx: &Ctx<'_>,
    gate: Buf,
    up: Buf,
    out: BufMut,
    width: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "mlp/gated.wgsl",
            entrypoint: "silu_mul_bfloat16",
            lanes: elementwise(*width, *rows)?,
        },
        &[gate.v(), up.v(), out.v()],
    )
}

/// gemma's activation: the TANH approximation of GELU, not the erf one.
///
/// A third symbol beside [`silu_mul`] and [`gptoss_swiglu`], and a text names
/// which — the three are not interchangeable and swapping them produces a
/// model that runs and is wrong.
///
/// **`params` is FORWARDED here, and `kernels-vulkan` takes it as `_params`
/// and drops it.** That is not a disagreement: `slangc` emits no binding for
/// a global its variant never reads, so vulkan's module has no slot to fill,
/// while WGSL declares its bindings in the source and `driver-wgpu` builds
/// the bind group layout from those declarations. A body that skipped it here
/// would shift every buffer after it. `kernels-wgpu`'s
/// `every_routine_binds_a_buffer_for_every_binding_its_module_declares` is
/// what holds that, and `refactor-bigplan.md` §8c is the argument.
///
/// # Errors
///
/// [`kernels::shader::elementwise`]'s.
pub fn geglu_tanh(
    ctx: &Ctx<'_>,
    gate: Buf,
    up: Buf,
    out: BufMut,
    params: Buf,
    width: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "mlp/gated.wgsl",
            entrypoint: "geglu_tanh_bfloat16",
            lanes: elementwise(*width, *rows)?,
        },
        &[gate.v(), up.v(), out.v(), params.v()],
    )
}

/// [`geglu_tanh`] over rows that are not contiguous.
///
/// gemma's PLE reads a narrow gate out of a wide buffer, so each of the three
/// operands states its own pitch — which is what `params` carries, and why
/// this one reads it where the dense form does not.
///
/// # Errors
///
/// [`kernels::shader::elementwise`]'s.
pub fn geglu_tanh_strided(
    ctx: &Ctx<'_>,
    gate: Buf,
    up: Buf,
    out: BufMut,
    params: Buf,
    width: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "mlp/gated.wgsl",
            entrypoint: "geglu_tanh_strided_bfloat16",
            lanes: elementwise(*width, *rows)?,
        },
        &[gate.v(), up.v(), out.v(), params.v()],
    )
}

/// gpt-oss's activation, which is not anyone else's.
///
/// The gate is clamped ABOVE only, the linear branch is clamped both ways and
/// carries a `+1`. [`silu_mul`] cannot serve it, so it is a symbol a text
/// names rather than a flag.
///
/// # Errors
///
/// [`kernels::shader::elementwise`]'s.
pub fn gptoss_swiglu(
    ctx: &Ctx<'_>,
    gate: Buf,
    up: Buf,
    out: BufMut,
    params: Buf,
    width: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "mlp/gated.wgsl",
            entrypoint: "gptoss_swiglu_bfloat16",
            lanes: elementwise(*width, *rows)?,
        },
        &[gate.v(), up.v(), out.v(), params.v()],
    )
}

pub static ROUTINES: &[Routine] = &[
    crate::routine!(geglu_tanh),
    crate::routine!(geglu_tanh_strided),
    crate::routine!(gptoss_swiglu),
    crate::routine!(silu_mul),
];
