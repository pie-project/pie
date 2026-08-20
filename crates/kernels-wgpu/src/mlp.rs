//! The dense FFN activations.
//!
//! `gptoss_swiglu` is one of the three kernels in this crate that earn a model
//! name: it bakes gpt-oss's asymmetric clamp, its `alpha` and its `(up + 1)`
//! term, and its own first line says so.

use kernels_macros::routine;

use crate::routine::{Asks, Bind, Const, Ctx, Fire, In, Out, Tensor, bf16, elementwise, elementwise_rows, keys};
use kernels::routine::Refusal;

/// `out = silu(gate) * up`, elementwise over the FFN intermediate.
///
/// Three buffers and no params: the element count is the grid's, and a body
/// that needed it would be asking the shader to recompute what the launch
/// already said.
///
/// # Errors
///
/// [`kernels::shader::elementwise`]'s, for a zero width or row count.
#[routine]
pub fn silu_mul(
    ctx: &Ctx<'_>,
    gate: In<Tensor<bf16>>,
    up: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    let width = gate.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("mlp/gated.wgsl", "silu_mul_bfloat16").apply(elementwise(width, rows)?),
        &[gate.arg(), up.arg(), out.arg()],
    )
}

/// gemma's activation: the TANH approximation of GELU, not the erf one.
///
/// A third symbol beside [`silu_mul`] and [`gptoss_swiglu`], and a text names
/// which — the three are not interchangeable and swapping them produces a
/// model that runs and is wrong.
///
/// **Three buffers and no params block**, which on this plane took a shader
/// edit rather than a body edit. `gated.wgsl` declared `struct GegluParams {
/// unused: u32 }` at `@group(0) @binding(3)` under `PIE_GEGLU` and this body
/// forwarded `ctx.params()` into it. The forwarding was never about the
/// SCALAR — the struct's one field was a per-row element count nothing read,
/// and the grid is the extent — it was about the SLOT: WGSL declares its
/// bindings in the source, `driver-wgpu` builds an explicit bind group layout
/// from those declarations, and a body that skipped a declared slot would
/// shift every buffer after it. So `kernels-vulkan` could take the same
/// operand as `_params` and drop it, because `slangc` emits no binding for a
/// global its variant never reads, and this crate could not.
///
/// Deleting the DECLARATION settles it in the same direction on all three
/// planes: the module now declares three `@group(0)` bindings and this body
/// binds three buffers. `kernels-wgpu`'s
/// `every_routine_binds_a_buffer_for_every_binding_its_module_declares` is
/// what holds those two numbers together, and `refactor-bigplan.md` §8c is
/// the argument.
///
/// # Errors
///
/// [`kernels::shader::elementwise`]'s.
#[routine]
pub fn geglu_tanh(
    ctx: &Ctx<'_>,
    gate: In<Tensor<bf16>>,
    up: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    let width = gate.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("mlp/gated.wgsl", "geglu_tanh_bfloat16").apply(elementwise(width, rows)?),
        &[gate.arg(), up.arg(), out.arg()],
    )
}

/// [`geglu_tanh`] over rows that are not contiguous.
///
/// gemma's PLE reads a narrow gate out of a wide buffer, so each of the three
/// operands states its own pitch — which is why this one states scalars where
/// the dense form states none.
///
/// All five are marks. `width` and `rows` are the launch's rectangle and this
/// body knows them independently, which is not a duplication this signature
/// introduced: `GegluStridedParams` already held both, the shader read the
/// staged words, and the grid came from the body. Both still do — the words
/// now reach the shader through the `@group(1)` block rather than a storage
/// descriptor.
///
/// # Errors
///
/// [`kernels::shader::elementwise`]'s.
#[routine]
pub fn geglu_tanh_strided(
    ctx: &Ctx<'_>,
    gate: In<Tensor<bf16>>,
    up: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    stated_width: Const<u32>,
    stated_rows: Const<u32>,
    gate_pitch: Const<u32>,
    up_pitch: Const<u32>,
    out_pitch: Const<u32>) -> Result<(), Refusal> {
    let width = gate.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("mlp/gated.wgsl", "geglu_tanh_strided_bfloat16").apply(elementwise(width, rows)?),
        &[
            gate.arg(),
            up.arg(),
            out.arg(),
            stated_width.arg(),
            stated_rows.arg(),
            gate_pitch.arg(),
            up_pitch.arg(),
            out_pitch.arg(),
        ],
    )
}

/// gpt-oss's activation, which is not anyone else's.
///
/// The gate is clamped ABOVE only, the linear branch is clamped both ways and
/// carries a `+1`. [`silu_mul`] cannot serve it, so it is a symbol a text
/// names rather than a flag.
///
/// # Why a mark is declared and not passed
///
/// `GptOssSwiGluParams` opened with a per-row element count nothing read —
/// dead the way `GegluParams`' one field was dead, and it outlived that struct
/// only because `limit` and `alpha` beside it are live. `Const` slots are the
/// statement's run counted in declaration order, so reaching `limit` at word 1
/// means naming word 0, and `_stated_elements` is that name. It is NOT passed:
/// the block the shader reads is packed from what this body hands `ctx.fire`,
/// so the dead word stops at the host instead of riding to the GPU as it did
/// inside the struct. The holder goes when the DSL stops stating the word.
///
/// `norm::rms_single_row` holds a slot the same way and for the same reason.
///
/// # Errors
///
/// [`kernels::shader::elementwise`]'s.
#[routine]
pub fn gptoss_swiglu(
    ctx: &Ctx<'_>,
    gate: In<Tensor<bf16>>,
    up: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    _stated_elements: Const<u32>,
    limit: Const<f32>,
    alpha: Const<f32>) -> Result<(), Refusal> {
    let width = gate.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("mlp/gated.wgsl", "gptoss_swiglu_bfloat16").apply(elementwise(width, rows)?),
        &[gate.arg(), up.arg(), out.arg(), limit.arg(), alpha.arg()],
    )
}

/// [`silu_mul`] over rows a `row_pitch` apart.
///
/// # The last kernel in the fleet to get a routine, and it did not need to be
///
/// `kernels-metal` calls this one DARK and `model-ir` carries a named
/// exception for it: on that backend the entrypoint leaves a buffer slot
/// empty, so it cannot be given a positional argument list at all. THAT IS A
/// FACT ABOUT MSL'S FLAT ARGUMENT TABLE, not about the kernel. Here
/// `gated.wgsl` declares `gate`, `up` and `out_` densely at `@group(0)` 0..2
/// and puts the pitch in a `@group(1)` uniform of its own, so there is no hole
/// and nothing to work around.
///
/// It was the fleet's last `kernel!` row for that reason — every backend had
/// inherited metal's conclusion. Reading this backend's own shader is what
/// says otherwise.
///
/// # Errors
///
/// [`kernels::shader::elementwise_rows`]'s.
#[routine]
pub fn silu_mul_strided(
    ctx: &Ctx<'_>,
    gate: In<Tensor<bf16>>,
    up: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<1>` named.
    // This body forwards `ctx.params()` as a STRUCT, so the run is the
    // shader's layout and no `Const` mark can name a word inside it; the
    // migration made this `keys::RowPitch`, which no driver answers.
    let row_pitch = ctx.param(1)?;
    let width = gate.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("mlp/gated.wgsl", "silu_mul_strided_bfloat16").apply(elementwise_rows(width, rows)?),
        &[gate.arg(), up.arg(), out.arg(), row_pitch.arg()],
    )
}

