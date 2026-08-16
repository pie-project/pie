//! The dense FFN activations.
//!
//! `gptoss_swiglu` is one of the three kernels in this crate that earn a model
//! name: it bakes gpt-oss's asymmetric clamp, its `alpha` and its `(up + 1)`
//! term, and its own first line says so.

use kernels::KernelSig;

/// EMPTY: this family's rows have been RETIRED.
///
/// `refactor-bigplan.md` §7 Stage 3. `mlp` was the first LIVE family to lose
/// its rows on this backend, and — one kernel later — the last family in the
/// crate to hold one. `silu_mul_strided` stayed because every backend had
/// inherited metal's finding that its entrypoint leaves a buffer slot empty
/// and so cannot take a positional argument list. That is true of MSL's flat
/// argument table and not of `gated.wgsl`, which numbers its three buffers
/// densely and gives the pitch a uniform of its own. Reading this backend's
/// own shader is what emptied the table.
///
/// Originally: `mlp` is the first LIVE family to lose its rows — every gated MLP names `silu_mul`, and the three
/// activations beside it are what gemma and gpt-oss run — so unlike `sample`
/// and `ptir` the crossing had to be MEASURED rather than argued.
/// `driver-wgpu`'s `every_launchs_scalars_land_where_its_module_reads_them`
/// derives every field of every rectangle twice, by the row and by the arm,
/// and compares.
pub static KERNELS: &[KernelSig] = &[];

/// The entrypoints this family's routines spell, now that its rows are gone.
///
/// See [`crate::sample::ENTRYPOINTS`].
pub static ENTRYPOINTS: &[&str] = &[
    "geglu_tanh_bfloat16",
    "geglu_tanh_strided_bfloat16",
    "gptoss_swiglu_bfloat16",
    "silu_mul_bfloat16",
    "silu_mul_strided_bfloat16",
];

use crate::routine::{keys, Ask, Bind, Block, Buf, BufMut, Ctx, Fire, Routine};
use crate::routine::{InSlot, OutSlot, Param};
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
    gate: InSlot<0, Buf>,
    up: InSlot<1, Buf>,
    out: OutSlot<0, BufMut>,
    width: Ask<keys::Width, i32>,
    rows: Ask<keys::Rows, i32>,
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
    gate: InSlot<0, Buf>,
    up: InSlot<1, Buf>,
    out: OutSlot<0, BufMut>,
    params: Block<Buf>,
    width: Ask<keys::Width, i32>,
    rows: Ask<keys::Rows, i32>,
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
    gate: InSlot<0, Buf>,
    up: InSlot<1, Buf>,
    out: OutSlot<0, BufMut>,
    params: Block<Buf>,
    width: Ask<keys::Width, i32>,
    rows: Ask<keys::Rows, i32>,
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
    gate: InSlot<0, Buf>,
    up: InSlot<1, Buf>,
    out: OutSlot<0, BufMut>,
    params: Block<Buf>,
    width: Ask<keys::Width, i32>,
    rows: Ask<keys::Rows, i32>,
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
pub fn silu_mul_strided(
    ctx: &Ctx<'_>,
    gate: InSlot<0, Buf>,
    up: InSlot<1, Buf>,
    out: OutSlot<0, BufMut>,
    row_pitch: Param<0, i32>,
    width: Ask<keys::Width, i32>,
    rows: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "mlp/gated.wgsl",
            entrypoint: "silu_mul_strided_bfloat16",
            lanes: kernels::shader::elementwise_rows(*width, *rows)?,
        },
        &[gate.v(), up.v(), out.v(), row_pitch.v()],
    )
}

pub static ROUTINES: &[Routine] = &[
    crate::routine!(geglu_tanh),
    crate::routine!(geglu_tanh_strided),
    crate::routine!(gptoss_swiglu),
    crate::routine!(silu_mul),
    crate::routine!(silu_mul_strided),
];
