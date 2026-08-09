#![allow(clippy::too_many_arguments)]

use crate::jit::{Ctx, Family, Launch, Routine};
use crate::routine;
use crate::x::Abi;
use crate::x::abi::{bf16, f16};
use kernels::Refusal;

use core::ptr::NonNull;

/// `mlp/swiglu.cuh` — the gated activations, flat and chunked.
pub mod swiglu {

    use crate::jit::Root;

    /// `mlp/swiglu.cuh` — the root these routines compile a symbol out of.
    pub static ROOT: Root =
        Root::new("mlp/swiglu", include_str!("../../csrc/src/mlp/swiglu.cuh"), "mlp/swiglu.cuh");

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    ///
    /// Absolute, because a routine body names the instantiation itself rather
    /// than a label some other table maps to one. The `<...>` argument is what
    /// used to be a row's `elem`, and it is `device::bf16` in every one of
    /// them: this family compiles no other element type.
    ///
    /// `pub(in crate::x)` and not `pub(super)`: `sigmoid_gate_inplace` is a
    /// driver op with no routine in this family, so the body that names its
    /// instantiation is `x::driver_internal`'s.
    pub(in crate::x) mod inst {
        /// `swiglu.cuh:135` — `y = silu(gate) * up`, flat.
        pub const SWIGLU: &str = "::pie_cuda_driver::kernels::mlp::device::swiglu\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `swiglu.cuh:185` — the same with the gate clamped to `±limit`.
        pub const SWIGLU_CLAMP: &str = "::pie_cuda_driver::kernels::mlp::device::swiglu_clamp\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `swiglu.cuh:206` — SiTU, which is not a swiglu variant.
        pub const SITU: &str = "::pie_cuda_driver::kernels::mlp::device::situ\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `swiglu.cuh:230` — GeGLU-tanh, which is not one either.
        pub const GEGLU_TANH: &str = "::pie_cuda_driver::kernels::mlp::device::geglu_tanh\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `swiglu.cuh:247` — `y = max(x, 0)^2`.
        pub const RELU2: &str = "::pie_cuda_driver::kernels::mlp::device::relu2\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `swiglu.cuh:161` — gpt-oss's clamped GLU.
        pub const GPT_OSS_GLU: &str = "::pie_cuda_driver::kernels::mlp::device::gpt_oss_glu\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `swiglu.cuh:261` — `x *= sigmoid(gate)`, in place.
        pub const SIGMOID_GATE_INPLACE: &str = "::pie_cuda_driver::kernels::mlp::device::sigmoid_gate_inplace\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `swiglu.cuh:343` — the packed gate‖up bank, the gate half first.
        pub const CHUNKED_SWIGLU: &str = "::pie_cuda_driver::kernels::mlp::device::chunked_swiglu\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `swiglu.cuh:349` — the same at `GateSecond = true`.
        pub const CHUNKED_SWIGLU_GATE_SECOND: &str = "::pie_cuda_driver::kernels::mlp::device::chunked_swiglu_gate_second\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `swiglu.cuh:435` — the packed form with the gate clamped.
        pub const CHUNKED_SWIGLU_CLAMP: &str = "::pie_cuda_driver::kernels::mlp::device::chunked_swiglu_clamp\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `swiglu.cuh:378` — SiTU over a packed bank.
        pub const CHUNKED_SITU: &str = "::pie_cuda_driver::kernels::mlp::device::chunked_situ\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `swiglu.cuh:387` — the same at `GateSecond = true`.
        pub const CHUNKED_SITU_GATE_SECOND: &str = "::pie_cuda_driver::kernels::mlp::device::chunked_situ_gate_second\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `swiglu.cuh:417` — GeGLU-tanh over a packed bank.
        pub const CHUNKED_GEGLU_TANH: &str = "::pie_cuda_driver::kernels::mlp::device::chunked_geglu_tanh\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `swiglu.cuh:425` — the same at `GateSecond = true`.
        pub const CHUNKED_GEGLU_TANH_GATE_SECOND: &str = "::pie_cuda_driver::kernels::mlp::device::chunked_geglu_tanh_gate_second\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `swiglu.cuh:607` — `out += y * sigmoid(x · gate_w)`.
        pub const SIGMOID_DOT_SCALAR_GATE_ADD: &str = "::pie_cuda_driver::kernels::mlp::device::sigmoid_dot_scalar_gate_add\
             <::pie_cuda_driver::kernels::device::bf16>";
    }
}

/// `mlp/gaussian_topk.cuh` — AltUp's activation sparsity, alone in its file
pub mod gaussian_topk {

    use crate::jit::Root;

    /// `mlp/gaussian_topk.cuh` — the root this one routine compiles out of.
    pub static ROOT: Root = Root::new(
        "mlp/gaussian_topk",
        include_str!("../../csrc/src/mlp/gaussian_topk.cuh"),
        "mlp/gaussian_topk.cuh",
    );

    /// The template-id NVRTC is handed, spelled as it is handed it.
    pub(super) mod inst {
        /// `gaussian_topk.cuh:72` — gemma-3n's AltUp sparsity, in place.
        pub const GAUSSIAN_TOPK: &str = "::pie_cuda_driver::kernels::mlp::device::gaussian_topk\
             <::pie_cuda_driver::kernels::device::bf16>";
    }
}

/// Threads per block, everywhere in this family.
const BLOCK: u32 = 256;

/// Threads per warp — the unit the reductions' shared scratch is counted in.
const WARP: u32 = 32;

/// The dynamic shared memory the two reducing kernels fold through.
const RMS_SMEM: u32 = (BLOCK / WARP) * 4;

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
pub fn swiglu_bf16(
    ctx: &Ctx,
    gate: *const bf16,
    up: *const bf16,
    y: *mut bf16,
    n: i32,
) -> Result<(), Refusal> {
    if n <= 0 {
        return Err(Refusal::Empty { what: "num_elements" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &swiglu::ROOT,
            swiglu::inst::SWIGLU,
            elementwise(n),
            &[gate.arg(), up.arg(), y.arg(), n.arg()],
        )
    }
}

/// The same with the gate clamped to `±limit` — `mlp::swiglu_clamp_bf16`.
///
/// # Safety
///
/// [`swiglu_bf16`]'s.
pub fn swiglu_clamp_bf16(
    ctx: &Ctx,
    gate: *const bf16,
    up: *const bf16,
    y: *mut bf16,
    n: i32,
    limit: f32,
) -> Result<(), Refusal> {
    if n <= 0 {
        return Err(Refusal::Empty { what: "num_elements" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &swiglu::ROOT,
            swiglu::inst::SWIGLU_CLAMP,
            elementwise(n),
            &[gate.arg(), up.arg(), y.arg(), n.arg(), limit.arg()],
        )
    }
}

/// SiTU — `mlp::situ_bf16`.
///
/// # Safety
///
/// [`swiglu_bf16`]'s.
pub fn situ_bf16(
    ctx: &Ctx,
    gate: *const bf16,
    up: *const bf16,
    y: *mut bf16,
    n: i32,
    beta: f32,
    linear_beta: f32,
) -> Result<(), Refusal> {
    if n <= 0 {
        return Err(Refusal::Empty { what: "num_elements" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &swiglu::ROOT,
            swiglu::inst::SITU,
            elementwise(n),
            &[gate.arg(), up.arg(), y.arg(), n.arg(), beta.arg(), linear_beta.arg()],
        )
    }
}

/// GeGLU-tanh — `mlp::geglu_tanh_bf16`.
///
/// # Safety
///
/// [`swiglu_bf16`]'s. `y` may alias `gate`.
pub fn geglu_tanh_bf16(
    ctx: &Ctx,
    gate: *const bf16,
    up: *const bf16,
    y: *mut bf16,
    n: i32,
) -> Result<(), Refusal> {
    if n <= 0 {
        return Err(Refusal::Empty { what: "num_elements" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &swiglu::ROOT,
            swiglu::inst::GEGLU_TANH,
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
pub fn relu2_bf16(ctx: &Ctx, x: *const bf16, y: *mut bf16, n: i32) -> Result<(), Refusal> {
    if n <= 0 {
        return Err(Refusal::Empty { what: "num_elements" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(&swiglu::ROOT, swiglu::inst::RELU2, elementwise(n), &[x.arg(), y.arg(), n.arg()])
    }
}

/// gpt-oss's clamped GLU — `mlp::gpt_oss_glu_bf16`.
///
/// # Safety
///
/// [`swiglu_bf16`]'s, plus: when `y_fp16` is `Some`, it must address `n`
/// writable **fp16** elements. `y` may alias `gate`.
pub fn gpt_oss_glu_bf16(
    ctx: &Ctx,
    gate: *const bf16,
    up: *const bf16,
    y: *mut bf16,
    y_fp16: Option<NonNull<f16>>,
    n: i32,
    limit: f32,
    alpha: f32,
) -> Result<(), Refusal> {
    if n <= 0 {
        return Err(Refusal::Empty { what: "num_elements" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &swiglu::ROOT,
            swiglu::inst::GPT_OSS_GLU,
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
pub fn chunked_swiglu_bf16(
    ctx: &Ctx,
    packed: *const bf16,
    y: *mut bf16,
    rows: i32,
    i: i32,
    gate_second: bool,
) -> Result<(), Refusal> {
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if i <= 0 {
        return Err(Refusal::Empty { what: "intermediate" });
    }
    let instantiation = if gate_second {
        swiglu::inst::CHUNKED_SWIGLU_GATE_SECOND
    } else {
        swiglu::inst::CHUNKED_SWIGLU
    };
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &swiglu::ROOT,
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
/// [`chunked_swiglu_bf16`]'s.
pub fn chunked_swiglu_clamp_bf16(
    ctx: &Ctx,
    packed: *const bf16,
    y: *mut bf16,
    rows: i32,
    i: i32,
    limit: f32,
) -> Result<(), Refusal> {
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if i <= 0 {
        return Err(Refusal::Empty { what: "intermediate" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &swiglu::ROOT,
            swiglu::inst::CHUNKED_SWIGLU_CLAMP,
            elementwise_rows(rows, i),
            &[packed.arg(), y.arg(), i.arg(), limit.arg()],
        )
    }
}

/// SiTU over a packed bank — `mlp::chunked_situ_bf16`.
///
/// # Safety
///
/// [`chunked_swiglu_bf16`]'s.
pub fn chunked_situ_bf16(
    ctx: &Ctx,
    packed: *const bf16,
    y: *mut bf16,
    rows: i32,
    i: i32,
    beta: f32,
    linear_beta: f32,
    gate_second: bool,
) -> Result<(), Refusal> {
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if i <= 0 {
        return Err(Refusal::Empty { what: "intermediate" });
    }
    let instantiation = if gate_second {
        swiglu::inst::CHUNKED_SITU_GATE_SECOND
    } else {
        swiglu::inst::CHUNKED_SITU
    };
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &swiglu::ROOT,
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
/// [`chunked_swiglu_bf16`]'s.
pub fn chunked_geglu_tanh_bf16(
    ctx: &Ctx,
    packed: *const bf16,
    y: *mut bf16,
    rows: i32,
    i: i32,
    gate_second: bool,
) -> Result<(), Refusal> {
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if i <= 0 {
        return Err(Refusal::Empty { what: "intermediate" });
    }
    let instantiation = if gate_second {
        swiglu::inst::CHUNKED_GEGLU_TANH_GATE_SECOND
    } else {
        swiglu::inst::CHUNKED_GEGLU_TANH
    };
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &swiglu::ROOT,
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
pub fn sigmoid_dot_scalar_gate_add_bf16(
    ctx: &Ctx,
    x: *const bf16,
    gate_w: *const bf16,
    out: *mut bf16,
    y: *const bf16,
    rows: i32,
    h: i32,
) -> Result<(), Refusal> {
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if h <= 0 {
        return Err(Refusal::Empty { what: "hidden" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &swiglu::ROOT,
            swiglu::inst::SIGMOID_DOT_SCALAR_GATE_ADD,
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
pub fn gaussian_topk_bf16(
    ctx: &Ctx,
    x: *mut bf16,
    rows: i32,
    dim: i32,
    std_multiplier: f32,
) -> Result<(), Refusal> {
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if dim <= 0 {
        return Err(Refusal::Empty { what: "dim" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &gaussian_topk::ROOT,
            gaussian_topk::inst::GAUSSIAN_TOPK,
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
pub static ROUTINES: &[Routine] = &[
    routine!(swiglu_bf16),
    routine!(swiglu_clamp_bf16),
    routine!(situ_bf16),
    routine!(geglu_tanh_bf16, in_place = &[(0, 0)]),
    routine!(relu2_bf16),
    routine!(gpt_oss_glu_bf16, in_place = &[(0, 0)]),
    routine!(chunked_swiglu_bf16, in_place = &[(0, 1)]),
    routine!(chunked_swiglu_clamp_bf16),
    routine!(chunked_situ_bf16),
    routine!(chunked_geglu_tanh_bf16),
    routine!(sigmoid_dot_scalar_gate_add_bf16, in_place = &[(0, 1)]),
    routine!(gaussian_topk_bf16, in_place = &[(0, 0)]),
];

/// `mlp`, as a trace names it.
pub static FAMILY: Family = Family { namespace: "mlp", routines: ROUTINES };
