#![allow(clippy::too_many_arguments)]

use crate::unit::Unit;
use crate::x::abi::{bf16, f16};
use crate::x::launch::Launch;

#[cfg(feature = "_cuda")]
use crate::x::contract::{Fired, Refusal};
#[cfg(feature = "_cuda")]
use core::ffi::c_void;
use core::ptr::NonNull;

/// `mlp/swiglu.cuh` — the gated activations, flat and chunked.
pub mod swiglu {
    use super::{bf16, f16};
    use core::ptr::NonNull;

    unit! {
        /// `mlp`'s gated activations: fifteen instantiations over as many
        unit SWIGLU = "mlp/swiglu",
            text = include_str!("../../csrc/src/mlp/swiglu.cuh"),
            file = "mlp/swiglu.cuh";

        /// `swiglu.cuh:135` — `y = silu(gate) * up`, flat.
        fn swiglu = "mlp::device::swiglu" <T> (
            gate: *const T,
            up: *const T,
            y: *mut T,
            n: i32,
        ) where *const T, *mut T {
            "mlp::swiglu_bf16" => where [T = bf16] "device::bf16",
        }

        /// `swiglu.cuh:185` — the same with the gate clamped to `±limit`.
        fn swiglu_clamp = "mlp::device::swiglu_clamp" <T> (
            gate: *const T,
            up: *const T,
            y: *mut T,
            n: i32,
            limit: f32,
        ) where *const T, *mut T {
            "mlp::swiglu_clamp_bf16" => where [T = bf16] "device::bf16",
        }

        /// `swiglu.cuh:206` — SiTU, which is not a swiglu variant.
        fn situ = "mlp::device::situ" <T> (
            gate: *const T,
            up: *const T,
            y: *mut T,
            n: i32,
            beta: f32,
            linear_beta: f32,
        ) where *const T, *mut T {
            "mlp::situ_bf16" => where [T = bf16] "device::bf16",
        }

        /// `swiglu.cuh:230` — GeGLU-tanh, which is not one either:
        fn geglu_tanh = "mlp::device::geglu_tanh" <T> (
            gate: *const T,
            up: *const T,
            y: *mut T,
            n: i32,
        ) where *const T, *mut T {
            "mlp::geglu_tanh_bf16" => where [T = bf16] "device::bf16",
        }

        /// `swiglu.cuh:247` — `y = max(x, 0)^2`.
        fn relu2 = "mlp::device::relu2" <T> (
            x: *const T,
            y: *mut T,
            n: i32,
        ) where *const T, *mut T {
            "mlp::relu2_bf16" => where [T = bf16] "device::bf16",
        }

        /// `swiglu.cuh:161` — gpt-oss's clamped GLU, with an optional fp16
        fn gpt_oss_glu = "mlp::device::gpt_oss_glu" <T> (
            gate: *const T,
            up: *const T,
            y: *mut T,
            y_fp16: Option<NonNull<f16>>,
            n: i32,
            limit: f32,
            alpha: f32,
        ) where *const T, *mut T {
            "mlp::gpt_oss_glu_bf16" => where [T = bf16] "device::bf16",
        }

        /// `swiglu.cuh:261` — `x *= sigmoid(gate)`, in place.
        fn sigmoid_gate_inplace = "mlp::device::sigmoid_gate_inplace" <T> (
            x: *mut T,
            gate: *const T,
            n: i32,
        ) where *const T, *mut T {
            "mlp::sigmoid_gate_inplace_bf16" => where [T = bf16] "device::bf16",
        }

        /// `swiglu.cuh:343` — the packed gate‖up bank, one row per block row.
        fn chunked_swiglu = "mlp::device::chunked_swiglu" <T> (
            packed: *const T,
            y: *mut T,
            i: i32,
        ) where *const T, *mut T {
            "mlp::chunked_swiglu_bf16" => where [T = bf16] "device::bf16",
        }

        /// `swiglu.cuh:349` — the same at `GateSecond = true`.
        fn chunked_swiglu_gate_second = "mlp::device::chunked_swiglu_gate_second" <T> (
            packed: *const T,
            y: *mut T,
            i: i32,
        ) where *const T, *mut T {
            "mlp::chunked_swiglu_gate_second_bf16" => where [T = bf16] "device::bf16",
        }

        /// `swiglu.cuh:435` — the packed form with the gate clamped.
        fn chunked_swiglu_clamp = "mlp::device::chunked_swiglu_clamp" <T> (
            packed: *const T,
            y: *mut T,
            i: i32,
            limit: f32,
        ) where *const T, *mut T {
            "mlp::chunked_swiglu_clamp_bf16" => where [T = bf16] "device::bf16",
        }

        /// `swiglu.cuh:378` — SiTU over a packed bank.
        fn chunked_situ = "mlp::device::chunked_situ" <T> (
            packed: *const T,
            y: *mut T,
            i: i32,
            beta: f32,
            linear_beta: f32,
        ) where *const T, *mut T {
            "mlp::chunked_situ_bf16" => where [T = bf16] "device::bf16",
        }

        /// `swiglu.cuh:387` — the same at `GateSecond = true`.
        fn chunked_situ_gate_second = "mlp::device::chunked_situ_gate_second" <T> (
            packed: *const T,
            y: *mut T,
            i: i32,
            beta: f32,
            linear_beta: f32,
        ) where *const T, *mut T {
            "mlp::chunked_situ_gate_second_bf16" => where [T = bf16] "device::bf16",
        }

        /// `swiglu.cuh:417` — GeGLU-tanh over a packed bank.
        fn chunked_geglu_tanh = "mlp::device::chunked_geglu_tanh" <T> (
            packed: *const T,
            y: *mut T,
            i: i32,
        ) where *const T, *mut T {
            "mlp::chunked_geglu_tanh_bf16" => where [T = bf16] "device::bf16",
        }

        /// `swiglu.cuh:425` — the same at `GateSecond = true`.
        fn chunked_geglu_tanh_gate_second = "mlp::device::chunked_geglu_tanh_gate_second" <T> (
            packed: *const T,
            y: *mut T,
            i: i32,
        ) where *const T, *mut T {
            "mlp::chunked_geglu_tanh_gate_second_bf16" => where [T = bf16] "device::bf16",
        }

        /// `swiglu.cuh:607` — `out += y * sigmoid(x · gate_w)`, the shared
        fn sigmoid_dot_scalar_gate_add = "mlp::device::sigmoid_dot_scalar_gate_add" <T> (
            x: *const T,
            gate_w: *const T,
            out: *mut T,
            y: *const T,
            h: i32,
        ) where *const T, *mut T {
            "mlp::sigmoid_dot_scalar_gate_add_bf16" => where [T = bf16] "device::bf16",
        }
    }
}

/// `mlp/gaussian_topk.cuh` — AltUp's activation sparsity, alone in its file
pub mod gaussian_topk {
    use super::bf16;

    unit! {
        /// gemma-3n's AltUp sparsity, one instantiation.
        unit GAUSSIAN_TOPK = "mlp/gaussian_topk",
            text = include_str!("../../csrc/src/mlp/gaussian_topk.cuh"),
            file = "mlp/gaussian_topk.cuh";

        /// `gaussian_topk.cuh:72` — zero everything below
        fn gaussian_topk = "mlp::device::gaussian_topk" <T> (
            x: *mut T,
            dim: i32,
            std_multiplier: f32,
        ) where *mut T {
            "mlp::gaussian_topk_bf16" => where [T = bf16] "device::bf16",
        }
    }
}

/// The units `mlp` compiles.
pub static UNITS: &[Unit] = &[swiglu::SWIGLU, gaussian_topk::GAUSSIAN_TOPK];

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
    Launch {
        grid: [rows.unsigned_abs(), width.unsigned_abs().div_ceil(BLOCK), 1],
        block: [BLOCK, 1, 1],
        smem: 0,
        smem_opt_in: false,
    }
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
/// `gate` and `up` must address `n` live bf16 elements, `y` `n` writable
/// ones, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn swiglu_bf16(
    gate: *const bf16,
    up: *const bf16,
    y: *mut bf16,
    n: i32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_elements" });
    }
    unsafe {
        swiglu::raw::swiglu("mlp::swiglu_bf16", elementwise(n), gate, up, y, n, stream);
    }
    Fired::Launched
}

/// The same with the gate clamped to `±limit` — `mlp::swiglu_clamp_bf16`.
///
/// # Safety
///
/// [`swiglu_bf16`]'s.
#[cfg(feature = "_cuda")]
pub unsafe fn swiglu_clamp_bf16(
    gate: *const bf16,
    up: *const bf16,
    y: *mut bf16,
    n: i32,
    limit: f32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_elements" });
    }
    unsafe {
        swiglu::raw::swiglu_clamp(
            "mlp::swiglu_clamp_bf16",
            elementwise(n),
            gate,
            up,
            y,
            n,
            limit,
            stream,
        );
    }
    Fired::Launched
}

/// SiTU — `mlp::situ_bf16`.
///
/// # Safety
///
/// [`swiglu_bf16`]'s.
#[cfg(feature = "_cuda")]
pub unsafe fn situ_bf16(
    gate: *const bf16,
    up: *const bf16,
    y: *mut bf16,
    n: i32,
    beta: f32,
    linear_beta: f32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_elements" });
    }
    unsafe {
        swiglu::raw::situ(
            "mlp::situ_bf16",
            elementwise(n),
            gate,
            up,
            y,
            n,
            beta,
            linear_beta,
            stream,
        );
    }
    Fired::Launched
}

/// GeGLU-tanh — `mlp::geglu_tanh_bf16`.
///
/// # Safety
///
/// [`swiglu_bf16`]'s. `y` may alias `gate`.
#[cfg(feature = "_cuda")]
pub unsafe fn geglu_tanh_bf16(
    gate: *const bf16,
    up: *const bf16,
    y: *mut bf16,
    n: i32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_elements" });
    }
    unsafe {
        swiglu::raw::geglu_tanh(
            "mlp::geglu_tanh_bf16",
            elementwise(n),
            gate,
            up,
            y,
            n,
            stream,
        );
    }
    Fired::Launched
}

/// `y = max(x, 0)^2` — `mlp::relu2_bf16`.
///
/// # Safety
///
/// `x` must address `n` live bf16 elements, `y` `n` writable ones, and
/// `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn relu2_bf16(
    x: *const bf16,
    y: *mut bf16,
    n: i32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_elements" });
    }
    unsafe {
        swiglu::raw::relu2("mlp::relu2_bf16", elementwise(n), x, y, n, stream);
    }
    Fired::Launched
}

/// gpt-oss's clamped GLU — `mlp::gpt_oss_glu_bf16`.
///
/// # Safety
///
/// [`swiglu_bf16`]'s, plus: when `y_fp16` is `Some`, it must address `n`
/// writable **fp16** elements. `y` may alias `gate`.
#[cfg(feature = "_cuda")]
pub unsafe fn gpt_oss_glu_bf16(
    gate: *const bf16,
    up: *const bf16,
    y: *mut bf16,
    y_fp16: Option<NonNull<f16>>,
    n: i32,
    limit: f32,
    alpha: f32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_elements" });
    }
    unsafe {
        swiglu::raw::gpt_oss_glu(
            "mlp::gpt_oss_glu_bf16",
            elementwise(n),
            gate,
            up,
            y,
            y_fp16,
            n,
            limit,
            alpha,
            stream,
        );
    }
    Fired::Launched
}

/// SwiGLU over a packed gate‖up bank — `mlp::chunked_swiglu_bf16`.
///
/// # Safety
///
/// `packed` must address `rows * 2 * i` live bf16 elements and `y`
/// `rows * i` writable ones. `y` may alias the second half of `packed`,
/// which is what `in_place = &[(0, 1)]` declares. `stream` must be live
/// across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn chunked_swiglu_bf16(
    packed: *const bf16,
    y: *mut bf16,
    rows: i32,
    i: i32,
    gate_second: bool,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if i <= 0 {
        return Fired::Declined(Refusal::Empty { what: "intermediate" });
    }
    let launch = elementwise_rows(rows, i);
    unsafe {
        if gate_second {
            swiglu::raw::chunked_swiglu_gate_second(
                "mlp::chunked_swiglu_gate_second_bf16",
                launch,
                packed,
                y,
                i,
                stream,
            );
        } else {
            swiglu::raw::chunked_swiglu(
                "mlp::chunked_swiglu_bf16",
                launch,
                packed,
                y,
                i,
                stream,
            );
        }
    }
    Fired::Launched
}

/// The packed form with the gate clamped —
///
/// # Safety
///
/// [`chunked_swiglu_bf16`]'s.
#[cfg(feature = "_cuda")]
pub unsafe fn chunked_swiglu_clamp_bf16(
    packed: *const bf16,
    y: *mut bf16,
    rows: i32,
    i: i32,
    limit: f32,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if i <= 0 {
        return Fired::Declined(Refusal::Empty { what: "intermediate" });
    }
    unsafe {
        swiglu::raw::chunked_swiglu_clamp(
            "mlp::chunked_swiglu_clamp_bf16",
            elementwise_rows(rows, i),
            packed,
            y,
            i,
            limit,
            stream,
        );
    }
    Fired::Launched
}

/// SiTU over a packed bank — `mlp::chunked_situ_bf16`.
///
/// # Safety
///
/// [`chunked_swiglu_bf16`]'s.
#[cfg(feature = "_cuda")]
pub unsafe fn chunked_situ_bf16(
    packed: *const bf16,
    y: *mut bf16,
    rows: i32,
    i: i32,
    beta: f32,
    linear_beta: f32,
    gate_second: bool,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if i <= 0 {
        return Fired::Declined(Refusal::Empty { what: "intermediate" });
    }
    let launch = elementwise_rows(rows, i);
    unsafe {
        if gate_second {
            swiglu::raw::chunked_situ_gate_second(
                "mlp::chunked_situ_gate_second_bf16",
                launch,
                packed,
                y,
                i,
                beta,
                linear_beta,
                stream,
            );
        } else {
            swiglu::raw::chunked_situ(
                "mlp::chunked_situ_bf16",
                launch,
                packed,
                y,
                i,
                beta,
                linear_beta,
                stream,
            );
        }
    }
    Fired::Launched
}

/// GeGLU-tanh over a packed bank — `mlp::chunked_geglu_tanh_bf16`.
///
/// # Safety
///
/// [`chunked_swiglu_bf16`]'s.
#[cfg(feature = "_cuda")]
pub unsafe fn chunked_geglu_tanh_bf16(
    packed: *const bf16,
    y: *mut bf16,
    rows: i32,
    i: i32,
    gate_second: bool,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if i <= 0 {
        return Fired::Declined(Refusal::Empty { what: "intermediate" });
    }
    let launch = elementwise_rows(rows, i);
    unsafe {
        if gate_second {
            swiglu::raw::chunked_geglu_tanh_gate_second(
                "mlp::chunked_geglu_tanh_gate_second_bf16",
                launch,
                packed,
                y,
                i,
                stream,
            );
        } else {
            swiglu::raw::chunked_geglu_tanh(
                "mlp::chunked_geglu_tanh_bf16",
                launch,
                packed,
                y,
                i,
                stream,
            );
        }
    }
    Fired::Launched
}

/// `out += y * sigmoid(x · gate_w)` — `mlp::sigmoid_dot_scalar_gate_add_bf16`.
///
/// # Safety
///
/// `x`, `y` and `out` must each address `rows * h` live bf16 elements —
/// `out` writable, and it IS the residual stream the statement takes as its
/// second operand, which is what `in_place = &[(0, 1)]` declares. `gate_w`
/// must address `h` live bf16 elements. `stream` must be live across the
/// launch.
#[cfg(feature = "_cuda")]
pub unsafe fn sigmoid_dot_scalar_gate_add_bf16(
    x: *const bf16,
    gate_w: *const bf16,
    out: *mut bf16,
    y: *const bf16,
    rows: i32,
    h: i32,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "hidden" });
    }
    unsafe {
        swiglu::raw::sigmoid_dot_scalar_gate_add(
            "mlp::sigmoid_dot_scalar_gate_add_bf16",
            rms(rows),
            x,
            gate_w,
            out,
            y,
            h,
            stream,
        );
    }
    Fired::Launched
}

/// AltUp's activation sparsity, in place — `mlp::gaussian_topk_bf16`.
///
/// # Safety
///
/// `x` must address `rows * dim` live and writable bf16 elements and
/// `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn gaussian_topk_bf16(
    x: *mut bf16,
    rows: i32,
    dim: i32,
    std_multiplier: f32,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "dim" });
    }
    unsafe {
        gaussian_topk::raw::gaussian_topk(
            "mlp::gaussian_topk_bf16",
            rms(rows),
            x,
            dim,
            std_multiplier,
            stream,
        );
    }
    Fired::Launched
}

contract! {
    /// SwiGLU over a packed gate‖up bank.
    CHUNKED_SWIGLU_BF16 = "mlp::chunked_swiglu_bf16" as chunked_swiglu {
        in_place: &[(0, 1)],
    }

    /// SwiGLU over two narrow buffers.
    SWIGLU_BF16 = "mlp::swiglu_bf16" as swiglu

    /// SwiGLU with the gate clamped — gpt-oss's `swiglu_limit`.
    SWIGLU_CLAMP_BF16 = "mlp::swiglu_clamp_bf16" as swiglu_clamp

    /// The packed form of the same.
    CHUNKED_SWIGLU_CLAMP_BF16 = "mlp::chunked_swiglu_clamp_bf16" as chunked_swiglu_clamp

    /// `y = max(x, 0)^2`.
    RELU2_BF16 = "mlp::relu2_bf16" as relu2

    /// SiTU, which is not a swiglu variant.
    SITU_BF16 = "mlp::situ_bf16" as situ

    /// The packed form of the same.
    CHUNKED_SITU_BF16 = "mlp::chunked_situ_bf16" as chunked_situ

    /// AltUp's activation sparsity, in place.
    GAUSSIAN_TOPK_BF16 = "mlp::gaussian_topk_bf16" as gaussian_topk {
        in_place: &[(0, 0)],
    }

    /// GeGLU-tanh over two narrow buffers, the gate half in place.
    GEGLU_TANH_BF16 = "mlp::geglu_tanh_bf16" as geglu_tanh {
        in_place: &[(0, 0)],
    }

    /// The packed form of the same.
    CHUNKED_GEGLU_TANH_BF16 = "mlp::chunked_geglu_tanh_bf16" as chunked_geglu_tanh

    /// gpt-oss's clamped GLU. `gate = glu(gate, up)`, so the gate half is
    GPT_OSS_GLU_BF16 = "mlp::gpt_oss_glu_bf16" as gpt_oss_glu {
        in_place: &[(0, 0)],
    }

    /// The shared expert's landing: `out += sigmoid(x · gate) * y`, where
    MOE_SHARED_GATE_DOT_BF16 = "mlp::sigmoid_dot_scalar_gate_add_bf16" as moe_shared_gate_dot {
        in_place: &[(0, 1)],
    }
}

#[cfg(feature = "_cuda")]
bind! {
    CHUNKED_SWIGLU_BF16 => { cx, stream => {
        unsafe {
            chunked_swiglu_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                cx.out_width(0)?,
                false,
                stream,
            )
        }
        .ok()
    }},

    SWIGLU_BF16 => { none: "this kernel reads the up projection's second \
        half, and a trace that leaves the projection packed never names it \
        -- the driver's op join supplies it, and a bind cannot ask the join \
        for anything. FLOOR: `up` is Source::Or(In(1), Aux(0)) and `Cx` \
        answers only In(1); needs `Facts::aux(i) -> Option<*mut c_void>`, \
        which is `join_aux(spec, i, frame, resolver)` and one defaulted \
        method" },

    SWIGLU_CLAMP_BF16 => { none: "this kernel needs the model's GLU clamp \
        limit and the up projection's second half, and a bind can ask for \
        neither. FLOOR: DispatchCtx::glu_limit (bind/mod.rs:1193) and the \
        join's foreign operands; needs `Facts::glu_limit()` and \
        `Facts::aux(i)`" },

    CHUNKED_SWIGLU_CLAMP_BF16 => { none: "this kernel needs the model's GLU \
        clamp limit, and a bind cannot ask for it. FLOOR: \
        DispatchCtx::glu_limit (bind/mod.rs:1193); needs \
        `Facts::glu_limit()`, one defaulted method over a field the driver \
        already holds" },

    RELU2_BF16 => { cx, stream => {
        let n = elements(cx)?;
        unsafe {
            relu2_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                n,
                stream,
            )
        }
        .ok()
    }},

    SITU_BF16 => { none: "this kernel needs the model's two SITU betas and \
        the up projection's second half, and a bind can ask for neither. \
        FLOOR: DispatchCtx::situ_beta and situ_linear_beta \
        (bind/mod.rs:1200-1202) and the join's foreign operands; needs \
        `Facts::situ() -> Option<(f32, f32)>` and `Facts::aux(i)`" },

    CHUNKED_SITU_BF16 => { none: "this kernel needs the model's two SITU \
        betas and which half of the packed projection is the gate, and a \
        bind can ask for neither. FLOOR: DispatchCtx::situ_beta, \
        situ_linear_beta (bind/mod.rs:1200-1202) and gate_second \
        (bind/mod.rs:1149); needs `Facts::situ()` and \
        `Facts::gate_second() -> bool`" },

    GAUSSIAN_TOPK_BF16 => { none: "this kernel needs the layer's altup \
        standard-deviation multiplier, which is a per-layer model constant, \
        and a bind cannot ask for it. FLOOR: the row bound \
        Source::CtxByLayer(\"altup_std_mult\") and \
        DispatchCtx::altup_std_mult(layer) (bind/mod.rs:1310) is the \
        accessor; needs `Facts::altup_std_mult(layer) -> Option<f32>`, and \
        `Cx::layer()` already answers the index" },

    GEGLU_TANH_BF16 => { cx, stream => {
        let n = elements(cx)?;
        unsafe {
            geglu_tanh_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                n,
                stream,
            )
        }
        .ok()
    }},

    CHUNKED_GEGLU_TANH_BF16 => { cx, stream => {
        unsafe {
            chunked_geglu_tanh_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                cx.out_width(0)?,
                false,
                stream,
            )
        }
        .ok()
    }},

    GPT_OSS_GLU_BF16 => { cx, stream => {
        let n = elements(cx)?;
        unsafe {
            gpt_oss_glu_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                None,
                n,
                cx.param_f32(0)?,
                GPT_OSS_GLU_ALPHA,
                stream,
            )
        }
        .ok()
    }},

    MOE_SHARED_GATE_DOT_BF16 => { cx, stream => {
        unsafe {
            sigmoid_dot_scalar_gate_add_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.weight(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_in(2)?.cast_const().cast::<bf16>(),
                cx.rows().count,
                cx.out_width(0)?,
                stream,
            )
        }
        .ok()
    }},
}

/// `Source::OutElements(0)`, in the vocabulary `Cx` has.
#[cfg(feature = "_cuda")]
fn elements(cx: &crate::x::Cx<'_>) -> Result<i32, Refusal> {
    let rows = cx.rows().count;
    let width = cx.out_width(0)?;
    Ok(rows.saturating_mul(width))
}
