use core::ffi::c_void;

use crate::x::abi::bf16;
use crate::x::contract::{Fired, Refusal};
use crate::x::launch::Launch;

/// `runtime/launch.rs:578` — `const BLOCK: u32 = 256;`, the block every
const BLOCK: u32 = 256;

/// `runtime/launch.rs:584` — `const WARP: u32 = 32;`.
const WARP: u32 = 32;

/// `runtime/launch.rs:581` — `const MAX_BLOCK: u32 = 1024;`, the cap
const MAX_BLOCK: u32 = 1024;

/// The QKV split — `attn::split_qkv_bf16`, `attn/split_packed.cuh:74`.
///
/// # Safety
///
/// `packed` is `[n_tokens, q_dim + 2 * kv_dim]` bf16; `q_out` is
/// `[n_tokens, q_dim]` and `k_out`/`v_out` are `[n_tokens, kv_dim]`, all
/// bf16 and all writable. All four live on `stream`, which must outlive the
/// launch.
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn split_qkv_bf16(
    packed: *const c_void,
    q_out: *mut c_void,
    k_out: *mut c_void,
    v_out: *mut c_void,
    n_tokens: i32,
    q_dim: i32,
    kv_dim: i32,
    stream: *mut c_void,
) -> Fired {
    if n_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "n_tokens" });
    }
    if q_dim <= 0 && kv_dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "q_dim and kv_dim" });
    }
    #[allow(clippy::cast_sign_loss)]
    let width = q_dim.max(kv_dim) as u32;
    #[allow(clippy::cast_sign_loss)]
    let launch = Launch {
        grid: [width.div_ceil(BLOCK), n_tokens as u32, 1],
        block: [BLOCK, 1, 1],
        smem: 0,
        smem_opt_in: false,
    };
    // SAFETY: the caller's assertion, forwarded. The device row states six
    unsafe {
        crate::x::attn::split_packed::raw::split_qkv(
            "attn::split_qkv_bf16",
            launch,
            packed.cast::<bf16>(),
            q_out.cast::<bf16>(),
            k_out.cast::<bf16>(),
            v_out.cast::<bf16>(),
            q_dim,
            kv_dim,
            stream,
        );
    }
    Fired::Launched
}

/// The bias add — `norm::add_bias_bf16`, `norm/add_bias.cuh`.
///
/// # Safety
///
/// `out` is `[num_rows, dim]` bf16 and writable, `bias` is `[dim]` bf16.
/// Both live on `stream`.
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn add_bias_bf16(
    out: *mut c_void,
    bias: *const c_void,
    num_rows: i32,
    dim: i32,
    stream: *mut c_void,
) -> Fired {
    if num_rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    if dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "dim" });
    }
    #[allow(clippy::cast_sign_loss)]
    let block = (dim as u32).div_ceil(WARP).max(1) * WARP;
    #[allow(clippy::cast_sign_loss)]
    let launch = Launch::per_row(num_rows as u32, block.min(MAX_BLOCK));
    // SAFETY: the caller's assertion, forwarded. Three operands — the row
    unsafe {
        crate::x::norm::add_bias::raw::add_bias(
            "norm::add_bias_bf16",
            launch,
            out.cast::<bf16>(),
            bias.cast::<bf16>(),
            dim,
            stream,
        );
    }
    Fired::Launched
}

/// Qwen3.5's post-convolution split — `ssm::qwen_gdn_post_conv_prep_bf16`,
///
/// # Safety
///
/// `qkv_post` is `[N, conv_dim]` bf16; `a`, `b` and `dt_bias` are bf16 over
/// `[N, V_h]`, `[N, V_h]` and `[V_h]`; `a_log` is `[V_h]` fp32; the five
/// outputs are writable for `[N, K_h, K_d]`, `[N, K_h, K_d]`,
/// `[N, V_h, V_d]`, `[N, V_h]` and `[N, V_h]`. All live on `stream`.
#[cfg(feature = "_cuda")]
#[must_use]
#[allow(clippy::too_many_arguments)]
pub unsafe fn qwen_gdn_post_conv_prep_bf16(
    qkv_post: *const c_void,
    a: *const c_void,
    b: *const c_void,
    a_log: *const c_void,
    dt_bias: *const c_void,
    q_norm_kh: *mut f32,
    k_norm_kh: *mut f32,
    v_fp32: *mut f32,
    g_log_out: *mut f32,
    beta_out: *mut f32,
    n: i32,
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    conv_dim: i32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "N" });
    }
    if k_h <= 0 || v_h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "K_h or V_h" });
    }
    if k_d <= 0 || v_d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "K_d or V_d" });
    }
    /// `gated_delta_net.cu:154` — `constexpr int BLOCK = 128;`.
    const PREP_BLOCK: u32 = 128;
    #[allow(clippy::cast_precision_loss)]
    let q_scale = (k_d as f32).sqrt().recip();
    #[allow(clippy::cast_sign_loss)]
    let qk_launch = Launch {
        grid: [n as u32, k_h as u32, 1],
        block: [PREP_BLOCK, 1, 1],
        smem: 0,
        smem_opt_in: false,
    };
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        crate::x::ssm::gated_delta_net_prep::raw::qwen_gdn_qk_norm(
            "ssm::qwen_gdn_post_conv_prep_bf16#qk_norm",
            qk_launch,
            qkv_post,
            q_norm_kh,
            k_norm_kh,
            k_h,
            k_d,
            conv_dim,
            q_scale,
            stream,
        );
    }
    #[allow(clippy::cast_sign_loss)]
    let vg_launch = Launch {
        grid: [n as u32, v_h as u32, 1],
        block: [PREP_BLOCK, 1, 1],
        smem: 0,
        smem_opt_in: false,
    };
    // SAFETY: the caller's assertion, forwarded. No barrier between the two
    unsafe {
        crate::x::ssm::gated_delta_net_prep::raw::qwen_gdn_v_g_beta(
            "ssm::qwen_gdn_post_conv_prep_bf16#v_g_beta",
            vg_launch,
            qkv_post,
            a,
            b,
            a_log.cast::<f32>(),
            dt_bias,
            v_fp32,
            g_log_out,
            beta_out,
            k_h,
            v_h,
            k_d,
            v_d,
            conv_dim,
            stream,
        );
    }
    Fired::Launched
}

/// The per-head query/gate split — `layout::split_q_gate_bf16`,
///
/// # Safety
///
/// `packed` is `[n, num_heads, 2 * head_dim]` bf16; `q_out` and `gate_out`
/// are `[n, num_heads, head_dim]` bf16 and writable. All live on `stream`.
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn split_q_gate_bf16(
    packed: *const c_void,
    q_out: *mut c_void,
    gate_out: *mut c_void,
    n: i32,
    num_heads: i32,
    head_dim: i32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "n" });
    }
    if num_heads <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_heads" });
    }
    if head_dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "head_dim" });
    }
    let block = if head_dim < 128 { 64 } else { 128 };
    #[allow(clippy::cast_sign_loss)]
    let launch = Launch {
        grid: [n as u32, num_heads as u32, 1],
        block: [block, 1, 1],
        smem: 0,
        smem_opt_in: false,
    };
    // SAFETY: the caller's assertion, forwarded. Six operands — the stream
    unsafe {
        crate::x::layout::deinterleave::raw::split_q_gate(
            "layout::split_q_gate_bf16",
            launch,
            packed.cast::<bf16>(),
            q_out.cast::<bf16>(),
            gate_out.cast::<bf16>(),
            n,
            num_heads,
            head_dim,
            stream,
        );
    }
    Fired::Launched
}

/// That gate applied — `mlp::sigmoid_gate_inplace_bf16`, `mlp/swiglu.cuh:261`.
///
/// # Safety
///
/// `x` and `gate` are both `num_elements` bf16 elements; `x` is writable and
/// is read and written by the same threads. Both live on `stream`.
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn sigmoid_gate_inplace_bf16(
    x: *mut c_void,
    gate: *const c_void,
    num_elements: i32,
    stream: *mut c_void,
) -> Fired {
    if num_elements <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_elements" });
    }
    #[allow(clippy::cast_sign_loss)]
    let launch = Launch::flat(num_elements as u32, BLOCK);
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        crate::x::mlp::swiglu::raw::sigmoid_gate_inplace(
            "mlp::sigmoid_gate_inplace_bf16",
            launch,
            x.cast::<bf16>(),
            gate.cast::<bf16>(),
            num_elements,
            stream,
        );
    }
    Fired::Launched
}

/// The gated norm with an FP32 `x` — `norm::rmsnorm_gated_fp32_in_bf16`,
///
/// # Safety
///
/// `x` is `[num_rows, hidden]` fp32; `gate` is `[num_rows, hidden]` bf16;
/// `weight` is `[hidden]` fp32; `y` is `[num_rows, hidden]` bf16 and
/// writable. All live on `stream`.
#[cfg(feature = "_cuda")]
#[must_use]
#[allow(clippy::too_many_arguments)]
pub unsafe fn rmsnorm_gated_fp32_in_bf16(
    x: *const c_void,
    gate: *const c_void,
    weight: *const c_void,
    y: *mut c_void,
    num_rows: i32,
    hidden: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    if num_rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    if hidden <= 0 {
        return Fired::Declined(Refusal::Empty { what: "hidden" });
    }
    #[allow(clippy::cast_sign_loss)]
    let launch = Launch::per_row(num_rows as u32, BLOCK);
    // SAFETY: the caller's assertion, forwarded. Six operands — the row
    unsafe {
        crate::x::norm::rmsnorm::raw::rmsnorm_gated_f32_in(
            "norm::rmsnorm_gated_fp32_in_bf16",
            launch,
            x.cast::<f32>(),
            gate.cast::<bf16>(),
            weight.cast::<f32>(),
            y.cast::<bf16>(),
            hidden,
            eps,
            stream,
        );
    }
    Fired::Launched
}
