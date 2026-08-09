//! Launchers the DRIVER reaches for directly, and NOT a family.
//!
//! Every kernel fired here belongs to `attn`, `norm`, `layout`, `mlp` or
//! `ssm`; what this module holds is the geometry a driver-side caller wants
//! for it. So there is no `FAMILY`, no `ROUTINES` and no line in `lib.rs`:
//! these are plain `pub fn`s `driver-cuda` calls by path, no statement names
//! them, and there is nothing for a trace to resolve.

use core::ffi::c_void;

use crate::jit::{Ctx, Launch};
use crate::x::Abi;
use crate::x::abi::bf16;
use crate::x::{attn, layout, mlp, norm, ssm};
use kernels::Refusal;

/// `runtime/launch.rs` — `const BLOCK: u32 = 256;`, the block the two
/// pointwise launches here take.
const BLOCK: u32 = 256;

/// The QKV split — `attn::split_qkv_bf16`, `attn/split_packed.cuh:74`.
///
/// # Safety
///
/// `packed` is `[n_tokens, q_dim + 2 * kv_dim]` bf16; `q_out` is
/// `[n_tokens, q_dim]` and `k_out`/`v_out` are `[n_tokens, kv_dim]`, all
/// bf16 and all writable. All four live on `ctx`'s stream, which must
/// outlive the launch.
pub fn split_qkv_bf16(
    ctx: &Ctx,
    packed: *const c_void,
    q_out: *mut c_void,
    k_out: *mut c_void,
    v_out: *mut c_void,
    n_tokens: i32,
    q_dim: i32,
    kv_dim: i32,
) -> Result<(), Refusal> {
    if n_tokens <= 0 {
        return Err(Refusal::Empty { what: "n_tokens" });
    }
    if q_dim <= 0 && kv_dim <= 0 {
        return Err(Refusal::Empty { what: "q_dim and kv_dim" });
    }
    let width = q_dim.max(kv_dim).unsigned_abs();
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            &attn::split_packed::ROOT,
            attn::split_packed::inst::SPLIT_QKV,
            Launch::grid([width.div_ceil(BLOCK), n_tokens.unsigned_abs(), 1], [BLOCK, 1, 1]),
            &[
                packed.cast::<bf16>().arg(),
                q_out.cast::<bf16>().arg(),
                k_out.cast::<bf16>().arg(),
                v_out.cast::<bf16>().arg(),
                q_dim.arg(),
                kv_dim.arg(),
            ],
        )
    }
}

/// The bias add — `norm::add_bias_bf16`, `norm/add_bias.cuh`.
///
/// The geometry is [`norm::add_bias_bf16`]'s own, so this is the cast and
/// nothing else: the driver hands `void*` where the routine takes `bf16*`.
///
/// # Safety
///
/// `out` is `[num_rows, dim]` bf16 and writable, `bias` is `[dim]` bf16.
/// Both live on `ctx`'s stream.
pub fn add_bias_bf16(
    ctx: &Ctx,
    out: *mut c_void,
    bias: *const c_void,
    num_rows: i32,
    dim: i32,
) -> Result<(), Refusal> {
    norm::add_bias_bf16(ctx, out.cast::<bf16>(), bias.cast::<bf16>(), num_rows, dim)
}

/// Qwen3.5's post-convolution split — `ssm::qwen_gdn_post_conv_prep_bf16`,
/// `gated_delta_net.cu:139-168`.
///
/// # Safety
///
/// `qkv_post` is `[N, conv_dim]` bf16; `a`, `b` and `dt_bias` are bf16 over
/// `[N, V_h]`, `[N, V_h]` and `[V_h]`; `a_log` is `[V_h]` fp32; the five
/// outputs are writable for `[N, K_h, K_d]`, `[N, K_h, K_d]`,
/// `[N, V_h, V_d]`, `[N, V_h]` and `[N, V_h]`. All live on `ctx`'s stream.
#[allow(clippy::too_many_arguments)]
pub fn qwen_gdn_post_conv_prep_bf16(
    ctx: &Ctx,
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
) -> Result<(), Refusal> {
    if n <= 0 {
        return Err(Refusal::Empty { what: "N" });
    }
    if k_h <= 0 || v_h <= 0 {
        return Err(Refusal::Empty { what: "K_h or V_h" });
    }
    if k_d <= 0 || v_d <= 0 {
        return Err(Refusal::Empty { what: "K_d or V_d" });
    }
    /// `gated_delta_net.cu:154` — `constexpr int BLOCK = 128;`.
    const PREP_BLOCK: u32 = 128;
    #[allow(clippy::cast_precision_loss)]
    let q_scale = (k_d as f32).sqrt().recip();
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            &ssm::gated_delta_net_prep::ROOT,
            ssm::gated_delta_net_prep::inst::QWEN_QK_NORM,
            Launch::grid([n.unsigned_abs(), k_h.unsigned_abs(), 1], [PREP_BLOCK, 1, 1]),
            &[
                qkv_post.arg(),
                q_norm_kh.arg(),
                k_norm_kh.arg(),
                k_h.arg(),
                k_d.arg(),
                conv_dim.arg(),
                q_scale.arg(),
            ],
        )?;
    }
    // SAFETY: the caller's assertion, forwarded. No barrier between the two
    // launches: the second reads `qkv_post` again rather than anything the
    // first wrote, so the stream's own ordering is all they need.
    unsafe {
        ctx.launch(
            &ssm::gated_delta_net_prep::ROOT,
            ssm::gated_delta_net_prep::inst::QWEN_V_G_BETA,
            Launch::grid([n.unsigned_abs(), v_h.unsigned_abs(), 1], [PREP_BLOCK, 1, 1]),
            &[
                qkv_post.arg(),
                a.arg(),
                b.arg(),
                a_log.cast::<f32>().arg(),
                dt_bias.arg(),
                v_fp32.arg(),
                g_log_out.arg(),
                beta_out.arg(),
                k_h.arg(),
                v_h.arg(),
                k_d.arg(),
                v_d.arg(),
                conv_dim.arg(),
            ],
        )
    }
}

/// The per-head query/gate split — `layout::split_q_gate_bf16`,
/// `layout/deinterleave.cuh:130`.
///
/// # Safety
///
/// `packed` is `[n, num_heads, 2 * head_dim]` bf16; `q_out` and `gate_out`
/// are `[n, num_heads, head_dim]` bf16 and writable. All live on `ctx`'s
/// stream.
pub fn split_q_gate_bf16(
    ctx: &Ctx,
    packed: *const c_void,
    q_out: *mut c_void,
    gate_out: *mut c_void,
    n: i32,
    num_heads: i32,
    head_dim: i32,
) -> Result<(), Refusal> {
    if n <= 0 {
        return Err(Refusal::Empty { what: "n" });
    }
    if num_heads <= 0 {
        return Err(Refusal::Empty { what: "num_heads" });
    }
    if head_dim <= 0 {
        return Err(Refusal::Empty { what: "head_dim" });
    }
    let block = if head_dim < 128 { 64 } else { 128 };
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            &layout::deinterleave::ROOT,
            layout::deinterleave::inst::SPLIT_Q_GATE,
            Launch::grid([n.unsigned_abs(), num_heads.unsigned_abs(), 1], [block, 1, 1]),
            &[
                packed.cast::<bf16>().arg(),
                q_out.cast::<bf16>().arg(),
                gate_out.cast::<bf16>().arg(),
                n.arg(),
                num_heads.arg(),
                head_dim.arg(),
            ],
        )
    }
}

/// That gate applied — `mlp::sigmoid_gate_inplace_bf16`, `mlp/swiglu.cuh:261`.
///
/// # Safety
///
/// `x` and `gate` are both `num_elements` bf16 elements; `x` is writable and
/// is read and written by the same threads. Both live on `ctx`'s stream.
pub fn sigmoid_gate_inplace_bf16(
    ctx: &Ctx,
    x: *mut c_void,
    gate: *const c_void,
    num_elements: i32,
) -> Result<(), Refusal> {
    if num_elements <= 0 {
        return Err(Refusal::Empty { what: "num_elements" });
    }
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            &mlp::swiglu::ROOT,
            mlp::swiglu::inst::SIGMOID_GATE_INPLACE,
            Launch::flat(num_elements.unsigned_abs(), BLOCK),
            &[x.cast::<bf16>().arg(), gate.cast::<bf16>().arg(), num_elements.arg()],
        )
    }
}

/// The gated norm with an FP32 `x` — `norm::rmsnorm_gated_fp32_in_bf16`,
/// `norm/rmsnorm.cuh:763`.
///
/// One block per row is [`norm::rmsnorm_gated_fp32_in_bf16`] at
/// `per_head_dim = 0`, so the geometry is that routine's. The two refusals
/// are not: the routine takes a row count it trusts, and the driver's
/// `hidden` and `num_rows` are read off a layer.
///
/// # Safety
///
/// `x` is `[num_rows, hidden]` fp32; `gate` is `[num_rows, hidden]` bf16;
/// `weight` is `[hidden]` fp32; `y` is `[num_rows, hidden]` bf16 and
/// writable. All live on `ctx`'s stream.
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_gated_fp32_in_bf16(
    ctx: &Ctx,
    x: *const c_void,
    gate: *const c_void,
    weight: *const c_void,
    y: *mut c_void,
    num_rows: i32,
    hidden: i32,
    eps: f32,
) -> Result<(), Refusal> {
    if num_rows <= 0 {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    if hidden <= 0 {
        return Err(Refusal::Empty { what: "hidden" });
    }
    norm::rmsnorm_gated_fp32_in_bf16(
        ctx,
        x.cast::<f32>(),
        gate.cast::<bf16>(),
        weight.cast::<f32>(),
        y.cast::<bf16>(),
        num_rows,
        hidden,
        0,
        eps,
    )
}
