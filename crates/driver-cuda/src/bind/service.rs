//! The rows the DRIVER executes itself — [`Execution::Service`], in Rust.
//!
//! [`Execution::Service`]: kernels_cuda_new::execution::Execution::Service
//!
//! # What this module is
//!
//! `kernels-cuda-new`'s `execution.rs` classifies every row as `Jit`,
//! `Composed` or `Service`, and says of the third: *a symbol whose body is
//! one library call and nothing else is not a kernel, and extracting it as
//! one is extracting nothing.* Fourteen rows are classified that way. Until
//! §45 that classification was **data with no consumer** — the calls were
//! still issued by `gemm/gemm.cpp`, a C++ translation unit, and a row served
//! by C++ is a row that cannot leave the archive.
//!
//! **`gemm/gemm.cpp` is now deleted.** §45 took the four pure-cuBLAS bodies
//! into this module; a later pass took the quantized router into
//! [`crate::bind::quant_gemm`]; and the last pass took the dense bf16
//! autotuner — the largest single thing in the file, and the reason it
//! outlived the rest — into [`crate::fire::gemm`]. Nothing in this tree
//! issues a cuBLAS call from C++ any more. The paragraph above is kept in the
//! past tense because it is the reason this module has the shape it has, not
//! because the condition still holds.
//!
//! This module is the consumer. It issues the same library calls from Rust,
//! through `cudarc`'s dynamically-loaded cuBLAS, and it exists so the C++
//! bodies could be deleted.
//!
//! # The constraint it is written under
//!
//! **The model compiler must not be able to tell whether a symbol is cuBLAS
//! or a JIT'd kernel.** Nothing above the dispatcher changes: [`KernelSig`]
//! is unchanged, the statement lowers the same way, and the arm that reaches
//! a function here is emitted by the same `abi::emit_dispatch` pass that
//! emits the JIT arms and the `pie_k_*` arms, from the same operand list.
//! The only difference is the callee's path, and that difference is decided
//! by one list — `execution::RUST_SERVED` — which no lowering reads.
//!
//! [`KernelSig`]: kernels::KernelSig
//!
//! # Why nothing new links
//!
//! Every entry below reaches cuBLAS through `cudarc::cublas::sys`, whose
//! `fallback-dynamic-loading` build resolves each symbol with `dlopen` on
//! first use. There is no `#[link]`, no `build.rs` flag and no header, so
//! `cargo check -p driver-cuda` with no CUDA toolkit on PATH still passes —
//! the hard gate that made `cudarc` the right seam and a C shim the wrong
//! one. The `cargo:rustc-link-lib=cublas` in `build.rs` is for the C++
//! ARCHIVE's remaining callers, not for this file, and it is why lifting it
//! out of the `bridge` block changes nothing about what this module needs.
//!
//! # A failure is a refusal, never a fallback
//!
//! Each C++ body this replaces ends in `check(status, ...)` or an explicit
//! `throw std::runtime_error(...)`, and the shim's `catch` turns that into an
//! abort with the cuBLAS status in the message. The ports below panic with
//! the same status number and the same shape identification. A non-success
//! status is **not** retried on another algorithm and **not** swallowed: the
//! one place the archive retried — `gemm_bf16_impl`'s
//! `CUBLAS_STATUS_NOT_SUPPORTED` second attempt — is in a body that stays in
//! C++ for an unrelated reason, and none of the four calls here ever had one.
//! `gemm_grouped_bf16_impl` says why in its own comment: *"a failed call
//! inside a graph capture invalidates the capture"*, so a speculative first
//! attempt is worse than no attempt.

use std::ffi::c_void;

use cudarc::cublas::sys::{
    cublasComputeType_t, cublasContext, cublasGemmAlgo_t, cublasGemmEx, cublasGemmGroupedBatchedEx,
    cublasGemmStridedBatchedEx, cublasOperation_t, cublasStatus_t, cudaDataType,
};

use super::DispatchCtx;
use super::abi::AttentionWorkspaceView;
use kernels_cuda_new::x::KvLayer;

use super::abi::KvCacheLayerView;
use super::abi::MlaCacheLayerView;

/// `gemm.cpp:55` — `cublasComputeType_t bf16_compute_type() { return
/// CUBLAS_COMPUTE_32F; }`.
///
/// A function there rather than a constant because the archive once chose
/// between `CUBLAS_COMPUTE_32F` and `CUBLAS_COMPUTE_32F_FAST_16BF` on the
/// device; it does not any more, and the one-line body is the whole of what
/// crosses. **fp32 accumulation of bf16 multiplies** — the arithmetic every
/// parity check in this tree is written against.
const COMPUTE: cublasComputeType_t = cublasComputeType_t::CUBLAS_COMPUTE_32F;

/// `CUBLAS_GEMM_DEFAULT_TENSOR_OP`, which the archive pinned on every one of
/// these calls.
const ALGO: cublasGemmAlgo_t = cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP;

/// The archive's `check(status, api)` — `gemm.cpp:47`.
///
/// Panics rather than returning, and that is the port being faithful and not
/// the port being lazy: the C++ threw, the generated shim caught, printed
/// `pie_k_*: <what>` and called `std::abort`. A `Result` here would have to
/// be swallowed by a dispatch arm that has nothing to do with it, which is a
/// fallback, which §45's rules forbid.
#[track_caller]
fn check(status: cublasStatus_t, what: &str) {
    assert!(
        status == cublasStatus_t::CUBLAS_STATUS_SUCCESS,
        "cuBLAS error ({}): {what}",
        status as i32
    );
}

// ── `gemm_act_x_wt_bf16_out_fp32` MOVED TO `kernels_cuda_new::x::gemm::act_x_wt_bf16_out_fp32` ──
//
// §5 step 5. Its body is verbatim where the device text and the tuner
// are; what stood here was the ~120 lines of argument assembly this
// crate no longer owns. **Its documentation is kept, unedited, as
// comments** — every measurement in it is a measurement about the body,
// and the body did not change when it changed crates:
//
// /// `gemm::act_x_wt_bf16_out_fp32` — one `cublasGemmEx`, bf16 in, fp32 out.
// ///
// /// Ported from `gemm.cpp:1030-1058` (`gemm_bf16_out_fp32_impl`, reached
// /// through the one-line `act_x_wt_bf16_out_fp32` at `:2327`). Row-major
// /// `y[M, N] = act[M, K] @ W[N, K]^T`, written column-major as the transpose,
// /// which is where `OP_T/OP_N` and the `m=N, n=M` swap come from.
// ///
// /// # Safety
// ///
// /// `act` and `w` must address `M*K` and `N*K` live bf16 elements, `y` must
// /// address `M*N` live floats, and all three must outlive the launch — which
// /// is asynchronous on the handle's stream, so "outlive" ends at the next
// /// synchronisation and not at this call's return.

// ── `gemm_grouped_act_x_wt_bf16` MOVED TO `kernels_cuda_new::x::gemm::grouped_act_x_wt_bf16` ──
//
// §5 step 5. Its body is verbatim where the device text and the tuner
// are; what stood here was the ~120 lines of argument assembly this
// crate no longer owns. **Its documentation is kept, unedited, as
// comments** — every measurement in it is a measurement about the body,
// and the body did not change when it changed crates:
//
// /// `gemm::grouped_act_x_wt_bf16` — one `cublasGemmGroupedBatchedEx`.
// ///
// /// Ported from `gemm.cpp:1242-1294` (`gemm_grouped_bf16_impl`, reached
// /// through `grouped_act_x_wt_bf16` at `:1632`). Every group shares `N`, `K`
// /// and the three leading dimensions; only `M` differs, which is why the
// /// arrays are filled from one scalar each and `n[]` from `M_array_host`.
// ///
// /// **This entry takes the handle rather than a [`DispatchCtx`]**, and it is
// /// the one that does. Its row states `Source::Unbound` for every operand — a
// /// group boundary is fire-global and no `Source` names one — so
// /// `emit_dispatch` writes no arm for it and its only consumer is
// /// `fire::lora`'s hand-written staged apply, which holds a `cublasHandle_t`
// /// and no context.
// ///
// /// # Safety
// ///
// /// The three pointer arrays must be HOST arrays of `group_count` device
// /// addresses (cuBLAS reads them on the host for the grouped form), and
// /// `m_array` a host array of `group_count` row counts.

/// The absorb pair's shared call — `cublasGemmStridedBatchedEx` over the head
/// axis, `batchCount = heads`.
///
/// Both MLA absorptions are the same strided-batched GEMM with a different
/// slice of `kv_b_proj` and a different transpose, so the argument assembly
/// is written once. `stride_a` is `(qk_nope_dim + v_head_dim) * kv_lora_rank`
/// for both — the FULL bank stride, because both read a slice of a bank
/// whose per-head pitch includes the other half.
///
/// # Safety
///
/// The caller's, per entry point below.
#[allow(clippy::too_many_arguments)]
unsafe fn absorb(
    handle: *mut c_void,
    op_a: cublasOperation_t,
    a: *const c_void,
    b: *const c_void,
    c: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    stride_a: i64,
    ldb: i32,
    stride_b: i64,
    ldc: i32,
    stride_c: i64,
    heads: i32,
    what: &str,
) {
    let alpha = 1.0f32;
    let beta = 0.0f32;
    // SAFETY: the caller's obligation.
    let status = unsafe {
        cublasGemmStridedBatchedEx(
            handle.cast::<cublasContext>(),
            op_a,
            cublasOperation_t::CUBLAS_OP_N,
            m,
            n,
            k,
            std::ptr::from_ref(&alpha).cast(),
            a,
            cudaDataType::CUDA_R_16BF,
            lda,
            stride_a,
            b,
            cudaDataType::CUDA_R_16BF,
            ldb,
            stride_b,
            std::ptr::from_ref(&beta).cast(),
            c,
            cudaDataType::CUDA_R_16BF,
            ldc,
            stride_c,
            heads,
            COMPUTE,
            ALGO,
        )
    };
    check(status, what);
}

/// `gemm::mla_absorb_q_to_latent_bf16` — `gemm.cpp:2419-2442`.
///
/// Row-major `C[T, kv_lora] = A[T, nope] @ B[nope, kv_lora]` per head,
/// written column-major as `C^T = B^T @ A^T` — which is why both operands
/// are `OP_N` and `kv_b_proj` is the *first*.
///
/// The `tokens <= 0 || heads <= 0` early return is the archive's, kept: it is
/// a HOST decision made before any launch, not a fallback.
///
/// # Safety
///
/// `q_nope` must address `tokens * heads * qk_nope_dim` bf16 elements,
/// `kv_b_proj` the whole `heads * (qk_nope_dim + v_head_dim) * kv_lora_rank`
/// bank, and `q_latent` `tokens * heads * kv_lora_rank` writable elements.
pub unsafe fn gemm_mla_absorb_q_to_latent_bf16(
    ctx: &DispatchCtx,
    q_nope: *const c_void,
    kv_b_proj: *const c_void,
    q_latent: *mut c_void,
    tokens: i32,
    heads: i32,
    qk_nope_dim: i32,
    v_head_dim: i32,
    kv_lora_rank: i32,
) {
    if tokens <= 0 || heads <= 0 {
        return;
    }
    // SAFETY: the caller's obligation, above.
    unsafe {
        absorb(
            ctx.cublas,
            cublasOperation_t::CUBLAS_OP_N,
            kv_b_proj,
            q_nope,
            q_latent,
            kv_lora_rank,
            tokens,
            qk_nope_dim,
            kv_lora_rank,
            i64::from(qk_nope_dim + v_head_dim) * i64::from(kv_lora_rank),
            heads * qk_nope_dim,
            i64::from(qk_nope_dim),
            heads * kv_lora_rank,
            i64::from(kv_lora_rank),
            heads,
            "mla_absorb_q_to_latent_bf16",
        );
    }
}

/// `gemm::mla_absorb_latent_to_v_bf16` — `gemm.cpp:2444-2468`.
///
/// The mirror: row-major `C[T, v_dim] = A[T, kv_lora] @ W[v_dim, kv_lora]^T`
/// per head, so `OP_T` on the weight, and the weight is the SECOND half of
/// each head's bank — `kv_b_proj + qk_nope_dim * kv_lora_rank`, in bf16
/// elements, which is the one pointer arithmetic step this port must not get
/// wrong.
///
/// # Safety
///
/// As [`gemm_mla_absorb_q_to_latent_bf16`], with `attn_latent` in place of
/// `q_nope` and `attn_v` (`tokens * heads * v_head_dim`) as the output.
pub unsafe fn gemm_mla_absorb_latent_to_v_bf16(
    ctx: &DispatchCtx,
    attn_latent: *const c_void,
    kv_b_proj: *const c_void,
    attn_v: *mut c_void,
    tokens: i32,
    heads: i32,
    qk_nope_dim: i32,
    v_head_dim: i32,
    kv_lora_rank: i32,
) {
    if tokens <= 0 || heads <= 0 {
        return;
    }
    // The `__nv_bfloat16*` arithmetic of `gemm.cpp:2452`, in bytes: two per
    // element, and the element count is `qk_nope_dim * kv_lora_rank`.
    // SAFETY: the offset lands inside the same bank the caller guaranteed —
    // `qk_nope_dim * kv_lora_rank` elements into a head pitch of
    // `(qk_nope_dim + v_head_dim) * kv_lora_rank`.
    let wv = unsafe {
        kv_b_proj
            .cast::<u8>()
            .add(2 * (qk_nope_dim as usize) * (kv_lora_rank as usize))
            .cast::<c_void>()
    };
    // SAFETY: the caller's obligation, above.
    unsafe {
        absorb(
            ctx.cublas,
            cublasOperation_t::CUBLAS_OP_T,
            wv,
            attn_latent,
            attn_v,
            v_head_dim,
            tokens,
            kv_lora_rank,
            kv_lora_rank,
            i64::from(qk_nope_dim + v_head_dim) * i64::from(kv_lora_rank),
            heads * kv_lora_rank,
            i64::from(kv_lora_rank),
            heads * v_head_dim,
            i64::from(v_head_dim),
            heads,
            "mla_absorb_latent_to_v_bf16",
        );
    }
}

// ── `gemm_act_x_wt_bf16` MOVED TO `kernels_cuda_new::x::gemm::act_x_wt_bf16` ──
//
// §5 step 5. Its body is verbatim where the device text and the tuner
// are; what stood here was the ~120 lines of argument assembly this
// crate no longer owns. **Its documentation is kept, unedited, as
// comments** — every measurement in it is a measurement about the body,
// and the body did not change when it changed crates:
//
// /// `gemm::act_x_wt_bf16` — the dense bf16 GEMM. Body in
// /// [`crate::fire::gemm::act_x_wt_bf16`].
// ///
// /// `y[M, N] = act[M, K] @ W[N, K]^T + beta * y`, all bf16, fp32 accumulate.
// /// The hottest row in the tree: every linear layer of every model lands here.
// ///
// /// **This is not one cuBLAS call and that is why it took so long to arrive.**
// /// It is a runtime autotuner over three kernel families — the warp-per-row
// /// GEMV, `cublasGemmEx`, and each algorithm cuBLASLt's heuristic offers —
// /// with a per-device tactic memo, an on-disk tactic cache and a fallback
// /// ladder behind it. All of it host code, all of it now Rust; the module
// /// carries the measurements.
// ///
// /// The thing that held it in C++ for three arcs was that `gemm_bf16_impl`
// /// called `gemv_bf16`, whose `bool` meant *"I did not launch"*, and a row
// /// cannot decline. The resolution was not to make the row decline: a
// /// **driver-owned launch is not a row**, so [`crate::fire::gemv::gemv_bf16`]
// /// spells its refusal as a type and the tuner's GEMV candidate is a
// /// `matches!(.., Gemv::Launched)` in the same short-circuiting position the
// /// C++ put it in.
// ///
// /// # Why the handle is an operand and `ctx` is not enough
// ///
// /// The row states `handle: CublasHandle <- Source::Ctx("cublas")`, so the
// /// emitted arm passes both `ctx` and the bound handle — the same redundancy
// /// [`gemm_act_x_wt_bias_bf16`] documents, and for the same reason: the
// /// composition takes this row as its first step and `Composition::agrees`
// /// type-checks `Take::From(i)` against the operands as stated. They are the
// /// same pointer; `ctx.cublas` is the engine's handle, created once at boot by
// /// `device::cublas`.
// ///
// /// # Safety
// ///
// /// `act`, `w` and `y` must address `M*K`, `N*K` and `M*N` live bf16 elements
// /// and outlive the launch — asynchronous on the handle's stream, so "outlive"
// /// ends at the next synchronisation and not at this call's return.
// #[allow(clippy::too_many_arguments)]

// ── `gemm_act_x_wt_bias_bf16` MOVED TO `kernels_cuda_new::x::gemm::act_x_wt_bias_bf16` ──
//
// §5 step 5. Its body is verbatim where the device text and the tuner
// are; what stood here was the ~120 lines of argument assembly this
// crate no longer owns. **Its documentation is kept, unedited, as
// comments** — every measurement in it is a measurement about the body,
// and the body did not change when it changed crates:
//
// /// `gemm::act_x_wt_bias_bf16` — the COMPOSITION, not a service.
// ///
// /// `execution::COMPOSED` already stated this row, step for step, and cited
// /// `gemm.cpp:2395-2398` for it: a `gemm::act_x_wt_bf16` and then a
// /// `norm::add_bias_bf16` over the result. This is that statement, executed.
// /// It is in this module because the seam is the same one — a row the driver
// /// runs itself, with no entry in the C++ shim.
// ///
// /// # What is lost, exactly
// ///
// /// The archive had a second arm: at `M == 1` with a bias, it asked
// /// `dense_tactic_for` whether the tuner's chosen tactic could absorb the bias
// /// into its epilogue, and `run_dense_tactic` declines every tactic except the
// /// warp-per-row GEMV. So the fused arm fired **only** on the GEMV, and its
// /// kernels state what they compute: `out[n] = bf16(bf16(dot) + bias[n])`, the
// /// double rounding deliberate, *"bit-identical to running `add_bias_bf16`
// /// afterwards"*. (That was `gemv.hpp`'s wording; the header is deleted and
// /// the sentence is now at both epilogues of
// /// `kernels-cuda-new/csrc/src/gemm/gemv.cuh`, which is the text NVRTC
// /// compiles.) The composition therefore produces THE SAME BYTES and costs one
// /// extra launch per biased `M == 1` projection.
// ///
// /// That is the whole cost and it is stated rather than measured away: the
// /// fusion was worth 11.9% of gpt-oss-20b's decode time when it was added
// /// (`gemm.hpp`), and what buys it back is a bias epilogue on a JIT'd GEMV.
// /// **That kernel now exists** — the `gemm/gemv` unit's four rows all take
// /// `bias` and fold it, and `fire::gemv::gemv_bf16` passes it through — so what
// /// is missing is no longer a kernel but a Rust caller that reaches it at
// /// `M == 1` instead of reaching `pie_k_gemm_act_x_wt_bf16`, which means the
// /// dense tactic enumeration in Rust. **That enumeration now exists** —
// /// [`crate::fire::gemm`] — so the remaining work is a `fire::gemm` entry that
// /// takes a `bias` and, when the tuned tactic for the shape is
// /// `GemmKind::Gemv`, passes it down instead of adding it afterwards.
// /// [`crate::fire::gemm::dense_tactic_is_gemv`] is the side-effect-free peek
// /// that arm needs, ported and waiting.
// ///
// /// # Safety
// ///
// /// `act`, `w`, `bias` and `y` must address live device memory of the extents
// /// `M`, `N` and `K` describe, and `y` must be writable.
// ///
// /// # Why this one still takes a handle and a stream
// ///
// /// The other four dropped `handle: CublasHandle` from their rows, because a
// /// service carries its own. This row cannot: `execution::COMPOSED` states its
// /// first step as `gemm::act_x_wt_bf16`, whose row DOES take a handle, and
// /// `Composition::agrees` type-checks each `Take::From(i)` against the
// /// composed row's operands. Remove the handle here and the composition can no
// /// longer supply its own first step. So the row keeps the operands the
// /// composition needs, the arm binds them, and `ctx` arrives as well because
// /// every service arm is emitted the same way — the redundancy is the
// /// emitter's uniformity, and `ctx.cublas`/`ctx.stream` are what
// /// `Source::Ctx("cublas")`/`Source::Ctx("stream")` bind to anyway.
// #[allow(clippy::too_many_arguments)]

// ─────────────────────────────────────────────────────────────────────────
// The quantized rows — `gemm.cpp`'s three `WeightView` entry points
// ─────────────────────────────────────────────────────────────────────────
//
// Bodies in [`super::quant_gemm`]; the spellings are here because
// `every_rust_served_symbol_is_spelled_here` reads THIS file's text. Each is
// the `gemm.hpp` inline it replaces: build a `WeightView` from the row's
// operands, then call the one router. `execution::WALKED` states them as
// `Control::Switch { on: "w_dtype" }`, which is what the router is.

/// `gemm::act_x_wt_channel_scaled` — `gemm.hpp:160`.
///
/// `y[M, N] = act[M, K] x W[N, K]^T`, with `W` quantized per output channel:
/// one scale per row of `W`. Serves both FP8 E4M3 and INT8 weights, and the
/// two take completely different routes inside — FP8 per-channel always
/// dequants to bf16 (cuBLASLt has no per-channel FP8 scale mode this tree
/// targets), INT8 per-channel runs the native `CUBLAS_COMPUTE_32I` path.
///
/// `channel_axis` is accepted and NOT read, exactly as the archive's inline
/// accepted and did not read it: the row states it because a per-channel
/// scale has an axis, and every weight this driver materialises is `[N, K]`
/// row-major with the channel on axis 0. A non-zero value is not refused
/// here because the C++ did not refuse it either — recording that is worth
/// more than inventing a check the archive never made.
///
/// # Safety
///
/// Every pointer must be a device address on the current device, `w` must
/// hold at least `N * K` elements of `w_dtype` and `scale` at least `N`
/// values; `y` must be writable for `M * N` bf16. Checked as far as
/// `validate_quant_weight_view` can check it, which is the byte counts.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_act_x_wt_channel_scaled(
    _ctx: &DispatchCtx,
    handle: *mut c_void,
    act: *const c_void,
    w: *const c_void,
    w_dtype: i32,
    w_nbytes: usize,
    scale: *const c_void,
    scale_dtype: i32,
    scale_numel: usize,
    _zero_point: *const c_void,
    _channel_axis: i32,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
) {
    let view = super::quant_gemm::WeightView {
        data: w,
        dtype: w_dtype,
        nbytes: w_nbytes,
        scale_data: scale,
        scale_dtype,
        scale_numel,
        quant_kind: super::quant_gemm::quant_kind::PER_CHANNEL,
        group_size: 0,
    };
    // SAFETY: the caller's obligation, above.
    unsafe {
        super::quant_gemm::act_x_w(
            handle,
            act,
            view,
            y,
            m,
            n,
            k,
            beta,
            super::quant_gemm::dtype::BF16,
            super::quant_gemm::dtype::BF16,
        );
    }
}

/// `gemm::act_x_wt_grouped_scaled` — `gemm.hpp:182`.
///
/// The same GEMM with `W` quantized per group along `K`. `group_size` is the
/// group extent, and for FP8 it is also the extent along `N`: DeepSeek's
/// `weight_block_size = [128, 128]` is a 2-D block scale, which is why
/// `validate_quant_weight_view` counts `ceil(N/gs) * ceil(K/gs)` scales for
/// FP8 and `N * ceil(K/gs)` for everything else.
///
/// **This is the row that reaches the block-scaled W8A8 path** — the one
/// arm here that does not dequant the weight, and the reason it exists is a
/// measurement: re-expanding a block-quantized FP8 weight to bf16 costs 5x
/// the weight bandwidth of the matmul and dominates decode.
///
/// # Safety
///
/// As [`gemm_act_x_wt_channel_scaled`], with the scale count above.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_act_x_wt_grouped_scaled(
    _ctx: &DispatchCtx,
    handle: *mut c_void,
    act: *const c_void,
    w: *const c_void,
    w_dtype: i32,
    w_nbytes: usize,
    scale: *const c_void,
    scale_dtype: i32,
    scale_numel: usize,
    _zero_point: *const c_void,
    group_size: i32,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
) {
    let view = super::quant_gemm::WeightView {
        data: w,
        dtype: w_dtype,
        nbytes: w_nbytes,
        scale_data: scale,
        scale_dtype,
        scale_numel,
        quant_kind: super::quant_gemm::quant_kind::PER_GROUP,
        group_size,
    };
    // SAFETY: the caller's obligation, above.
    unsafe {
        super::quant_gemm::act_x_w(
            handle,
            act,
            view,
            y,
            m,
            n,
            k,
            beta,
            super::quant_gemm::dtype::BF16,
            super::quant_gemm::dtype::BF16,
        );
    }
}

/// `gemm::act_x_wt_mxfp4_marlin` — `gemm.hpp:206`.
///
/// MXFP4: four-bit elements packed two per byte with one raw E8M0 exponent
/// byte per 32-element block. The scale dtype is UINT8 and the group size is
/// 32, and both are asserted rather than defaulted.
///
/// **"marlin" in the name is the checkpoint format's, not a kernel's.** The
/// vendored marlin tree went in §54; this row dequants to bf16 and runs the
/// classic GEMM, which is what the archive's arm did after the removal too.
///
/// # Safety
///
/// `w` must hold at least `ceil(N * K / 2)` bytes and `scale` at least
/// `N * ceil(K / 32)` bytes; `y` writable for `M * N` bf16.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_act_x_wt_mxfp4_marlin(
    _ctx: &DispatchCtx,
    handle: *mut c_void,
    act: *const c_void,
    w: *const c_void,
    w_nbytes: usize,
    scale: *const c_void,
    scale_numel: usize,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
) {
    let view = super::quant_gemm::WeightView {
        data: w,
        dtype: super::quant_gemm::dtype::MXFP4_PACKED,
        nbytes: w_nbytes,
        scale_data: scale,
        scale_dtype: super::quant_gemm::dtype::UINT8,
        scale_numel,
        quant_kind: super::quant_gemm::quant_kind::PER_GROUP,
        group_size: 32,
    };
    // SAFETY: the caller's obligation, above.
    unsafe {
        super::quant_gemm::act_x_w(
            handle,
            act,
            view,
            y,
            m,
            n,
            k,
            beta,
            super::quant_gemm::dtype::BF16,
            super::quant_gemm::dtype::BF16,
        );
    }
}


// `moe::moe_grouped_gemm_bf16` STOOD HERE, as `pub unsafe fn
// moe_moe_grouped_gemm_bf16`, and is DELETED. §5 step 5 took `moe` into
// fn-world: the host program is `x::moe::moe_grouped_gemm_bf16`, a `bind!`
// arm fires it, and the refusal this wrapper had to drop -- the generated
// arm returned `bool` and its `true` meant "a branch ran" -- is now the
// value the fire reports with the symbol named. The symbol left
// `execution::RUST_SERVED` with the family.

// `moe::flashinfer_cutlass_moe_bf16` STOOD HERE, as
// `pub unsafe fn moe_flashinfer_cutlass_moe_bf16`, and is DELETED with the
// fused CUTLASS leg it was the seam to.
//
// It was the driver-op shape entire: a `contract!` in `x::moe` with no
// `Entry`, reached by name through this wrapper into
// `fire::flashinfer_moe::bf16`, because the body needed a workspace query, an
// allocation and an arch probe -- a device surface `Cx` must not grow. That
// shape is what made the retirement a deletion rather than an unpicking:
// nothing but this function reached it, so removing the leg removed the
// whole reachable set.
//
// The decision was the owner's, on a measurement: carrying CUTLASS so NVRTC
// could compile the GEMM is a 505-file, 13,891,303-byte `include_str!`
// closure, against the 429-file, 4,376,255-byte carry this tree already
// refused in writing for cub. Same mechanism, 3.2x a line already drawn.
//
// **What it leaves behind is one bind, and it is not optional.**
// `moe::build_moe_ptrs_aligned_bf16` declares `gu_stage`/`act_stage`/
// `out_stage` -- the destinations every op in the aligned leg writes into --
// and it has never had an arm in either world. Every condition that turned
// the fused leg off already returned the aligned one, so this deletion makes
// the aligned leg the ONLY leg, and it cannot start until that binds.
//
// One behaviour is recorded rather than carried, because it is gone with the
// only caller that could reach it: the C++ `to_cutlass_activation` ended in
// `case Relu2: default:`, so an enumerator this driver had not been taught
// became `Relu2` rather than an error. This wrapper reproduced that and did
// not widen it. Only `Swiglu` was ever reachable through the statement, so
// nothing observed the other two.

// `sample::lm_head_gemv_argmax_int8` — `sample/argmax.hpp:37` — STOOD HERE,
// as `pub unsafe fn sample_lm_head_gemv_argmax_int8`, and is DELETED.
//
// Its doc said, and every sentence is still true of the thing it now names:
//
// > Greedy decode straight off an int8 LM head: for each of `num_rows`
// > hidden vectors, the vocab index whose dequantized dot product is
// > largest, written as one i32. The vocab-wide logit row is never
// > materialised, which is why `table::sample` states this as its own symbol
// > rather than as an `lm_head` GEMM followed by an argmax over its output.
// >
// > # Why it is here rather than behind a `pie_k_` shim
// >
// > Two kernels, a device scratch between them that the row's operand list
// > does not mention, and a grid extent read off
// > `cudaDevAttrMultiProcessorCount`. `execution::WALKED` classifies it as a
// > `Walk` for exactly that — host control flow whose shape comes from the
// > input and from the machine. What reaches this function is one call with
// > eight operands, the same eight the C++ launcher took, and no model text
// > can tell that two `__global__`s run behind it.
//
// §5 step 5 took `sample` into fn-world. The whole program is
// `kernels_cuda_new::x::sample::lm_head_gemv_argmax_int8`, and this wrapper
// is gone because there is nothing left for it to do: it existed to turn a
// generated dispatch arm's argument list into a `fire::` call, and a bind
// body reads a `Cx` instead.
//
// **There is no bind either, and that is the honest end of the last
// paragraph above.** "What reaches this function is one call with eight
// operands" was never true — all eight of the row's operands were
// `Source::Unbound`, `emit_rust_dispatch` skipped the row whole, and nothing
// in `crates/model` states the symbol. `x::sample`'s contract carries a
// written refusal naming the one fact that is still missing: the int8 head
// and its per-row dequant scale are named weights, and no model text names
// them. The refusal is made at model load now instead of being an
// `UnknownKernel` at fire time.


// ── `norm/`: SIX WRAPPERS STOOD HERE AND ARE GONE ─────────────────────
//
// `norm` crossed into fn-world (`.wiki/kernel-x/northstar.md` §5 step 5).
// Its host programs are `kernels-cuda-new/src/x/norm.rs`, beside the six
// `csrc/src/norm/*.cuh` roots they fire, and `bind/mod.rs::dispatch` reaches
// them through `kernels_cuda_new::x::entry` — one lookup, no wrapper.
//
// The six were: `norm_rmsnorm_bf16_with_fp16`,
// `norm_rmsnorm_residual_add_scale_rmsnorm_bf16`,
// `norm_hc_pre_postprocess_bf16`, `norm_hc_post_bf16`,
// `norm_hc_head_postprocess_bf16` and `norm_hc_rmsnorm_to_f32`. See the
// `rope/` note below for why a ported family needs none of them: a family
// that states no `operands` is on neither side of the `RUST_SERVED` fork.
//
// Every measurement in their doc comments moved with the fns. Two claims
// they made are worth repeating because `x::norm` now proves them:
//
//   * *"the middle arm is still TWO launches with the bf16 result as the
//     intermediate, which is what `Composition`'s `Take` cannot spell"* —
//     §2.3's `Composed` shape, and `x::norm::rmsnorm_bf16_with_fp16` is the
//     first body in the tree to write it. Its second launch is
//     `quant::bf16_to_fp16`, another FAMILY's kernel, which is the one thing
//     §2.3 does not cover.
//   * *"the fused arm no longer silently degrades to a different reduction
//     order, which was the §21.14 failure the refusal was protecting."*
//
// `norm::add_bias_bf16` is NOT one of the six and is still fired from this
// file, by the gemm wrapper above: that call goes through `super::jit::fire`
// by symbol and keeps resolving, because `x::norm` declares the same symbol.


// ── `rope/`: NINE WRAPPERS STOOD HERE AND ARE GONE ──────────────────────
//
// `rope` crossed into fn-world (`.wiki/kernel-x/northstar.md` §5 step 3).
// Its host programs are `kernels-cuda-new/src/x/rope.rs`, beside the
// `rope.cuh` they fire, and `bind/mod.rs::dispatch` reaches them through
// `kernels_cuda_new::x::entry` — one lookup, no wrapper.
//
// # Why the wrappers existed, and why nothing needs them now
//
// A wrapper here was the price of `execution::RUST_SERVED`: that list is
// what decides whether `abi::emit_rust_dispatch` writes an arm calling
// `bind::service::<sym>` or `emit_c_shim` writes a `pie_k_*`, so a symbol on
// it had to be spelled here in EXACTLY the row's operand order, including
// the row's `Ty::Stream` position. Three signatures of one kernel —
// the row, the wrapper, and the `fire::` fn — had to agree, and only the
// numeric smoke could tell you when they stopped.
//
// A ported family states no `operands` at all, so it is on neither side of
// that fork: no shim, no generated arm, no wrapper. The host program's
// parameter list is the ONLY host-side spelling of the kernel's signature,
// and the typecheck TU checks it against the `__global__`.
//
// The nine were: `rope_rope_bf16`, `rope_rope_write_kv_bf16`,
// `rope_qk_rmsnorm_rope_bf16_devwin`, `rope_qk_rmsnorm_mrope_bf16`,
// `rope_qk_rmsnorm_rope_bf16_rounded`, `rope_rope_yarn_bf16`,
// `rope_rope_yarn_original_bf16`, `rope_rope_partial_bf16_position_delta`
// and `rope_rope_partial_last_bf16`. Every measurement in their doc
// comments — the `heads_per_block`/`cache_pairs` host conditionals most of
// all — moved with the fns to `x::rope`, which is where the launch that
// uses them now is.

// ── `ssm/`: ELEVEN WRAPPERS STOOD HERE AND ARE GONE ─────────────────────
//
// `ssm` crossed into fn-world (`.wiki/kernel-x/northstar.md` §5 step 5).
// Its host programs are `kernels-cuda-new/src/x/ssm.rs` — twenty-seven of
// them, in five inline `pub mod`s beside the five `.cuh` they fire — and
// `bind/mod.rs::dispatch` reaches them through `kernels_cuda_new::x::entry`,
// one lookup, no wrapper. `driver-cuda/src/fire/{causal_conv1d,
// gated_delta_net,kda,nemotron_h}.rs` are deleted with them.
//
// The eleven were: `ssm_causal_conv1d_prefill_batched_bf16`,
// `ssm_qwen_gdn_post_conv_prep_bf16`,
// `ssm_recurrent_gated_delta_step_batched_gqa_state_bf16`, the four
// `ssm_chunk_gated_delta_prefill_batched{,_state_bf16,_cached,
// _cached_state_bf16}`, `ssm_nemotron_mamba_split_bf16`,
// `ssm_nemotron_mamba_ssm_batched_bf16`, `ssm_kda_recurrent_step_batched`
// and `ssm_kda_prefill_batched`. Every measurement in their doc comments
// moved with the fns to `x::ssm` — the KDA prefill's block width most of
// all, `min(D, 32) * 32`, one warp per state `v` row, **2.2x at T=2048,
// 26.2 ms -> 12.0 ms per layer at K3's widths**, which is on
// `x::ssm::kda_prefill_batched` beside the `<<<>>>` that uses it.
//
// `ssm_qwen_gdn_post_conv_prep_bf16` is the one that did not go to `x::ssm`:
// it is `x::driver_internal::qwen_gdn_post_conv_prep_bf16`, a `fn` with two
// `fire` calls and no `contract!`, called by `bind/mod.rs`'s GDN path
// directly.
//
// # THE PARAGRAPH THIS BLOCK EXISTED TO WRITE DOWN, and it is now history
//
// *"The parameter lists are the TABLE ROW's, not the C++ launcher's, and
// where they differ the table wins. `abi::emit_rust_dispatch` writes the
// operands in row order including the `Ty::Stream` one, so
// `ssm_causal_conv1d_prefill_batched_bf16` takes its stream in the MIDDLE —
// after `k`, before `write_state` — because that is where `table::ssm` put
// it. A signature that 'tidied' the stream to the end would compile and
// would pass a `bool` where a `cudaStream_t` goes."*
//
// **That hazard is gone by construction and it is the sharpest single
// argument for the port this file can make.** There is no row, so there is
// no row order; the host program's parameter list is the ONLY host-side
// spelling of the kernel's signature and it is the one the `unit!` raw stub
// checks against the `__global__`. Three signatures had to agree here and
// only the numeric smoke could tell you when they stopped; one signature
// cannot disagree with itself.
//
// TWO OF THE ELEVEN WERE UNREACHABLE — the KDA pair state no `Source` on any
// operand, so `emit_rust_dispatch` skipped those rows whole and wrote no arm
// to them. They are `none:` arms in `x::ssm::kda` now, which is the same
// fact said out loud: `Route::Unbound` at MODEL LOAD with the sentence,
// rather than a wrapper nothing called and a comment explaining why.


// `moe::moe_gate_up_decode_gemv_bf16` STOOD HERE, as `pub unsafe fn
// moe_moe_gate_up_decode_gemv_bf16`, and is DELETED with `fire::moe_dispatch`
// -- `x::moe::moe_gate_up_decode_gemv_bf16` is the host program and its
// `bind!` arm is the fire.


// `moe::moe_down_decode_gemv_bf16` STOOD HERE, as `pub unsafe fn
// moe_moe_down_decode_gemv_bf16`, and is DELETED for its twin's reason.


// `moe::transpose_expert_scales_u8` STOOD HERE, as `pub unsafe fn
// moe_transpose_expert_scales_u8`, and is DELETED. `x::moe` keeps the host
// program and declares the symbol a `none:`: weight preparation is not a
// trace statement, which is what its five unsourced operands were saying.


// `moe::build_moe_ptrs_aligned_bf16` STOOD HERE, as `pub unsafe fn
// moe_build_moe_ptrs_aligned_bf16`, and is DELETED. `x::moe` keeps the host
// program -- twenty-one parameters, six of them pointer arrays -- and the
// symbol is a DRIVER OP: a `contract!` with no `Entry` at all,
// `Service::DriverOp` in `execution::SERVED`, a body in `fire::moe_ptrs` and
// an arm in `bind/mod.rs`'s driver-op table. It was a `none:` arm for one day
// in between, and the day is the interesting part -- see the gate the arm's
// comment carries.
//
// THE REASON IS THE SENTENCE THIS COMMENT ALREADY HAD and it did not change:
// **the aligned staging is the driver's arena and not the trace's.** What
// changed is what follows from it. Six pointer arrays with no stated
// consumer -- their only reader is the batched-cuBLAS arm INSIDE
// `moe_grouped_gemm_bf16`, a lowering and not a statement -- are six trace
// values `lower.rs:1911` frees at the first op past the build, so declaring
// them would hand that reader bytes the next allocation owns. A wrong
// answer, not a refusal.


// `moe::reorder_moe_aligned_output_bf16` STOOD HERE, as `pub unsafe fn
// moe_reorder_moe_aligned_output_bf16`, and is DELETED. `x::moe` keeps the
// host program, including the vectorisability fork that chooses between two
// symbols before a single launch, and it BINDS: `Cx::in_rows` landed in
// a41a1df0a and the arm reads `cx.in_rows(1)` for the sorted map's row
// count. This comment said `none: until Cx can be asked for an operand's
// row count` for as long as that was true and for a day after it was not,
// which is the failure mode a tombstone has: it is written beside the
// deletion and nothing re-derives it when the sentence it names comes true.
// `x::moe`'s four remaining `none:` arms are `add_moe_route_bias`,
// `transpose_expert_scales`, `moe_bucket_exact` and `scatter_add_weighted`;
// fourteen bind and two are driver ops. It said FIVE until
// `build_moe_ptrs_aligned` took `Service::DriverOp`, which is this same
// paragraph's subject happening to this same paragraph.

// `ssm_build_nemotron_moe_ptrs_decode_batched_bf16` AND
// `ssm_build_nemotron_moe_ptrs_aligned_bf16` STOOD HERE, filed with `moe/`
// rather than with `ssm/` because that is what they feed, and both are GONE
// with the rest of `ssm` — §5 step 5, see the `ssm/` block above.
//
// They are `x::ssm::nemotron_h::build_nemotron_moe_ptrs_{decode_batched,
// aligned}_bf16` now, and they are the family's other two `none:` arms.
//
// # What they recorded, because it is a FINDING and not a status line
//
// *"The `table::ssm` row stays unbound and that is deliberate. §52.3's
// missing `Source::Scratch(name, extent)` still has no word for a slab this
// driver allocated, so no operand is sourced, `emit_rust_dispatch` writes no
// arm, and nothing in a model trace reaches this. What `RUST_SERVED` changed
// is only that the shim no longer emits an entry — which is what let
// `ssm/nemotron_h.cu` be deleted."*
//
// **Still true, and the port does not fix it** — it only moves where the
// sentence is said. A `none:` arm surfaces at MODEL LOAD as `Route::Unbound`
// carrying that reason, instead of a wrapper that compiles, is exported, and
// is called by nothing. The `attn::write_kv_to_pages` note below draws the
// contrast against these two and it still draws it.
//
// The two shapes, kept because they are the only prose on the pair's
// geometry outside `x::ssm`: the DECODE form is one thread per route with
// **`routes = n * top_k` and not `n`** as the bound — the trap
// `nemotron_h.cu:53-94` documents — filling six device-pointer arrays plus
// the router weight copied out as f32; the ALIGNED form is one thread per
// padded block of the sorted MoE layout, `nemotron_h.cu:96-137`, with **four
// guard terms rather than one**, because `block_size`, `hidden` and
// `intermediate` are multipliers inside the kernel's address arithmetic and
// a zero pitch aliases every pointer in the array onto row zero.

/// `attn::write_kv_to_pages` — was `attn/kv_paged.hpp`, and that file's
/// launcher at `kv_paged.cu:109` is DELETED.
///
/// The paged KV append every fire makes once per layer. One
/// `if (layer.is_native_bf16())` over two programs: the native bf16 appender
/// with its envelope refresh, and the five-way `switch (layer.scheme)`.
///
/// Body: [`kernels_cuda_new::x::attn::kv_paged::write_kv_to_pages`], MOVED.
/// Classified
/// `execution::Execution::Walk` — `Control::Switch { on: "layer.scheme" }` —
/// before it was taken over, which is what
/// `every_taken_over_row_was_classified_first` requires.
///
/// **The row stays fully sourced and stays reachable.** Unlike the
/// `ssm::build_nemotron_moe_ptrs_*` pair above, every operand of
/// `table::attn`'s row has a `Source`, so `emit_rust_dispatch` writes an arm
/// and a model trace reaches this function. What `RUST_SERVED` changed is
/// only which language the arm lands in.
///
/// # Panics
///
/// If `first_token != 0` on a cache that is not native bf16 —
/// `kv_paged.cu:130-134`'s `throw`, and see the `Walk`'s refusals.
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch,
/// and `stream` is the fire's stream, held live for the same window.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_write_kv_to_pages(
    _ctx: &DispatchCtx,
    layer: KvCacheLayerView,
    k_curr: *const c_void,
    v_curr: *const c_void,
    qo_indptr: *const u32,
    kv_page_indices: *const u32,
    kv_page_indptr: *const u32,
    kv_last_page_lens: *const u32,
    total_tokens: i32,
    num_requests: i32,
    stream: *mut c_void,
    row_valid: *const u8,
    first_token: i32,
) {
    // The layer view, as the seventeen facts the moved body takes. The `Err`
    // arm is stated once for all four entries in this file: it means a
    // producer put a dtype in a KV page that `KvDType` says a KV page cannot
    // hold (`fire/kv_paged.rs`'s `TryFrom` argues it). Nothing is launched,
    // which is what the bind arm replacing this entry will also do, because
    // `Cx::kv_layer()` returns `None` on exactly the same input.
    let Ok(layer) = KvLayer::try_from(&layer) else {
        return;
    };
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    let _ = unsafe {
        kernels_cuda_new::x::attn::kv_paged::write_kv_to_pages(
            &layer,
            k_curr.cast(),
            v_curr.cast(),
            qo_indptr,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_lens,
            total_tokens,
            num_requests,
            stream,
            row_valid,
            first_token,
        )
    };
}

/// `attn::write_kv_explicit_bf16` — was `attn/kv_paged.hpp`, and that file's
/// launcher at `kv_paged.cu:304` is DELETED.
///
/// The explicit append: the fire states each token's destination page and
/// offset rather than deriving them from the CSR, and the envelope refresh
/// behind it merges from that descriptor instead of from the page list.
///
/// Body: [`kernels_cuda_new::x::attn::kv_paged::write_kv_explicit_bf16`],
/// MOVED. Classified
/// `Execution::Walk` first — the conditional second launch is the control
/// flow, and `families/attn.rs`' `_dev` symbol split is what makes
/// `unit_of("attn::write_kv_explicit_bf16")` `None` so §52.11 holds.
///
/// # Panics
///
/// If the cache is not native bf16 — `kv_paged.cu:314-317`'s `throw`.
///
/// # Safety
///
/// As [`attn_write_kv_to_pages`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_write_kv_explicit_bf16(
    _ctx: &DispatchCtx,
    layer: KvCacheLayerView,
    k_curr: *const c_void,
    v_curr: *const c_void,
    w_page: *const u32,
    w_off: *const u32,
    b: i32,
    stream: *mut c_void,
    row_valid: *const u8,
) {
    let Ok(layer) = KvLayer::try_from(&layer) else {
        return;
    };
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    let _ = unsafe {
        kernels_cuda_new::x::attn::kv_paged::write_kv_explicit_bf16(
            &layer,
            k_curr.cast(),
            v_curr.cast(),
            w_page,
            w_off,
            b,
            stream,
            row_valid,
        )
    };
}

/// `attn::dequant_kv_cache_layer_to_bf16_active` — was `attn/kv_paged.hpp`,
/// and `attn/kv_paged.cu` is DELETED **with the whole of its four `<<<>>>`,
/// which were the last in the tree.**
///
/// Widen a quantised layer's ACTIVE pages to bf16 so an attention kernel that
/// only reads bf16 can read them. A five-way `switch (layer.scheme)`, one
/// launch per arm, one grid for all four.
///
/// Body:
/// [`kernels_cuda_new::x::attn::kv_paged::dequant_kv_cache_layer_to_bf16_active`],
/// MOVED.
/// Classified `execution::Execution::Walk` — `Control::Switch { on:
/// "layer.scheme" }` — before it was taken over, which is what
/// `every_taken_over_row_was_classified_first` requires. It is the third
/// `kv_paged` symbol on that list and the third with the same discriminant;
/// see the sibling above.
///
/// # This function has FOUR sibling callers and they do not come through here
///
/// [`attn_dispatch_attention_flashinfer_decode`] and the three other FA2
/// entry points below call
/// `x::attn::kv_paged::dequant_kv_cache_layer_to_bf16_active` DIRECTLY,
/// because the C++ they replaced did: `attention_flashinfer.cu:648`, `:675`,
/// `:1098` and `:1244` were C++ calling C++ by symbol with no shim between,
/// and a dispatch cannot be inserted into the middle of another dispatch's
/// body. This function is the FIFTH caller and the only one a model trace
/// reaches — `model-compiler/src/dsl.rs:7750` states the symbol and
/// `lower.rs:1100` names it, so a plan that wants the widening ahead of
/// something other than FA2 has a row to fire.
///
/// Five callers of one Rust function is not five copies of a switch, which
/// is the thing `fire/kv_paged.rs`' header refuses. It is the opposite: the
/// reason that file could finally hold the switch at all was that the four
/// C++ call sites became calls into it.
///
/// # Declines, and why nothing here reports one
///
/// A `Fired::Declined` is returned for a
/// native-bf16 layer, an empty batch, and a `Native` scheme reached at the
/// switch. All three mean NOTHING WAS LAUNCHED AND NOTHING NEEDED TO BE —
/// the C++ spelled the first two `return` and the third `break` — so the
/// value is dropped here exactly as the two siblings above drop theirs. A
/// decline that meant a caller must do something else would be a panic, as
/// it is for every FA2 entry point below.
///
/// # Panics
///
/// If the kernel table and this driver disagree —
/// [`crate::fire::hand::fire`]'s contract. There is no `throw` in the C++ to
/// reproduce: this launcher had none.
///
/// # Safety
///
/// As [`attn_write_kv_to_pages`]: every pointer in `layer` is a device
/// address of the extent the layer describes, `kv_page_indices` holds
/// `num_pages_in_batch` entries, and `stream` outlives the launch.
pub unsafe fn attn_dequant_kv_cache_layer_to_bf16_active(
    _ctx: &DispatchCtx,
    layer: KvCacheLayerView,
    kv_page_indices: *const u32,
    num_pages_in_batch: i32,
    stream: *mut c_void,
) {
    let Ok(layer) = KvLayer::try_from(&layer) else {
        return;
    };
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    let _ = unsafe {
        kernels_cuda_new::x::attn::kv_paged::dequant_kv_cache_layer_to_bf16_active(
            &layer,
            kv_page_indices,
            num_pages_in_batch,
            stream,
        )
    };
}

/// `attn::mla_prepare_bf16` — was `attn/mla_paged.hpp`, and that file is
/// DELETED.
///
/// The MLA prologue in one kernel: the `kv_a` RMSNorm, the `k_pe` rotation,
/// the paged write of both, and the query-side nope/pe split.
///
/// Body: [`crate::fire::mla_paged::mla_prepare_bf16`]. Classified
/// `Execution::Walk` with `Control::Supplies` first — `heads_per_block` is an
/// operand AND the grid's head axis, which is the case that variant exists
/// for. The `table::attn` row is unsourced on every operand, so this took the
/// shim entry and nothing else (§60.7).
///
/// # Safety
///
/// The caller's; every pointer is a device address live across the launch.
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub unsafe fn attn_mla_prepare_bf16(
    _ctx: &DispatchCtx,
    layer: MlaCacheLayerView,
    kv_a: *const c_void,
    kv_a_norm_weight: *const c_void,
    q_b: *const c_void,
    kv_c: *mut c_void,
    k_pe: *mut c_void,
    q_nope: *mut c_void,
    q_pe: *mut c_void,
    positions: *const i32,
    qo_indptr: *const u32,
    kv_page_indices: *const u32,
    kv_page_indptr: *const u32,
    kv_last_page_lens: *const u32,
    total_tokens: i32,
    num_requests: i32,
    heads: i32,
    qk_nope_head_dim: i32,
    eps: f32,
    theta: f32,
    interleaved: bool,
    kv_a_row_stride: i32,
    yarn: Option<crate::fire::mla_paged::YarnOriginal>,
    stream: *mut c_void,
    row_valid: *const u8,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    let _ = unsafe {
        crate::fire::mla_paged::mla_prepare_bf16(
            layer,
            kv_a,
            kv_a_norm_weight,
            q_b,
            kv_c,
            k_pe,
            q_nope,
            q_pe,
            positions,
            qo_indptr,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_lens,
            total_tokens,
            num_requests,
            heads,
            qk_nope_head_dim,
            eps,
            theta,
            interleaved,
            kv_a_row_stride,
            yarn,
            stream,
            row_valid,
        )
    };
}

/// `attn::write_mla_to_pages` — was `attn/mla_paged.hpp`, and that file is
/// DELETED.
///
/// Appends one step's compressed latent and rope plane to the paged MLA
/// cache. The C++ forwarder's callee `write_mla_to_pages_bf16` had an empty
/// consumer set and is deleted outright; its `<<<>>>` is in the Rust.
///
/// Body: [`crate::fire::mla_paged::write_mla_to_pages`]. Classified
/// `Execution::Walk` with `Control::Supplies` — the three layer fields the
/// kernel takes and the dispatch passes as one view.
///
/// # Safety
///
/// [`attn_mla_prepare_bf16`]'s.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_write_mla_to_pages(
    _ctx: &DispatchCtx,
    layer: MlaCacheLayerView,
    ckv_curr: *const c_void,
    kpe_curr: *const c_void,
    qo_indptr: *const u32,
    kv_page_indices: *const u32,
    kv_page_indptr: *const u32,
    kv_last_page_lens: *const u32,
    total_tokens: i32,
    num_requests: i32,
    stream: *mut c_void,
    row_valid: *const u8,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    let _ = unsafe {
        crate::fire::mla_paged::write_mla_to_pages(
            layer,
            ckv_curr,
            kpe_curr,
            qo_indptr,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_lens,
            total_tokens,
            num_requests,
            stream,
            row_valid,
        )
    };
}

// `layout::embed_bf16` — was `layout/embed.hpp`, and that file is DELETED
// with its `.cu` and the whole of `kernels-cuda/csrc/src/layout/` — STOOD
// HERE, as `pub unsafe fn layout_embed_bf16`, and is DELETED.
//
// Its doc read:
//
// > The first launch of every fire: one row of the vocabulary table per
// > token.
// >
// > Body: `crate::fire::embed::embed_bf16`. Classified `Execution::Walk`
// > with `Control::Switch` — `embed<true>` or `embed<false>`, chosen from a
// > 16-byte alignment test on `weight` and `y` plus `hidden % 8`, which also
// > sizes the grid.
//
// §5 step 5 took `layout` into fn-world. The program is
// `kernels_cuda_new::x::layout::embed_bf16`; the switch is
// `x::layout::vectorisable`, a `pub fn` a caller staging its own buffers can
// ask directly; and the bind that reads the operands off a `Cx` is
// `x::layout`'s `EMBED` arm. Nothing calls this wrapper any more: it existed
// to turn a generated dispatch arm's argument list into a `fire::` call.
//
// The one caller-visible fact worth keeping in reach: `weight` and `y` were
// `*const c_void`/`*mut c_void` here and are `*const bf16`/`*mut bf16` in
// the declaration. The opaque spelling was the shim's, because a `pie_k_`
// entry point is `extern "C"`; the typed one is the `.cuh`'s, and it is what
// the typecheck translation unit compares.

/// `attn::split_qkv_bf16_devwin` — was `attn/split_packed.hpp`, and that file
/// is DELETED with its `.cu`.
///
/// The device-window split: Q, K and V out of a packed activation, visiting
/// only the rows a device-resident window admits.
///
/// Body: [`crate::fire::split_packed::split_qkv_bf16_devwin`]. Classified
/// `Execution::Walk` with `Control::Supplies`. The row moved from
/// `table::driver_internal` to `table::attn` in the same change so
/// `RUST_SERVED` could take it.
///
/// # Safety
///
/// The caller's. The four buffer pointers must be BASE pointers — the kernel
/// windows them itself from `win_d`, so a pre-windowed pointer is windowed
/// twice.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_split_qkv_bf16_devwin(
    _ctx: &DispatchCtx,
    packed: *const c_void,
    q_out: *mut c_void,
    k_out: *mut c_void,
    v_out: *mut c_void,
    win_d: *const u32,
    n_max: i32,
    q_dim: i32,
    kv_dim: i32,
    stream: *mut c_void,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    let _ = unsafe {
        crate::fire::split_packed::split_qkv_bf16_devwin(
            packed, q_out, k_out, v_out, win_d, n_max, q_dim, kv_dim, stream,
        )
    };
}

/// `attn::compact_page_csr` — was `attn/page_compact.hpp`, and that file is
/// DELETED with its `.cu`.
///
/// Drops the pages a keep-mask rejects and rewrites the CSR so the survivors
/// are contiguous. Two launches on one stream, the second reading the
/// `scratch_counts` buffer the first fills.
///
/// Body: [`crate::fire::page_compact::compact_page_csr`]. Classified
/// `Execution::Composed` — the composition that fires end to end — since the
/// split; this is what takes the row over.
///
/// # Safety
///
/// The caller's; `scratch_counts` must stay live across BOTH launches.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_compact_page_csr(
    _ctx: &DispatchCtx,
    page_indices_in: *const u32,
    page_indptr_in: *const u32,
    last_page_lens_in: *const u32,
    keep: *const u8,
    scratch_counts: *mut u32,
    keep_stride: u32,
    num_requests: i32,
    page_indices_out: *mut u32,
    page_indptr_out: *mut u32,
    last_page_lens_out: *mut u32,
    stream: *mut c_void,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    let _ = unsafe {
        crate::fire::page_compact::compact_page_csr(
            page_indices_in,
            page_indptr_in,
            last_page_lens_in,
            keep,
            scratch_counts,
            keep_stride,
            num_requests,
            page_indices_out,
            page_indptr_out,
            last_page_lens_out,
            stream,
        )
    };
}

/// `attn::mtp_shift_hidden_bf16` — was `attn/attention_naive.hpp`, and that
/// file is DELETED with its `.cu`.
///
/// The previous step's pending hidden state becomes this step's first token,
/// per request.
///
/// Body: [`crate::fire::attention_naive::mtp_shift_hidden_bf16`]. Classified
/// `Execution::Walk` with `Control::Supplies`; the device row was renamed
/// `attn::mtp_shift_hidden_dev` (§60.6) so `unit_of` on this symbol is
/// `None`.
///
/// # Safety
///
/// The caller's; every pointer is a device address live across the launch.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_mtp_shift_hidden_bf16(
    _ctx: &DispatchCtx,
    target_hidden: *const c_void,
    pending_hidden: *const c_void,
    qo_indptr: *const u32,
    slot_ids: *const i32,
    out: *mut c_void,
    total_tokens: i32,
    num_requests: i32,
    hidden_size: i32,
    stream: *mut c_void,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    let _ = unsafe {
        crate::fire::attention_naive::mtp_shift_hidden_bf16(
            target_hidden,
            pending_hidden,
            qo_indptr,
            slot_ids,
            out,
            total_tokens,
            num_requests,
            hidden_size,
            stream,
        )
    };
}

/// `attn::mtp_update_pending_hidden_bf16` — was `attn/attention_naive.hpp`,
/// and that file is DELETED with its `.cu`.
///
/// Stashes each request's last hidden state into the pending buffer.
///
/// Body: [`crate::fire::attention_naive::mtp_update_pending_hidden_bf16`].
///
/// # Safety
///
/// [`attn_mtp_shift_hidden_bf16`]'s.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_mtp_update_pending_hidden_bf16(
    _ctx: &DispatchCtx,
    target_hidden: *const c_void,
    pending_hidden: *mut c_void,
    qo_indptr: *const u32,
    slot_ids: *const i32,
    num_requests: i32,
    hidden_size: i32,
    stream: *mut c_void,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    let _ = unsafe {
        crate::fire::attention_naive::mtp_update_pending_hidden_bf16(
            target_hidden,
            pending_hidden,
            qo_indptr,
            slot_ids,
            num_requests,
            hidden_size,
            stream,
        )
    };
}

/// `attn::dsa_index_knorm_rope_bf16` — was `attn/dsa_indexer.hpp`, and that
/// file is DELETED with its `.cu`.
///
/// Body: [`crate::fire::dsa_indexer::dsa_index_knorm_rope_bf16`].
///
/// # Safety
///
/// The caller's; every pointer is a device address live across the launch.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_dsa_index_knorm_rope_bf16(
    _ctx: &DispatchCtx,
    idx_k: *mut c_void,
    k_norm_weight: *const c_void,
    k_norm_bias: *const c_void,
    positions: *const i32,
    tokens: i32,
    head_dim: i32,
    rope_dim: i32,
    theta: f32,
    eps: f32,
    stream: *mut c_void,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    let _ = unsafe {
        crate::fire::dsa_indexer::dsa_index_knorm_rope_bf16(
            idx_k, k_norm_weight, k_norm_bias, positions, tokens, head_dim,
            rope_dim, theta, eps, stream,
        )
    };
}

/// `attn::dsa_index_q_rope_bf16` — was `attn/dsa_indexer.hpp`, and that file
/// is DELETED with its `.cu`.
///
/// Body: [`crate::fire::dsa_indexer::dsa_index_q_rope_bf16`]. The block width
/// is `round_up(n_heads, 32)` with a one-warp floor, which is why this is a
/// `Walk` and not a rule.
///
/// # Safety
///
/// [`attn_dsa_index_knorm_rope_bf16`]'s.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_dsa_index_q_rope_bf16(
    _ctx: &DispatchCtx,
    idx_q: *mut c_void,
    positions: *const i32,
    tokens: i32,
    n_heads: i32,
    head_dim: i32,
    rope_dim: i32,
    theta: f32,
    stream: *mut c_void,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    let _ = unsafe {
        crate::fire::dsa_indexer::dsa_index_q_rope_bf16(
            idx_q, positions, tokens, n_heads, head_dim, rope_dim, theta,
            stream,
        )
    };
}

/// `attn::dsa_index_topk_mask` — was `attn/dsa_indexer.hpp`, and that file is
/// DELETED with its `.cu`.
///
/// The causal top-k mask. Its row is fully sourced, so this function is a
/// LIVE dispatch target and not merely a shim entry being dropped.
///
/// Body: [`crate::fire::dsa_indexer::dsa_index_topk_mask`].
///
/// # Safety
///
/// [`attn_dsa_index_knorm_rope_bf16`]'s.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_dsa_index_topk_mask(
    _ctx: &DispatchCtx,
    idx_q: *const c_void,
    idx_k: *const c_void,
    idx_w: *const c_void,
    mask: *mut u8,
    tokens: i32,
    n_heads: i32,
    head_dim: i32,
    topk: i32,
    stream: *mut c_void,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    let _ = unsafe {
        crate::fire::dsa_indexer::dsa_index_topk_mask(
            idx_q, idx_k, idx_w, mask, tokens, n_heads, head_dim, topk, stream,
        )
    };
}

// `attn_combine_attn_outputs_bf16` STOOD HERE, and it is gone rather than
// moved: the symbol crossed into fn-world as
// `kernels_cuda_new::x::attn`'s `COMBINE_ATTN_OUTPUTS`, so its `table::attn`
// row is deleted, `emit_rust_dispatch` writes no arm that could call this,
// and a seam with nothing on either side of it is not a seam. Its
// `RUST_SERVED` entry and its `execution::WALKED` classification went in the
// same change.

/// `attn::dsv4_boundary_meta_decode` — was `attn/dsv4_compress.hpp`, and that
/// file is DELETED with its `.cu`.
///
/// Body: [`crate::fire::dsv4_compress::dsv4_boundary_meta_decode`].
///
/// # Safety
///
/// The caller's; every pointer is a device address live across the launch.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_dsv4_boundary_meta_decode(
    _ctx: &DispatchCtx,
    positions: *const i32,
    out_pos: *mut i32,
    out_req: *mut i32,
    out_rope: *mut i32,
    n: i32,
    ratio: i32,
    stream: *mut c_void,
    row_valid: *const u8,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    let _ = unsafe {
        crate::fire::dsv4_compress::dsv4_boundary_meta_decode(
            positions, out_pos, out_req, out_rope, n, ratio, stream, row_valid,
        )
    };
}

/// `attn::dsv4_boundary_meta_paged` — was `attn/dsv4_compress.hpp`, and that
/// file is DELETED with its `.cu`.
///
/// Body: [`crate::fire::dsv4_compress::dsv4_boundary_meta_paged`].
///
/// # Safety
///
/// [`attn_dsv4_boundary_meta_decode`]'s.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_dsv4_boundary_meta_paged(
    _ctx: &DispatchCtx,
    positions: *const i32,
    qo_indptr: *const u32,
    out_pos: *mut i32,
    out_req: *mut i32,
    out_rope: *mut i32,
    n: i32,
    num_requests: i32,
    ratio: i32,
    stream: *mut c_void,
    row_valid: *const u8,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    let _ = unsafe {
        crate::fire::dsv4_compress::dsv4_boundary_meta_paged(
            positions, qo_indptr, out_pos, out_req, out_rope, n, num_requests,
            ratio, stream, row_valid,
        )
    };
}

/// `attn::attention_compressed_paged_bf16` — was `attn/dsv4_compress.hpp`,
/// and that file is DELETED with its `.cu`.
///
/// Attention against the compressed KV pages. `qo_indptr` is accepted and not
/// forwarded, exactly as the C++ did — it was `/*qo_indptr*/` in the
/// launcher's own parameter list.
///
/// Body: [`crate::fire::dsv4_compress::attention_compressed_paged_bf16`].
///
/// # Safety
///
/// [`attn_dsv4_boundary_meta_decode`]'s.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_attention_compressed_paged_bf16(
    _ctx: &DispatchCtx,
    q: *const c_void,
    comp_kv_pages: *const c_void,
    o: *mut c_void,
    lse_out: *mut f32,
    positions: *const i32,
    qo_indptr: *const u32,
    kv_page_indices: *const u32,
    kv_page_indptr: *const u32,
    req_of_token: *const i32,
    total_tokens: i32,
    num_q_heads: i32,
    head_dim: i32,
    ratio: i32,
    page_size: i32,
    sm_scale: f32,
    stream: *mut c_void,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    let _ = unsafe {
        crate::fire::dsv4_compress::attention_compressed_paged_bf16(
            q, comp_kv_pages, o, lse_out, positions, qo_indptr,
            kv_page_indices, kv_page_indptr, req_of_token, total_tokens,
            num_q_heads, head_dim, ratio, page_size, sm_scale, stream,
        )
    };
}

/// `attn::qkv_decode_qk_norm_rope_write_kv_bf16` — was `attn/qkv_fused.hpp`,
/// and that file is DELETED with its `.cu`.
///
/// **The one dispatch on this list that was already live.** Its table row is
/// fully sourced — 23 of 23, `stream` included — so this is not a
/// `RUST_SERVED` that only frees a `.cu` (§60.7); it moves a real dispatch
/// off the generated C shim and into Rust. If anything in this pass shows up
/// as a behaviour change rather than a link error, it is this function.
///
/// One launcher over four kernels: `head_dim` picks the warp form (64, 128,
/// 256) or falls through to the block form, and `rope_table != nullptr` picks
/// the `USE_ROPE_TABLE` arm inside whichever it picked.
///
/// Body: [`crate::fire::qkv_fused::qkv_decode_qk_norm_rope_write_kv_bf16`].
///
/// # Safety
///
/// Every pointer is a live device address; `rope_table`, `w_page`, `w_off`
/// and `row_valid` may be null and the kernels test each.
#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
pub unsafe fn attn_qkv_decode_qk_norm_rope_write_kv_bf16(
    _ctx: &DispatchCtx,
    packed: *const c_void,
    q_out: *mut c_void,
    k_pages: *mut c_void,
    v_pages: *mut c_void,
    q_weight: *const c_void,
    k_weight: *const c_void,
    positions: *const i32,
    rope_table: *const f32,
    kv_page_indices: *const u32,
    kv_page_indptr: *const u32,
    kv_last_page_lens: *const u32,
    w_page: *const u32,
    w_off: *const u32,
    row_valid: *const u8,
    num_requests: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    page_size: i32,
    hnd_layout: bool,
    theta: f32,
    eps: f32,
    stream: *mut c_void,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    let _ = unsafe {
        crate::fire::qkv_fused::qkv_decode_qk_norm_rope_write_kv_bf16(
            packed, q_out, k_pages, v_pages, q_weight, k_weight, positions,
            rope_table, kv_page_indices, kv_page_indptr, kv_last_page_lens,
            w_page, w_off, row_valid, num_requests, num_q_heads, num_kv_heads,
            head_dim, page_size, hnd_layout, theta, eps, stream,
        )
    };
}

/// `attn::write_kv_explicit_bf16_devwin` — was `attn/kv_paged.hpp`, and that
/// launcher is DELETED from `attn/kv_paged.cu`.
///
/// The windowed explicit append: each token's destination page and offset are
/// stated by the fire, and `win_d` is a DEVICE window the kernel reads per
/// row, so the grid spans every lane and out-of-window rows early out. That
/// is what makes a captured launch replay across splits.
///
/// §58 held this symbol back for a pass on the reading that a
/// `Specialisation` is already the walk; §60.6 dissolved it by moving the
/// DEVICE rows to `_dev`, which the sibling `attn::write_kv_explicit_bf16`
/// had already done. Its row is fully sourced, so this moves a live dispatch.
///
/// Body:
/// [`kernels_cuda_new::x::attn::kv_paged::write_kv_explicit_bf16_devwin`],
/// MOVED.
///
/// # Safety
///
/// Every pointer is a live device address, `win_d` is NOT nullable — the
/// kernel reads `win[0]` and `win[1]` before any guard — and `row_valid` may
/// be null.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_write_kv_explicit_bf16_devwin(
    _ctx: &DispatchCtx,
    layer: KvCacheLayerView,
    k_curr: *const c_void,
    v_curr: *const c_void,
    w_page: *const u32,
    w_off: *const u32,
    win_d: *const u32,
    n_max: i32,
    stream: *mut c_void,
    row_valid: *const u8,
) {
    let Ok(layer) = KvLayer::try_from(&layer) else {
        return;
    };
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    let _ = unsafe {
        kernels_cuda_new::x::attn::kv_paged::write_kv_explicit_bf16_devwin(
            &layer,
            k_curr.cast(),
            v_curr.cast(),
            w_page,
            w_off,
            win_d,
            n_max,
            stream,
            row_valid,
        )
    };
}

/// `comm::all_reduce_bf16` — the custom P2P all-reduce.
///
/// Ported from `comm/custom_all_reduce.cu:603-621` by way of
/// [`crate::fire::all_reduce`], which holds the whole lifecycle. **That file
/// is deleted**, and with it `custom_all_reduce.hpp` and
/// `custom_all_reduce_stub.cpp`.
///
/// This is the first row in the tree that is on `execution::SERVED` and
/// `execution::RUST_SERVED` at once, and the pairing is the point: `SERVED`
/// says *the body is one library call*, `RUST_SERVED` says *Rust issues it*.
/// Every other `SERVED` row's library is cuBLAS; this one's is a header-only
/// P2P kernel in a CPM-fetched flashinfer tree, and until that text is
/// vendored the call **declines** — see
/// [`crate::fire::all_reduce::Decline::NoDeviceText`].
///
/// # A decline here is a panic, and that is faithful
///
/// The C++ threw `"custom_all_reduce: not initialised"` and the shim's
/// `catch` aborted. A decline that this arm swallowed would be a silent
/// wrong answer — the reduction would not have happened and every rank would
/// read stale activations. The panic names the refusal, which is the
/// specification for what would fix it.
///
/// # Safety
///
/// `car` is an opaque [`crate::fire::all_reduce::CustomAllReduce`] handle;
/// `input` and `output` address at least `count` bf16 elements on the device.
pub unsafe fn comm_all_reduce_bf16(
    _ctx: &DispatchCtx,
    car: *mut c_void,
    input: *const c_void,
    output: *mut c_void,
    count: usize,
    stream: *mut c_void,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    let outcome = unsafe { crate::fire::all_reduce::all_reduce_bf16(car, input, output, count, stream) };
    if let crate::fire::all_reduce::AllReduce::Declined(why) = outcome {
        panic!("comm::all_reduce_bf16 declined: {why}");
    }
}

/// `comm::all_reduce_residual_rmsnorm_bf16` — all-reduce, residual add and
/// RMSNorm in one landing.
///
/// Ported from `comm/custom_all_reduce.cu:623-662`. The four runtime values
/// that select flashinfer's template point are computed in
/// [`crate::fire::all_reduce::CustomAllReduce::all_reduce_residual_rmsnorm_bf16`],
/// so a decline names the exact instantiation rather than the family.
///
/// # A decline here is a panic
///
/// As above, and more so: this row has no unfused spelling at the call site.
/// `custom_all_reduce.hpp` said it — *"the fused landing IS this kernel, and
/// there is no other way to spell it"* — which is why the header threw on a
/// null handle instead of returning `false`.
///
/// # Safety
///
/// `car` is an opaque handle; `input`, `residual_inout` and `norm_out`
/// address at least `tokens * hidden` bf16 elements, and `rms_gamma` at
/// least `hidden`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn comm_all_reduce_residual_rmsnorm_bf16(
    _ctx: &DispatchCtx,
    car: *mut c_void,
    input: *const c_void,
    residual_inout: *mut c_void,
    rms_gamma: *const c_void,
    norm_out: *mut c_void,
    tokens: i32,
    hidden: i32,
    eps: f32,
    stream: *mut c_void,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    let outcome = unsafe {
        crate::fire::all_reduce::all_reduce_residual_rmsnorm_bf16(
            car,
            input,
            residual_inout,
            rms_gamma,
            norm_out,
            tokens,
            hidden,
            eps,
            stream,
        )
    };
    if let crate::fire::all_reduce::AllReduce::Declined(why) = outcome {
        panic!("comm::all_reduce_residual_rmsnorm_bf16 declined: {why}");
    }
}

// ── FlashInfer FA2 — north star §5 step 7's six rows ────────────────────────
//
// `attention_flashinfer.cu` (1,258 lines) and `plan_lifecycle.cpp` (105) are
// DELETED, and these six functions are the whole of what stood behind them.
// The measured census that justified it: `__global__` 0, `__device__` 0, one
// real `<<<>>>` and that one was `device::attn_score_fold_heads`, ours and
// already rowed (`fire/attn_score.rs:279`).
//
// The split is deliberate and is `fa2-nvrtc`'s: `fire/flashinfer_fa2.rs`
// plans, `fire/flashinfer_fa2_dispatch.rs` decides a symbol, a grid and a
// filled params block, and only these functions -- which own a module and a
// stream -- launch. Everything above the launch is testable without a GPU,
// which nothing calling `cudaLaunchKernel` inline ever was.
//
// # Why every refusal here is a panic
//
// The C++ threw `std::runtime_error` / `std::invalid_argument` from exactly
// these points, and a generated dispatch arm returns `()`. `Decline` is a
// type one layer down, where it can be asserted about in a unit test; this is
// the boundary where it stops being one, and it stops being one loudly.
//
// # `Fired::Split` FOLDS, and this is the record of the pass where it did not
//
// A split fire leaves partials in `tmp_v`/`tmp_s` that
// `VariableLengthMergeStates` has to fold into `o`. That kernel came from
// `attention/cascade.cuh` compiled INTO `attention_flashinfer.cu`, and when
// that file was deleted it had no unit and no row -- so for one pass every
// arm below was a `panic!`, prefill was kept away from it by
// `plan_prefill` setting `disable_split_kv` unconditionally, and decode
// could still reach it. Firing anyway would have put un-merged partials in
// `o`: a silent wrong answer, which is the one outcome worse than a stop.
//
// It runs now. `kernels_cuda_new::families::cascade` compiles
// `PersistentVariableLengthMergeStatesKernel` out of the vendored
// `cascade.cuh` under NVRTC, and `fire/merge_states.rs` fires it. Every
// function below that can split does this, in this order:
//
//   1. `fa2::fire_{decode,prefill}` -- the attention kernel, writing
//      partials, because `make_*_params` redirected `params.o`/`params.lse`
//      to `tmp_v`/`tmp_s` (`prefill.cuh:4339-4342`, `decode.cuh:809-812`).
//   2. `merge_states::variable_length` -- the fold, same stream, writing the
//      caller's real `o` and `lse` (`prefill.cuh:4350-4352`,
//      `decode.cuh:822-824`).
//
// **Two things the old note got wrong, recorded so the next reader does not
// inherit them.** Decode did NOT reach the panic only through the env-gated
// windowed planner: `DecodePlanCache::can_use_static_nonsplit` covers
// batches of 512 or fewer on cc >= 8, so any batch ABOVE 512 took the real
// planner and could split. And `disable_split_kv` is a PREFILL flag, so
// flipping it was never going to be the whole fix -- the decode arms had to
// fold too.

use crate::fire::flashinfer_fa2 as fa2;
use crate::fire::flashinfer_fa2_dispatch as fa2d;
use crate::fire::merge_states;

/// The workspace addresses every FA2 fire reads, widened once.
fn fa2_buffers(
    q: *const c_void,
    k_pages: *mut c_void,
    v_pages: *mut c_void,
    o: *mut c_void,
    kv_page_indices: *const u32,
    kv_page_indptr: *const u32,
    kv_last_page_lens: *const u32,
    qo_indptr: *const u32,
    lse: *mut f32,
    workspace: AttentionWorkspaceView,
) -> fa2d::Buffers {
    fa2d::Buffers {
        q: q as u64,
        k_pages: k_pages as u64,
        v_pages: v_pages as u64,
        o: o as u64,
        kv_page_indices: kv_page_indices as u64,
        kv_page_indptr: kv_page_indptr as u64,
        kv_last_page_lens: kv_last_page_lens as u64,
        qo_indptr: qo_indptr as u64,
        lse: lse as u64,
        int_buffer: workspace.int_buffer as u64,
        float_buffer: workspace.float_buffer as u64,
    }
}

/// `dispatch_attention_flashinfer_decode`, `attention_flashinfer.cu:660-684`.
///
/// Two statements, in the C++'s order: dequantise the layer's active pages
/// into `k_bf16_pages`/`v_bf16_pages`, then fire FA2 over those. The KV width
/// axis is why -- `KvWidth::BF16` is the only width the lattice instantiates,
/// so every scheme is widened before FA2 sees a page.
///
/// # Panics
///
/// If the plan is empty (`:504-508`'s `throw`), if the head dim or GQA group
/// has no unit, or if the plan splits -- see this section's banner.
///
/// # Safety
///
/// `cache` is a live [`crate::fire::flashinfer_fa2::DecodePlanCache`]; every
/// other pointer is a device address the caller keeps live across the launch;
/// `stream` is the fire's stream.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_dispatch_attention_flashinfer_decode(
    _ctx: &DispatchCtx,
    cache: *const c_void,
    q: *const c_void,
    kv_layer: KvCacheLayerView,
    o: *mut c_void,
    kv_page_indices_d: *const u32,
    kv_page_indptr_d: *const u32,
    kv_last_page_lens_d: *const u32,
    workspace: AttentionWorkspaceView,
    stream: *mut c_void,
    window_left: i32,
    logits_soft_cap: f32,
    sm_scale: f32,
    lse_out: *mut f32,
) {
    // SAFETY: the caller's contract -- `bind::DecodePlan::as_ptr` is the only
    // producer of this pointer and it hands out its own boxed cache.
    let plan = unsafe { &*cache.cast::<fa2::DecodePlanCache>() };

    // The dequant prelude, moved. A layer whose dtype `KvDType` does not
    // name skips the prelude and the attention below still runs — which is
    // the shape the `Declined` it used to return already had, because every
    // one of these four call sites consumed that return with `let _ =`.
    if let Ok(l) = KvLayer::try_from(&kv_layer) {
        // SAFETY: forwarded unchanged; `:675`.
        let _ = unsafe {
            kernels_cuda_new::x::attn::kv_paged::dequant_kv_cache_layer_to_bf16_active(
                &l,
                kv_page_indices_d,
                plan.num_pages_in_batch,
                stream,
            )
        };
    }

    let bufs = fa2_buffers(
        q,
        kv_layer.k_bf16_pages,
        kv_layer.v_bf16_pages,
        o,
        kv_page_indices_d,
        kv_page_indptr_d,
        kv_last_page_lens_d,
        core::ptr::null(),
        lse_out,
        workspace,
    );
    let fired = fa2d::decode(
        plan,
        &bufs,
        fa2::fa_device(),
        window_left,
        logits_soft_cap,
        sm_scale,
        // `attention_flashinfer.hpp:136`'s default; the outer dispatch never
        // passed it.
        false,
    );
    let (mut dispatch, partials) = match fired {
        fa2d::Fired::Whole(d) => (d, None),
        // The plan split KV. The fire writes per-chunk partials --
        // `make_*_params` pointed `params.o`/`params.lse` at them -- and
        // the fold after the launch below turns them into the caller's
        // `o`. Both are on this stream, in this order.
        fa2d::Fired::Split(d, split) => (d, Some(split)),
        fa2d::Fired::Declined(why) => {
            panic!("attn::dispatch_attention_flashinfer_decode declined: {why}")
        }
    };
    // SAFETY: the caller's contract, plus the plan's own: `int_upload` was
    // carved against `workspace.int_bytes` by the planner that filled it.
    unsafe {
        fa2::fire_decode(
            &mut dispatch,
            fa2::PlanUpload {
                bytes: &plan.int_upload,
                int_buffer: workspace.int_buffer as u64,
                int_base_bytes: plan.int_base_bytes,
            },
            stream,
        )
    }
    .unwrap_or_else(|why| panic!("attn::dispatch_attention_flashinfer_decode: {why}"));

    if let Some(split) = partials {
        // SAFETY: `split` names the plan's own float workspace and the
        // `o`/`lse` this call was handed; the stream is the caller's, as
        // above. `decode.cuh:822-824` fires exactly this, in exactly this position.
        unsafe { merge_states::variable_length(split.merge(), stream) }
            .expect_launched("attn::dispatch_attention_flashinfer_decode");
    }
}

/// `dispatch_attention_flashinfer_decode_capture`, `:631-658`.
///
/// [`attn_dispatch_attention_flashinfer_decode`] writing the pre-softmax
/// logits to a ragged sink as it goes. The params block is
/// [`kernels_cuda_new::fa2::params::DecodeScoreParams`] rather than
/// `DecodeParams`, which is why this is a separate function and not a flag.
///
/// The C++ threw on a null sink BEFORE choosing a variant, and so does the
/// arm helper: [`crate::fire::flashinfer_fa2_dispatch::Decline::CaptureSinkMissing`].
///
/// The post-kernels (`attn::attn_score_normalize`, `attn::attn_score_fold_heads`)
/// are NOT fired here and were not fired by the C++ either -- they belong to
/// `fire/attn_score.rs`' `LayerScoreCapture::publish`, on this stream,
/// immediately after this returns.
///
/// # Panics
///
/// As [`attn_dispatch_attention_flashinfer_decode`], plus: a soft cap, a
/// window, or a null score sink, none of which compose with capture.
///
/// # Safety
///
/// As [`attn_dispatch_attention_flashinfer_decode`]; `score_out` addresses
/// `score_indptr[batch]` floats and `score_indptr` addresses `batch + 1`
/// `i32`s.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_dispatch_attention_flashinfer_decode_capture(
    _ctx: &DispatchCtx,
    cache: *const c_void,
    q: *const c_void,
    kv_layer: KvCacheLayerView,
    o: *mut c_void,
    kv_page_indices_d: *const u32,
    kv_page_indptr_d: *const u32,
    kv_last_page_lens_d: *const u32,
    workspace: AttentionWorkspaceView,
    stream: *mut c_void,
    score_out: *mut f32,
    score_indptr_d: *const i32,
    window_left: i32,
    logits_soft_cap: f32,
    sm_scale: f32,
    lse_out: *mut f32,
) {
    // SAFETY: as above.
    let plan = unsafe { &*cache.cast::<fa2::DecodePlanCache>() };

    // The dequant prelude, moved. A layer whose dtype `KvDType` does not
    // name skips the prelude and the attention below still runs — which is
    // the shape the `Declined` it used to return already had, because every
    // one of these four call sites consumed that return with `let _ =`.
    if let Ok(l) = KvLayer::try_from(&kv_layer) {
        // SAFETY: forwarded unchanged; `:648`.
        let _ = unsafe {
            kernels_cuda_new::x::attn::kv_paged::dequant_kv_cache_layer_to_bf16_active(
                &l,
                kv_page_indices_d,
                plan.num_pages_in_batch,
                stream,
            )
        };
    }

    let bufs = fa2_buffers(
        q,
        kv_layer.k_bf16_pages,
        kv_layer.v_bf16_pages,
        o,
        kv_page_indices_d,
        kv_page_indptr_d,
        kv_last_page_lens_d,
        core::ptr::null(),
        lse_out,
        workspace,
    );
    let capture = fa2d::Capture {
        score_out: score_out as u64,
        score_indptr: score_indptr_d as u64,
        // A decode step has exactly one query row per request, so there is no
        // window to observe. The C++ capture params for decode carry no
        // `score_window` field at all -- see `DecodeScoreParams`.
        score_window: 0,
    };
    let fired = fa2d::decode_capture(
        plan,
        &bufs,
        &capture,
        fa2::fa_device(),
        window_left,
        logits_soft_cap,
        sm_scale,
        false,
    );
    let (mut dispatch, partials) = match fired {
        fa2d::Fired::Whole(d) => (d, None),
        // The plan split KV. The fire writes per-chunk partials --
        // `make_*_params` pointed `params.o`/`params.lse` at them -- and
        // the fold after the launch below turns them into the caller's
        // `o`. Both are on this stream, in this order.
        fa2d::Fired::Split(d, split) => (d, Some(split)),
        fa2d::Fired::Declined(why) => {
            panic!("attn::dispatch_attention_flashinfer_decode_capture declined: {why}")
        }
    };
    // SAFETY: as above.
    unsafe {
        fa2::fire_decode(
            &mut dispatch,
            fa2::PlanUpload {
                bytes: &plan.int_upload,
                int_buffer: workspace.int_buffer as u64,
                int_base_bytes: plan.int_base_bytes,
            },
            stream,
        )
    }
    .unwrap_or_else(|why| panic!("attn::dispatch_attention_flashinfer_decode_capture: {why}"));

    if let Some(split) = partials {
        // SAFETY: `split` names the plan's own float workspace and the
        // `o`/`lse` this call was handed; the stream is the caller's, as
        // above. `decode.cuh:822-824` fires exactly this, in exactly this position.
        unsafe { merge_states::variable_length(split.merge(), stream) }
            .expect_launched("attn::dispatch_attention_flashinfer_decode_capture");
    }
}

/// `dispatch_attention_flashinfer_prefill_bf16`, `:775-810`.
///
/// The one FA2 row whose KV comes in ALREADY bf16: the fire states `k_pages`
/// and `v_pages` rather than a [`KvCacheLayerView`], so there is no dequant
/// here and there was none in the C++ either.
///
/// # Panics
///
/// As [`attn_dispatch_attention_flashinfer_decode`], plus
/// [`crate::fire::flashinfer_fa2_dispatch::Decline::Sm90Unported`] if the
/// plan ever names the Hopper route. It cannot today --
/// `fire::flashinfer_fa2::plan_prefill` writes `use_sm90 = false` -- and the
/// refusal is kept so that wiring an SM90 family is one conditional and not
/// an audit.
///
/// # Safety
///
/// As [`attn_dispatch_attention_flashinfer_decode`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_dispatch_attention_flashinfer_prefill_bf16(
    _ctx: &DispatchCtx,
    cache: *const c_void,
    q: *const c_void,
    k_pages: *mut c_void,
    v_pages: *mut c_void,
    o: *mut c_void,
    qo_indptr_d: *const u32,
    kv_page_indices_d: *const u32,
    kv_page_indptr_d: *const u32,
    kv_last_page_lens_d: *const u32,
    workspace: AttentionWorkspaceView,
    stream: *mut c_void,
    logits_soft_cap: f32,
    sm_scale: f32,
    lse_out: *mut f32,
) {
    // SAFETY: `bind::PrefillPlan::as_ptr` is the only producer.
    let plan = unsafe { &*cache.cast::<fa2::PrefillPlanCache>() };
    let bufs = fa2_buffers(
        q,
        k_pages,
        v_pages,
        o,
        kv_page_indices_d,
        kv_page_indptr_d,
        kv_last_page_lens_d,
        qo_indptr_d,
        lse_out,
        workspace,
    );
    // `:786-790`. The arm reads the plan's own variant and mask flags, which
    // is what lets one row serve a causal decoder layer and a bidirectional
    // ViT: `tower/qwen3_vl` plans with `causal_mask: false` and fires this.
    let arm = fa2d::prefill_arm(plan.full_attention_variant, plan.causal_mask, logits_soft_cap);
    let fired =
        fa2d::prefill(plan, &bufs, fa2::fa_device(), arm, logits_soft_cap, sm_scale);
    let (mut dispatch, partials) = match fired {
        fa2d::Fired::Whole(d) => (d, None),
        // The plan split KV. The fire writes per-chunk partials --
        // `make_*_params` pointed `params.o`/`params.lse` at them -- and
        // the fold after the launch below turns them into the caller's
        // `o`. Both are on this stream, in this order.
        fa2d::Fired::Split(d, split) => (d, Some(split)),
        fa2d::Fired::Declined(why) => {
            panic!("attn::dispatch_attention_flashinfer_prefill_bf16 declined: {why}")
        }
    };
    // SAFETY: as above.
    unsafe {
        fa2::fire_prefill(
            &mut dispatch,
            fa2::PlanUpload {
                bytes: &plan.int_upload,
                int_buffer: workspace.int_buffer as u64,
                int_base_bytes: plan.int_base_bytes,
            },
            stream,
        )
    }
    .unwrap_or_else(|why| panic!("attn::dispatch_attention_flashinfer_prefill_bf16: {why}"));

    if let Some(split) = partials {
        // SAFETY: `split` names the plan's own float workspace and the
        // `o`/`lse` this call was handed; the stream is the caller's, as
        // above. `prefill.cuh:4350-4352` fires exactly this, in exactly this position.
        unsafe { merge_states::variable_length(split.merge(), stream) }
            .expect_launched("attn::dispatch_attention_flashinfer_prefill_bf16");
    }
}

/// `dispatch_attention_flashinfer_prefill_capture_bf16`, `:1255-1258` onwards.
///
/// [`attn_dispatch_attention_flashinfer_prefill_bf16`] with the score sink and
/// the observation window, on
/// [`kernels_cuda_new::fa2::params::PrefillScoreParams`].
///
/// `folded_out` is bound by the row and **not read here**: folding is
/// `attn::attn_score_fold_heads`, a separate row fired by
/// `fire/attn_score.rs`' `LayerPrefillScoreCapture::publish` after this
/// returns. It stays in the signature because the row states it and because
/// dropping it would make the operand list disagree with `table/attn.rs`.
///
/// # Panics
///
/// As [`attn_dispatch_attention_flashinfer_prefill_bf16`], plus a soft cap, a
/// window, a null sink, or a zero window -- the C++'s four `throw`s.
///
/// # Safety
///
/// As [`attn_dispatch_attention_flashinfer_decode_capture`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_dispatch_attention_flashinfer_prefill_capture_bf16(
    _ctx: &DispatchCtx,
    cache: *const c_void,
    q: *const c_void,
    k_pages: *mut c_void,
    v_pages: *mut c_void,
    o: *mut c_void,
    qo_indptr_d: *const u32,
    kv_page_indices_d: *const u32,
    kv_page_indptr_d: *const u32,
    kv_last_page_lens_d: *const u32,
    workspace: AttentionWorkspaceView,
    stream: *mut c_void,
    score_out: *mut f32,
    folded_out: *mut f32,
    score_indptr_d: *const i32,
    window: i32,
    logits_soft_cap: f32,
    sm_scale: f32,
    lse_out: *mut f32,
) {
    let _ = folded_out;
    // SAFETY: as above.
    let plan = unsafe { &*cache.cast::<fa2::PrefillPlanCache>() };
    let bufs = fa2_buffers(
        q,
        k_pages,
        v_pages,
        o,
        kv_page_indices_d,
        kv_page_indptr_d,
        kv_last_page_lens_d,
        qo_indptr_d,
        lse_out,
        workspace,
    );
    let capture = fa2d::Capture {
        score_out: score_out as u64,
        score_indptr: score_indptr_d as u64,
        score_window: window.max(0) as u32,
    };
    let fired = fa2d::prefill_capture(
        plan,
        &bufs,
        &capture,
        fa2::fa_device(),
        plan.causal_mask,
        logits_soft_cap,
        sm_scale,
    );
    let (mut dispatch, partials) = match fired {
        fa2d::Fired::Whole(d) => (d, None),
        // The plan split KV. The fire writes per-chunk partials --
        // `make_*_params` pointed `params.o`/`params.lse` at them -- and
        // the fold after the launch below turns them into the caller's
        // `o`. Both are on this stream, in this order.
        fa2d::Fired::Split(d, split) => (d, Some(split)),
        fa2d::Fired::Declined(why) => panic!(
            "attn::dispatch_attention_flashinfer_prefill_capture_bf16 declined: {why}"
        ),
    };
    // SAFETY: as above.
    unsafe {
        fa2::fire_prefill(
            &mut dispatch,
            fa2::PlanUpload {
                bytes: &plan.int_upload,
                int_buffer: workspace.int_buffer as u64,
                int_base_bytes: plan.int_base_bytes,
            },
            stream,
        )
    }
    .unwrap_or_else(|why| {
        panic!("attn::dispatch_attention_flashinfer_prefill_capture_bf16: {why}")
    });

    if let Some(split) = partials {
        // SAFETY: `split` names the plan's own float workspace and the
        // `o`/`lse` this call was handed; the stream is the caller's, as
        // above. `prefill.cuh:4350-4352` fires exactly this, in exactly this position.
        unsafe { merge_states::variable_length(split.merge(), stream) }
            .expect_launched("attn::dispatch_attention_flashinfer_prefill_capture_bf16");
    }
}

/// `dispatch_attention_flashinfer_prefill_custom`, `:1225-1252`.
///
/// The arbitrary-mask prefill: the fire supplies a packed bit per
/// `(qo_row, kv_pos)` and the kernel reads it instead of deriving causality.
/// Dequantises like decode -- it takes a [`KvCacheLayerView`], not raw pages
/// -- with `num_pages_in_batch` read off the plan's own KV indptr tail rather
/// than off a device pointer, exactly as `:1244` did.
///
/// `window_left` is **not** a parameter and is not read from the plan:
/// `:1163` writes `params.window_left = -1` literally, because a custom mask
/// already says everything a window would.
///
/// # Panics
///
/// As [`attn_dispatch_attention_flashinfer_prefill_bf16`]. The C++'s
/// *"custom prefill dispatch requires a prepared non-SM90 plan"* is
/// `Decline::Unplanned` / `Decline::Sm90Unported` here.
///
/// # Safety
///
/// As [`attn_dispatch_attention_flashinfer_decode`]; `mask_d` addresses the
/// packed bits `mask_indptr_d` indexes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_dispatch_attention_flashinfer_prefill_custom(
    _ctx: &DispatchCtx,
    cache: *const c_void,
    q: *const c_void,
    kv_layer: KvCacheLayerView,
    o: *mut c_void,
    qo_indptr_d: *const u32,
    kv_page_indices_d: *const u32,
    kv_page_indptr_d: *const u32,
    kv_last_page_lens_d: *const u32,
    mask_d: *const u8,
    mask_indptr_d: *const i32,
    workspace: AttentionWorkspaceView,
    stream: *mut c_void,
    logits_soft_cap: f32,
    sm_scale: f32,
    lse_out: *mut f32,
) {
    // SAFETY: as above.
    let plan = unsafe { &*cache.cast::<fa2::PrefillPlanCache>() };

    // `:1244`, whole: the page count comes off the plan's widened KV indptr,
    // because the device copy cannot be read from the host.
    let num_pages_in_batch = if plan.num_requests > 0 {
        plan.kv_h_buf.get(plan.num_requests as usize).copied().unwrap_or(0)
    } else {
        0
    };
    // The dequant prelude, moved. A layer whose dtype `KvDType` does not
    // name skips the prelude and the attention below still runs — which is
    // the shape the `Declined` it used to return already had, because every
    // one of these four call sites consumed that return with `let _ =`.
    if let Ok(l) = KvLayer::try_from(&kv_layer) {
        // SAFETY: forwarded unchanged.
        let _ = unsafe {
            kernels_cuda_new::x::attn::kv_paged::dequant_kv_cache_layer_to_bf16_active(
                &l,
                kv_page_indices_d,
                num_pages_in_batch,
                stream,
            )
        };
    }

    let bufs = fa2_buffers(
        q,
        kv_layer.k_bf16_pages,
        kv_layer.v_bf16_pages,
        o,
        kv_page_indices_d,
        kv_page_indptr_d,
        kv_last_page_lens_d,
        qo_indptr_d,
        lse_out,
        workspace,
    );
    let mask = fa2d::CustomMask { mask: mask_d as u64, mask_indptr: mask_indptr_d as u64 };
    let fired =
        fa2d::prefill_custom(plan, &bufs, &mask, fa2::fa_device(), logits_soft_cap, sm_scale);
    let (mut dispatch, partials) = match fired {
        fa2d::Fired::Whole(d) => (d, None),
        // The plan split KV. The fire writes per-chunk partials --
        // `make_*_params` pointed `params.o`/`params.lse` at them -- and
        // the fold after the launch below turns them into the caller's
        // `o`. Both are on this stream, in this order.
        fa2d::Fired::Split(d, split) => (d, Some(split)),
        fa2d::Fired::Declined(why) => {
            panic!("attn::dispatch_attention_flashinfer_prefill_custom declined: {why}")
        }
    };
    // SAFETY: as above.
    unsafe {
        fa2::fire_prefill(
            &mut dispatch,
            fa2::PlanUpload {
                bytes: &plan.int_upload,
                int_buffer: workspace.int_buffer as u64,
                int_base_bytes: plan.int_base_bytes,
            },
            stream,
        )
    }
    .unwrap_or_else(|why| panic!("attn::dispatch_attention_flashinfer_prefill_custom: {why}"));

    if let Some(split) = partials {
        // SAFETY: `split` names the plan's own float workspace and the
        // `o`/`lse` this call was handed; the stream is the caller's, as
        // above. `prefill.cuh:4350-4352` fires exactly this, in exactly this position.
        unsafe { merge_states::variable_length(split.merge(), stream) }
            .expect_launched("attn::dispatch_attention_flashinfer_prefill_custom");
    }
}

/// `attention_flashinfer_prefill`, `:1077-1113` — the PLANLESS prefill.
///
/// `Prepare::FireWide` with `whole = true`: no cache crosses, so this plans
/// into a cache of its own and throws it away. The C++ did the same with a
/// function-local `PrefillPlanInfo` and two `std::vector<IdType>`; the only
/// difference here is that the vectors live on a `PrefillPlanCache` that is
/// dropped at the end of the call, which costs one allocation per fire and
/// buys sharing every line of the planned path.
///
/// `:1063-1067` fixes three flags this path never varies:
/// `enable_cuda_graph = false`, `full_attention_variant = false`,
/// `causal_mask = true`. So the arm is always
/// `prefill_arm(false, true, soft_cap)` -- `CausalSoftcap` or `CausalWindow`.
///
/// # Panics
///
/// As [`attn_dispatch_attention_flashinfer_prefill_bf16`]. `num_requests <= 0`
/// is `Decline::NoRequests` and not a silent return, because the C++ reached
/// `PrefillPlan` with it and `PrefillPlan` failed.
///
/// # Safety
///
/// As [`attn_dispatch_attention_flashinfer_decode`], plus: `qo_indptr_h` and
/// `kv_page_indptr_h` address `num_requests + 1` readable HOST `u32`s.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_attention_flashinfer_prefill(
    _ctx: &DispatchCtx,
    q: *const c_void,
    kv_layer: KvCacheLayerView,
    o: *mut c_void,
    qo_indptr_d: *const u32,
    kv_page_indices_d: *const u32,
    kv_page_indptr_d: *const u32,
    kv_last_page_lens_d: *const u32,
    qo_indptr_h: *const u32,
    kv_page_indptr_h: *const u32,
    total_tokens: i32,
    num_requests: i32,
    num_q_heads: i32,
    workspace: AttentionWorkspaceView,
    stream: *mut c_void,
    window_left: i32,
    logits_soft_cap: f32,
    sm_scale: f32,
    lse_out: *mut f32,
) {
    if num_requests <= 0 {
        panic!("attn::attention_flashinfer_prefill declined: empty batch");
    }
    let n = num_requests as usize + 1;
    // SAFETY: the caller's contract -- both are host CSRs of `num_requests + 1`
    // entries, which is what `Prepare::FireWide` publishes.
    let (qo_h, kv_h) = unsafe {
        (
            core::slice::from_raw_parts(qo_indptr_h, n),
            core::slice::from_raw_parts(kv_page_indptr_h, n),
        )
    };

    // `:1098`.
    let num_pages_in_batch = kv_h[num_requests as usize] as i32;
    // The dequant prelude, moved. A layer whose dtype `KvDType` does not
    // name skips the prelude and the attention below still runs — which is
    // the shape the `Declined` it used to return already had, because every
    // one of these four call sites consumed that return with `let _ =`.
    if let Ok(l) = KvLayer::try_from(&kv_layer) {
        // SAFETY: forwarded unchanged.
        let _ = unsafe {
            kernels_cuda_new::x::attn::kv_paged::dequant_kv_cache_layer_to_bf16_active(
                &l,
                kv_page_indices_d,
                num_pages_in_batch,
                stream,
            )
        };
    }

    let mut plan = fa2::PrefillPlanCache::new();
    let device = fa2::plan_device();
    let planned = fa2::plan_prefill(
        &mut plan,
        qo_h,
        kv_h,
        total_tokens,
        num_requests,
        num_q_heads,
        kv_layer.num_kv_heads,
        kv_layer.head_dim,
        kv_layer.page_size,
        kernels_cuda_new::plan::Workspace {
            float_bytes: workspace.float_bytes,
            int_bytes: workspace.int_bytes,
        },
        &device,
        // `:1000`.
        false,
        window_left,
        // `:1066-1067`.
        false,
        kv_layer.hnd_layout,
        true,
        false,
        false,
    );
    if let fa2::Planned::Declined(why) = planned {
        panic!("attn::attention_flashinfer_prefill declined: {why}");
    }

    let bufs = fa2_buffers(
        q,
        kv_layer.k_bf16_pages,
        kv_layer.v_bf16_pages,
        o,
        kv_page_indices_d,
        kv_page_indptr_d,
        kv_last_page_lens_d,
        qo_indptr_d,
        lse_out,
        workspace,
    );
    let arm = fa2d::prefill_arm(false, true, logits_soft_cap);
    let fired =
        fa2d::prefill(&plan, &bufs, fa2::fa_device(), arm, logits_soft_cap, sm_scale);
    let (mut dispatch, partials) = match fired {
        fa2d::Fired::Whole(d) => (d, None),
        // The plan split KV. The fire writes per-chunk partials --
        // `make_*_params` pointed `params.o`/`params.lse` at them -- and
        // the fold after the launch below turns them into the caller's
        // `o`. Both are on this stream, in this order.
        fa2d::Fired::Split(d, split) => (d, Some(split)),
        fa2d::Fired::Declined(why) => {
            panic!("attn::attention_flashinfer_prefill declined: {why}")
        }
    };
    // SAFETY: as above. `plan` outlives the H2D because the copy is issued
    // from a pageable source, which `cudaMemcpyAsync` stages synchronously --
    // see `fire::flashinfer_fa2::upload_int_plan`'s note. That is what makes a
    // function-local plan legal here and it is the reason the note exists.
    unsafe {
        fa2::fire_prefill(
            &mut dispatch,
            fa2::PlanUpload {
                bytes: &plan.int_upload,
                int_buffer: workspace.int_buffer as u64,
                int_base_bytes: plan.int_base_bytes,
            },
            stream,
        )
    }
    .unwrap_or_else(|why| panic!("attn::attention_flashinfer_prefill: {why}"));

    if let Some(split) = partials {
        // SAFETY: `split` names the plan's own float workspace and the
        // `o`/`lse` this call was handed; the stream is the caller's, as
        // above. `prefill.cuh:4350-4352` fires exactly this, in exactly this position.
        unsafe { merge_states::variable_length(split.merge(), stream) }
            .expect_launched("attn::attention_flashinfer_prefill");
    }
}

#[cfg(test)]
mod tests {
    //! What can be checked without a device: that the classification and
    //! this module agree about which symbols are here.

    /// EVERY `RUST_SERVED` SYMBOL HAS A FUNCTION HERE, AND THE NAMES MATCH.
    ///
    /// The list in `execution.rs` is what makes `emit_c_shim` drop a row's
    /// shim entry. A symbol on that list with no function here is a row with
    /// NO executor at all: the C++ body is deleted, the shim entry is gone,
    /// and the arm the emitter writes names a path that does not exist —
    /// which is a compile error, but only in the crate that includes the
    /// generated file, and only in one feature combination. Saying it here
    /// says it in the crate that owns the answer.
    #[test]
    fn every_rust_served_symbol_is_spelled_here() {
        let text = include_str!("service.rs");
        for symbol in kernels_cuda_new::execution::RUST_SERVED {
            let name = kernels_cuda_new::abi::entry_name(symbol);
            let bare = name.strip_prefix("pie_k_").expect("`entry_name` prefixes");
            assert!(
                text.contains(&format!("pub unsafe fn {bare}(")),
                "`execution::RUST_SERVED` names `{symbol}`, so `emit_c_shim` drops its shim \
                 entry and the C++ body is deletable -- but `bind::service` has no \
                 `pub unsafe fn {bare}`. The row would have no executor on any path."
            );
        }
    }
}
