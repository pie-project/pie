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
    cublasComputeType_t, cublasContext, cublasGemmAlgo_t, cublasGemmEx,
    cublasGemmGroupedBatchedEx, cublasGemmStridedBatchedEx, cublasOperation_t, cublasStatus_t,
    cudaDataType,
};

use super::DispatchCtx;

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

/// `gemm::act_x_wt_bf16_out_fp32` — one `cublasGemmEx`, bf16 in, fp32 out.
///
/// Ported from `gemm.cpp:1030-1058` (`gemm_bf16_out_fp32_impl`, reached
/// through the one-line `act_x_wt_bf16_out_fp32` at `:2327`). Row-major
/// `y[M, N] = act[M, K] @ W[N, K]^T`, written column-major as the transpose,
/// which is where `OP_T/OP_N` and the `m=N, n=M` swap come from.
///
/// # Safety
///
/// `act` and `w` must address `M*K` and `N*K` live bf16 elements, `y` must
/// address `M*N` live floats, and all three must outlive the launch — which
/// is asynchronous on the handle's stream, so "outlive" ends at the next
/// synchronisation and not at this call's return.
pub unsafe fn gemm_act_x_wt_bf16_out_fp32(
    ctx: &DispatchCtx,
    act: *const c_void,
    w: *const c_void,
    y: *mut f32,
    m: i32,
    n: i32,
    k: i32,
) {
    let alpha = 1.0f32;
    let beta = 0.0f32;
    // SAFETY: the caller's obligation, above. The handle is the engine's,
    // created once at boot by `device::cublas`.
    let status = unsafe {
        cublasGemmEx(
            ctx.cublas.cast::<cublasContext>(),
            cublasOperation_t::CUBLAS_OP_T,
            cublasOperation_t::CUBLAS_OP_N,
            n,
            m,
            k,
            std::ptr::from_ref(&alpha).cast(),
            w,
            cudaDataType::CUDA_R_16BF,
            k,
            act,
            cudaDataType::CUDA_R_16BF,
            k,
            std::ptr::from_ref(&beta).cast(),
            y.cast(),
            cudaDataType::CUDA_R_32F,
            n,
            COMPUTE,
            ALGO,
        )
    };
    check(status, &format!("cublasGemmEx[bf16->fp32] M={m} N={n} K={k}"));
}

/// `gemm::grouped_act_x_wt_bf16` — one `cublasGemmGroupedBatchedEx`.
///
/// Ported from `gemm.cpp:1242-1294` (`gemm_grouped_bf16_impl`, reached
/// through `grouped_act_x_wt_bf16` at `:1632`). Every group shares `N`, `K`
/// and the three leading dimensions; only `M` differs, which is why the
/// arrays are filled from one scalar each and `n[]` from `M_array_host`.
///
/// **This entry takes the handle rather than a [`DispatchCtx`]**, and it is
/// the one that does. Its row states `Source::Unbound` for every operand — a
/// group boundary is fire-global and no `Source` names one — so
/// `emit_dispatch` writes no arm for it and its only consumer is
/// `fire::lora`'s hand-written staged apply, which holds a `cublasHandle_t`
/// and no context.
///
/// # Safety
///
/// The three pointer arrays must be HOST arrays of `group_count` device
/// addresses (cuBLAS reads them on the host for the grouped form), and
/// `m_array` a host array of `group_count` row counts.
pub unsafe fn gemm_grouped_act_x_wt_bf16(
    handle: *mut c_void,
    act_ptrs_host: *const *const c_void,
    w_ptrs_host: *const *const c_void,
    y_ptrs_host: *const *mut c_void,
    m_array_host: *const i32,
    group_count: i32,
    n: i32,
    k: i32,
    beta: f32,
) {
    if group_count <= 0 {
        return;
    }
    let groups = group_count as usize;
    let transa = vec![cublasOperation_t::CUBLAS_OP_T; groups];
    let transb = vec![cublasOperation_t::CUBLAS_OP_N; groups];
    let m_arr = vec![n; groups];
    // SAFETY: `m_array_host` is a host array of `group_count` ints, per the
    // caller's obligation above.
    let n_arr = unsafe { std::slice::from_raw_parts(m_array_host, groups) }.to_vec();
    let k_arr = vec![k; groups];
    let lda = vec![k; groups];
    let ldb = vec![k; groups];
    let ldc = vec![n; groups];
    let group_size = vec![1i32; groups];
    let alpha = vec![1.0f32; groups];
    let beta_values = vec![beta; groups];

    // No `CUBLAS_COMPUTE_32F_FAST_16BF` attempt first: it has no algorithm
    // for these shapes, and a failed call inside a graph capture invalidates
    // the capture. (`gemm.cpp:1288`, kept verbatim because the reason is not
    // obvious from the code.)
    // SAFETY: every array above is `group_count` long and lives across the
    // call; cuBLAS reads them synchronously.
    let status = unsafe {
        cublasGemmGroupedBatchedEx(
            handle.cast::<cublasContext>(),
            transa.as_ptr(),
            transb.as_ptr(),
            m_arr.as_ptr(),
            n_arr.as_ptr(),
            k_arr.as_ptr(),
            alpha.as_ptr().cast(),
            w_ptrs_host,
            cudaDataType::CUDA_R_16BF,
            lda.as_ptr(),
            act_ptrs_host,
            cudaDataType::CUDA_R_16BF,
            ldb.as_ptr(),
            beta_values.as_ptr().cast(),
            y_ptrs_host,
            cudaDataType::CUDA_R_16BF,
            ldc.as_ptr(),
            group_count,
            group_size.as_ptr(),
            COMPUTE,
        )
    };
    check(
        status,
        &format!("cublasGemmGroupedBatchedEx[bf16] groups={group_count} N={n} K={k}"),
    );
}

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

/// `gemm::act_x_wt_bias_bf16` — the COMPOSITION, not a service.
///
/// `execution::COMPOSED` already stated this row, step for step, and cited
/// `gemm.cpp:2395-2398` for it: a `gemm::act_x_wt_bf16` and then a
/// `norm::add_bias_bf16` over the result. This is that statement, executed.
/// It is in this module because the seam is the same one — a row the driver
/// runs itself, with no entry in the C++ shim.
///
/// # What is lost, exactly
///
/// The archive had a second arm: at `M == 1` with a bias, it asked
/// `dense_tactic_for` whether the tuner's chosen tactic could absorb the bias
/// into its epilogue, and `run_dense_tactic` declines every tactic except the
/// warp-per-row GEMV. So the fused arm fired **only** on the GEMV, and
/// `gemv.hpp` states what it computes: `out[n] = bf16(bf16(dot) + bias[n])`,
/// the double rounding deliberate, *"bit-identical to running
/// `add_bias_bf16` afterwards"*. The composition therefore produces THE SAME
/// BYTES and costs one extra launch per biased `M == 1` projection.
///
/// That is the whole cost and it is stated rather than measured away: the
/// fusion was worth 11.9% of gpt-oss-20b's decode time when it was added
/// (`gemm.hpp`), and what buys it back is a bias epilogue on a JIT'd GEMV,
/// which is a kernel this tree does not have yet. The alternative was to keep
/// a 2,470-line C++ translation unit calling `norm::add_bias_bf16` — and a
/// C++ caller is exactly what no Rust dispatch can intercept, so the row
/// could never be routed and `norm/add_bias.cuh` could never be the only
/// copy.
///
/// # Safety
///
/// `act`, `w`, `bias` and `y` must address live device memory of the extents
/// `M`, `N` and `K` describe, and `y` must be writable.
///
/// # Why this one still takes a handle and a stream
///
/// The other four dropped `handle: CublasHandle` from their rows, because a
/// service carries its own. This row cannot: `execution::COMPOSED` states its
/// first step as `gemm::act_x_wt_bf16`, whose row DOES take a handle, and
/// `Composition::agrees` type-checks each `Take::From(i)` against the
/// composed row's operands. Remove the handle here and the composition can no
/// longer supply its own first step. So the row keeps the operands the
/// composition needs, the arm binds them, and `ctx` arrives as well because
/// every service arm is emitted the same way — the redundancy is the
/// emitter's uniformity, and `ctx.cublas`/`ctx.stream` are what
/// `Source::Ctx("cublas")`/`Source::Ctx("stream")` bind to anyway.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_act_x_wt_bias_bf16(
    _ctx: &DispatchCtx,
    handle: *mut c_void,
    act: *const c_void,
    w: *const c_void,
    bias: *const c_void,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    stream: *mut c_void,
    beta: f32,
) {
    // Step one: `gemm_bf16_impl(handle, act, W, y, M, N, K, beta)`, which is
    // `gemm::act_x_wt_bf16`'s own body — the runtime autotuner, and the one
    // part of `gemm.cpp` that STAYS in C++ (it chooses `gemv_bf16` on some
    // shapes, and `gemv_bf16` returns `bool`; a row cannot decline).
    // SAFETY: the caller's obligation, above.
    unsafe {
        super::abi::ffi::pie_k_gemm_act_x_wt_bf16(handle, act, w, y, m, n, k, beta);
    }
    if bias.is_null() {
        return;
    }
    // Step two: `norm::add_bias_bf16(y, bias, M, N, stream)`. Fired through
    // the JIT rather than through `ffi::pie_k_norm_add_bias_bf16`, and the
    // difference is the entire point of the change: a `pie_k_*` call is a
    // consumer of the C++ launcher, so making one here would hold the row
    // exactly as `gemm.cpp:2393` did. `runtime::fire` resolves the symbol
    // through `unit_of`, not through `JIT_DISPATCHED`, so this works whether
    // or not the row has been routed — which matters, because routing it is
    // someone else's change and this one must not depend on the order.
    //
    // The JIT row is `(out, bias, dim)` with `LaunchRule::RouteRows`: the
    // launcher's `num_rows` became the rule and its `stream` became an
    // argument of the fire. `execution.rs`'s `sig_of` documents why the
    // device row wins over the table row here, and binding the table's five
    // operands to this three-parameter kernel is the failure it names.
    // SAFETY: `y` was just written by the GEMM above and is `m * n` bf16
    // elements; `bias` is `n`; the stream is the fire's.
    unsafe {
        super::jit::fire(
            "norm::add_bias_bf16",
            kernels_cuda_new::Dims {
                rows: m.max(0) as u32,
                width: n.max(0) as u32,
                ..kernels_cuda_new::Dims::default()
            },
            &[
                super::device::ArgValue::Ptr(y),
                super::device::ArgValue::Ptr(bias.cast_mut()),
                super::device::ArgValue::I32(n),
            ],
            stream,
        );
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
