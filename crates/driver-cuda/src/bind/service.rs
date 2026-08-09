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
    check(
        status,
        &format!("cublasGemmEx[bf16->fp32] M={m} N={n} K={k}"),
    );
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

/// `gemm::act_x_wt_bf16` — the dense bf16 GEMM. Body in
/// [`crate::fire::gemm::act_x_wt_bf16`].
///
/// `y[M, N] = act[M, K] @ W[N, K]^T + beta * y`, all bf16, fp32 accumulate.
/// The hottest row in the tree: every linear layer of every model lands here.
///
/// **This is not one cuBLAS call and that is why it took so long to arrive.**
/// It is a runtime autotuner over three kernel families — the warp-per-row
/// GEMV, `cublasGemmEx`, and each algorithm cuBLASLt's heuristic offers —
/// with a per-device tactic memo, an on-disk tactic cache and a fallback
/// ladder behind it. All of it host code, all of it now Rust; the module
/// carries the measurements.
///
/// The thing that held it in C++ for three arcs was that `gemm_bf16_impl`
/// called `gemv_bf16`, whose `bool` meant *"I did not launch"*, and a row
/// cannot decline. The resolution was not to make the row decline: a
/// **driver-owned launch is not a row**, so [`crate::fire::gemv::gemv_bf16`]
/// spells its refusal as a type and the tuner's GEMV candidate is a
/// `matches!(.., Gemv::Launched)` in the same short-circuiting position the
/// C++ put it in.
///
/// # Why the handle is an operand and `ctx` is not enough
///
/// The row states `handle: CublasHandle <- Source::Ctx("cublas")`, so the
/// emitted arm passes both `ctx` and the bound handle — the same redundancy
/// [`gemm_act_x_wt_bias_bf16`] documents, and for the same reason: the
/// composition takes this row as its first step and `Composition::agrees`
/// type-checks `Take::From(i)` against the operands as stated. They are the
/// same pointer; `ctx.cublas` is the engine's handle, created once at boot by
/// `device::cublas`.
///
/// # Safety
///
/// `act`, `w` and `y` must address `M*K`, `N*K` and `M*N` live bf16 elements
/// and outlive the launch — asynchronous on the handle's stream, so "outlive"
/// ends at the next synchronisation and not at this call's return.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_act_x_wt_bf16(
    _ctx: &DispatchCtx,
    handle: *mut c_void,
    act: *const c_void,
    w: *const c_void,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
) {
    // SAFETY: the caller's obligation, above.
    unsafe { crate::fire::gemm::act_x_wt_bf16(handle, act, w, y, m, n, k, beta) }
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
/// warp-per-row GEMV. So the fused arm fired **only** on the GEMV, and its
/// kernels state what they compute: `out[n] = bf16(bf16(dot) + bias[n])`, the
/// double rounding deliberate, *"bit-identical to running `add_bias_bf16`
/// afterwards"*. (That was `gemv.hpp`'s wording; the header is deleted and
/// the sentence is now at both epilogues of
/// `kernels-cuda-new/csrc/src/gemm/gemv.cuh`, which is the text NVRTC
/// compiles.) The composition therefore produces THE SAME BYTES and costs one
/// extra launch per biased `M == 1` projection.
///
/// That is the whole cost and it is stated rather than measured away: the
/// fusion was worth 11.9% of gpt-oss-20b's decode time when it was added
/// (`gemm.hpp`), and what buys it back is a bias epilogue on a JIT'd GEMV.
/// **That kernel now exists** — the `gemm/gemv` unit's four rows all take
/// `bias` and fold it, and `fire::gemv::gemv_bf16` passes it through — so what
/// is missing is no longer a kernel but a Rust caller that reaches it at
/// `M == 1` instead of reaching `pie_k_gemm_act_x_wt_bf16`, which means the
/// dense tactic enumeration in Rust. **That enumeration now exists** —
/// [`crate::fire::gemm`] — so the remaining work is a `fire::gemm` entry that
/// takes a `bias` and, when the tuned tactic for the shape is
/// `GemmKind::Gemv`, passes it down instead of adding it afterwards.
/// [`crate::fire::gemm::dense_tactic_is_gemv`] is the side-effect-free peek
/// that arm needs, ported and waiting.
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
    // Step one: `gemm::act_x_wt_bf16`'s own body — the runtime autotuner and
    // the fallback ladder — which is [`crate::fire::gemm::act_x_wt_bf16`] and
    // no longer C++ at all. It used to be `ffi::pie_k_gemm_act_x_wt_bf16`,
    // and the note here used to say the tactic enumeration was what kept
    // `gemm.cpp` alive. `gemm.cpp` is deleted; the enumeration is
    // `fire::gemm`.
    // SAFETY: the caller's obligation, above.
    unsafe {
        crate::fire::gemm::act_x_wt_bf16(handle, act, w, y, m, n, k, beta);
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

/// `moe::moe_grouped_gemm_bf16` — the short-K grouped GEMM, JIT'd.
///
/// # This one is not a library call, and that is the point
///
/// The four rows above are cuBLAS. This one is a `__global__` of ours,
/// compiled by NVRTC from `moe/moe_grouped_gemm.cuh`, fired on a grid this
/// crate builds by hand. It reaches `bind::service` by the same door for the
/// reason the module header gives and `execution::RUST_SERVED` restates:
/// **the model compiler must not be able to tell which it is.** `table::moe`
/// states one row, `emit_dispatch` writes one arm, and what is behind the
/// arm is this crate's business.
///
/// The body is [`crate::fire::moe::moe_grouped_gemm_bf16`], which carries
/// the geometry with `moe/moe_grouped_gemm.cu`'s line numbers beside it.
/// Here there is only the ABI: the operand order is `table::moe`'s
/// `moe_grouped_gemm` row, verbatim.
///
/// # The decline is dropped HERE and nowhere deeper
///
/// The generated arm returns `bool` and its `true` means "a branch ran", not
/// "the kernel launched" — `emit_dispatch` writes `true` unconditionally for
/// a `RUST_SERVED` row because there is no shim entry to fall through to.
/// The shape refusal therefore cannot be reported through the arm, and the
/// C++ this replaces could not report it either: it returned `void` after an
/// early `return`. What is new is that the refusal is now a VALUE at the
/// only place that can act on it — see `fire::moe::Decline`.
///
/// # Safety
///
/// The four pointers must be device allocations of the row's shapes, live on
/// `ctx.stream` until the launch completes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn moe_moe_grouped_gemm_bf16(
    _ctx: &DispatchCtx,
    a: *const c_void,
    weight_base: *const c_void,
    c: *mut c_void,
    expert_ids: *const i32,
    max_blocks: i32,
    m: i32,
    n: i32,
    k: i32,
    stream: *mut c_void,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    let _ = unsafe {
        crate::fire::moe::moe_grouped_gemm_bf16(
            a,
            weight_base,
            c,
            expert_ids,
            max_blocks,
            m,
            n,
            k,
            stream,
        )
    };
}

/// `moe::flashinfer_cutlass_moe_bf16` — the fused CUTLASS MoE, and the ONE
/// `bool`-returning row in the generated shim.
///
/// The whole routed block as one call: permute, both grouped GEMMs, the gated
/// activation and the weighted finalize. `crates/model/src/qwen_3_5/forward/
/// mod.rs:362` calls `dsl::cuda::moe_fused_cutlass` **unconditionally**, and
/// the doc above it says *"the fused CUTLASS call is the fourth and the one
/// decode actually takes."* A survey once reported this symbol uncalled
/// because the symbol string and the DSL wrapper name are different tokens.
///
/// The body is [`crate::fire::flashinfer_moe::bf16`], which carries
/// `moe/flashinfer_moe.cu`'s measurements beside every constant. Here there
/// is only the ABI: the operand order is `table::moe`'s `moe_fused_cutlass`
/// row, verbatim, and `activation` arrives as the `u32` the generated
/// bindings spell `MoeActivation` as.
///
/// # The `bool` is a REFUSAL, and that is why it is not a `bool` underneath
///
/// `false` never meant "the kernel failed". It meant the fused path
/// **declined** — a null operand, a row count outside a configured window, or
/// a workspace the arch probe says cannot exist here — and the caller is
/// correct only because it means that. So `fire::flashinfer_moe` answers
/// `Fused::{Ran, Declined}` and the two-state value is flattened HERE, at the
/// ABI, because the row's `returns = "bool"` is what `KernelSig` states and
/// `KernelSig` is unchanged. Anything that is not a refusal — a runner that
/// will not build, a `setTactic` that fails — panics in the body with the
/// symbol named rather than arriving here as `false`.
///
/// # Why this one is gated
///
/// Every other function in this file reaches its library by `dlopen` through
/// `cudarc`, so nothing links. This one reaches `CutlassMoeFCRunner` through
/// the five-function `extern "C"` seam that is all that is left of
/// `csrc/src/moe/flashinfer_moe.cu`, and that seam is in
/// `libpie_kernels_cuda.a` — which `bridge` is exactly the feature that
/// links. `every_rust_served_symbol_is_spelled_here` reads this file's TEXT,
/// so the gate costs the row nothing it had.
///
/// # Safety
///
/// Every pointer must be a device allocation of the row's shapes on the
/// current device, live on `stream` until the launch completes; `workspace`
/// must be writable for `workspace_bytes`.
#[cfg(feature = "bridge")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn moe_flashinfer_cutlass_moe_bf16(
    _ctx: &DispatchCtx,
    activation: u32,
    input: *const u16,
    token_selected_experts: *const i32,
    token_final_scales: *const f32,
    fc1_expert_weights: *const u16,
    fc2_expert_weights: *const u16,
    output: *mut u16,
    workspace: *mut u8,
    workspace_bytes: usize,
    unpermuted_row_to_permuted_row: *mut i32,
    num_rows: i32,
    hidden_size: i32,
    inter_size: i32,
    num_experts: i32,
    experts_per_token: i32,
    tp_size: i32,
    tp_rank: i32,
    stream: *mut c_void,
) -> bool {
    // The C++ `to_cutlass_activation` ended in `case Relu2: default:`, so an
    // enumerator this driver has not been taught became `Relu2` rather than
    // an error. Reproduced, and not widened: a port may not invent a check
    // the archive never made.
    let activation = match activation {
        1 => super::abi::MoeActivation::Swiglu,
        2 => super::abi::MoeActivation::Geglu,
        _ => super::abi::MoeActivation::Relu2,
    };
    // SAFETY: the caller's obligation, above.
    let fused = unsafe {
        crate::fire::flashinfer_moe::bf16(
            activation,
            input,
            token_selected_experts,
            token_final_scales,
            fc1_expert_weights,
            fc2_expert_weights,
            output,
            workspace,
            workspace_bytes,
            unpermuted_row_to_permuted_row,
            num_rows,
            hidden_size,
            inter_size,
            num_experts,
            experts_per_token,
            tp_size,
            tp_rank,
            stream,
        )
    };
    matches!(fused, crate::fire::flashinfer_moe::Fused::Ran)
}

/// `sample::lm_head_gemv_argmax_int8` — `sample/argmax.hpp:37`.
///
/// Greedy decode straight off an int8 LM head: for each of `num_rows` hidden
/// vectors, the vocab index whose dequantized dot product is largest, written
/// as one i32. The vocab-wide logit row is never materialised, which is why
/// `table::sample` states this as its own symbol rather than as an `lm_head`
/// GEMM followed by an argmax over its output.
///
/// The body is [`crate::fire::lm_head_argmax::lm_head_gemv_argmax_int8`],
/// which carries `sample/argmax.cu`'s line numbers beside every constant.
/// Here there is only the ABI: the operand order is `table::sample`'s one
/// row, verbatim.
///
/// # Why it is here rather than behind a `pie_k_` shim
///
/// Two kernels, a device scratch between them that the row's operand list
/// does not mention, and a grid extent read off
/// `cudaDevAttrMultiProcessorCount`. `execution::WALKED` classifies it as a
/// `Walk` for exactly that — host control flow whose shape comes from the
/// input and from the machine. What reaches this function is one call with
/// eight operands, the same eight the C++ launcher took, and no model text
/// can tell that two `__global__`s run behind it.
///
/// # Safety
///
/// The four pointers must be device allocations of the row's shapes —
/// `hidden_states` bf16 `[num_rows, hidden]`, `lm_head_weight` int8 `[vocab,
/// hidden]`, `scale_inv` fp32 `[vocab]`, `token_ids` writable for `num_rows`
/// i32 — live on `stream` until both launches complete.
#[allow(clippy::too_many_arguments)]
pub unsafe fn sample_lm_head_gemv_argmax_int8(
    _ctx: &DispatchCtx,
    hidden_states: *const c_void,
    lm_head_weight: *const i8,
    scale_inv: *const f32,
    token_ids: *mut i32,
    num_rows: i32,
    hidden: i32,
    vocab: i32,
    stream: *mut c_void,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    unsafe {
        crate::fire::lm_head_argmax::lm_head_gemv_argmax_int8(
            hidden_states,
            lm_head_weight,
            scale_inv,
            token_ids,
            num_rows,
            hidden,
            vocab,
            stream,
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// `norm/rmsnorm.cu` and `norm/dsv4_hc.cu` — six rows
// ═══════════════════════════════════════════════════════════════════════════
//
// None of the six below is a library call, and none of them is a
// composition: each is a host program that chooses an instantiation, computes
// a rectangle, or supplies a value the `Source` grammar cannot state, and
// then fires kernels NVRTC compiled out of
// `kernels-cuda-new/csrc/src/norm/*.cuh`.
//
// `execution::WALKED` states each one with the archive line it came from and
// what it refuses; the programs are `crate::fire::{rmsnorm, dsv4_hc}`.
// The functions here are thin on purpose — a service entry's whole job is to
// receive the operands `abi::emit_dispatch` bound and hand them on in order,
// so that the ONE thing that decides whether a symbol goes to cuBLAS, to a
// generated `pie_k_*`, or here, stays `execution::RUST_SERVED` and stays
// invisible to every lowering.
//
// **`rope/rope.cu`'s nine used to be counted in this heading and are not
// any more.** They did not become library calls or generated arms; the
// family crossed into fn-world (`.wiki/kernel-x/northstar.md` §5 step 3)
// and left the `RUST_SERVED` fork entirely. See the note where they stood,
// below.

/// RMSNorm with an optional fp16 copy of the result — **the row whose three
/// arms `execution::Step` measured and could not state.**
///
/// `Step`'s header refuses a `Choose` variant and names this row as the
/// reason the refusal is about ROWS and not about predicates: *"`norm::
/// rmsnorm_bf16_with_fp16`'s three arms are all statable — `Term::Present` on
/// `y_fp16`, `Term::Multiple { of: 8 }` on `hidden`, `Term::Aligned { bytes:
/// 16 }` on three pointers, every one of them already proven — and the op is
/// still refused, because the arm those predicates SELECT (`rmsnorm_vec8<512,
/// false, EMIT_FP16=true>`) is an instantiation `families/norm.rs` does not
/// carry."*
///
/// **`families/norm.rs` carries it now**, as
/// `norm::rmsnorm_bf16_with_fp16#vec8_512`. That does not make the row a
/// composition — the middle arm is still TWO launches with the bf16 result as
/// the intermediate, which is what `Composition`'s `Take` cannot spell — but
/// it does mean the fused arm no longer silently degrades to a different
/// reduction order, which was the §21.14 failure the refusal was protecting.
///
/// # Safety
///
/// `x` and `weight` must be live bf16 device memory of `[num_rows, hidden]`
/// and `[hidden]`; `y` must be writable for the same extent; `y_fp16`, if not
/// null, must be writable for `num_rows * hidden` halves. All live on
/// `stream` until the launches complete.
#[allow(clippy::too_many_arguments)]
pub unsafe fn norm_rmsnorm_bf16_with_fp16(
    _ctx: &DispatchCtx,
    x: *const c_void,
    weight: *const c_void,
    y: *mut c_void,
    y_fp16: *mut c_void,
    num_rows: i32,
    hidden: i32,
    eps: f32,
    stream: *mut c_void,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    unsafe {
        crate::fire::rmsnorm::with_fp16(x, weight, y, y_fp16, num_rows, hidden, eps, stream);
    }
}

/// gemma-4's residual-add, scale and two norms in one launch.
///
/// **The one row from these two files that takes over a LIVE generated
/// arm.** It was one of four; the other three were `rope::rope_bf16`,
/// `rope::qk_rmsnorm_rope_bf16_rounded` and `rope::rope_yarn_original_bf16`,
/// which have since crossed into fn-world and are bound in
/// `kernels-cuda-new/src/x/rope.rs`. The rest of the fifteen stated no
/// `Source` on some operand, so `abi::emit_dispatch` skipped them whole and
/// they were unreachable before this change for a reason it did not create.
///
/// This row is fully sourced and gemma-4 fires it four times a layer, so the
/// arm that used to call
/// `ffi::pie_k_norm_rmsnorm_residual_add_scale_rmsnorm_bf16` now calls this,
/// and the bits must not move: the port keeps all three instantiations,
/// including the 2560 threshold that chooses between them.
///
/// # Safety
///
/// `x`, `weight`, `next_weight` live bf16; `hidden` is read AND written
/// (`in_place = &[(0, 1)]`); `norm_out` writable for `[num_rows,
/// hidden_size]`. All live on `stream`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn norm_rmsnorm_residual_add_scale_rmsnorm_bf16(
    _ctx: &DispatchCtx,
    x: *const c_void,
    weight: *const c_void,
    hidden: *mut c_void,
    scale: f32,
    next_weight: *const c_void,
    norm_out: *mut c_void,
    num_rows: i32,
    hidden_size: i32,
    eps: f32,
    stream: *mut c_void,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    unsafe {
        crate::fire::rmsnorm::residual_add_scale_rmsnorm(
            x,
            weight,
            hidden,
            scale,
            next_weight,
            norm_out,
            num_rows,
            hidden_size,
            eps,
            stream,
        );
    }
}

/// deepseek-v4 hyper-connections: the per-token mixing matrix, and the
/// scratch two later kernels read.
///
/// # Safety
///
/// `mixes`, `scale`, `base` are fp32 device slabs the layer owns;
/// `post_mix`/`comb_mix` are writable fp32 scratch this launch fills for
/// [`norm_hc_post_bf16`]; `residual` is bf16 `[n, hc_mult * hidden_size]` and
/// `layer_input` is writable bf16 `[n, hidden_size]`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn norm_hc_pre_postprocess_bf16(
    _ctx: &DispatchCtx,
    mixes: *const f32,
    scale: *const f32,
    base: *const f32,
    residual: *const c_void,
    post_mix: *mut f32,
    comb_mix: *mut f32,
    layer_input: *mut c_void,
    n: i32,
    hc_mult: i32,
    hidden_size: i32,
    hc_eps: f32,
    hc_post_alpha: f32,
    sinkhorn_iters: i32,
    stream: *mut c_void,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    unsafe {
        crate::fire::dsv4_hc::hc_pre_postprocess_bf16(
            mixes,
            scale,
            base,
            residual,
            post_mix,
            comb_mix,
            layer_input,
            n,
            hc_mult,
            hidden_size,
            hc_eps,
            hc_post_alpha,
            sinkhorn_iters,
            stream,
        );
    }
}

/// The write-back half of a hyper-connection layer.
///
/// **This one PANICS where the C++ returned silently.** `dsv4_hc.cu:59` was
/// `if (hc_mult > MAX_HC_MULT) return;`, and `hc_post` keeps its `M` residual
/// values in a register array of that width — so a larger multiplier is not a
/// slower launch, it is a residual stream that never gets written and a layer
/// that reads its own uninitialised memory. §54: a refusal is never a
/// fallback.
///
/// # Safety
///
/// `x` bf16 `[n, hidden_size]`; `residual` and `out_residual` bf16 `[n,
/// hc_mult * hidden_size]`; `post_mix`/`comb_mix` the fp32 scratch
/// [`norm_hc_pre_postprocess_bf16`] filled on the same stream.
#[allow(clippy::too_many_arguments)]
pub unsafe fn norm_hc_post_bf16(
    _ctx: &DispatchCtx,
    x: *const c_void,
    residual: *const c_void,
    post_mix: *const f32,
    comb_mix: *const f32,
    out_residual: *mut c_void,
    n: i32,
    hc_mult: i32,
    hidden_size: i32,
    stream: *mut c_void,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    unsafe {
        crate::fire::dsv4_hc::hc_post_bf16(
            x,
            residual,
            post_mix,
            comb_mix,
            out_residual,
            n,
            hc_mult,
            hidden_size,
            stream,
        );
    }
}

/// The final collapse of `hc_mult` residual streams into the one the LM head
/// reads.
///
/// # Safety
///
/// As [`norm_hc_pre_postprocess_bf16`], with `out` writable bf16 `[n,
/// hidden_size]` and no scratch. Note the operand ORDER: `hc_eps` follows
/// `stream`, because the launcher's C++ signature did and `KernelSig` states
/// launchers.
#[allow(clippy::too_many_arguments)]
pub unsafe fn norm_hc_head_postprocess_bf16(
    _ctx: &DispatchCtx,
    mixes: *const f32,
    scale: *const f32,
    base: *const f32,
    residual: *const c_void,
    out: *mut c_void,
    n: i32,
    hc_mult: i32,
    hidden_size: i32,
    stream: *mut c_void,
    hc_eps: f32,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    unsafe {
        crate::fire::dsv4_hc::hc_head_postprocess_bf16(
            mixes,
            scale,
            base,
            residual,
            out,
            n,
            hc_mult,
            hidden_size,
            stream,
            hc_eps,
        );
    }
}

/// RMSNorm from bf16 INTO fp32 — the widened input the mixing matrices are
/// computed from.
///
/// # Safety
///
/// `input` bf16 `[n, dim]`; `output` writable for `n * dim` fp32.
pub unsafe fn norm_hc_rmsnorm_to_f32(
    _ctx: &DispatchCtx,
    input: *const c_void,
    output: *mut f32,
    n: i32,
    dim: i32,
    eps: f32,
    stream: *mut c_void,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    unsafe {
        crate::fire::dsv4_hc::hc_rmsnorm_to_f32(input, output, n, dim, eps, stream);
    }
}

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

// ── `ssm/`: eleven launchers, four files, 25 live `<<<>>>` ───────────────
//
// Nothing below calls a library. Every one of these forwards to
// `crate::fire::{causal_conv1d,gated_delta_net,kda,nemotron_h}`, which fires
// NVRTC-compiled device text out of `kernels-cuda-new/csrc/src/ssm/*.cuh`.
// They are here rather than in `bind/mod.rs` for the reason the `norm`,
// `rope` and `moe` blocks above are: `execution::RUST_SERVED` is the ONE
// list that decides whether `abi::emit_rust_dispatch` writes an arm to
// `bind::service` or `emit_c_shim` writes a `pie_k_*`, and a symbol on that
// list must be spelled here or the generated arm names nothing.
//
// **The parameter lists are the TABLE ROW's, not the C++ launcher's**, and
// where they differ the table wins. `abi::emit_rust_dispatch` writes the
// operands in row order including the `Ty::Stream` one, so
// `ssm_causal_conv1d_prefill_batched_bf16` takes its stream in the MIDDLE —
// after `k`, before `write_state` — because that is where `table::ssm` put
// it. A signature that "tidied" the stream to the end would compile and
// would pass a `bool` where a `cudaStream_t` goes.
//
// TWO OF THE ELEVEN ARE UNREACHABLE and stay that way: the KDA pair state
// no `Source` on any operand, so `emit_rust_dispatch` skips those rows whole
// and writes no arm. They are written anyway — the geometry they carry is
// the only surviving copy of `kda.cu`'s measurements, and the alternative
// was to delete the launcher without a home for them.

/// The depthwise causal convolution over a prefill's token runs.
///
/// # Safety
///
/// `x`/`y` are `[qo_indptr[R], C]` bf16, `weight` is `[C, K]` bf16, `bias` is
/// `[C]` bf16 or null, `state_out_base` is a slot arena of
/// `slot_stride_elems` elements per slot, `slot_ids` is `[R]`, `qo_indptr` is
/// `[R + 1]`, and `commit_len`/`write_state_mask` are `[R]` or null.
#[allow(clippy::too_many_arguments)]
pub unsafe fn ssm_causal_conv1d_prefill_batched_bf16(
    _ctx: &DispatchCtx,
    x: *const c_void,
    weight: *const c_void,
    bias: *const c_void,
    y: *mut c_void,
    state_out_base: *mut c_void,
    slot_ids: *const i32,
    qo_indptr: *const u32,
    slot_stride_elems: ::core::ffi::c_longlong,
    r: i32,
    c: i32,
    k: i32,
    stream: *mut c_void,
    write_state: bool,
    commit_len: *const i32,
    write_state_mask: *const u8,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    unsafe {
        crate::fire::causal_conv1d::prefill_batched_bf16(
            x,
            weight,
            bias,
            y,
            state_out_base,
            slot_ids,
            qo_indptr,
            slot_stride_elems,
            r,
            c,
            k,
            stream,
            write_state,
            commit_len,
            write_state_mask,
        );
    }
}

/// Qwen3.5's post-convolution split: q/k norm, then v/gate/beta.
///
/// A `driver_internal` row — `table::driver_internal:156` — so no model
/// statement names it directly; `bind/mod.rs`'s GDN path does. Two launches
/// and one host quantity, `q_scale = rsqrtf(K_d)`.
///
/// # Safety
///
/// `qkv_post` is `[N, conv_dim]` bf16; `a`/`b`/`dt_bias` are bf16;
/// `a_log` is `[V_h]` fp32; the five outputs are writable for their
/// rectangles. All live on `stream`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn ssm_qwen_gdn_post_conv_prep_bf16(
    _ctx: &DispatchCtx,
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
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    unsafe {
        crate::fire::gated_delta_net::post_conv_prep_bf16(
            qkv_post, a, b, a_log, dt_bias, q_norm_kh, k_norm_kh, v_fp32, g_log_out,
            beta_out, n, k_h, v_h, k_d, v_d, conv_dim, stream,
        );
    }
}

/// Qwen3.5's gated-delta decode step, GQA-aware, bf16 state.
///
/// **This one takes over a live generated arm and the bits must not move.**
/// The switch it carries is `V_d == 128 && K_d == 128`, which §30 measured as
/// selecting between two byte-identical kernels — the choice is 34% of the
/// step's time and none of its output.
///
/// # Safety
///
/// `q_norm_kh`/`k_norm_kh` are `[R, K_h, K_d]` fp32; `v`/`g_log`/`beta` are
/// fp32 over `V_h`; `state_base` is a slot arena of `slot_stride_elems`
/// **bf16** per slot; `slot_ids` is `[R]`; `out` is `[R, V_h, V_d]`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn ssm_recurrent_gated_delta_step_batched_gqa_state_bf16(
    _ctx: &DispatchCtx,
    q_norm_kh: *const f32,
    k_norm_kh: *const f32,
    v: *const f32,
    g_log: *const f32,
    beta: *const f32,
    state_base: *mut c_void,
    slot_ids: *const i32,
    slot_stride_elems: ::core::ffi::c_longlong,
    out: *mut f32,
    r: i32,
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    stream: *mut c_void,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    unsafe {
        crate::fire::gated_delta_net::recurrent_step_batched_gqa_state_bf16(
            q_norm_kh,
            k_norm_kh,
            v,
            g_log,
            beta,
            state_base,
            slot_ids,
            slot_stride_elems,
            out,
            r,
            k_h,
            v_h,
            k_d,
            v_d,
            stream,
        );
    }
}

/// The chunked gated-delta prefill, fp32 state.
///
/// # Safety
///
/// As [`ssm_chunk_gated_delta_prefill_batched_state_bf16`], with `state_base`
/// an arena of `slot_stride_elems` **fp32** per slot.
#[allow(clippy::too_many_arguments)]
pub unsafe fn ssm_chunk_gated_delta_prefill_batched(
    _ctx: &DispatchCtx,
    q_norm: *const f32,
    k_norm: *const f32,
    v: *const f32,
    g_log: *const f32,
    beta: *const f32,
    state_base: *mut f32,
    slot_ids: *const i32,
    qo_indptr: *const u32,
    slot_stride_elems: ::core::ffi::c_longlong,
    out: *mut f32,
    r: i32,
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    stream: *mut c_void,
    write_state: bool,
    commit_len: *const i32,
    write_state_mask: *const u8,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    unsafe {
        crate::fire::gated_delta_net::chunk_prefill_batched(
            q_norm,
            k_norm,
            v,
            g_log,
            beta,
            state_base,
            slot_ids,
            qo_indptr,
            slot_stride_elems,
            out,
            r,
            k_h,
            v_h,
            k_d,
            v_d,
            stream,
            write_state,
            commit_len,
            write_state_mask,
        );
    }
}

/// The chunked gated-delta prefill, bf16 state.
///
/// The FLA arm is 9× the legacy arm at production shapes (47.5 ms → 5.3 ms)
/// and bit-identical; the legacy arm is **not GQA-aware** and takes four
/// fewer operands, which is why the two are a `Switch` and not a knob.
///
/// # Safety
///
/// `q_norm`/`k_norm` are `[T, K_h, K_d]` fp32 over `T = qo_indptr[R]`;
/// `v`/`g_log`/`beta` are fp32 over `V_h`; `state_base` is a slot arena of
/// `slot_stride_elems` bf16; `out` is `[T, V_h, V_d]`; `commit_len` and
/// `write_state_mask` are `[R]` or null.
#[allow(clippy::too_many_arguments)]
pub unsafe fn ssm_chunk_gated_delta_prefill_batched_state_bf16(
    _ctx: &DispatchCtx,
    q_norm: *const f32,
    k_norm: *const f32,
    v: *const f32,
    g_log: *const f32,
    beta: *const f32,
    state_base: *mut c_void,
    slot_ids: *const i32,
    qo_indptr: *const u32,
    slot_stride_elems: ::core::ffi::c_longlong,
    out: *mut f32,
    r: i32,
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    stream: *mut c_void,
    write_state: bool,
    commit_len: *const i32,
    write_state_mask: *const u8,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    unsafe {
        crate::fire::gated_delta_net::chunk_prefill_batched_state_bf16(
            q_norm,
            k_norm,
            v,
            g_log,
            beta,
            state_base,
            slot_ids,
            qo_indptr,
            slot_stride_elems,
            out,
            r,
            k_h,
            v_h,
            k_d,
            v_d,
            stream,
            write_state,
            commit_len,
            write_state_mask,
        );
    }
}

/// The state-cached gated-delta prefill, fp32 state.
///
/// # Safety
///
/// As [`ssm_chunk_gated_delta_prefill_batched_cached_state_bf16`], with
/// `state_base` an arena of `slot_stride_elems` **fp32** per slot.
#[allow(clippy::too_many_arguments)]
pub unsafe fn ssm_chunk_gated_delta_prefill_batched_cached(
    _ctx: &DispatchCtx,
    q_norm: *const f32,
    k_norm: *const f32,
    v: *const f32,
    g_log: *const f32,
    beta: *const f32,
    state_base: *mut f32,
    slot_ids: *const i32,
    qo_indptr: *const u32,
    slot_stride_elems: ::core::ffi::c_longlong,
    out: *mut f32,
    r: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    stream: *mut c_void,
    write_state: bool,
    write_state_mask: *const u8,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    unsafe {
        crate::fire::gated_delta_net::chunk_prefill_batched_cached(
            q_norm,
            k_norm,
            v,
            g_log,
            beta,
            state_base,
            slot_ids,
            qo_indptr,
            slot_stride_elems,
            out,
            r,
            v_h,
            k_d,
            v_d,
            stream,
            write_state,
            write_state_mask,
        );
    }
}

/// The state-cached gated-delta prefill, bf16 state.
///
/// Holds the whole `K_d × V_d` state tile in shared memory for the length of
/// the token run: **65,536 bytes at production shapes**, against a 48 KiB
/// default. The opt-in the C++ did with a file-local `cudaFuncSetAttribute`
/// is now `runtime::module::raise_dynamic_smem_cap`, at the fire, for every
/// kernel over the cap rather than for this one.
///
/// Takes no `commit_len`: the state it writes is the one it has been holding,
/// so there is no partial commit to express. Takes no `K_h` either — it is
/// not GQA-aware and requires the expanded layout.
///
/// # Safety
///
/// As [`ssm_chunk_gated_delta_prefill_batched_state_bf16`], minus
/// `commit_len` and `k_h`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn ssm_chunk_gated_delta_prefill_batched_cached_state_bf16(
    _ctx: &DispatchCtx,
    q_norm: *const f32,
    k_norm: *const f32,
    v: *const f32,
    g_log: *const f32,
    beta: *const f32,
    state_base: *mut c_void,
    slot_ids: *const i32,
    qo_indptr: *const u32,
    slot_stride_elems: ::core::ffi::c_longlong,
    out: *mut f32,
    r: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    stream: *mut c_void,
    write_state: bool,
    write_state_mask: *const u8,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    unsafe {
        crate::fire::gated_delta_net::chunk_prefill_batched_cached_state_bf16(
            q_norm,
            k_norm,
            v,
            g_log,
            beta,
            state_base,
            slot_ids,
            qo_indptr,
            slot_stride_elems,
            out,
            r,
            v_h,
            k_d,
            v_d,
            stream,
            write_state,
            write_state_mask,
        );
    }
}

/// Nemotron-H's fused in-projection, cut into gate, conv input and `dt`.
///
/// `gate` null chooses `mamba_split_conv_dt`, a kernel with no `gate`
/// parameter at all — the absence of an output, not a mode flag. The row
/// carries `publishes_aux = &[(0, 2)]`.
///
/// # Safety
///
/// `projected` is `[N, projection_dim]` bf16; `conv_in` and `dt` are writable
/// for `[N, conv_dim]` and `[N, num_heads]`; `gate` is writable for
/// `[N, intermediate]` or null.
#[allow(clippy::too_many_arguments)]
pub unsafe fn ssm_nemotron_mamba_split_bf16(
    _ctx: &DispatchCtx,
    projected: *const c_void,
    gate: *mut c_void,
    conv_in: *mut c_void,
    dt: *mut c_void,
    n: i32,
    projection_dim: i32,
    intermediate: i32,
    conv_dim: i32,
    num_heads: i32,
    stream: *mut c_void,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    unsafe {
        crate::fire::nemotron_h::mamba_split_bf16(
            projected,
            gate,
            conv_in,
            dt,
            n,
            projection_dim,
            intermediate,
            conv_dim,
            num_heads,
            stream,
        );
    }
}

/// Nemotron-H's Mamba-2 selective scan, batched over paged state slots.
///
/// `sequence_prefill` is bound from
/// `Source::Ne(&Source::Rows, &Source::Attn("num_requests"))` — a fire with
/// more rows than requests IS a prefill — and it picks a 512-wide block with
/// a three-axis grid over the 256-wide two-axis decode form.
///
/// `dt_precomputed` and `da_precomputed` may be null; both kernels recompute
/// from `dt`, `a` and `dt_bias` when they are. Nemotron-H fires
/// `ssm::nemotron_prepare_mamba_dt_da` to fill them and Zamba does not.
///
/// # Safety
///
/// `conv_out`/`dt` are bf16 over the token run; `a`/`d`/`dt_bias` are
/// `[num_heads]` fp32; `ssm_state_base` is a slot arena; `slot_ids` is `[R]`;
/// `qo_indptr` is `[R + 1]`; `y` is writable for the token run.
#[allow(clippy::too_many_arguments)]
pub unsafe fn ssm_nemotron_mamba_ssm_batched_bf16(
    _ctx: &DispatchCtx,
    conv_out: *const c_void,
    dt: *const c_void,
    a: *const f32,
    d: *const f32,
    dt_bias: *const f32,
    dt_precomputed: *const f32,
    da_precomputed: *const f32,
    ssm_state_base: *mut c_void,
    slot_ids: *const i32,
    qo_indptr: *const u32,
    y: *mut c_void,
    r: i32,
    num_heads: i32,
    head_dim: i32,
    state_size: i32,
    n_groups: i32,
    conv_dim: i32,
    intermediate: i32,
    time_step_min: f32,
    sequence_prefill: bool,
    stream: *mut c_void,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    unsafe {
        crate::fire::nemotron_h::mamba_ssm_batched_bf16(
            conv_out,
            dt,
            a,
            d,
            dt_bias,
            dt_precomputed,
            da_precomputed,
            ssm_state_base,
            slot_ids,
            qo_indptr,
            y,
            r,
            num_heads,
            head_dim,
            state_size,
            n_groups,
            conv_dim,
            intermediate,
            time_step_min,
            sequence_prefill,
            stream,
        );
    }
}

/// Kimi Delta Attention's decode step.
///
/// **UNREACHABLE from a model trace today**, and not because of this
/// function: `table::ssm`'s row states no `Source` on any operand —
/// `state_base` is a driver-owned slab and `Source` has no `Scratch`
/// (`.wiki/driver/new-horizon.md` §52.3) — so `abi::emit_rust_dispatch`
/// skips the row whole and writes no arm to here. It is spelled because
/// `execution::RUST_SERVED` names the symbol, which is what dropped the shim
/// entry and let `kda.cu` be deleted, and because a caller that HAS the
/// pointers can reach it.
///
/// # Safety
///
/// `q_norm`/`k_norm`/`v`/`gate`/`beta` are fp32 over `[R, H, D]`;
/// `state_base` is a slot arena of `slot_stride_elems` fp32 per slot;
/// `slot_ids` is `[R]`; `out` is writable for `[R, H, D]`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn ssm_kda_recurrent_step_batched(
    _ctx: &DispatchCtx,
    q_norm: *const f32,
    k_norm: *const f32,
    v: *const f32,
    gate: *const f32,
    beta: *const f32,
    state_base: *mut f32,
    slot_ids: *const i32,
    slot_stride_elems: ::core::ffi::c_longlong,
    out: *mut f32,
    r: i32,
    h: i32,
    d: i32,
    stream: *mut c_void,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    unsafe {
        crate::fire::kda::recurrent_step_batched(
            q_norm,
            k_norm,
            v,
            gate,
            beta,
            state_base,
            slot_ids,
            slot_stride_elems,
            out,
            r,
            h,
            d,
            stream,
        );
    }
}

/// Kimi Delta Attention's prefill scan.
///
/// Unreachable for the same reason as
/// [`ssm_kda_recurrent_step_batched`]. Its block width — `min(32, D) * 32`,
/// one warp per state `v` row — is the archive's own measurement: **2.2× at
/// T=2048, 26.2 ms → 12.0 ms per layer at K3's widths.**
///
/// # Safety
///
/// As [`ssm_kda_recurrent_step_batched`], plus `qo_indptr` readable for
/// `[R + 1]`, and the input rectangles are over `qo_indptr[R]` tokens rather
/// than `R`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn ssm_kda_prefill_batched(
    _ctx: &DispatchCtx,
    q_norm: *const f32,
    k_norm: *const f32,
    v: *const f32,
    gate: *const f32,
    beta: *const f32,
    state_base: *mut f32,
    slot_ids: *const i32,
    qo_indptr: *const u32,
    slot_stride_elems: ::core::ffi::c_longlong,
    out: *mut f32,
    r: i32,
    h: i32,
    d: i32,
    stream: *mut c_void,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    unsafe {
        crate::fire::kda::prefill_batched(
            q_norm,
            k_norm,
            v,
            gate,
            beta,
            state_base,
            slot_ids,
            qo_indptr,
            slot_stride_elems,
            out,
            r,
            h,
            d,
            stream,
        );
    }
}

/// `moe::moe_gate_up_decode_gemv_bf16` — `moe/moe_dispatch.hpp`.
///
/// The gate/up leg of a decode-shaped MoE: one fused GEMV per route over the
/// concatenated `[gate | up]` expert weight, `num_tokens * top_k` routes on
/// `grid.y` and `ceil(2 * i_moe / 4)` output-column groups on `grid.x`.
///
/// The body is [`crate::fire::moe_dispatch::moe_gate_up_decode_gemv_bf16`],
/// which carries `moe_dispatch.cu:85-110` beside every constant. Here there
/// is only the ABI: `table::moe`'s one row, verbatim, in its order.
///
/// # Why it is here rather than behind a `pie_k_` shim
///
/// A two-dimensional block — `dim3(32, kGemvWarps)` — which no `LaunchRule`
/// states and which §10.5 forbids adding one for, plus three host products
/// (`routes`, `N = 2 * i_moe`, and a 64-bit expert stride) that are not
/// extents of any value the fire named. `execution::WALKED` classifies it for
/// exactly that. **The generated arm changes target with this function and
/// must not change behaviour** — `emit_rust_dispatch` writes a live arm for
/// this row because every operand is sourced, and a model trace reaches it.
///
/// # Safety
///
/// Every pointer is a device allocation of the row's shape, live on `stream`
/// until the launch completes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn moe_moe_gate_up_decode_gemv_bf16(
    _ctx: &DispatchCtx,
    topk_idx: *const i32,
    norm_x: *const c_void,
    gate_up_base: *const c_void,
    expert_gate_up: *mut c_void,
    num_tokens: i32,
    top_k: i32,
    h: i32,
    i_moe: i32,
    stream: *mut c_void,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    unsafe {
        crate::fire::moe_dispatch::moe_gate_up_decode_gemv_bf16(
            topk_idx,
            norm_x,
            gate_up_base,
            expert_gate_up,
            num_tokens,
            top_k,
            h,
            i_moe,
            stream,
        );
    }
}

/// `moe::moe_down_decode_gemv_bf16` — `moe/moe_dispatch.hpp`.
///
/// The down leg, and the mirror of [`moe_moe_gate_up_decode_gemv_bf16`]: the
/// reduction extent and the output width swap, so the divisibility refusal
/// moves from `h` to `i_moe` and the grid from `2 * i_moe` to `h`. Same
/// kernel template, the `ActByToken = false` instantiation.
///
/// Body: [`crate::fire::moe_dispatch::moe_down_decode_gemv_bf16`],
/// `moe_dispatch.cu:112-137`.
///
/// # Safety
///
/// As [`moe_moe_gate_up_decode_gemv_bf16`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn moe_moe_down_decode_gemv_bf16(
    _ctx: &DispatchCtx,
    topk_idx: *const i32,
    expert_act: *const c_void,
    down_base: *const c_void,
    expert_out: *mut c_void,
    num_tokens: i32,
    top_k: i32,
    h: i32,
    i_moe: i32,
    stream: *mut c_void,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    unsafe {
        crate::fire::moe_dispatch::moe_down_decode_gemv_bf16(
            topk_idx,
            expert_act,
            down_base,
            expert_out,
            num_tokens,
            top_k,
            h,
            i_moe,
            stream,
        );
    }
}

/// `moe::transpose_expert_scales_u8` — `moe/moe_dispatch.hpp`.
///
/// The MXFP4 group-scale relayout, `[e][n][kg] -> [e][kg][n]`, one E8M0 byte
/// per scale. A THREE-dimensional grid over a two-dimensional block, which is
/// two axes past the whole `LaunchRule` vocabulary — `families::moe`'s header
/// has said so since the split.
///
/// Body: [`crate::fire::moe_dispatch::transpose_expert_scales_u8`],
/// `moe_dispatch.cu:187-199`.
///
/// **This row is deliberately unsourced in `table::moe`**, so
/// `emit_rust_dispatch` writes no arm and no model trace reaches it. It is
/// here because `execution::RUST_SERVED` naming it is what drops the shim
/// entry, and the shim entry is what kept `moe_dispatch.cu`'s launcher alive.
///
/// # Safety
///
/// `src` and `dst` are each `num_experts * n * k_groups` device bytes and
/// must not overlap.
pub unsafe fn moe_transpose_expert_scales_u8(
    _ctx: &DispatchCtx,
    src: *const c_void,
    dst: *mut c_void,
    num_experts: i32,
    n: i32,
    k_groups: i32,
    stream: *mut c_void,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    unsafe {
        crate::fire::moe_dispatch::transpose_expert_scales_u8(
            src,
            dst,
            num_experts,
            n,
            k_groups,
            stream,
        );
    }
}

/// `moe::build_moe_ptrs_aligned_bf16` — `moe/moe_dispatch.hpp`.
///
/// Fills the six pointer arrays a pair of batched GEMMs reads, one thread per
/// padded block of the aligned MoE layout.
///
/// Body: [`crate::fire::moe_dispatch::build_moe_ptrs_aligned_bf16`],
/// `moe_dispatch.cu:204-250`. The line worth knowing about is `:246-248`,
/// which rewrites the `routed_blocks` OPERAND when either shared-expert base
/// is null — a host decision taken from a pointer's nullity, which is why
/// this is a walk and not a rule.
///
/// Unsourced in `table::moe`, so no generated arm; here for the shim-entry
/// reason [`moe_transpose_expert_scales_u8`] gives.
///
/// # Safety
///
/// The six pointer arrays hold at least `max_blocks` pointers each.
/// `shared_gate_up_base` and `shared_down_base` may be null.
#[allow(clippy::too_many_arguments)]
pub unsafe fn moe_build_moe_ptrs_aligned_bf16(
    _ctx: &DispatchCtx,
    expert_ids: *const i32,
    gate_up_base: *const c_void,
    down_base: *const c_void,
    aligned_in: *const c_void,
    aligned_gate_up: *mut c_void,
    aligned_act: *mut c_void,
    aligned_out: *mut c_void,
    a_gu_ptrs: *mut *const c_void,
    b_gu_ptrs: *mut *const c_void,
    c_gu_ptrs: *mut *mut c_void,
    a_dn_ptrs: *mut *const c_void,
    b_dn_ptrs: *mut *const c_void,
    c_dn_ptrs: *mut *mut c_void,
    max_blocks: i32,
    block_size: i32,
    h: i32,
    i_moe: i32,
    routed_blocks: i32,
    shared_gate_up_base: *const c_void,
    shared_down_base: *const c_void,
    stream: *mut c_void,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    unsafe {
        crate::fire::moe_dispatch::build_moe_ptrs_aligned_bf16(
            expert_ids,
            gate_up_base,
            down_base,
            aligned_in,
            aligned_gate_up,
            aligned_act,
            aligned_out,
            a_gu_ptrs,
            b_gu_ptrs,
            c_gu_ptrs,
            a_dn_ptrs,
            b_dn_ptrs,
            c_dn_ptrs,
            max_blocks,
            block_size,
            h,
            i_moe,
            routed_blocks,
            shared_gate_up_base,
            shared_down_base,
            stream,
        );
    }
}

/// `moe::reorder_moe_aligned_output_bf16` — `moe/moe_dispatch.hpp`.
///
/// Scatters an aligned GEMM's output rows back to route order, optionally
/// folding a shared-expert row on the way. ONE ABI symbol, TWO device rows:
/// the launcher forks on a pointer alignment, and `moe_dispatch.cu:271-273`
/// is the only place in the file where a `__global__` is chosen at run time
/// from a fact about an allocation rather than about a shape.
///
/// Body: [`crate::fire::moe_dispatch::reorder_moe_aligned_output_bf16`],
/// `moe_dispatch.cu:252-286`, where the §30 reading of that fork is written
/// out: the arms differ *structurally* — the vector kernel faults on a
/// misaligned base rather than running slower — so the branch is a port.
///
/// **`crates/model/src/qwen_3_5/forward/mod.rs:222` states this**, and the
/// row is fully sourced, so `emit_rust_dispatch` writes a live arm. The
/// target changes with this function; the behaviour must not.
///
/// # Safety
///
/// `aligned_out` is `[aligned_rows, hidden]` bf16, `sorted_route_ids`
/// `[aligned_rows]` i32, `route_out` writable for `[num_routes, hidden]`
/// bf16. `shared_out` may be null.
#[allow(clippy::too_many_arguments)]
pub unsafe fn moe_reorder_moe_aligned_output_bf16(
    _ctx: &DispatchCtx,
    aligned_out: *const c_void,
    sorted_route_ids: *const i32,
    route_out: *mut c_void,
    num_routes: i32,
    aligned_rows: i32,
    hidden: i32,
    shared_row_begin: i32,
    num_tokens: i32,
    shared_out: *mut c_void,
    stream: *mut c_void,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    unsafe {
        crate::fire::moe_dispatch::reorder_moe_aligned_output_bf16(
            aligned_out,
            sorted_route_ids,
            route_out,
            num_routes,
            aligned_rows,
            hidden,
            shared_row_begin,
            num_tokens,
            shared_out,
            stream,
        );
    }
}

/// `ssm::build_nemotron_moe_ptrs_decode_batched_bf16` — `ssm/nemotron_h.hpp`.
///
/// Nemotron-H's MoE decode pointer builder: one thread per route, six device
/// pointer arrays filled for a pair of batched GEMMs, plus the router weight
/// copied out as f32.
///
/// Body:
/// [`crate::fire::nemotron_h::build_nemotron_moe_ptrs_decode_batched_bf16`],
/// `nemotron_h.cu:53-94`. The trap it documents: the kernel's bound is
/// `routes = n * top_k`, not `n`.
///
/// **The `table::ssm` row stays unbound and that is deliberate.** §52.3's
/// missing `Source::Scratch(name, extent)` still has no word for a slab this
/// driver allocated, so no operand is sourced, `emit_rust_dispatch` writes no
/// arm, and nothing in a model trace reaches this. What `RUST_SERVED` changed
/// is only that the shim no longer emits an entry — which is what let
/// `ssm/nemotron_h.cu` be deleted.
///
/// # Safety
///
/// The two weight-pointer arrays are device arrays of at least `num_experts`
/// pointers; the six output arrays hold at least `n * top_k` pointers each.
#[allow(clippy::too_many_arguments)]
pub unsafe fn ssm_build_nemotron_moe_ptrs_decode_batched_bf16(
    _ctx: &DispatchCtx,
    topk_idx: *const i32,
    topk_w: *const f32,
    up_weight_ptrs: *const *const c_void,
    down_weight_ptrs: *const *const c_void,
    norm_x: *const c_void,
    expert_up: *mut c_void,
    expert_act: *mut c_void,
    expert_out: *mut c_void,
    a_up_ptrs: *mut *const c_void,
    b_up_ptrs: *mut *const c_void,
    c_up_ptrs: *mut *mut c_void,
    a_down_ptrs: *mut *const c_void,
    b_down_ptrs: *mut *const c_void,
    c_down_ptrs: *mut *mut c_void,
    weights_out: *mut f32,
    n: i32,
    top_k: i32,
    hidden: i32,
    intermediate: i32,
    stream: *mut c_void,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    unsafe {
        crate::fire::nemotron_h::build_nemotron_moe_ptrs_decode_batched_bf16(
            topk_idx,
            topk_w,
            up_weight_ptrs,
            down_weight_ptrs,
            norm_x,
            expert_up,
            expert_act,
            expert_out,
            a_up_ptrs,
            b_up_ptrs,
            c_up_ptrs,
            a_down_ptrs,
            b_down_ptrs,
            c_down_ptrs,
            weights_out,
            n,
            top_k,
            hidden,
            intermediate,
            stream,
        );
    }
}

/// `ssm::build_nemotron_moe_ptrs_aligned_bf16` — `ssm/nemotron_h.hpp`.
///
/// The aligned-batch form: one thread per padded block of the sorted MoE
/// layout, and four guard terms rather than one, because `block_size`,
/// `hidden` and `intermediate` are multipliers inside the kernel's address
/// arithmetic.
///
/// Body: [`crate::fire::nemotron_h::build_nemotron_moe_ptrs_aligned_bf16`],
/// `nemotron_h.cu:96-137`. Unbound for the reason
/// [`ssm_build_nemotron_moe_ptrs_decode_batched_bf16`] gives.
///
/// # Safety
///
/// As the decode form, with `max_blocks` in place of `n * top_k`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn ssm_build_nemotron_moe_ptrs_aligned_bf16(
    _ctx: &DispatchCtx,
    expert_ids: *const i32,
    up_weight_ptrs: *const *const c_void,
    down_weight_ptrs: *const *const c_void,
    aligned_in: *const c_void,
    aligned_up: *mut c_void,
    aligned_act: *mut c_void,
    aligned_out: *mut c_void,
    a_up_ptrs: *mut *const c_void,
    b_up_ptrs: *mut *const c_void,
    c_up_ptrs: *mut *mut c_void,
    a_down_ptrs: *mut *const c_void,
    b_down_ptrs: *mut *const c_void,
    c_down_ptrs: *mut *mut c_void,
    max_blocks: i32,
    block_size: i32,
    hidden: i32,
    intermediate: i32,
    stream: *mut c_void,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    unsafe {
        crate::fire::nemotron_h::build_nemotron_moe_ptrs_aligned_bf16(
            expert_ids,
            up_weight_ptrs,
            down_weight_ptrs,
            aligned_in,
            aligned_up,
            aligned_act,
            aligned_out,
            a_up_ptrs,
            b_up_ptrs,
            c_up_ptrs,
            a_down_ptrs,
            b_down_ptrs,
            c_down_ptrs,
            max_blocks,
            block_size,
            hidden,
            intermediate,
            stream,
        );
    }
}

/// `attn::write_kv_to_pages` — was `attn/kv_paged.hpp`, and that file's
/// launcher at `kv_paged.cu:109` is DELETED.
///
/// The paged KV append every fire makes once per layer. One
/// `if (layer.is_native_bf16())` over two programs: the native bf16 appender
/// with its envelope refresh, and the five-way `switch (layer.scheme)`.
///
/// Body: [`crate::fire::kv_paged::write_kv_to_pages`]. Classified
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
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    let _ = unsafe {
        crate::fire::kv_paged::write_kv_to_pages(
            layer,
            k_curr,
            v_curr,
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
/// Body: [`crate::fire::kv_paged::write_kv_explicit_bf16`]. Classified
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
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    let _ = unsafe {
        crate::fire::kv_paged::write_kv_explicit_bf16(
            layer, k_curr, v_curr, w_page, w_off, b, stream, row_valid,
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

/// `layout::embed_bf16` — was `layout/embed.hpp`, and that file is DELETED
/// with its `.cu` and the whole of `kernels-cuda/csrc/src/layout/`.
///
/// The first launch of every fire: one row of the vocabulary table per token.
///
/// Body: [`crate::fire::embed::embed_bf16`]. Classified `Execution::Walk`
/// with `Control::Switch` — `embed<true>` or `embed<false>`, chosen from a
/// 16-byte alignment test on `weight` and `y` plus `hidden % 8`, which also
/// sizes the grid.
///
/// # Safety
///
/// The caller's; every pointer is a device address live across the launch.
pub unsafe fn layout_embed_bf16(
    _ctx: &DispatchCtx,
    token_ids: *const i32,
    weight: *const c_void,
    y: *mut c_void,
    num_tokens: i32,
    hidden: i32,
    vocab: i32,
    stream: *mut c_void,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    let _ = unsafe {
        crate::fire::embed::embed_bf16(
            token_ids, weight, y, num_tokens, hidden, vocab, stream,
        )
    };
}

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

/// `attn::combine_attn_outputs_bf16` — was `attn/dsv4_compress.hpp`, and that
/// file is DELETED with its `.cu`.
///
/// Body: [`crate::fire::dsv4_compress::combine_attn_outputs_bf16`]. The block
/// clamp is `[32, 256]` and `LaunchRule::PerHeadElementwise`'s is `[32, 128]`,
/// which is why no row states this geometry.
///
/// # Safety
///
/// The caller's; every pointer is a device address live across the launch.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_combine_attn_outputs_bf16(
    _ctx: &DispatchCtx,
    o1: *const c_void,
    lse1: *const f32,
    o2: *const c_void,
    lse2: *const f32,
    o_out: *mut c_void,
    lse_out: *mut f32,
    n: i32,
    num_heads: i32,
    head_dim: i32,
    stream: *mut c_void,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    let _ = unsafe {
        crate::fire::dsv4_compress::combine_attn_outputs_bf16(
            o1, lse1, o2, lse2, o_out, lse_out, n, num_heads, head_dim, stream,
        )
    };
}

/// `attn::dsv4_boundary_meta_decode` — was `attn/dsv4_compress.hpp`, and that
/// file is DELETED with its `.cu`.
///
/// Body: [`crate::fire::dsv4_compress::dsv4_boundary_meta_decode`].
///
/// # Safety
///
/// [`attn_combine_attn_outputs_bf16`]'s.
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
/// [`attn_combine_attn_outputs_bf16`]'s.
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
/// [`attn_combine_attn_outputs_bf16`]'s.
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
/// Body: [`crate::fire::kv_paged::write_kv_explicit_bf16_devwin`].
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
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    let _ = unsafe {
        crate::fire::kv_paged::write_kv_explicit_bf16_devwin(
            layer, k_curr, v_curr, w_page, w_off, win_d, n_max, stream,
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
