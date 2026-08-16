//! MLA's absorb pair — two `cublasGemmStridedBatchedEx` over the head axis.
//!
//! `gemm.cpp:2419-2468`. They project Q into the compressed latent space
//! before attention and the attention result back out of it afterwards, and
//! both do it by slicing ONE `kv_b_proj` bank per head: the weight is
//! `[heads, qk_nope_dim + v_head_dim, kv_lora_rank]`, the first absorb reads
//! its `W_k` half and the second its `W_v` half, at a byte offset this file
//! computes rather than at a pointer the caller passes.
//!
//! # Why these live in `gemm` and not in `attn`
//!
//! **They carry `gemm`'s namespace, and a routine's symbol is its family's
//! namespace plus its own name.** [`crate::jit::Family::symbol`] is the whole
//! of that join, so a host program in `attn` can only ever answer
//! `attn::…`; these two are stated by a trace as
//! `gemm::mla_absorb_q_to_latent_bf16` and `gemm::mla_absorb_latent_to_v_bf16`,
//! and while they sat beside MLA's attention kernels **no `Family` resolved
//! them at all** and both had to be carried by hand in
//! `not_yet_crossed::NOT_YET_CROSSED`.
//!
//! Moving the host program is the fix rather than giving `Family` a second
//! namespace. A second namespace would let one family claim symbols in
//! another's, which is exactly the ambiguity
//! `lib.rs`'s `no_symbol_is_declared_twice` exists to refuse — it can only
//! check that two families do not collide, not adjudicate which of two
//! claimants to a namespace is right. And the device work here IS a GEMM: it
//! is `cublasGemmStridedBatchedEx` with a bf16 A, a bf16 B and a strided head
//! batch, which is the same call [`crate::gemm::dense`] makes for every
//! other dense matmul in this crate. Only its CALLER is attention's.
//!
//! # The handle is the context's, not an argument
//!
//! The C++ took a `cublasHandle_t` as its first parameter and so did the first
//! Rust port. A routine cannot: a `cublasHandle_t` is not an
//! [`crate::jit::ArgValue`] and a trace statement could not name one if it
//! were. It comes off [`Ctx::cublas`] instead, which is where every other
//! cuBLAS routine in this family gets it, and which refuses in a sentence for
//! a context that was built without one.

use core::ffi::c_void;

use crate::jit::Ctx;
use kernels::Refusal;

#[cfg(feature = "_cuda")]
use cudarc::cublas::sys::{
    cublasComputeType_t, cublasContext, cublasGemmAlgo_t, cublasGemmStridedBatchedEx,
    cublasOperation_t, cublasStatus_t, cudaDataType,
};

/// `CUBLAS_COMPUTE_32F` — see [`crate::gemm::dense`]'s `COMPUTE` for the
/// tp > 1 argument that pins it.
#[cfg(feature = "_cuda")]
const ABSORB_COMPUTE: cublasComputeType_t = cublasComputeType_t::CUBLAS_COMPUTE_32F;

/// `CUBLAS_GEMM_DEFAULT_TENSOR_OP`, which the archive pinned on both calls.
#[cfg(feature = "_cuda")]
const ABSORB_ALGO: cublasGemmAlgo_t = cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP;

/// The absorb pair's shared call — `cublasGemmStridedBatchedEx` over the head
/// axis, with `alpha = 1` and `beta = 0`.
///
/// # Safety
///
/// The caller's, per entry point below.
#[cfg(feature = "_cuda")]
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
    /// The archive's `check(status, api)` — `gemm.cpp:47`.
    #[cfg(feature = "_cuda")]
    fn absorb_check(status: cublasStatus_t, what: &str) {
    assert!(
    status == cublasStatus_t::CUBLAS_STATUS_SUCCESS,
    "cuBLAS error ({}): {what}",
    status as i32
    );
    }

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
            core::ptr::from_ref(&alpha).cast(),
            a,
            cudaDataType::CUDA_R_16BF,
            lda,
            stride_a,
            b,
            cudaDataType::CUDA_R_16BF,
            ldb,
            stride_b,
            core::ptr::from_ref(&beta).cast(),
            c,
            cudaDataType::CUDA_R_16BF,
            ldc,
            stride_c,
            heads,
            ABSORB_COMPUTE,
            ABSORB_ALGO,
        )
    };
    absorb_check(status, what);
}

/// `gemm::mla_absorb_q_to_latent_bf16` — `gemm.cpp:2419-2442`.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty batch or head count;
/// [`Refusal::Absent`] if this context carries no cuBLAS handle.
///
/// # Safety
///
/// `q_nope` must address `tokens * heads * qk_nope_dim` bf16 elements,
/// `kv_b_proj` the whole `heads * (qk_nope_dim + v_head_dim) * kv_lora_rank`
/// bank, and `q_latent` `tokens * heads * kv_lora_rank` writable elements —
/// all live across the launch, which is asynchronous on the handle's stream.
#[allow(clippy::too_many_arguments)]
pub fn mla_absorb_q_to_latent_bf16(
    ctx: &Ctx,
    q_nope: *const c_void,
    kv_b_proj: *const c_void,
    q_latent: *mut c_void,
    tokens: i32,
    heads: i32,
    qk_nope_dim: i32,
    v_head_dim: i32,
    kv_lora_rank: i32,
) -> Result<(), Refusal> {
    let handle = ctx.cublas()?;
    // SAFETY: `call()`'s contract -- the three matrices address live device
    // memory of the extents above, and `handle`'s stream is this fire's.
    #[cfg(feature = "_cuda")]
    unsafe {
        absorb(
            handle,
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
    #[cfg(not(feature = "_cuda"))]
    let _ =
        (handle, q_nope, kv_b_proj, q_latent, tokens, heads, qk_nope_dim, v_head_dim, kv_lora_rank);
    Ok(())
}

/// `gemm::mla_absorb_latent_to_v_bf16` — `gemm.cpp:2444-2468`.
///
/// # Errors
///
/// As [`mla_absorb_q_to_latent_bf16`]'s.
///
/// # Safety
///
/// As [`mla_absorb_q_to_latent_bf16`]'s, with `attn_latent` in place of
/// `q_nope` and `attn_v` (`tokens * heads * v_head_dim`) as the output.
#[allow(clippy::too_many_arguments)]
pub fn mla_absorb_latent_to_v_bf16(
    ctx: &Ctx,
    attn_latent: *const c_void,
    kv_b_proj: *const c_void,
    attn_v: *mut c_void,
    tokens: i32,
    heads: i32,
    qk_nope_dim: i32,
    v_head_dim: i32,
    kv_lora_rank: i32,
) -> Result<(), Refusal> {
    let handle = ctx.cublas()?;
    // SAFETY: the offset lands inside the same bank the caller guaranteed —
    // `W_v` begins after `W_k`'s `qk_nope_dim * kv_lora_rank` bf16 elements,
    // which is twice that many BYTES.
    let wv = unsafe {
        kv_b_proj
            .cast::<u8>()
            .add(2 * (qk_nope_dim as usize) * (kv_lora_rank as usize))
            .cast::<c_void>()
    };
    // SAFETY: `call()`'s contract -- the three matrices address live device
    // memory of the extents above, and `handle`'s stream is this fire's.
    #[cfg(feature = "_cuda")]
    unsafe {
        absorb(
            handle,
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
    #[cfg(not(feature = "_cuda"))]
    let _ =
        (handle, wv, attn_latent, attn_v, tokens, heads, qk_nope_dim, v_head_dim, kv_lora_rank);
    Ok(())
}
