//! MLA's absorb pair — two `cublasGemmStridedBatchedEx` over the head axis.
//!
//! `gemm.cpp:2419-2468`. They project Q into the compressed latent space
//! before attention and the attention result back out of it afterwards, and
//! both do it by slicing ONE `kv_b_proj` bank per head: the weight is
//! `[heads, qk_nope_dim + v_head_dim, kv_lora_rank]`, the first absorb reads
//! its `W_k` half and the second its `W_v` half, at a byte offset this file
//! computes rather than at a pointer the caller passes.
//!
//! They carry `gemm`'s namespace rather than `attn`'s because
//! [`crate::jit::Family::symbol`] joins a family's namespace to a routine's
//! own name, and the device work IS a GEMM — only its caller is attention's.
//!
//! The `cublasHandle_t` is not a parameter: it is not an
//! [`crate::jit::ArgValue`], so no trace statement could name one. It comes
//! off [`Ctx::cublas`], which refuses for a context built without one.

use core::ffi::c_void;

use crate::jit::Ctx;
use kernels::keys;
use kernels::routine::{Bank, Env, In, Out};
use kernels::Refusal;
// Both launchers spell the attribute in full as `#[kernels_macros::routine]`:
// there is deliberately no `use crate::routine` here, so `layout.rs:13-20`'s
// collision cannot arise.

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
#[kernels_macros::routine]
pub fn mla_absorb_q_to_latent_bf16(
    ctx: &Ctx,
    // `bind/mod.rs`'s `mla_absorb` passes `b.args[0]`, which is input zero of
    // the trace's operand run (inputs, outputs, weights, in that order).
    q_nope: In<0, c_void>,
    // THE POSITIONAL BANK, NOT THE NAMED ONE. This is bound from
    // `b.args[spec.n_in + spec.n_out]` — a position in the statement's own
    // operand list. `Weight<0, _>` would reach `f.weight_named(0)`, a
    // different table holding a different pointer.
    kv_b_proj: Bank<0, c_void>,
    // `b.args[spec.n_in]` — output zero of the same run, read positionally.
    q_latent: Out<0, c_void>,
    // `mla_absorb` passes `rows`, which is this: `f.rows.count`, the rows this
    // launch serves — NOT `f.rows.total`, the whole fire's count. `keys::Rows`
    // is the fallback spelling for a signature with no region to read the row
    // count off, and this one has none: the head pitch is not any operand's
    // extent, so a region here would carry a true row count and a fictional
    // width.
    tokens: Env<keys::Rows>,
    // THE FOUR THAT KEEP THIS ROW OFF THE TABLE PATH, and the reason moved
    // in Stage 3. `mla_absorb` reads them as `spec.params[0..4]`, which
    // `operand` now ANSWERS (`bind/table.rs:1054-1070`) -- so `Param<0, i32>`
    // through `Param<3, i32>` would be true of every one of them, and the
    // family header records why they still say `i32`: `bind/mod.rs:1404-1414`
    // spells this signature out as a `call:` fn-pointer type with four bare
    // `i32`s, and that type is the driver's. They stay unsourced until the
    // two edits can land together.
    //
    // What has NOT changed is why they ride the param channel at all: each
    // absorb takes the WHOLE `kv_b_proj` bank and slices it itself, so the
    // head pitch is not any operand's extent and no `InWidth`/`OutWidth`
    // could reach them.
    heads: i32,
    qk_nope_dim: i32,
    v_head_dim: i32,
    kv_lora_rank: i32,
) -> Result<(), Refusal> {
    let handle = ctx.cublas()?;
    // The archive's `if (tokens <= 0 || heads <= 0) return;`, which was a
    // bare return under `()`. This function's own `# Errors` has always said
    // `Refusal::Empty` for it and `bind::mod`'s `mla_absorb` has always
    // matched `Err(Refusal::Empty { .. }) => Ok(())` for it, calling it "the
    // archive's `tokens <= 0 || heads <= 0`" -- so the doc, the caller and
    // `gemm_service_parity`'s degenerate rows all named a refusal that the
    // body never made. It fell through to cuBLAS with a zero extent instead.
    if **tokens <= 0 {
        return Err(Refusal::Empty { what: "tokens" });
    }
    if heads <= 0 {
        return Err(Refusal::Empty { what: "heads" });
    }
    // SAFETY: `call()`'s contract -- the three matrices address live device
    // memory of the extents above, and `handle`'s stream is this fire's.
    #[cfg(feature = "_cuda")]
    unsafe {
        absorb(
            handle,
            cublasOperation_t::CUBLAS_OP_N,
            kv_b_proj.ptr,
            q_nope.ptr,
            q_latent.ptr,
            kv_lora_rank,
            **tokens,
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
        (handle, q_nope.ptr, kv_b_proj.ptr, q_latent.ptr, *tokens, heads, qk_nope_dim, v_head_dim, kv_lora_rank);
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
#[kernels_macros::routine]
pub fn mla_absorb_latent_to_v_bf16(
    ctx: &Ctx,
    // As [`mla_absorb_q_to_latent_bf16`]'s `q_nope`: `bind/mod.rs` runs both
    // symbols through one `mla_absorb` helper, so one binding decides both.
    attn_latent: In<0, c_void>,
    // The same positional bank, decided by the same binding.
    kv_b_proj: Bank<0, c_void>,
    attn_v: Out<0, c_void>,
    // The twin of [`mla_absorb_q_to_latent_bf16`]'s `tokens`; the reasoning is
    // written once, there.
    tokens: Env<keys::Rows>,
    // `spec.params[0..4]`, as above -- answerable as `Param<0..3, i32>` and
    // unsourced until `bind/mod.rs:1404-1414`'s fn-pointer type can move with
    // them. This row is the PROOF of the reason they ride the param channel:
    // `qk_nope_dim` and `kv_lora_rank` are used below to OFFSET INSIDE the
    // bank rather than to describe an operand, and an extent that names a
    // slice of a weight is not a fact about any region the statement placed.
    heads: i32,
    qk_nope_dim: i32,
    v_head_dim: i32,
    kv_lora_rank: i32,
) -> Result<(), Refusal> {
    let handle = ctx.cublas()?;
    // The archive's `if (tokens <= 0 || heads <= 0) return;`, which was a
    // bare return under `()`. This function's own `# Errors` has always said
    // `Refusal::Empty` for it and `bind::mod`'s `mla_absorb` has always
    // matched `Err(Refusal::Empty { .. }) => Ok(())` for it, calling it "the
    // archive's `tokens <= 0 || heads <= 0`" -- so the doc, the caller and
    // `gemm_service_parity`'s degenerate rows all named a refusal that the
    // body never made. It fell through to cuBLAS with a zero extent instead.
    if **tokens <= 0 {
        return Err(Refusal::Empty { what: "tokens" });
    }
    if heads <= 0 {
        return Err(Refusal::Empty { what: "heads" });
    }
    // SAFETY: the offset lands inside the same bank the caller guaranteed —
    // `W_v` begins after `W_k`'s `qk_nope_dim * kv_lora_rank` bf16 elements,
    // which is twice that many BYTES.
    let wv = unsafe {
        kv_b_proj
            .ptr
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
            attn_latent.ptr,
            attn_v.ptr,
            v_head_dim,
            **tokens,
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
        (handle, wv, attn_latent.ptr, attn_v.ptr, *tokens, heads, qk_nope_dim, v_head_dim, kv_lora_rank);
    Ok(())
}
