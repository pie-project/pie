use core::ffi::c_void;
use kernels_macros::routine;

use crate::jit::Ctx;
use crate::jit::abi::Tensor;
use kernels::Refusal;
use kernels::routine::{Const, In, Out};

#[cfg(feature = "_cuda")]
use cudarc::cublas::sys::{
    cublasComputeType_t, cublasContext, cublasGemmAlgo_t, cublasGemmStridedBatchedEx,
    cublasOperation_t, cublasStatus_t, cudaDataType,
};

#[cfg(feature = "_cuda")]
const ABSORB_COMPUTE: cublasComputeType_t = cublasComputeType_t::CUBLAS_COMPUTE_32F;

#[cfg(feature = "_cuda")]
const ABSORB_ALGO: cublasGemmAlgo_t = cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP;

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

#[routine(driver, canon = "mla_absorb.q")]
pub fn mla_absorb_q_to_latent_bf16(
    ctx: &Ctx<'_>,
    q_nope: In<Tensor<c_void>>,
    kv_b_proj: Const<Tensor<c_void>>,
    q_latent: Out<Tensor<c_void>>,
    // THE STATEMENT CARRIES THESE FOUR, which is what `Const` says: the driver
    // arm reads them off `spec.params` and each absorb takes the WHOLE
    // `kv_b_proj` bank and slices it itself, so no operand's rectangle spells
    // the head pitch.
    heads: Const<i32>,
    qk_nope_dim: Const<i32>,
    v_head_dim: Const<i32>,
    kv_lora_rank: Const<i32>,
    tokens: Const<i32>,
) -> Result<(), Refusal> {
    let (heads, qk_nope_dim, v_head_dim, kv_lora_rank) =
        (heads.v, qk_nope_dim.v, v_head_dim.v, kv_lora_rank.v);
    let tokens = *tokens;
    let handle = ctx.cublas()?;

    if tokens <= 0 {
        return Err(Refusal::Empty { what: "tokens" });
    }
    if heads <= 0 {
        return Err(Refusal::Empty { what: "heads" });
    }

    #[cfg(feature = "_cuda")]
    unsafe {
        absorb(
            handle,
            cublasOperation_t::CUBLAS_OP_N,
            kv_b_proj.v,
            q_nope.ptr,
            q_latent.ptr,
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
    let _ = (
        handle,
        q_nope.ptr,
        kv_b_proj.v,
        q_latent.ptr,
        tokens,
        heads,
        qk_nope_dim,
        v_head_dim,
        kv_lora_rank,
    );
    Ok(())
}

#[routine(driver, canon = "mla_absorb.out")]
pub fn mla_absorb_latent_to_v_bf16(
    ctx: &Ctx<'_>,
    attn_latent: In<Tensor<c_void>>,
    kv_b_proj: Const<Tensor<c_void>>,
    attn_v: Out<Tensor<c_void>>,
    // [`mla_absorb_q_to_latent_bf16`]'s four, for its reason.
    heads: Const<i32>,
    qk_nope_dim: Const<i32>,
    v_head_dim: Const<i32>,
    kv_lora_rank: Const<i32>,
    tokens: Const<i32>,
) -> Result<(), Refusal> {
    let (heads, qk_nope_dim, v_head_dim, kv_lora_rank) =
        (heads.v, qk_nope_dim.v, v_head_dim.v, kv_lora_rank.v);
    let tokens = *tokens;
    let handle = ctx.cublas()?;

    if tokens <= 0 {
        return Err(Refusal::Empty { what: "tokens" });
    }
    if heads <= 0 {
        return Err(Refusal::Empty { what: "heads" });
    }

    let wv = unsafe {
        kv_b_proj
            .v
            .cast::<u8>()
            .add(2 * (qk_nope_dim as usize) * (kv_lora_rank as usize))
            .cast::<c_void>()
    };

    #[cfg(feature = "_cuda")]
    unsafe {
        absorb(
            handle,
            cublasOperation_t::CUBLAS_OP_T,
            wv,
            attn_latent.ptr,
            attn_v.ptr,
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
    let _ = (
        handle,
        wv,
        attn_latent.ptr,
        attn_v.ptr,
        tokens,
        heads,
        qk_nope_dim,
        v_head_dim,
        kv_lora_rank,
    );
    Ok(())
}
