use crate::error::Error;
use dtype::Dtype;

use crate::attn::kv;
use crate::attn::plan::MlaPlan;
use crate::jit::{Ctx, Launch, count, dtype_dispatch, refuse, stated};
use crate::tensor::{KvPool, RaggedTensor, Tensor};
use prep::split_kv_a_norm;

pub(crate) mod decode;
pub mod expand;
pub mod prep;
#[cfg(feature = "cuda")]
mod sparse;

const BLOCK: u32 = 256;

pub(crate) const fn rms(rows: u32) -> Launch {
    Launch::per_row(rows, BLOCK).smem((BLOCK / 32) * 4)
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Layer {
    ckv_pages: u64,
    kpe_pages: u64,
    page_size: i32,
    kv_lora_rank: i32,
    rope_dim: i32,
}

impl Layer {
    fn of(
        op: &'static str,
        pool: &KvPool,
        kv_lora_rank: u32,
        rope_dim: i32,
    ) -> Result<Self, Error> {
        Ok(Self {
            ckv_pages: pool.keys.ptr,
            kpe_pages: pool.values.ptr,
            page_size: pool.page_size,
            kv_lora_rank: stated(op, kv_lora_rank)?,
            rope_dim,
        })
    }
}

/// Whether `pool` holds vLLM's standard fp8 MLA rows (`attn/mla.cuh`'s
/// `mla_kv_append_fp8`): one u8 plane, 512 latent and 64 or no rotated e4m3
/// codes a cell under one scale of 1.
fn fp8_rows(pool: &KvPool) -> bool {
    pool.keys.dtype == Dtype::U8 && matches!(pool.keys.width, 512 | 576)
}

fn rope_per_head(op: &'static str, q_pe: Tensor, heads: i32) -> Result<i32, Error> {
    if heads <= 0 {
        return Err(refuse(op, "the stated head count is zero"));
    }
    let width = stated(op, q_pe.width)?;
    if width < 0 || width % heads != 0 {
        return Err(refuse(
            op,
            format!("the {width}-wide rotated half does not divide by the {heads} stated heads"),
        ));
    }
    Ok(width / heads)
}

/// Where a scoring lands its rows' per-head base-2 log-sum-exp, if asked:
/// an f32 plane of one entry per head per row.
pub(crate) fn lse_plane(
    op: &'static str,
    lse: Option<&mut Tensor>,
    rows: u32,
    heads: i32,
) -> Result<u64, Error> {
    let Some(lse) = lse else {
        return Ok(0);
    };
    if lse.dtype != Dtype::F32 || lse.rows != rows || i64::from(lse.width) != i64::from(heads) {
        return Err(refuse(
            op,
            format!(
                "the log-sum-exp plane is {:?} [{}, {}] and the scores want f32 [{rows}, {heads}]",
                lse.dtype, lse.rows, lse.width
            ),
        ));
    }
    if lse.ptr == 0 {
        return Err(refuse(op, "the log-sum-exp plane is null"));
    }
    Ok(lse.ptr)
}

pub fn latents(
    ctx: &Ctx,
    kv_a: Tensor,
    weight: Tensor,
    eps: f32,
    kv_lora_rank: u32,
    kv_c: &mut Tensor,
    k_pe: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attn.mla_latents";
    debug_assert_eq!(
        kv_c.width, kv_lora_rank,
        "the latent output is the stated rank wide"
    );
    split_kv_a_norm(ctx, OP, kv_a, weight, eps, kv_c, k_pe)
}

#[allow(clippy::too_many_arguments)]
pub fn latents_rope(
    ctx: &Ctx,
    kv_a: Tensor,
    positions: Tensor,
    weight: Tensor,
    eps: f32,
    kv_lora_rank: u32,
    rope_dim: u32,
    theta: f32,
    kv_c: &mut Tensor,
    k_pe: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attn.mla_latents_rope";
    dtype_dispatch!(OP, kv_a.dtype, { Bf16 => () });
    debug_assert_eq!(
        kv_c.width, kv_lora_rank,
        "the latent output is the stated rank wide"
    );
    split_kv_a_norm(ctx, OP, kv_a, weight, eps, kv_c, k_pe)?;
    crate::elemwise::rope::partial_q(
        ctx,
        k_pe,
        positions,
        rope_dim,
        rope_dim,
        theta,
        rope_dim / 2,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn absorb_q(
    ctx: &Ctx,
    q_nope: Tensor,
    kv_b: Tensor,
    heads: u32,
    kv_lora_rank: u32,
    nope_dim: u32,
    v_head_dim: u32,
    q_latent: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attn.mla_absorb_q";
    dtype_dispatch!(OP, q_nope.dtype, { Bf16 => () });
    let heads = count(OP, "the head count this absorb states", heads)?;
    let nope = stated(OP, nope_dim)?;
    let v_dim = stated(OP, v_head_dim)?;
    let rank = stated(OP, kv_lora_rank)?;
    let tokens = count(OP, "rows", q_nope.rows)?;
    let handle = ctx.cublas(OP)?;

    #[cfg(feature = "cuda")]
    unsafe {
        absorb(
            handle,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            kv_b.ptr,
            q_nope.ptr,
            q_latent.ptr,
            rank,
            tokens,
            nope,
            rank,
            i64::from(nope + v_dim) * i64::from(rank),
            heads * nope,
            i64::from(nope),
            heads * rank,
            i64::from(rank),
            heads,
        )
        .map_err(|status| cublas_refused(OP, status))
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (handle, kv_b, q_latent, heads, nope, v_dim, rank, tokens);
        Err(crate::jit::runtimeless(OP))
    }
}

#[allow(clippy::too_many_arguments)]
pub fn absorb_out(
    ctx: &Ctx,
    latent: Tensor,
    kv_b: Tensor,
    heads: u32,
    kv_lora_rank: u32,
    v_head_dim: u32,
    nope_dim: u32,
    o: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attn.mla_absorb_out";
    dtype_dispatch!(OP, latent.dtype, { Bf16 => () });
    let heads = count(OP, "the head count this absorb states", heads)?;
    let nope = stated(OP, nope_dim)?;
    let v_dim = stated(OP, v_head_dim)?;
    let rank = stated(OP, kv_lora_rank)?;
    let tokens = count(OP, "rows", latent.rows)?;
    let handle = ctx.cublas(OP)?;

    let wv = kv_b
        .ptr
        .wrapping_add(2 * u64::from(nope.unsigned_abs()) * u64::from(rank.unsigned_abs()));

    #[cfg(feature = "cuda")]
    unsafe {
        absorb(
            handle,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            wv,
            latent.ptr,
            o.ptr,
            v_dim,
            tokens,
            rank,
            rank,
            i64::from(nope + v_dim) * i64::from(rank),
            heads * rank,
            i64::from(rank),
            heads * v_dim,
            i64::from(v_dim),
            heads,
        )
        .map_err(|status| cublas_refused(OP, status))
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (handle, wv, o, heads, v_dim, rank, tokens);
        Err(crate::jit::runtimeless(OP))
    }
}

#[cfg(feature = "cuda")]
fn cublas_refused(op: &'static str, status: i32) -> Error {
    refuse(
        op,
        format!("`cublasGemmStridedBatchedEx` answered {status}"),
    )
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
unsafe fn absorb(
    handle: *mut core::ffi::c_void,
    op_a: cudarc::cublas::sys::cublasOperation_t,
    a: u64,
    b: u64,
    c: u64,
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
) -> Result<(), i32> {
    use cudarc::cublas::sys::{
        cublasComputeType_t, cublasContext, cublasGemmAlgo_t, cublasGemmStridedBatchedEx,
        cublasOperation_t, cublasStatus_t, cudaDataType,
    };

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
            a as usize as *const core::ffi::c_void,
            cudaDataType::CUDA_R_16BF,
            lda,
            stride_a,
            b as usize as *const core::ffi::c_void,
            cudaDataType::CUDA_R_16BF,
            ldb,
            stride_b,
            core::ptr::from_ref(&beta).cast(),
            c as usize as *mut core::ffi::c_void,
            cudaDataType::CUDA_R_16BF,
            ldc,
            stride_c,
            heads,
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
    };
    if status == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(status as i32)
    }
}

pub fn kv_append(
    ctx: &Ctx,
    kv_c: RaggedTensor,
    k_pe: Tensor,
    pool: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attn.mla_kv_append";
    let _ = (write_page, write_offset);
    dtype_dispatch!(OP, kv_c.data.dtype, { Bf16 => () });
    #[cfg(feature = "cuda")]
    if sparse::fp8_rows(pool) {
        return prep::write_mla_fp8_rows(ctx, OP, kv_c.data, k_pe, kv_c.indptr, pool);
    }
    // The latent rows land and are read as bf16 (or as standard fp8 rows,
    // above): a pool laid out at another width would take them past its end.
    if pool.values.dtype != Dtype::Bf16 {
        return Err(refuse(
            OP,
            format!(
                "the latent pool holds {:?} rows, and MLA appends and reads bf16 ones; \
                 serve this deployment with bf16 kv",
                pool.values.dtype
            ),
        ));
    }
    debug_assert_eq!(
        k_pe.rows, kv_c.data.rows,
        "the rope plane is appended beside the latent plane, one row each"
    );
    prep::write_mla_to_pages(
        ctx,
        OP,
        kv_c.data,
        k_pe.arg(),
        pool.values.arg(),
        kv_c.indptr,
        pool,
        stated(OP, kv_c.data.width)?,
        stated(OP, k_pe.width)?,
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch_dense(
    ctx: &Ctx,
    op: &'static str,
    plan: &MlaPlan,
    q_nope: Tensor,
    q_pe: Tensor,
    layer: &Layer,
    pool: &KvPool,
    qo_indptr: Tensor,
    num_heads: i32,
    sm_scale: f32,
    causal: bool,
    window: Option<u32>,
    o: &mut Tensor,
    lse: u64,
) -> Result<(), Error> {
    // `kernel.mla.path`: Blackwell profiles pick `naive`; otherwise the FA2
    // arm the plan's device fits. The FA2 arms read a 512-wide latent and
    // every row, so a window or another width is the naive kernel's too.
    // A rope part other than 64 wide (GLM-5.3-Flash's none) is the naive
    // kernel's too: the FA2 arms read a 64-wide one beside the latent.
    // An fp8 pool is the naive arm's tensor-core kernel's, which widens it.
    if ctx.tuning().mla_naive
        || window.is_some()
        || layer.kv_lora_rank != 512
        || layer.rope_dim != 64
        || fp8_rows(pool)
    {
        let window = match window {
            None => 0,
            Some(w) => i32::try_from(w)
                .ok()
                .filter(|&w| w > 0)
                .ok_or_else(|| refuse(op, format!("a {w}-token window is not one it reads")))?,
        };
        let num_requests = kv::lanes_of(op, qo_indptr)?;
        return decode::naive::fire(
            ctx,
            op,
            decode::naive::Ptrs {
                q_nope: q_nope.ptr,
                q_pe: q_pe.ptr,
                ckv_pages: layer.ckv_pages,
                kpe_pages: layer.kpe_pages,
                qo_indptr: qo_indptr.ptr,
                kv_page_indices: pool.page_indices.ptr,
                kv_page_indptr: pool.page_indptr.ptr,
                kv_last_page_lens: pool.last_page_lens.ptr,
                o: o.ptr,
                lse,
                selection: 0,
            },
            decode::naive::Shape {
                kv_lora_rank: layer.kv_lora_rank,
                rope_dim: layer.rope_dim,
                page_size: layer.page_size,
                total_tokens: stated(op, o.rows)?,
                num_requests,
                num_heads,
                sm_scale,
                causal,
                top_k: 0,
                window,
                fp8: fp8_rows(pool),
            },
        );
    }

    let Some(arm) = decode::fa2::arm_index(plan.device.max_smem_per_sm) else {
        return Err(refuse(
            op,
            "no `DISPATCH_SMEM_CONFIG` arm fits this device's shared memory per SM",
        ));
    };

    let params = decode::fa2::pack(
        plan,
        layer,
        num_heads.unsigned_abs(),
        sm_scale,
        decode::fa2::Buffers {
            q_nope: q_nope.ptr,
            q_pe: q_pe.ptr,
            out: o.ptr,
            kv_page_indices: pool.page_indices.ptr,
            lse,
        },
        lse != 0,
    );
    decode::fa2::fire(
        ctx,
        op,
        arm,
        causal,
        &params,
        decode::fa2::grid(&plan.info, decode::fa2::ARMS[arm]),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn attention_decode(
    ctx: &Ctx,
    q: RaggedTensor,
    plan: &MlaPlan,
    q_pe: Tensor,
    pool: &KvPool,
    heads: u32,
    kv_lora_rank: u32,
    sm_scale: f32,
    window: Option<u32>,
    o: &mut Tensor,
    lse: Option<&mut Tensor>,
) -> Result<(), Error> {
    const OP: &str = "attn.mla_decode";
    dtype_dispatch!(OP, q.data.dtype, { Bf16 => () });
    let heads = count(OP, "the head count this attention states", heads)?;
    let rope = rope_per_head(OP, q_pe, heads)?;
    let layer = Layer::of(OP, pool, kv_lora_rank, rope)?;
    dispatch_dense(
        ctx,
        OP,
        plan,
        q.data,
        q_pe,
        &layer,
        pool,
        q.indptr,
        heads,
        sm_scale,
        false,
        window,
        o,
        lse_plane(OP, lse, o.rows, heads)?,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn attention_prefill(
    ctx: &Ctx,
    q: RaggedTensor,
    plan: &MlaPlan,
    q_pe: Tensor,
    pool: &KvPool,
    heads: u32,
    kv_lora_rank: u32,
    sm_scale: f32,
    window: Option<u32>,
    o: &mut Tensor,
    lse: Option<&mut Tensor>,
) -> Result<(), Error> {
    const OP: &str = "attn.mla_prefill";
    dtype_dispatch!(OP, q.data.dtype, { Bf16 => () });
    let heads = count(OP, "the head count this attention states", heads)?;
    let rope = rope_per_head(OP, q_pe, heads)?;
    let layer = Layer::of(OP, pool, kv_lora_rank, rope)?;
    dispatch_dense(
        ctx,
        OP,
        plan,
        q.data,
        q_pe,
        &layer,
        pool,
        q.indptr,
        heads,
        sm_scale,
        true,
        window,
        o,
        lse_plane(OP, lse, o.rows, heads)?,
    )
}

#[allow(clippy::too_many_arguments)]
fn selected(
    ctx: &Ctx,
    op: &'static str,
    q: RaggedTensor,
    q_pe: Tensor,
    selection: Tensor,
    pool: &KvPool,
    heads: u32,
    kv_lora_rank: u32,
    sm_scale: f32,
    causal: bool,
    o: &mut Tensor,
    lse: Option<&mut Tensor>,
) -> Result<(), Error> {
    dtype_dispatch!(op, q.data.dtype, { Bf16 => () });
    debug_assert_eq!(selection.dtype, Dtype::I32, "the selection is i32 page ids");
    let heads = count(op, "the head count this attention states", heads)?;
    let rope = rope_per_head(op, q_pe, heads)?;
    let layer = Layer::of(op, pool, kv_lora_rank, rope)?;
    if selection.ptr == 0 {
        return Err(refuse(
            op,
            "the selection this attention attends over is null",
        ));
    }
    if selection.rows != o.rows {
        return Err(refuse(
            op,
            "the selection does not carry one row per query row",
        ));
    }
    let num_requests = kv::lanes_of(op, q.indptr)?;
    let lse = lse_plane(op, lse, o.rows, heads)?;
    #[cfg(feature = "cuda")]
    if let Some(fired) = sparse::fire(
        ctx,
        op,
        &sparse::Shape {
            q: q.data,
            qo_indptr: q.indptr.ptr,
            q_pe,
            selection,
            pool,
            heads,
            kv_lora_rank: layer.kv_lora_rank,
            rope: layer.rope_dim,
            requests: num_requests,
            sm_scale,
            causal,
            lse,
        },
        o,
    ) {
        return fired;
    }
    decode::naive::fire(
        ctx,
        op,
        decode::naive::Ptrs {
            q_nope: q.data.ptr,
            q_pe: q_pe.ptr,
            ckv_pages: layer.ckv_pages,
            kpe_pages: layer.kpe_pages,
            qo_indptr: q.indptr.ptr,
            kv_page_indices: pool.page_indices.ptr,
            kv_page_indptr: pool.page_indptr.ptr,
            kv_last_page_lens: pool.last_page_lens.ptr,
            o: o.ptr,
            lse,
            selection: selection.ptr,
        },
        decode::naive::Shape {
            kv_lora_rank: layer.kv_lora_rank,
            rope_dim: layer.rope_dim,
            page_size: layer.page_size,
            total_tokens: stated(op, o.rows)?,
            num_requests,
            num_heads: heads,
            sm_scale,
            causal,
            top_k: stated(op, selection.width)?,
            window: 0,
            fp8: fp8_rows(pool),
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub fn attention_decode_selected(
    ctx: &Ctx,
    q: RaggedTensor,
    plan: &MlaPlan,
    q_pe: Tensor,
    selection: Tensor,
    pool: &KvPool,
    heads: u32,
    kv_lora_rank: u32,
    sm_scale: f32,
    o: &mut Tensor,
    lse: Option<&mut Tensor>,
) -> Result<(), Error> {
    let _ = plan;
    selected(
        ctx,
        "attn.mla_decode_selected",
        q,
        q_pe,
        selection,
        pool,
        heads,
        kv_lora_rank,
        sm_scale,
        false,
        o,
        lse,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn attention_prefill_selected(
    ctx: &Ctx,
    q: RaggedTensor,
    plan: &MlaPlan,
    q_pe: Tensor,
    selection: Tensor,
    pool: &KvPool,
    heads: u32,
    kv_lora_rank: u32,
    sm_scale: f32,
    o: &mut Tensor,
    lse: Option<&mut Tensor>,
) -> Result<(), Error> {
    let _ = plan;
    selected(
        ctx,
        "attn.mla_prefill_selected",
        q,
        q_pe,
        selection,
        pool,
        heads,
        kv_lora_rank,
        sm_scale,
        true,
        o,
        lse,
    )
}

#[cfg(feature = "cuda")]
#[cfg(test)]
mod tests {
    use super::decode::fa2::{ARMS, BLOCK, arm_index};
    use crate::attn::plan::Device;

    #[test]
    fn the_ada_ceiling_takes_the_sixteen_row_arm_and_it_fits_one_block() {
        let arm = arm_index(Device::L40S.max_smem_per_sm).expect("an arm fits the L40S");
        assert_eq!(ARMS[arm].cta_tile_kv, 16);
        assert!(ARMS[arm].smem <= Device::L40S.max_smem_per_block_optin);
        assert_eq!(
            BLOCK,
            [32, 4, 2],
            "a lane, a warp in its warpgroup, a warpgroup"
        );
    }
}
