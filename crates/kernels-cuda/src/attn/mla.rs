use crate::error::Error;
use dtype::Dtype;

use crate::attn::kv;
use crate::attn::plan::MlaPlan;
use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, count, dtype_dispatch, refuse, stated};
use crate::tensor::{KvPool, RaggedTensor, Tensor};

const BLOCK: u32 = 256;

const fn rms(rows: u32) -> Launch {
    Launch::per_row(rows, BLOCK).smem((BLOCK / 32) * 4)
}

#[derive(Clone, Copy, Debug)]
struct Layer {
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
fn lse_plane(
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

fn split_kv_a_norm(
    ctx: &Ctx,
    op: &'static str,
    kv_a: Tensor,
    weight: Tensor,
    eps: f32,
    kv_c: &mut Tensor,
    k_pe: &mut Tensor,
) -> Result<(), Error> {
    let t = dtype_dispatch!(op, kv_a.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let kv_lora = count(op, "the latent width this cut states", kv_c.width)?;
    let rope = stated(op, k_pe.width)?;
    let src_row_stride = stated(op, kv_a.width)?;
    if src_row_stride < kv_lora + rope {
        return Err(refuse(
            op,
            format!(
                "the {src_row_stride}-wide source row does not hold the {kv_lora}-wide \
                 latent beside the {rope}-wide rope plane"
            ),
        ));
    }
    ctx.fire(
        op,
        Fire::at(
            "attn/mla.cuh",
            crate::jit::symbol(&format!("::pie::attn::mla_latents<{t}, 256>")),
        )
        .apply(rms(kv_a.rows)),
        &[
            kv_a.arg(),
            weight.arg(),
            kv_c.arg(),
            k_pe.arg(),
            kv_lora.arg(),
            rope.arg(),
            src_row_stride.arg(),
            eps.arg(),
            ctx.stage(),
        ],
    )
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
    const OP: &str = "attention.mla_latents";
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
    const OP: &str = "attention.mla_latents_rope";
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

pub fn split_q_b(
    ctx: &Ctx,
    q_b: Tensor,
    heads: u32,
    nope_dim: u32,
    rope_dim: u32,
    q_nope: &mut Tensor,
    q_pe: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.mla_split_q_b";
    let t = dtype_dispatch!(OP, q_b.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let heads = count(OP, "the head count this cut states", heads)?;
    let nope = stated(OP, nope_dim)?;
    let rope = stated(OP, rope_dim)?;
    let width = i64::from(heads) * (i64::from(nope) + i64::from(rope));
    // Whole 16-byte runs a half: eight elements a thread, and an i64 count
    // (a prefill's cut passes i32 elements).
    if nope % 8 == 0 && rope % 8 == 0 && [q_b.ptr, q_nope.ptr, q_pe.ptr].iter().all(|p| p % 16 == 0)
    {
        let runs = i64::from(q_b.rows) * width / 8;
        let blocks = u32::try_from((runs + i64::from(BLOCK) - 1) / i64::from(BLOCK))
            .map_err(|_| refuse(OP, format!("{runs} runs overflow a launch")))?;
        return ctx.fire(
            OP,
            Fire::at(
                "attn/mla.cuh",
                crate::jit::symbol(&format!("::pie::attn::mla_split_q_b_vec8<{t}>")),
            )
            .apply(Launch::grid([blocks, 1, 1], [BLOCK, 1, 1])),
            &[
                q_b.arg(),
                q_nope.arg(),
                q_pe.arg(),
                ArgValue::I64(runs),
                heads.arg(),
                nope.arg(),
                rope.arg(),
                ctx.stage(),
            ],
        );
    }
    let total = i64::from(q_b.rows) * width;
    let total = i32::try_from(total).map_err(|_| {
        refuse(
            OP,
            format!("{total} split elements do not fit the kernel's int"),
        )
    })?;
    ctx.fire(
        OP,
        Fire::at(
            "attn/mla.cuh",
            crate::jit::symbol(&format!("::pie::attn::mla_split_q_b<{t}>")),
        )
        .apply(Launch::flat(total.unsigned_abs(), BLOCK)),
        &[
            q_b.arg(),
            q_nope.arg(),
            q_pe.arg(),
            total.arg(),
            heads.arg(),
            nope.arg(),
            rope.arg(),
            ctx.stage(),
        ],
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
    const OP: &str = "attention.mla_absorb_q";
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
    const OP: &str = "attention.mla_absorb_out";
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
    const OP: &str = "attention.mla_kv_append";
    let _ = (write_page, write_offset);
    dtype_dispatch!(OP, kv_c.data.dtype, { Bf16 => () });
    #[cfg(feature = "cuda")]
    if sparse::fp8_rows(pool) {
        return kv::write_mla_fp8_rows(ctx, OP, kv_c.data, k_pe, kv_c.indptr, pool);
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
    kv::write_mla_to_pages(
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

pub(crate) mod naive {
    use super::{ArgValue, Ctx, Error, Fire, Launch, refuse};
    use crate::jit::Arg;

    pub const NAIVE_BLOCK: u32 = 256;

    pub const NAIVE_WARPS: i32 = NAIVE_BLOCK as i32 / 32;

    pub const NAIVE_MAX_PER: i32 = 32;

    pub const NAIVE_MAX_PE_PER: i32 = 4;

    pub const WAVE_TARGET: i64 = 296;

    pub const MMA_BM: i32 = 16;

    pub const MMA_SMEM_BYTES: u32 = 100_032;

    #[must_use]
    pub const fn naive_smem_bytes(kv_lora_rank: i32) -> u32 {
        let per = NAIVE_WARPS as i64 * kv_lora_rank as i64 + 2 * NAIVE_WARPS as i64;
        let bytes = per * 4;
        if bytes < 0 { 0 } else { bytes as u32 }
    }

    #[must_use]
    pub const fn mma_supported(kv_lora_rank: i32, rope_dim: i32, num_heads: i32) -> bool {
        kv_lora_rank == 512 && rope_dim == 64 && num_heads % MMA_BM == 0
    }

    #[must_use]
    pub fn head_group(num_heads: i32, total_tokens: i32) -> i32 {
        let mut g = NAIVE_WARPS;
        while g > 1
            && (num_heads % g != 0
                || i64::from(total_tokens) * i64::from(num_heads / g) < WAVE_TARGET)
        {
            g >>= 1;
        }
        g
    }

    #[derive(Clone, Copy, Debug)]
    pub struct Shape {
        pub kv_lora_rank: i32,
        pub rope_dim: i32,
        pub page_size: i32,
        pub total_tokens: i32,
        pub num_requests: i32,
        pub num_heads: i32,
        pub sm_scale: f32,
        pub causal: bool,
        pub top_k: i32,
        pub window: i32,
    }

    #[derive(Clone, Copy, Debug)]
    pub struct Ptrs {
        pub q_nope: u64,
        pub q_pe: u64,
        pub ckv_pages: u64,
        pub kpe_pages: u64,
        pub qo_indptr: u64,
        pub kv_page_indices: u64,
        pub kv_page_indptr: u64,
        pub kv_last_page_lens: u64,
        pub o: u64,
        pub lse: u64,
        pub selection: u64,
    }

    pub fn fire(ctx: &Ctx, op: &'static str, ptrs: Ptrs, shape: Shape) -> Result<(), Error> {
        const MMA_THREADS: u32 = 256;

        if shape.total_tokens <= 0 {
            return Err(refuse(op, "the query this attention was handed is empty"));
        }
        if ptrs.qo_indptr == 0 || ptrs.kv_page_indptr == 0 || ptrs.kv_last_page_lens == 0 {
            return Err(refuse(
                op,
                "the CSR triple this attention resolves its pages from is null",
            ));
        }
        let selected = ptrs.selection != 0;

        if !selected
            && shape.window == 0
            && mma_supported(shape.kv_lora_rank, shape.rope_dim, shape.num_heads)
        {
            #[allow(clippy::cast_sign_loss)]
            let launch = Launch::grid(
                [
                    (shape.num_heads / MMA_BM).max(0) as u32,
                    shape.total_tokens.max(0) as u32,
                    1,
                ],
                [MMA_THREADS, 1, 1],
            )
            .smem(MMA_SMEM_BYTES);
            return ctx.fire(
                op,
                Fire::at(
                    "attn/mla.cuh",
                    "::pie::attn::mla_naive::mma_detail::mla_mma_paged_kernel",
                )
                .apply(launch),
                &[
                    ArgValue::Ptr(ptrs.q_nope),
                    ArgValue::Ptr(ptrs.q_pe),
                    ArgValue::Ptr(ptrs.ckv_pages),
                    ArgValue::Ptr(ptrs.kpe_pages),
                    ArgValue::Ptr(ptrs.qo_indptr),
                    ArgValue::Ptr(ptrs.kv_page_indices),
                    ArgValue::Ptr(ptrs.kv_page_indptr),
                    ArgValue::Ptr(ptrs.kv_last_page_lens),
                    ArgValue::Ptr(ptrs.o),
                    ArgValue::Ptr(ptrs.lse),
                    shape.num_requests.arg(),
                    shape.num_heads.arg(),
                    shape.page_size.arg(),
                    shape.sm_scale.arg(),
                    shape.causal.arg(),
                    ctx.stage(),
                ],
            );
        }

        let ckv = shape.kv_lora_rank;
        let kpe = shape.rope_dim;
        if ckv % 32 != 0 || ckv / 32 > NAIVE_MAX_PER {
            return Err(refuse(
                op,
                format!(
                    "the latent rank {ckv} is not one this kernel can lane-split \
                     (a multiple of 32, at most 1024)"
                ),
            ));
        }
        if kpe % 32 != 0 || kpe / 32 > NAIVE_MAX_PE_PER {
            return Err(refuse(
                op,
                format!(
                    "the rope width {kpe} is not one this kernel can lane-split \
                     (a multiple of 32, at most 128)"
                ),
            ));
        }

        let g = head_group(shape.num_heads, shape.total_tokens);
        #[allow(clippy::cast_sign_loss)]
        let launch = Launch::grid(
            [
                shape.total_tokens.max(0) as u32,
                (shape.num_heads / g.max(1)).max(1) as u32,
                1,
            ],
            [NAIVE_BLOCK, 1, 1],
        )
        .smem(naive_smem_bytes(ckv));
        ctx.fire(
            op,
            Fire::at(
                "attn/mla.cuh",
                if ckv / 32 > 16 {
                    "::pie::attn::mla_naive::mla_naive_paged_kernel<32>"
                } else {
                    "::pie::attn::mla_naive::mla_naive_paged_kernel<16>"
                },
            )
            .apply(launch),
            &[
                ArgValue::Ptr(ptrs.q_nope),
                ArgValue::Ptr(ptrs.q_pe),
                ArgValue::Ptr(ptrs.ckv_pages),
                ArgValue::Ptr(ptrs.kpe_pages),
                ArgValue::Ptr(ptrs.qo_indptr),
                ArgValue::Ptr(ptrs.kv_page_indices),
                ArgValue::Ptr(ptrs.kv_page_indptr),
                ArgValue::Ptr(ptrs.kv_last_page_lens),
                ArgValue::Ptr(ptrs.o),
                ArgValue::Ptr(ptrs.lse),
                ArgValue::Ptr(ptrs.selection),
                shape.top_k.arg(),
                shape.num_requests.arg(),
                shape.num_heads.arg(),
                shape.kv_lora_rank.arg(),
                shape.rope_dim.arg(),
                shape.page_size.arg(),
                shape.sm_scale.arg(),
                shape.causal.arg(),
                g.arg(),
                shape.window.arg(),
                ctx.stage(),
            ],
        )
    }
}

/// The selected attention a block a row on tcgen05 (`attn/mla_sparse.cuh`).
#[cfg(feature = "cuda")]
mod sparse {

    use super::{Ctx, Error, KvPool, Tensor};
    use crate::jit::{Arg, ArgValue, Fire, Launch, aligned16, symbol};
    use crate::linear::tc5::supported;
    use dtype::Dtype;

    const FILE: &str = "attn/mla_sparse.cuh";

    /// `pie::attn::mlas::Params`, field for field.
    #[repr(C)]
    struct Params {
        o: u64,
        lse: u64,
        q: u64,
        q_pe: u64,
        qo_indptr: u64,
        page_indptr: u64,
        page_indices: u64,
        last_page_lens: u64,
        selection: u64,
        live: u64,
        requests: i32,
        causal: i32,
        runs: i32,
        page: i32,
        top_k: i32,
        rows: i32,
        q_stride: i32,
        qpe_stride: i32,
        o_stride: i32,
        scale_log2: f32,
        part_o: u64,
        part_lse: u64,
        splits: i32,
        dense: i32,
    }

    /// `pie::attn::mlas::Layout<kRope, kE4>::kSmem`: the ring (two slots of
    /// 64 keys' bf16 latent and rotated dims twice, or three of their e4m3
    /// codes, five without rotated dims), P twice, the score exchange, six key lists, barriers, and the
    /// 1024-byte alignment.
    const fn smem(rope: i32, e4: bool) -> u32 {
        let ring = if e4 {
            if rope > 0 { 3 * (32768 + 16384) } else { 5 * 32768 }
        } else {
            2 * (65536 + if rope > 0 { 16384 } else { 0 })
        };
        ring + 2 * 8192 + 16384 + 6 * 64 * 4 + 1024 + 1024
    }

    pub(super) struct Shape<'a> {
        pub q: Tensor,
        pub qo_indptr: u64,
        pub q_pe: Tensor,
        pub selection: Tensor,
        pub pool: &'a KvPool,
        pub heads: i32,
        pub kv_lora_rank: i32,
        pub rope: i32,
        pub requests: i32,
        pub sm_scale: f32,
        pub causal: bool,
        pub lse: u64,
    }

    /// Fires the pass where the profile says and the shape is the one it is
    /// built for: 64 heads over a 512-wide bf16 latent and a 64-wide rotated
    /// part; `None` where it does not serve.
    pub(super) fn fire(
        ctx: &Ctx,
        op: &'static str,
        s: &Shape<'_>,
        o: &mut Tensor,
    ) -> Option<Result<(), Error>> {
        let pool = s.pool;
        let cells = pool.keys.rows;
        let fp8 = fp8_rows(pool);
        let fits = ctx.tuning().mla_sparse
            && supported()
            && s.heads == 64
            && s.kv_lora_rank == 512
            && (s.rope == 0 || s.rope == 64)
            && s.q.dtype == Dtype::Bf16
            && (pool.keys.dtype == Dtype::Bf16 || fp8)
            && pool.page_size > 0
            && cells > 0
            && s.selection.width > 0
            && o.width == 64 * 512
            && s.q.width == 64 * 512
            && [s.q.ptr, o.ptr, pool.keys.ptr]
                .iter()
                .all(|&at| aligned16(at))
            && (!fp8 || u32::try_from(s.rope).is_ok_and(|rope| pool.keys.width == 512 + rope))
            && (s.rope == 0
                || ((fp8 || (pool.values.dtype == Dtype::Bf16 && aligned16(pool.values.ptr)))
                    && aligned16(s.q_pe.ptr)
                    && s.q_pe.width == 64 * 64));
        if !fits {
            // An fp8 pool has no other reader.
            return fp8.then(|| {
                Err(crate::jit::refuse(
                    op,
                    "fp8 latent rows are read by the tcgen05 sparse pass alone, and this \
                     shape or device is not one it serves",
                ))
            });
        }
        Some(launch(ctx, op, s, o))
    }

    /// Whether `pool` holds vLLM's standard fp8 rows (`attn/mla.cuh`'s
    /// `mla_kv_append_fp8`): one u8 plane, 512 latent and 64 or no rotated
    /// e4m3 codes a cell under one scale of 1.
    pub(super) fn fp8_rows(pool: &KvPool) -> bool {
        pool.keys.dtype == Dtype::U8 && matches!(pool.keys.width, 512 | 576)
    }

    fn launch(ctx: &Ctx, op: &'static str, s: &Shape<'_>, o: &mut Tensor) -> Result<(), Error> {
        let rows = o.rows;
        if rows == 0 {
            return Ok(());
        }
        let pool = s.pool;
        let cells = pool.keys.rows;
        let int = |v: u32| i32::try_from(v).unwrap_or(i32::MAX);
        let map = |ptr: u64, width: u32, box_rows: u32| {
            crate::attn::bf16_map(
                ptr,
                [width, cells, 0],
                [u64::from(width) * 2, 0],
                [64, box_rows, 1],
            )
        };
        let fp8 = fp8_rows(pool);
        // fp8 rows: one u8 view of the plane, 128 codes a box, stands for
        // every map (the rotated codes come from the same rows).
        let (map_ckv, map_ckv_run, map_kpe) = if fp8 {
            use crate::linear::tc5::{Map, encode};
            use cudarc::driver::sys as dr;
            let u8_map = |box_rows: u32| {
                encode(Map {
                    ptr: pool.keys.ptr,
                    ty: dr::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
                    dims: [pool.keys.width, cells, 0],
                    strides: [u64::from(pool.keys.width), 0],
                    boxes: [128, box_rows, 1],
                    swizzle: dr::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
                })
            };
            let rows = u8_map(1)?;
            (rows, u8_map(16)?, rows)
        } else {
            let map_ckv = map(pool.keys.ptr, 512, 1)?;
            let map_ckv_run = map(pool.keys.ptr, 512, 16)?;
            // No rotated part: the map is never read; any valid one stands in.
            let map_kpe = if s.rope > 0 {
                map(pool.values.ptr, 64, 1)?
            } else {
                map_ckv
            };
            (map_ckv, map_ckv_run, map_kpe)
        };
        let live = match ctx.stage() {
            ArgValue::Ptr(at) => at,
            _ => 0,
        };
        // Few rows (a decode) deal each row's tiles over splits, so the
        // multiprocessors fill in one wave: as many as fit them whole, at
        // most a split a tile of the selection, and at most 16 (a split's
        // Q load and partial write cost about a tile's work of their own).
        let sms = ctx.multiprocessors().unwrap_or(148);
        let tiles = s.selection.width.div_ceil(64).max(1);
        let splits = (sms / rows.max(1)).min(tiles).clamp(1, 16);
        let (part_o, part_lse) = if splits > 1 {
            let units = rows as usize * splits as usize * 64;
            (
                ctx.scratch_shared(op, "attn.mla_sparse.part_o", units * 512 * 2)? as usize as u64,
                ctx.scratch_shared(op, "attn.mla_sparse.part_lse", units * 4)? as usize as u64,
            )
        } else {
            (0, 0)
        };
        let params = Params {
            o: o.ptr,
            lse: s.lse,
            q: s.q.ptr,
            q_pe: s.q_pe.ptr,
            qo_indptr: s.qo_indptr,
            page_indptr: pool.page_indptr.ptr,
            page_indices: pool.page_indices.ptr,
            last_page_lens: pool.last_page_lens.ptr,
            selection: s.selection.ptr,
            live,
            requests: s.requests,
            causal: i32::from(s.causal),
            runs: 1,
            page: pool.page_size,
            top_k: int(s.selection.width),
            rows: int(rows),
            q_stride: int(s.q.width),
            qpe_stride: int(s.q_pe.width),
            o_stride: int(o.width),
            scale_log2: s.sm_scale * core::f32::consts::LOG2_E,
            part_o,
            part_lse,
            splits: int(splits),
            dense: i32::from(ctx.tuning().mla_dense_short),
        };
        ctx.fire(
            op,
            Fire::at(
                FILE,
                symbol(&format!("::pie::attn::mlas::mla_sparse<{}, {fp8}>", s.rope)),
            )
            .apply(
                Launch::grid([(rows * splits).min(sms), 1, 1], [384, 1, 1]).smem(smem(s.rope, fp8)),
            ),
            &[
                ArgValue::of(&map_ckv),
                ArgValue::of(&map_ckv_run),
                ArgValue::of(&map_kpe),
                ArgValue::of(&params),
            ],
        )?;
        if splits == 1 {
            return Ok(());
        }
        ctx.fire(
            op,
            Fire::at(FILE, "::pie::attn::mlas::mla_sparse_fold")
                .apply(Launch::grid([rows, 64, 1], [128, 1, 1]).programmatic(ctx.tuning().pdl)),
            &[
                ArgValue::Ptr(part_o),
                ArgValue::Ptr(part_lse),
                o.arg(),
                ArgValue::Ptr(s.lse),
                int(splits).arg(),
                int(o.width).arg(),
                ctx.stage(),
            ],
        )
    }
}

mod mla_fa2 {
    use super::{Ctx, Error, Layer, refuse};
    use crate::attn::fa2_abi::{UintFastdiv, resolve};
    use crate::attn::plan::{MlaPlan, MlaPlanInfo};
    use crate::jit::{Fire, Launch};

    const FILE: &str = "attn/mla.cuh";

    pub const INST: [[&str; 2]; 3] = [
        [
            "::flashinfer::mla::BatchMLAPagedAttentionKernel\
             <::pie::attn::mla_fa2::Traits<false, 2u, true, 64u>, \
             ::pie::attn::mla_fa2::Params>",
            "::flashinfer::mla::BatchMLAPagedAttentionKernel\
             <::pie::attn::mla_fa2::Traits<true, 2u, true, 64u>, \
             ::pie::attn::mla_fa2::Params>",
        ],
        [
            "::flashinfer::mla::BatchMLAPagedAttentionKernel\
             <::pie::attn::mla_fa2::Traits<false, 2u, true, 32u>, \
             ::pie::attn::mla_fa2::Params>",
            "::flashinfer::mla::BatchMLAPagedAttentionKernel\
             <::pie::attn::mla_fa2::Traits<true, 2u, true, 32u>, \
             ::pie::attn::mla_fa2::Params>",
        ],
        [
            "::flashinfer::mla::BatchMLAPagedAttentionKernel\
             <::pie::attn::mla_fa2::Traits<false, 1u, false, 16u>, \
             ::pie::attn::mla_fa2::Params>",
            "::flashinfer::mla::BatchMLAPagedAttentionKernel\
             <::pie::attn::mla_fa2::Traits<true, 1u, false, 16u>, \
             ::pie::attn::mla_fa2::Params>",
        ],
    ];

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Arm {
        #[allow(dead_code)]
        pub stages: u32,
        pub cta_tile_kv: u32,
        #[allow(dead_code)]
        pub qk_shard: bool,
        pub smem: u32,
    }

    pub const ARMS: [Arm; 3] = [
        Arm {
            stages: 2,
            cta_tile_kv: 64,
            qk_shard: true,
            smem: 221_696,
        },
        Arm {
            stages: 2,
            cta_tile_kv: 32,
            qk_shard: true,
            smem: 147_968,
        },
        Arm {
            stages: 1,
            cta_tile_kv: 16,
            qk_shard: false,
            smem: 92_672,
        },
    ];

    #[must_use]
    pub const fn arm_index(smem_limit_per_sm: u32) -> Option<usize> {
        let mut i = 0;
        while i < ARMS.len() {
            if smem_limit_per_sm >= ARMS[i].smem {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct MlaParams {
        pub q_nope: u64,
        pub q_pe: u64,
        pub ckv: u64,
        pub kpe: u64,
        pub partial_o: u64,
        pub partial_lse: u64,
        pub final_o: u64,
        pub final_lse: u64,
        pub q_indptr: u64,
        pub kv_indptr: u64,
        pub partial_indptr: u64,
        pub merge_packed_offset_start: u64,
        pub merge_packed_offset_end: u64,
        pub merge_partial_packed_offset_start: u64,
        pub merge_partial_packed_offset_end: u64,
        pub merge_partial_stride: u64,
        pub kv_indices: u64,
        pub q_len: u64,
        pub kv_len: u64,
        pub q_start: u64,
        pub kv_start: u64,
        pub kv_end: u64,
        pub work_indptr: u64,
        pub block_size: UintFastdiv,
        pub num_heads: UintFastdiv,
        pub q_nope_stride_n: u32,
        pub q_nope_stride_h: u32,
        pub q_pe_stride_n: u32,
        pub q_pe_stride_h: u32,
        pub ckv_stride_page: u32,
        pub ckv_stride_n: u32,
        pub kpe_stride_page: u32,
        pub kpe_stride_n: u32,
        pub o_stride_n: u32,
        pub o_stride_h: u32,
        pub sm_scale: f32,
        pub ckv_scale: f32,
        pub kpe_scale: f32,
        pub return_lse_base_on_e: bool,
    }

    const _: () = assert!(
        core::mem::size_of::<MlaParams>() == 288,
        "MlaParams: sizeof disagrees with the measured ::flashinfer::MLAParams",
    );
    const _: () = assert!(
        core::mem::align_of::<MlaParams>() == 8,
        "MlaParams: alignof disagrees with the measured ::flashinfer::MLAParams",
    );
    const _: () = assert!(core::mem::offset_of!(MlaParams, q_nope) == 0);
    const _: () = assert!(core::mem::offset_of!(MlaParams, work_indptr) == 176);
    const _: () = assert!(core::mem::offset_of!(MlaParams, block_size) == 184);
    const _: () = assert!(core::mem::offset_of!(MlaParams, num_heads) == 208);
    const _: () = assert!(core::mem::offset_of!(MlaParams, q_nope_stride_n) == 232);
    const _: () = assert!(core::mem::offset_of!(MlaParams, o_stride_h) == 268);
    const _: () = assert!(core::mem::offset_of!(MlaParams, sm_scale) == 272);
    const _: () = assert!(core::mem::offset_of!(MlaParams, ckv_scale) == 276);
    const _: () = assert!(core::mem::offset_of!(MlaParams, kpe_scale) == 280);
    const _: () = assert!(core::mem::offset_of!(MlaParams, return_lse_base_on_e) == 284);

    #[derive(Clone, Copy, Debug)]
    pub struct Buffers {
        pub q_nope: u64,
        pub q_pe: u64,
        pub out: u64,
        pub kv_page_indices: u64,
        pub lse: u64,
    }

    #[must_use]
    #[allow(clippy::cast_sign_loss)]
    pub fn pack(
        plan: &MlaPlan,
        layer: &Layer,
        num_heads: u32,
        sm_scale: f32,
        buffers: Buffers,
        want_lse: bool,
    ) -> MlaParams {
        let info = &plan.info;
        let int_buf = plan.workspace.int_ptr;
        let float_buf = plan.workspace.float_ptr;
        let page_size = layer.page_size as u32;
        let rank = layer.kv_lora_rank as u32;
        let rope = layer.rope_dim as u32;
        MlaParams {
            q_nope: buffers.q_nope,
            q_pe: buffers.q_pe,
            ckv: layer.ckv_pages,
            kpe: layer.kpe_pages,
            partial_o: resolve(float_buf, info.partial_o_offset),
            partial_lse: resolve(float_buf, info.partial_lse_offset),
            final_o: buffers.out,
            final_lse: if want_lse { buffers.lse } else { 0 },
            q_indptr: resolve(int_buf, info.q_indptr_offset),
            kv_indptr: resolve(int_buf, info.kv_indptr_offset),
            partial_indptr: resolve(int_buf, info.partial_indptr_offset),
            merge_packed_offset_start: resolve(int_buf, info.merge_packed_offset_start_offset),
            merge_packed_offset_end: resolve(int_buf, info.merge_packed_offset_end_offset),
            merge_partial_packed_offset_start: resolve(
                int_buf,
                info.merge_partial_packed_offset_start_offset,
            ),
            merge_partial_packed_offset_end: resolve(
                int_buf,
                info.merge_partial_packed_offset_end_offset,
            ),
            merge_partial_stride: resolve(int_buf, info.merge_partial_stride_offset),
            kv_indices: buffers.kv_page_indices,
            q_len: resolve(int_buf, info.q_len_offset),
            kv_len: resolve(int_buf, info.kv_len_offset),
            q_start: resolve(int_buf, info.q_start_offset),
            kv_start: resolve(int_buf, info.kv_start_offset),
            kv_end: resolve(int_buf, info.kv_end_offset),
            work_indptr: resolve(int_buf, info.work_indptr_offset),
            block_size: UintFastdiv::new(page_size),
            num_heads: UintFastdiv::new(num_heads),
            q_nope_stride_n: num_heads * rank,
            q_nope_stride_h: rank,
            q_pe_stride_n: num_heads * rope,
            q_pe_stride_h: rope,
            ckv_stride_page: page_size * rank,
            ckv_stride_n: rank,
            kpe_stride_page: page_size * rope,
            kpe_stride_n: rope,
            o_stride_n: num_heads * rank,
            o_stride_h: rank,
            sm_scale,
            ckv_scale: 1.0,
            kpe_scale: 1.0,
            return_lse_base_on_e: false,
        }
    }

    // The kernel reads its warp from `threadIdx.y` and its warpgroup from
    // `threadIdx.z`; a flat 256-wide block hands every thread warp 0 of
    // warpgroup 0 and a lane index up to 255, and the tile loads then run 32
    // rows into the 16-row `CTA_TILE_KV` buffers.
    pub const BLOCK: [u32; 3] = [32, 4, 2];

    #[must_use]
    pub const fn grid(info: &MlaPlanInfo, arm: Arm) -> Launch {
        Launch::grid([info.num_blks_x as u32, info.num_blks_y as u32, 1], BLOCK)
            .smem(arm.smem)
            .cooperative()
    }

    pub fn fire(
        ctx: &Ctx,
        op: &'static str,
        arm: usize,
        causal: bool,
        params: &MlaParams,
        launch: Launch,
    ) -> Result<(), Error> {
        let Some(row) = INST.get(arm) else {
            return Err(refuse(
                op,
                "no `DISPATCH_SMEM_CONFIG` arm exists for this device",
            ));
        };

        ctx.fire(
            op,
            Fire::at(FILE, row[usize::from(causal)]).apply(launch),
            &[crate::jit::ArgValue::of(params)],
        )
    }
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
    if ctx.tuning().mla_naive
        || window.is_some()
        || layer.kv_lora_rank != 512
        || layer.rope_dim != 64
    {
        let window = match window {
            None => 0,
            Some(w) => i32::try_from(w)
                .ok()
                .filter(|&w| w > 0)
                .ok_or_else(|| refuse(op, format!("a {w}-token window is not one it reads")))?,
        };
        let num_requests = kv::lanes_of(op, qo_indptr)?;
        return naive::fire(
            ctx,
            op,
            naive::Ptrs {
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
            naive::Shape {
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
            },
        );
    }

    let Some(arm) = mla_fa2::arm_index(plan.device.max_smem_per_sm) else {
        return Err(refuse(
            op,
            "no `DISPATCH_SMEM_CONFIG` arm fits this device's shared memory per SM",
        ));
    };

    let params = mla_fa2::pack(
        plan,
        layer,
        num_heads.unsigned_abs(),
        sm_scale,
        mla_fa2::Buffers {
            q_nope: q_nope.ptr,
            q_pe: q_pe.ptr,
            out: o.ptr,
            kv_page_indices: pool.page_indices.ptr,
            lse,
        },
        lse != 0,
    );
    mla_fa2::fire(
        ctx,
        op,
        arm,
        causal,
        &params,
        mla_fa2::grid(&plan.info, mla_fa2::ARMS[arm]),
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
    const OP: &str = "attention.mla_decode";
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
    const OP: &str = "attention.mla_prefill";
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
    naive::fire(
        ctx,
        op,
        naive::Ptrs {
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
        naive::Shape {
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
        "attention.mla_decode_selected",
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
        "attention.mla_prefill_selected",
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

/// The queries an expanded prefill reads: `q_b`'s rows whole (each head's
/// `nope` part then its rope part, as [`split_q_b`] reads them), or the two
/// halves apart.
#[derive(Clone, Copy, Debug)]
pub enum ExpandedQ {
    Packed(Tensor),
    Split { q_nope: Tensor, q_pe: Tensor },
}

/// The shape of a latent attention's heads, as its absorbs state it.
#[derive(Clone, Copy, Debug)]
pub struct Heads {
    pub heads: u32,
    pub kv_lora_rank: u32,
    pub nope: u32,
    pub rope: u32,
    pub v: u32,
}

/// Whether [`prefill_expanded`] serves these heads on this device.
#[must_use]
pub fn expanded_takes(h: &Heads) -> bool {
    #[cfg(feature = "cuda")]
    {
        h.rope > 0
            && h.nope.is_multiple_of(8)
            && h.rope.is_multiple_of(8)
            && h.v.is_multiple_of(8)
            && (h.kv_lora_rank + h.rope).is_multiple_of(8)
            && crate::attn::tc5_fmha::planes_take(h.nope + h.rope, h.v)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = h;
        false
    }
}

/// A latent attention's whole prefill, its query absorb, its scoring over
/// every cached latent row and its output absorb, expanded as vLLM's MLA
/// prefill runs it: the fire's latent rows (each request's pages, prefix
/// and new alike) gathered into rows of their own, expanded through `kv_b`
/// into each head's key part and value, the keys packed beside the shared
/// rope part, and a causal prefill of one key head a query head at the
/// key width `nope + rope` and the value width `v` landing `o` (`[rows]
/// [heads * v]`). The same sums the absorbed form takes, in another order.
/// `Ok(false)` where it does not take the fire: the caller runs the three.
#[allow(clippy::too_many_arguments)]
pub fn prefill_expanded(
    ctx: &Ctx,
    q: ExpandedQ,
    qo_indptr: Tensor,
    pool: &KvPool,
    kv_b: Tensor,
    h: Heads,
    sm_scale: f32,
    o: &mut Tensor,
) -> Result<bool, Error> {
    #[cfg(feature = "cuda")]
    {
        use crate::attn::tc5_fmha::{Planes, prefill_planes};
        const OP: &str = "attention.mla_prefill_expanded";
        const LAT: &str = "attention.mla_expand_latents";
        const KPE: &str = "attention.mla_expand_rope";
        const IDENT: &str = "attention.mla_expand_pages";
        const LIVE: &str = "attention.mla_expand_live";
        const EXP: &str = "attention.mla_expand_rows";
        const KEYS: &str = "attention.mla_expand_keys";
        const QS: &str = "attention.mla_expand_queries";
        let width = h.heads * (h.nope + h.v);
        let dqk = h.nope + h.rope;
        let rows = o.rows;
        let page = u32::try_from(pool.page_size).unwrap_or(0);
        let pages = pool.page_indices.rows;
        let q_rows = match q {
            ExpandedQ::Packed(q) => q.rows,
            ExpandedQ::Split { q_nope, .. } => q_nope.rows,
        };
        if !expanded_takes(&h)
            || kv_b.dtype != Dtype::Bf16
            || kv_b.rows != width
            || kv_b.width != h.kv_lora_rank
            || o.dtype != Dtype::Bf16
            || o.width != h.heads * h.v
            || q_rows != rows
            || rows == 0
            || pages == 0
            || page == 0
            || !kv::native_bf16(pool)
        {
            return Ok(false);
        }
        let requests = kv::lanes_of(OP, qo_indptr)?.unsigned_abs();
        let cells = u64::from(pages) * u64::from(page);
        let cells32 = u32::try_from(cells).map_err(|_| refuse(OP, "the fire's cells overflow"))?;
        let bytes = |n: u64| usize::try_from(n * 2).unwrap_or(usize::MAX);
        let lat = ctx.scratch_stream(OP, LAT, bytes(cells * u64::from(h.kv_lora_rank)))? as u64;
        let kpe = ctx.scratch_stream(OP, KPE, bytes(cells * u64::from(h.rope)))? as u64;
        let ident = ctx.scratch_stream(OP, IDENT, pages as usize * 4)? as u64;
        let live = ctx.scratch_stream(OP, LIVE, 8)? as u64;
        let runs = cells * u64::from((h.kv_lora_rank + h.rope) / 8);
        let threads = u32::try_from(runs.max(u64::from(pages)))
            .map_err(|_| refuse(OP, "the gather overflows a launch"))?;
        let int = |v: u32| ArgValue::I32(i32::try_from(v).unwrap_or(i32::MAX));
        ctx.fire(
            OP,
            Fire::at(EXPAND_FILE, "::pie::attn::mla_expand::gather")
                .apply(Launch::flat(threads, BLOCK)),
            &[
                ArgValue::Ptr(pool.keys.ptr),
                ArgValue::Ptr(pool.values.ptr),
                pool.page_indices.arg(),
                int(pages),
                int(page),
                int(h.kv_lora_rank),
                int(h.rope),
                ArgValue::Ptr(lat),
                ArgValue::Ptr(kpe),
                ArgValue::Ptr(ident),
                ArgValue::Ptr(live),
                int(requests),
            ],
        )?;
        // Each cell's heads expanded: `[heads][nope + v]` a cell.
        let exp = ctx.scratch_stream(OP, EXP, bytes(cells * u64::from(width)))? as u64;
        let mut expanded = Tensor::new(exp, cells32, width, Dtype::Bf16);
        crate::linear::gemm::matmul(
            ctx,
            Tensor::new(lat, cells32, h.kv_lora_rank, Dtype::Bf16),
            kv_b,
            &mut expanded,
        )?;
        let keys = ctx.scratch_stream(OP, KEYS, bytes(cells * u64::from(h.heads * dqk)))? as u64;
        let key_runs = cells * u64::from(h.heads) * u64::from(dqk / 8);
        ctx.fire(
            OP,
            Fire::at(EXPAND_FILE, "::pie::attn::mla_expand::pack_keys").apply(Launch::flat(
                u32::try_from(key_runs)
                    .map_err(|_| refuse(OP, "the key pack overflows a launch"))?,
                BLOCK,
            )),
            &[
                ArgValue::Ptr(exp),
                ArgValue::Ptr(kpe),
                ArgValue::Ptr(keys),
                ArgValue::I64(i64::try_from(cells).unwrap_or(i64::MAX)),
                int(h.heads),
                int(h.nope),
                int(h.v),
                int(h.rope),
            ],
        )?;
        let (q_at, q_stride) = match q {
            ExpandedQ::Packed(q) => {
                if q.width != h.heads * dqk || q.dtype != Dtype::Bf16 {
                    return Ok(false);
                }
                (q.ptr, q.width)
            }
            ExpandedQ::Split { q_nope, q_pe } => {
                let qs =
                    ctx.scratch_stream(OP, QS, bytes(u64::from(rows) * u64::from(h.heads * dqk)))?
                        as u64;
                let q_runs = u64::from(rows) * u64::from(h.heads) * u64::from(dqk / 8);
                ctx.fire(
                    OP,
                    Fire::at(EXPAND_FILE, "::pie::attn::mla_expand::pack_queries").apply(
                        Launch::flat(
                            u32::try_from(q_runs)
                                .map_err(|_| refuse(OP, "the query pack overflows a launch"))?,
                            BLOCK,
                        ),
                    ),
                    &[
                        q_nope.arg(),
                        q_pe.arg(),
                        ArgValue::Ptr(qs),
                        ArgValue::I64(i64::from(rows)),
                        int(h.heads),
                        int(h.nope),
                        int(h.rope),
                    ],
                )?;
                (qs, h.heads * dqk)
            }
        };
        prefill_planes(
            ctx,
            OP,
            &Planes {
                q: q_at,
                q_stride,
                o: o.ptr,
                o_stride: o.width,
                k: keys,
                k_head: dqk,
                k_cell: h.heads * dqk,
                v: exp + u64::from(h.nope) * 2,
                v_head: h.nope + h.v,
                v_cell: width,
                cells: cells32,
                heads: h.heads,
                dqk,
                dv: h.v,
                qo_indptr: qo_indptr.ptr,
                kv_indptr: pool.page_indptr.ptr,
                kv_indices: ident,
                last_page_len: pool.last_page_lens.ptr,
                page,
                live,
                requests,
                rows,
                causal: true,
                sm_scale,
            },
        )?;
        Ok(true)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (ctx, q, qo_indptr, pool, kv_b, h, sm_scale, o);
        Ok(false)
    }
}

#[cfg(feature = "cuda")]
const EXPAND_FILE: &str = "attn/mla_expand.cuh";

#[cfg(test)]
mod tests {
    use super::mla_fa2::{ARMS, BLOCK, arm_index};
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
