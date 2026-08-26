//! `Mla`: multi-head latent attention. The latent split/absorb math, the
//! paged latent appender, and the two attention engines — FlashInfer's mla
//! fa2 (Hopper-class, by-value params, cooperative grid) and the naive
//! scalar/mma kernels (Blackwell and the selected paths). Engine selection
//! reads the arch from the context and the smem arms from the plan's device
//! facts — none of it leaks above these entries.

use kernels::KernelError;
use model_ir::Dtype;

use crate::attn::kv;
use crate::attn::plan::MlaPlan;
use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, count, dtype_dispatch, refuse, stated};
use crate::tensor::{KvPool, RaggedTensor, Tensor};

const BLOCK: u32 = 256;

/// One block per row, warp-shuffle reduction scratch beside it.
const fn rms(rows: u32) -> Launch {
    Launch::per_row(rows, BLOCK).smem((BLOCK / 32) * 4)
}

/// The pool facts an mla layer reads: latent pages ride `keys`, rope pages
/// ride `values`.
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
    ) -> Result<Self, KernelError> {
        Ok(Self {
            ckv_pages: pool.keys.ptr,
            kpe_pages: pool.values.ptr,
            page_size: pool.page_size,
            kv_lora_rank: stated(op, kv_lora_rank)?,
            rope_dim,
        })
    }
}

/// The per-head rope width a row's width spells at a stated head count —
/// `attn::row_heads`'s mirror (that one divides a width by the head width;
/// this one divides by the count), kept separate because the two quotients
/// are different quantities.
fn rope_per_head(op: &'static str, q_pe: Tensor, heads: i32) -> Result<i32, KernelError> {
    if heads <= 0 {
        return Err(refuse(op, "the stated head count is zero"));
    }
    let width = stated(op, q_pe.width)?;
    if width <= 0 || width % heads != 0 {
        return Err(refuse(
            op,
            format!("the {width}-wide rotated half does not divide by the {heads} stated heads"),
        ));
    }
    Ok(width / heads)
}

/// Splits `kv_a` into the rmsnormed compressed latent and the rope plane.
fn split_kv_a_norm(
    ctx: &Ctx,
    op: &'static str,
    kv_a: Tensor,
    weight: Tensor,
    eps: f32,
    kv_c: &mut Tensor,
    k_pe: &mut Tensor,
) -> Result<(), KernelError> {
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
) -> Result<(), KernelError> {
    const OP: &str = "attention.mla_latents";
    debug_assert_eq!(
        kv_c.width, kv_lora_rank,
        "the latent output is the stated rank wide"
    );
    split_kv_a_norm(ctx, OP, kv_a, weight, eps, kv_c, k_pe)
}

/// [`latents`], then a partial rotation of the rope plane in place.
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
) -> Result<(), KernelError> {
    const OP: &str = "attention.mla_latents_rope";
    dtype_dispatch!(OP, kv_a.dtype, { Bf16 => () });
    debug_assert_eq!(
        kv_c.width, kv_lora_rank,
        "the latent output is the stated rank wide"
    );
    split_kv_a_norm(ctx, OP, kv_a, weight, eps, kv_c, k_pe)?;
    crate::elemwise::rope::partial_q(ctx, k_pe, positions, rope_dim, rope_dim, theta)
}

/// Splits `q_b` into per-head nope/rope planes.
pub fn split_q_b(
    ctx: &Ctx,
    q_b: Tensor,
    heads: u32,
    nope_dim: u32,
    rope_dim: u32,
    q_nope: &mut Tensor,
    q_pe: &mut Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "attention.mla_split_q_b";
    let t = dtype_dispatch!(OP, q_b.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let heads = count(OP, "the head count this cut states", heads)?;
    let nope = stated(OP, nope_dim)?;
    let rope = stated(OP, rope_dim)?;
    let width = i64::from(heads) * (i64::from(nope) + i64::from(rope));
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
        ],
    )
}

/// Absorbs `kv_b`'s up-projection into q: per-head strided-batched GEMM,
/// mapping nope heads into latent space. cuBLAS is the engine, as before —
/// the old `gemm::absorb` helpers, now living with the family that fires
/// them.
pub fn absorb_q(
    ctx: &Ctx,
    q_nope: Tensor,
    kv_b: Tensor,
    heads: u32,
    kv_lora_rank: u32,
    nope_dim: u32,
    v_head_dim: u32,
    q_latent: &mut Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "attention.mla_absorb_q";
    dtype_dispatch!(OP, q_nope.dtype, { Bf16 => () });
    let heads = count(OP, "the head count this absorb states", heads)?;
    let nope = stated(OP, nope_dim)?;
    let v_dim = stated(OP, v_head_dim)?;
    let rank = stated(OP, kv_lora_rank)?;
    let tokens = count(OP, "rows", q_nope.rows)?;
    let handle = ctx.cublas(OP)?;

    #[cfg(feature = "_cuda")]
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
    #[cfg(not(feature = "_cuda"))]
    {
        let _ = (handle, kv_b, q_latent, heads, nope, v_dim, rank, tokens);
        Err(crate::jit::runtimeless(OP))
    }
}

/// The absorb's other half: latent attention output back through `kv_b`'s
/// value planes.
pub fn absorb_out(
    ctx: &Ctx,
    latent: Tensor,
    kv_b: Tensor,
    heads: u32,
    kv_lora_rank: u32,
    v_head_dim: u32,
    nope_dim: u32,
    o: &mut Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "attention.mla_absorb_out";
    dtype_dispatch!(OP, latent.dtype, { Bf16 => () });
    let heads = count(OP, "the head count this absorb states", heads)?;
    let nope = stated(OP, nope_dim)?;
    let v_dim = stated(OP, v_head_dim)?;
    let rank = stated(OP, kv_lora_rank)?;
    let tokens = count(OP, "rows", latent.rows)?;
    let handle = ctx.cublas(OP)?;

    // The value planes sit past the nope planes inside kv_b.
    let wv = kv_b
        .ptr
        .wrapping_add(2 * u64::from(nope.unsigned_abs()) * u64::from(rank.unsigned_abs()));

    #[cfg(feature = "_cuda")]
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
    #[cfg(not(feature = "_cuda"))]
    {
        let _ = (handle, wv, o, heads, v_dim, rank, tokens);
        Err(crate::jit::runtimeless(OP))
    }
}

#[cfg(feature = "_cuda")]
fn cublas_refused(op: &'static str, status: i32) -> KernelError {
    refuse(
        op,
        format!("`cublasGemmStridedBatchedEx` answered {status}"),
    )
}

/// The per-head strided-batched bf16 GEMM both absorbs ride.
#[cfg(feature = "_cuda")]
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

/// Appends latent rows (`kv_c` beside `k_pe`) into the pool's pages.
///
// MENLO-SEAM: the op states its write geometry (`write_page`/
// `write_offset`), but the latent writer (`mla_kv_append`) still re-derives
// each token's cell from the read-side CSR and the fire indptr riding in
// `kv_c` — the stated pair goes unread until the device text grows an
// explicit-descriptor latent writer.
pub fn kv_append(
    ctx: &Ctx,
    kv_c: RaggedTensor,
    k_pe: Tensor,
    pool: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "attention.mla_kv_append";
    let _ = (write_page, write_offset);
    dtype_dispatch!(OP, kv_c.data.dtype, { Bf16 => () });
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

// ── the two attention engines ───────────────────────────────────────────────

/// The naive scalar/mma kernels: the Blackwell path and the only engine the
/// selected (sparse) variants have.
mod naive {
    use super::{ArgValue, Ctx, Fire, KernelError, Launch, refuse};
    use crate::jit::Arg;

    pub const NAIVE_BLOCK: u32 = 256;

    pub const NAIVE_WARPS: i32 = NAIVE_BLOCK as i32 / 32;

    pub const NAIVE_MAX_PER: i32 = 16;

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
        pub selection: u64,
    }

    /// Fires the naive kernel that fits: mma when the shape supports it and
    /// no selection is stated, scalar otherwise. A shape neither can lane-
    /// split is refused — the old plane's silent decline was a no-launch
    /// that looked like success.
    pub fn fire(ctx: &Ctx, op: &'static str, ptrs: Ptrs, shape: Shape) -> Result<(), KernelError> {
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

        if !selected && mma_supported(shape.kv_lora_rank, shape.rope_dim, shape.num_heads) {
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
                    shape.num_requests.arg(),
                    shape.num_heads.arg(),
                    shape.page_size.arg(),
                    shape.sm_scale.arg(),
                    shape.causal.arg(),
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
                     (a multiple of 32, at most 512)"
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
                "::pie::attn::mla_naive::mla_naive_paged_kernel",
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
            ],
        )
    }
}

/// The FlashInfer mla fa2 engine: cooperative grid, one by-value parameter
/// block, smem arm chosen from the plan's device facts.
mod mla_fa2 {
    use super::{Ctx, KernelError, Layer, refuse};
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

    /// One `DISPATCH_SMEM_CONFIG` row. `stages`/`qk_shard` restate the
    /// trait parameters the instantiation strings are stamped with — kept
    /// as the record even though only `cta_tile_kv` and `smem` are read.
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

    /// `::flashinfer::MLAParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
    /// int32_t>`, measured at 288 bytes / align 8. Device pointers travel as
    /// the `u64` the handles carry; the offsets are pinned by the const
    /// asserts below — the safety payload the old `by_value!` macro bought,
    /// without the macro.
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
            return_lse_base_on_e: true,
        }
    }

    #[must_use]
    pub const fn grid(info: &MlaPlanInfo, arm: Arm) -> Launch {
        Launch::grid(
            [info.num_blks_x as u32, info.num_blks_y as u32, 1],
            [256, 1, 1],
        )
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
    ) -> Result<(), KernelError> {
        let Some(row) = INST.get(arm) else {
            return Err(refuse(
                op,
                "no `DISPATCH_SMEM_CONFIG` arm exists for this device",
            ));
        };

        ctx.fire(
            op,
            Fire::at(FILE, row[usize::from(causal)]).apply(launch),
            &[crate::attn::fa2::block(params)],
        )
    }
}

/// The dense-attention dispatch both plan-consuming entries share: naive on
/// cc >= 10 (the fa2 unit's Hopper intrinsics do not lower there), else the
/// mla fa2 arm the plan's smem facts admit.
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
    o: &mut Tensor,
) -> Result<(), KernelError> {
    let Some(major) = ctx.compute_capability_major() else {
        return Err(refuse(op, "the device's compute capability is unknowable"));
    };

    if major >= 10 {
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
            },
        );
    }

    let Some(arm) = mla_fa2::arm_index(plan.device.max_smem_per_sm) else {
        return Err(refuse(
            op,
            "no `DISPATCH_SMEM_CONFIG` arm fits this device's shared memory per SM",
        ));
    };
    if mla_fa2::ARMS[arm].cta_tile_kv < 32 {
        return Err(refuse(
            op,
            "the only `DISPATCH_SMEM_CONFIG` arm that fits this device's shared memory is \
             `CTA_TILE_KV = 16`, which writes past its own `SharedStorage` (measured)",
        ));
    }

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
            lse: 0,
        },
        false,
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

/// Latent attention over one token per lane; the fire indptr rides in `q`.
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
    o: &mut Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "attention.mla_decode";
    dtype_dispatch!(OP, q.data.dtype, { Bf16 => () });
    let heads = count(OP, "the head count this attention states", heads)?;
    let rope = rope_per_head(OP, q_pe, heads)?;
    let layer = Layer::of(OP, pool, kv_lora_rank, rope)?;
    dispatch_dense(
        ctx, OP, plan, q.data, q_pe, &layer, pool, q.indptr, heads, sm_scale, false, o,
    )
}

/// Latent attention over ragged prefixes, causal.
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
    o: &mut Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "attention.mla_prefill";
    dtype_dispatch!(OP, q.data.dtype, { Bf16 => () });
    let heads = count(OP, "the head count this attention states", heads)?;
    let rope = rope_per_head(OP, q_pe, heads)?;
    let layer = Layer::of(OP, pool, kv_lora_rank, rope)?;
    dispatch_dense(
        ctx, OP, plan, q.data, q_pe, &layer, pool, q.indptr, heads, sm_scale, true, o,
    )
}

/// The selected (sparse) paths: always the naive engine — the fa2 unit has
/// no selection seat — so the plan goes unread beyond its role as the op's
/// struct value.
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
) -> Result<(), KernelError> {
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
        },
    )
}

/// Decode over the sparse selection `index.topk` produced.
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
) -> Result<(), KernelError> {
    // MENLO-SEAM: the plan is accepted for the op's seat and goes unread —
    // the selected paths always run the naive engine (see [`selected`]).
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
) -> Result<(), KernelError> {
    // MENLO-SEAM: as `attention_decode_selected` — the plan goes unread on
    // the selected paths (see [`selected`]).
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
    )
}
