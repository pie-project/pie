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
            "attn/mla.hip",
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
            "attn/mla.hip",
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
    let handle = ctx.blas(OP)?;

    #[cfg(feature = "rocm")]
    unsafe {
        absorb(
            handle,
            ctx.stream(),
            crate::hip::sys::BLAS_OP_N,
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
        .map_err(|why| refuse(OP, why.to_string()))
    }
    #[cfg(not(feature = "rocm"))]
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
    let handle = ctx.blas(OP)?;

    let wv = kv_b
        .ptr
        .wrapping_add(2 * u64::from(nope.unsigned_abs()) * u64::from(rank.unsigned_abs()));

    #[cfg(feature = "rocm")]
    unsafe {
        absorb(
            handle,
            ctx.stream(),
            crate::hip::sys::BLAS_OP_T,
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
        .map_err(|why| refuse(OP, why.to_string()))
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = (handle, wv, o, heads, v_dim, rank, tokens);
        Err(crate::jit::runtimeless(OP))
    }
}

#[cfg(feature = "rocm")]
#[allow(clippy::too_many_arguments)]
unsafe fn absorb(
    handle: *mut core::ffi::c_void,
    stream: *mut core::ffi::c_void,
    op_a: crate::hip::sys::BlasOp,
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
) -> Result<(), crate::hip::blas::Refused> {
    use crate::hip::blas::{self, Operand};
    use crate::hip::sys;

    // SAFETY: the caller hands a live handle and device planes sized for
    // `heads` strided problems.
    unsafe {
        blas::bind(handle, stream)?;
        blas::gemm_strided(
            handle,
            op_a,
            sys::BLAS_OP_N,
            m,
            n,
            k,
            Operand::new(a, sys::BLAS_BF16, lda).strided(stride_a),
            Operand::new(b, sys::BLAS_BF16, ldb).strided(stride_b),
            Operand::new(c, sys::BLAS_BF16, ldc).strided(stride_c),
            heads,
        )
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

/// The reference walk of `kernels/attn/mla.hip` (`mla_naive`): any latent
/// and rope width, a wavefront per (query row, key stride).
pub(crate) mod naive {
    use super::{ArgValue, Ctx, Error, Fire, Launch, refuse};
    use crate::jit::Arg;

    pub const NAIVE_BLOCK: u32 = 256;

    pub const NAIVE_MAX_PER: i32 = 16;

    pub const NAIVE_MAX_PE_PER: i32 = 4;

    pub const WAVE_TARGET: i64 = 296;

    /// Wavefronts in the MLA walk's block.
    const WAVES: i32 = NAIVE_BLOCK as i32 / 64;

    /// The widest latent and rope rows a lane-split walk holds.
    pub const MAX_LATENT: i32 = 64 * NAIVE_MAX_PER;
    pub const MAX_ROPE: i32 = 64 * NAIVE_MAX_PE_PER;

    #[must_use]
    pub const fn naive_smem_bytes(kv_lora_rank: i32) -> u32 {
        let per = WAVES as i64 * kv_lora_rank as i64 + 2 * WAVES as i64;
        let bytes = per * 4;
        if bytes < 0 { 0 } else { bytes as u32 }
    }

    /// Query heads a block of the MLA walk takes: as many as its wavefronts
    /// while the grid still fills the device.
    #[must_use]
    pub fn wave_group(num_heads: i32, total_tokens: i32) -> i32 {
        group_of(WAVES, num_heads, total_tokens)
    }

    fn group_of(most: i32, num_heads: i32, total_tokens: i32) -> i32 {
        let mut g = most;
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

    pub fn fire(ctx: &Ctx, op: &'static str, ptrs: Ptrs, shape: Shape) -> Result<(), Error> {
        if shape.total_tokens <= 0 {
            return Err(refuse(op, "the query this attention was handed is empty"));
        }
        if ptrs.qo_indptr == 0 || ptrs.kv_page_indptr == 0 || ptrs.kv_last_page_lens == 0 {
            return Err(refuse(
                op,
                "the CSR triple this attention resolves its pages from is null",
            ));
        }
        let (ckv, kpe) = (shape.kv_lora_rank, shape.rope_dim);
        if ckv <= 0 || ckv > MAX_LATENT {
            return Err(refuse(
                op,
                format!("the latent rank {ckv} is not one this walk holds (1..={MAX_LATENT})"),
            ));
        }
        if !(0..=MAX_ROPE).contains(&kpe) {
            return Err(refuse(
                op,
                format!("the rope width {kpe} is not one this walk holds (0..={MAX_ROPE})"),
            ));
        }
        let g = wave_group(shape.num_heads, shape.total_tokens);
        #[allow(clippy::cast_sign_loss)]
        let launch = Launch::grid(
            [
                shape.total_tokens as u32,
                (shape.num_heads / g).max(1) as u32,
                1,
            ],
            [NAIVE_BLOCK, 1, 1],
        )
        .smem(naive_smem_bytes(ckv));
        ctx.fire(
            op,
            Fire::at(
                "attn/mla.hip",
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
                ctx.stage(),
            ],
        )
    }
}

/// The matrix-core walk of `kernels/attn/mla.hip` (`mla`): a 512-wide
/// latent beside a 64-wide (or no) rope plane, 64 query rows a block.
pub(crate) mod latent {
    use super::{ArgValue, Ctx, Error, Fire, Launch, Layer, refuse};
    use crate::attn::plan::MlaPlan;
    use crate::jit::symbol;

    const FILE: &str = "attn/mla.hip";

    pub const RANK: i32 = 512;

    /// Query rows a block owns, and its threads.
    pub const BLOCK_ROWS: u32 = 64;
    const THREADS: u32 = 512;

    /// Keys a split takes at least: two of the walk's 32-key tiles.
    const MIN_SHARE: u64 = 64;

    const MAX_SPLITS: u64 = 64;

    #[must_use]
    pub const fn supported(rank: i32, rope: i32) -> bool {
        rank == RANK && (rope == 0 || rope == 64)
    }

    fn scale_log2(sm_scale: f32) -> f32 {
        sm_scale * core::f32::consts::LOG2_E
    }

    fn by_value<P>(params: &P) -> ArgValue {
        ArgValue::Bytes {
            ptr: core::ptr::from_ref(params).cast::<u8>(),
            len: core::mem::size_of::<P>(),
        }
    }

    const fn at(base: u64, off: Option<u32>) -> u64 {
        match off {
            Some(off) => base + off as u64,
            None => 0,
        }
    }

    /// `pie::attn::mla::Work`, field for field.
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    struct Work {
        q_nope: u64,
        q_pe: u64,
        ckv: u64,
        kpe: u64,
        kv_indices: u64,
        q_indptr: u64,
        kv_indptr: u64,
        partial_indptr: u64,
        q_len: u64,
        kv_len: u64,
        q_start: u64,
        kv_start: u64,
        kv_end: u64,
        work_indptr: u64,
        o: u64,
        lse: u64,
        partial_o: u64,
        partial_lse: u64,
        num_heads: i32,
        page_size: i32,
        scale_log2: f32,
        causal: i32,
    }

    const _: () = assert!(core::mem::size_of::<Work>() == 160);

    /// `pie::attn::mla::Merge`, field for field.
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    struct Merge {
        start: u64,
        end: u64,
        pstart: u64,
        pend: u64,
        stride: u64,
        partial_o: u64,
        partial_lse: u64,
        o: u64,
        lse: u64,
        num_heads: i32,
    }

    const _: () = assert!(core::mem::size_of::<Merge>() == 80);

    /// `pie::attn::mla::Gather`, field for field.
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub(crate) struct Gather {
        pub q_nope: u64,
        pub q_pe: u64,
        pub ckv: u64,
        pub kpe: u64,
        pub page_indices: u64,
        pub page_indptr: u64,
        pub last_page_lens: u64,
        pub qo_indptr: u64,
        pub positions: u64,
        pub req_of_token: u64,
        pub selection: u64,
        pub o: u64,
        pub lse: u64,
        pub part_o: u64,
        pub part_lse: u64,
        pub win: u64,
        pub top_k: i32,
        pub num_requests: i32,
        pub num_heads: i32,
        pub page_size: i32,
        pub ratio: i32,
        pub splits: i32,
        pub causal: i32,
        pub scale_log2: f32,
    }

    const _: () = assert!(core::mem::size_of::<Gather>() == 160);

    /// The planned walk: the MLA scheduler's work list over the batch, then
    /// the fold of every split row. The fold always runs (its blocks with
    /// nothing to fold return): a captured step replays it whatever split
    /// the next plan stages.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn planned(
        ctx: &Ctx,
        op: &'static str,
        plan: &MlaPlan,
        q_nope: u64,
        q_pe: u64,
        layer: &Layer,
        kv_indices: u64,
        num_heads: i32,
        sm_scale: f32,
        causal: bool,
        o: u64,
    ) -> Result<(), Error> {
        if plan.num_heads != num_heads.unsigned_abs() {
            return Err(refuse(
                op,
                format!(
                    "the plan was cut for {} heads and this attention states {num_heads}",
                    plan.num_heads
                ),
            ));
        }
        let info = &plan.info;
        let int = plan.workspace.int_ptr;
        let float = plan.workspace.float_ptr;
        let work = Work {
            q_nope,
            q_pe,
            ckv: layer.ckv_pages,
            kpe: layer.kpe_pages,
            kv_indices,
            q_indptr: at(int, info.q_indptr_offset),
            kv_indptr: at(int, info.kv_indptr_offset),
            partial_indptr: at(int, info.partial_indptr_offset),
            q_len: at(int, info.q_len_offset),
            kv_len: at(int, info.kv_len_offset),
            q_start: at(int, info.q_start_offset),
            kv_start: at(int, info.kv_start_offset),
            kv_end: at(int, info.kv_end_offset),
            work_indptr: at(int, info.work_indptr_offset),
            o,
            lse: 0,
            partial_o: at(float, info.partial_o_offset),
            partial_lse: at(float, info.partial_lse_offset),
            num_heads,
            page_size: layer.page_size,
            scale_log2: scale_log2(sm_scale),
            causal: i32::from(causal),
        };
        #[allow(clippy::cast_sign_loss)]
        let grid = [info.num_blks_x as u32, info.num_blks_y as u32, 1];
        let entry = symbol(&format!("::pie::attn::mla::mla_paged<{}>", layer.rope_dim));
        ctx.fire(
            op,
            Fire::at(FILE, entry).apply(Launch::grid(grid, [THREADS, 1, 1])),
            &[by_value(&work)],
        )?;
        let merge = Merge {
            start: at(int, info.merge_packed_offset_start_offset),
            end: at(int, info.merge_packed_offset_end_offset),
            pstart: at(int, info.merge_partial_packed_offset_start_offset),
            pend: at(int, info.merge_partial_packed_offset_end_offset),
            stride: at(int, info.merge_partial_stride_offset),
            partial_o: work.partial_o,
            partial_lse: work.partial_lse,
            o,
            lse: 0,
            num_heads,
        };
        ctx.fire(
            op,
            Fire::at(FILE, "::pie::attn::mla::mla_merge")
                .apply(Launch::grid([plan.device.num_sm.max(1), 1, 1], [256, 1, 1])),
            &[by_value(&merge)],
        )
    }

    /// How many splits a gathered walk cuts each row's key list into: enough
    /// blocks to fill the device twice over, no split under two tiles.
    #[must_use]
    pub fn splits(ctx: &Ctx, rows: u32, heads: i32, max_list: i64) -> i32 {
        let cus = u64::from(ctx.multiprocessors().unwrap_or(304).max(1));
        let base = u64::from(rows).max(1) * u64::from(heads.unsigned_abs().div_ceil(BLOCK_ROWS));
        let want = (2 * cus).div_ceil(base.max(1));
        let cap = max_list.max(1).unsigned_abs().div_ceil(MIN_SHARE);
        #[allow(clippy::cast_possible_truncation)]
        let s = want.min(cap).clamp(1, MAX_SPLITS) as i32;
        s
    }

    /// The gathered walk: one block per (64 heads, token, split), then the
    /// fold of the splits when there are several. `g.splits` says how many;
    /// the partial planes come from this fire's scratch.
    pub fn gather(
        ctx: &Ctx,
        op: &'static str,
        rope: i32,
        rows: u32,
        mut g: Gather,
    ) -> Result<(), Error> {
        if rows == 0 {
            return Ok(());
        }
        if g.num_heads <= 0 || g.splits <= 0 {
            return Err(refuse(op, "the walk states no heads or no splits"));
        }
        let heads = g.num_heads.unsigned_abs();
        let splits = g.splits.unsigned_abs();
        if splits > 1 {
            let cells = rows as usize * heads as usize * splits as usize;
            let rank = RANK.unsigned_abs() as usize;
            g.part_o = ctx.scratch(op, "attn.mla_part_o", cells * rank * 4)? as usize as u64;
            g.part_lse = ctx.scratch(op, "attn.mla_part_lse", cells * 4)? as usize as u64;
        }
        if let ArgValue::Ptr(win) = ctx.stage() {
            g.win = win;
        }
        let entry = symbol(&format!("::pie::attn::mla::latent_gather<{rope}>"));
        ctx.fire(
            op,
            Fire::at(FILE, entry).apply(Launch::grid(
                [heads.div_ceil(BLOCK_ROWS), rows, splits],
                [THREADS, 1, 1],
            )),
            &[by_value(&g)],
        )?;
        if splits == 1 {
            return Ok(());
        }
        ctx.fire(
            op,
            Fire::at(FILE, "::pie::attn::mla::latent_merge")
                .apply(Launch::grid([rows, heads, 1], [64, 1, 1])),
            &[
                ArgValue::Ptr(g.part_o),
                ArgValue::Ptr(g.part_lse),
                ArgValue::Ptr(g.o),
                ArgValue::Ptr(g.lse),
                ArgValue::I32(g.num_heads),
                ArgValue::I32(g.splits),
                ctx.stage(),
            ],
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
    o: &mut Tensor,
) -> Result<(), Error> {
    // `kernel.mla.path = naive` takes the reference walk; otherwise the
    // matrix-core walk where it is built.
    if !ctx.tuning().mla_naive && latent::supported(layer.kv_lora_rank, layer.rope_dim) {
        return latent::planned(
            ctx,
            op,
            plan,
            q_nope.ptr,
            q_pe.ptr,
            layer,
            pool.page_indices.ptr,
            num_heads,
            sm_scale,
            causal,
            o.ptr,
        );
    }
    let num_requests = kv::lanes_of(op, qo_indptr)?;
    naive::fire(
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
    o: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.mla_decode";
    dtype_dispatch!(OP, q.data.dtype, { Bf16 => () });
    let heads = count(OP, "the head count this attention states", heads)?;
    let rope = rope_per_head(OP, q_pe, heads)?;
    let layer = Layer::of(OP, pool, kv_lora_rank, rope)?;
    dispatch_dense(
        ctx, OP, plan, q.data, q_pe, &layer, pool, q.indptr, heads, sm_scale, false, o,
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
    o: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.mla_prefill";
    dtype_dispatch!(OP, q.data.dtype, { Bf16 => () });
    let heads = count(OP, "the head count this attention states", heads)?;
    let rope = rope_per_head(OP, q_pe, heads)?;
    let layer = Layer::of(OP, pool, kv_lora_rank, rope)?;
    dispatch_dense(
        ctx, OP, plan, q.data, q_pe, &layer, pool, q.indptr, heads, sm_scale, true, o,
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
    let top_k = stated(op, selection.width)?;
    if !ctx.tuning().mla_naive && latent::supported(layer.kv_lora_rank, layer.rope_dim) {
        return latent::gather(
            ctx,
            op,
            layer.rope_dim,
            o.rows,
            latent::Gather {
                q_nope: q.data.ptr,
                q_pe: q_pe.ptr,
                ckv: layer.ckv_pages,
                kpe: layer.kpe_pages,
                page_indices: pool.page_indices.ptr,
                page_indptr: pool.page_indptr.ptr,
                last_page_lens: pool.last_page_lens.ptr,
                qo_indptr: q.indptr.ptr,
                selection: selection.ptr,
                o: o.ptr,
                top_k,
                num_requests,
                num_heads: heads,
                page_size: layer.page_size,
                ratio: 1,
                splits: latent::splits(ctx, o.rows, heads, i64::from(top_k)),
                causal: i32::from(causal),
                scale_log2: sm_scale * core::f32::consts::LOG2_E,
                ..latent::Gather::default()
            },
        );
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
            top_k,
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
    )
}

#[cfg(test)]
mod tests {
    use super::latent::supported;
    use super::naive::wave_group;

    #[test]
    fn the_matrix_core_walk_takes_the_512_latent_with_or_without_rope() {
        assert!(supported(512, 64));
        assert!(supported(512, 0));
        assert!(!supported(256, 64));
        assert!(!supported(512, 32));
    }

    #[test]
    fn a_head_group_divides_the_heads_and_fits_the_block() {
        assert_eq!(wave_group(64, 1), 1);
        assert_eq!(wave_group(64, 4096), 4);
        assert_eq!(wave_group(6, 4096), 2);
    }
}
