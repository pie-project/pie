//! The paged latent attention on mma.sync (`attn/mla/decode_sm80.cu`): the
//! naive arm a block a row and the FlashAttention-2 arm.

#![allow(clippy::too_many_arguments)]

use crate::error::Error;
use crate::jit::{ArgValue, Ctx, Fire, Launch, refuse};

use super::Layer;

const DECODE_SM80: &str = "attn/mla/decode_sm80.cu";

pub(crate) mod naive {
    use super::DECODE_SM80;
    use super::{ArgValue, Ctx, Error, Fire, Launch, refuse};
    use crate::jit::Arg;

    pub const NAIVE_BLOCK: u32 = 256;

    pub const NAIVE_WARPS: i32 = NAIVE_BLOCK as i32 / 32;

    pub const NAIVE_MAX_PER: i32 = 32;

    pub const NAIVE_MAX_PE_PER: i32 = 4;

    pub const WAVE_TARGET: i64 = 296;

    pub const MMA_BM: i32 = 16;

    /// `PIE_MLA_MMA_BK`: the keys an mma tile reads.
    pub const MMA_BK: u32 = 64;

    pub const MMA_SMEM_BYTES: u32 = 100_032;

    #[must_use]
    pub const fn naive_smem_bytes(kv_lora_rank: i32) -> u32 {
        let per = NAIVE_WARPS as i64 * kv_lora_rank as i64 + 2 * NAIVE_WARPS as i64;
        let bytes = per * 4;
        if bytes < 0 { 0 } else { bytes as u32 }
    }

    #[must_use]
    pub const fn mma_supported(kv_lora_rank: i32, rope_dim: i32, num_heads: i32) -> bool {
        // A 64-wide rotated part, or none (NoPE: GLM-5.3-Flash's DSA latents).
        kv_lora_rank == 512 && (rope_dim == 64 || rope_dim == 0) && num_heads % MMA_BM == 0
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
        /// The pool's rows are fp8 (`fp8_rows`): the tensor-core kernel
        /// widens them as it stages them, and no other kernel here reads them.
        pub fp8: bool,
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

        // A selection reads on the tensor cores as a dense window does: a
        // tile is the selection's positions rather than the window's.
        if shape.window == 0
            && (!selected || shape.top_k > 0)
            && mma_supported(shape.kv_lora_rank, shape.rope_dim, shape.num_heads)
        {
            #[allow(clippy::cast_sign_loss)]
            let (blocks, rows, heads) = (
                (shape.num_heads / MMA_BM).max(0) as u32,
                shape.total_tokens.max(0) as u32,
                shape.num_heads.max(0) as u32,
            );
            // Few rows (a decode) deal each row's key tiles over splits, so
            // the multiprocessors fill: as many as fit them, at most 32 (the
            // fold's lanes) and at most a split a tile of a selection.
            let sms = ctx.multiprocessors().unwrap_or(132);
            let mut splits = (sms / (blocks * rows).max(1)).clamp(1, 32);
            if selected {
                #[allow(clippy::cast_sign_loss)]
                let tiles = (shape.top_k.max(1) as u32).div_ceil(MMA_BK);
                splits = splits.min(tiles).max(1);
            }
            let (part_o, part_lse) = if splits > 1 {
                let units = rows as usize * splits as usize * heads as usize;
                (
                    ctx.scratch_shared(op, "attn.mla_mma.part_o", units * 512 * 2)? as u64,
                    ctx.scratch_shared(op, "attn.mla_mma.part_lse", units * 4)? as u64,
                )
            } else {
                (0, 0)
            };
            let launch =
                Launch::grid([blocks, rows, splits], [MMA_THREADS, 1, 1]).smem(MMA_SMEM_BYTES);
            ctx.fire(
                op,
                Fire::at(
                    DECODE_SM80,
                    crate::jit::symbol(&format!(
                        "::pie::attn::mla::mla_naive::mma_detail::mla_mma_paged_kernel<{}{}>",
                        shape.rope_dim,
                        if shape.fp8 { ", true" } else { "" }
                    )),
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
                    ArgValue::Ptr(part_o),
                    ArgValue::Ptr(part_lse),
                    ArgValue::Ptr(ptrs.selection),
                    shape.top_k.arg(),
                    i32::from(ctx.tuning().mla_dense_short).arg(),
                    shape.num_requests.arg(),
                    shape.num_heads.arg(),
                    shape.page_size.arg(),
                    shape.sm_scale.arg(),
                    shape.causal.arg(),
                    ctx.stage(),
                ],
            )?;
            if splits == 1 {
                return Ok(());
            }
            return ctx.fire(
                op,
                Fire::at(
                    DECODE_SM80,
                    "::pie::attn::mla::mla_naive::mma_detail::mla_mma_fold",
                )
                .apply(Launch::grid([rows, heads, 1], [128, 1, 1])),
                &[
                    ArgValue::Ptr(part_o),
                    ArgValue::Ptr(part_lse),
                    ArgValue::Ptr(ptrs.o),
                    ArgValue::Ptr(ptrs.lse),
                    #[allow(clippy::cast_possible_wrap)]
                    (splits as i32).arg(),
                    shape.num_heads.arg(),
                    ctx.stage(),
                ],
            );
        }

        if shape.fp8 {
            return Err(refuse(
                op,
                "fp8 latent rows are read by the tensor-core kernels alone (a 512-wide \
                 latent, a 64-wide rotated part or none, heads in whole MMA tiles, no window)",
            ));
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
                DECODE_SM80,
                if ckv / 32 > 16 {
                    "::pie::attn::mla::mla_naive::mla_naive_paged_kernel<32>"
                } else {
                    "::pie::attn::mla::mla_naive::mla_naive_paged_kernel<16>"
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

pub(crate) mod fa2 {
    use super::DECODE_SM80;
    use super::{Ctx, Error, Layer, refuse};
    use crate::attn::fa2::abi::{UintFastdiv, resolve};
    use crate::attn::plan::{MlaPlan, MlaPlanInfo};
    use crate::jit::{Fire, Launch};

    pub const INST: [[&str; 2]; 3] = [
        [
            "::pie::attn::fa2::mla::BatchMLAPagedAttentionKernel\
             <::pie::attn::mla::mla_fa2::Traits<false, 2u, true, 64u>, \
             ::pie::attn::mla::mla_fa2::Params>",
            "::pie::attn::fa2::mla::BatchMLAPagedAttentionKernel\
             <::pie::attn::mla::mla_fa2::Traits<true, 2u, true, 64u>, \
             ::pie::attn::mla::mla_fa2::Params>",
        ],
        [
            "::pie::attn::fa2::mla::BatchMLAPagedAttentionKernel\
             <::pie::attn::mla::mla_fa2::Traits<false, 2u, true, 32u>, \
             ::pie::attn::mla::mla_fa2::Params>",
            "::pie::attn::fa2::mla::BatchMLAPagedAttentionKernel\
             <::pie::attn::mla::mla_fa2::Traits<true, 2u, true, 32u>, \
             ::pie::attn::mla::mla_fa2::Params>",
        ],
        [
            "::pie::attn::fa2::mla::BatchMLAPagedAttentionKernel\
             <::pie::attn::mla::mla_fa2::Traits<false, 1u, false, 16u>, \
             ::pie::attn::mla::mla_fa2::Params>",
            "::pie::attn::fa2::mla::BatchMLAPagedAttentionKernel\
             <::pie::attn::mla::mla_fa2::Traits<true, 1u, false, 16u>, \
             ::pie::attn::mla::mla_fa2::Params>",
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
        "MlaParams: sizeof disagrees with the measured ::pie::attn::fa2::MLAParams",
    );
    const _: () = assert!(
        core::mem::align_of::<MlaParams>() == 8,
        "MlaParams: alignof disagrees with the measured ::pie::attn::fa2::MLAParams",
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
            Fire::at(DECODE_SM80, row[usize::from(causal)]).apply(launch),
            &[crate::jit::ArgValue::of(params)],
        )
    }
}
