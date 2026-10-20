//! The hardware-dependent decisions the HIP kernels read, and [`Tuning`]:
//! the values one load runs under.

use profile::{Anywhere, Decision, Profile, Registry};

/// Whether routed matmuls over bf16 or f16 banks take the grouped GEMM once
/// an expert's routes pay for it; off, every route reads its expert alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.grouped", flag, default = true, tune = search)]
pub struct MoeGrouped;

/// Whether a dense bank's grouped prefill may take hipBLASLt's grouped GEMM
/// kernels (hip/lt_grouped.rs), from the routes an expert below.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.lt_grouped", flag, default = false, tune = search)]
pub struct MoeLtGrouped;

/// Routes an expert from which a routed swiglu MLP keeps its rows by expert
/// from the gate-up to the fold over hipBLASLt's grouped GEMM (0: never).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.lt_mlp_from", constant(range(0..=1_048_576)), default = 0, tune = search)]
pub struct MoeLtMlpFrom(pub u32);

/// The grouped GEMM tile hipBLASLt's leg takes over a contraction of at
/// least `kernel.moe.lt_deep_k`, as MT0·10⁶ + MT1·10³ + depth (256_192_064
/// is MT256x192x64).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.lt_tile_deep", constant(range(0..=4_000_000_000)), default = 256_192_064, tune = search)]
pub struct MoeLtTileDeep(pub u32);

/// The same over a shorter contraction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.lt_tile_shallow", constant(range(0..=4_000_000_000)), default = 256_160_032, tune = search)]
pub struct MoeLtTileShallow(pub u32);

/// The contraction from which hipBLASLt's leg takes the deep tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.lt_deep_k", constant(range(0..=1_048_576)), default = 2048, tune = search)]
pub struct MoeLtDeepK(pub u32);

/// Routes an expert from which a routed swiglu MLP keeps its routes in
/// expert order between the native grouped legs (0: never).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.ordered_from", constant(range(0..=1_048_576)), default = 0, tune = search)]
pub struct MoeOrderedFrom(pub u32);

/// The same for mxfp4 banks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.mxfp4_grouped", flag, default = true, tune = search)]
pub struct MoeMxfp4Grouped;

/// Whether the grouped mxfp4 kernel multiplies bf16 rows with tensor cores
/// (WMMA) rather than the scalar grouped kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.mxfp4_wmma", flag, default = true, tune = search)]
pub struct MoeMxfp4Wmma;

/// The fewest keys a split decode chunk carries (the flag above lowers it
/// to 128 at most).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.decode.chunk_floor_tokens", constant(range(16..=4096)), default = 64, tune = search)]
pub struct DecodeChunkFloorTokens(pub u32);

/// The widest bf16 GEMV that splits its contraction across warps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.gemv.splitk_max_n", constant(range(0..=1_048_576)), default = 4096, tune = search)]
pub struct GemvSplitkMaxN(pub u32);

/// The fewest rows a held affine-quantised bank decodes to bf16 for
/// hipBLASLt, fewer taking the native kernels (on MI300X the decode starts
/// to win near 128 rows: below, the native kernels read a quarter of the
/// bytes), and the fewest a tiled bank takes the tiled GEMM for (past the
/// GEMV's 16 rows it takes it regardless).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.qmm.gemm_from", constant(range(1..=4096)), default = 128, tune = search)]
pub struct QmmGemmFrom(pub u32);

/// The fewest rows the tiled quantised GEMM takes its long tile (64 x 128,
/// 2 stages) for; fewer take the short one (32 x 128, 4 stages).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.qmm.long_rows", constant(range(1..=65_536)), default = 512, tune = search)]
pub struct QmmLongRows(pub u32);

/// Warps a split-K tiled GEMM aims to put on each SM.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.qmm.split_warps_per_sm", constant(range(1..=1024)), default = 58, tune = search)]
pub struct QmmSplitWarpsPerSm(pub u32);

/// The deepest split-K for 8 rows or fewer, and for more.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.qmm.split_thin_max", constant(set(8, 16, 32)), default = 32, tune = search)]
pub struct QmmSplitThinMax(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.qmm.split_wide_max", constant(set(8, 16, 32)), default = 16, tune = search)]
pub struct QmmSplitWideMax(pub u32);

/// Prefill attention CTAs per SM: the grid a split prefill fills, and the
/// floor a recorded prefill pads its tiles to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.prefill.ctas_per_sm", constant(range(1..=8)), default = 2, tune = search)]
pub struct PrefillCtasPerSm(pub u32);

/// The most requests an eager decode serves with the static, unsplit
/// schedule (which skips planning).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.decode.static_max_requests", constant(range(0..=65_536)), default = 512, tune = search)]
pub struct DecodeStaticMaxRequests(pub u32);

/// The most rows the fused skinny bf16 GEMM (GEGLU, soft-capped heads)
/// serves; past it cuBLAS does. Its tile holds 64; 0 never takes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.gemm.skinny_rows", constant(range(0..=64)), default = 64, tune = search)]
pub struct GemmSkinnyRows(pub u32);

/// The fewest activation rows the skinny projection runs on whole products
/// (four rows an MFMA) rather than the diagonal kernel; past 8 never.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.gemm.skinny_quad_from", constant(range(1..=9)), default = 3, tune = search)]
pub struct GemmSkinnyQuadFrom(pub u32);

/// The most rows a dense bf16 projection gives the skinny kernels whatever
/// its weight; past it hipBLASLt or the batch engine take it (a small
/// weight stays skinny to eight rows).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.gemm.skinny_dense_rows", constant(range(0..=8)), default = 4, tune = search)]
pub struct GemmSkinnyDenseRows(pub u32);

/// Past the skinny rows (up to eight), a bank this wide (output columns)
/// takes the batch engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.gemm.skinny_batch_min_n", constant(range(0..=4_194_304)), default = 65536, tune = search)]
pub struct GemmSkinnyBatchMinN(pub u32);

/// Routes an expert must average before bf16/f16 routed matmuls group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.group_at", constant(range(1..=4096)), default = 4, tune = search)]
pub struct MoeGroupAt(pub u32);

/// Routes per expert above which mxfp4 routed matmuls group: the grouped
/// kernels read an expert once per batch, the by-route one once per route
/// at the card's streaming rate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.mxfp4_grouped_reuse", constant(range(0..=64)), default = 2, tune = search)]
pub struct MoeMxfp4GroupedReuse(pub u32);

/// Whether routed matmuls over affine (mlx u4) banks take the grouped
/// kernel once enough routes come, and whether bf16 rows there take tensor
/// cores (WMMA).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.affine_grouped", flag, default = true, tune = search)]
pub struct MoeAffineGrouped;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.affine_wmma", flag, default = true, tune = search)]
pub struct MoeAffineWmma;

/// The fewest routes an affine routed matmul groups at.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.affine_grouped_from", constant(range(1..=65_536)), default = 16, tune = search)]
pub struct MoeAffineGroupedFrom(pub u32);

/// Whether a routed matmul whose experts see few routes each (a decode
/// batch) takes the batch kernels, which read each expert once for all its
/// routes, over the per-route stream and the grouped GEMM.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.batch", flag, default = true, tune = search)]
pub struct MoeBatch;

/// The fewest tokens (routes over the fan-out) a batch fire takes; fewer
/// stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.batch_from", constant(range(1..=65_536)), default = 2, tune = search)]
pub struct MoeBatchFrom(pub u32);

/// The most routes an expert averages (routes over experts, rounded up)
/// that a batch fire takes; more group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.batch_until", constant(range(0..=4096)), default = 32, tune = search)]
pub struct MoeBatchUntil(pub u32);

/// The same for a bf16 / f16 bank, whose 64-route items keep the matrix
/// core ahead of the grouped GEMM's up to ~4 such items an expert.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.dense_batch_until", constant(range(0..=4096)), default = 160, tune = search)]
pub struct MoeDenseBatchUntil(pub u32);

/// The routes an expert averages from which a bf16 / f16 batch fire takes
/// the wide leg (an item an expert's routes, the waves splitting them, the
/// weight read once) over the batch items; 0, never.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.dense_wide_from", constant(range(0..=4096)), default = 26, tune = search)]
pub struct MoeDenseWideFrom(pub u32);

/// The rows from which, and up to which, a bf16 projection (and the fused
/// gate/up SwiGLU) takes the batch engine over hipBLASLt; until 0, never.
/// On MI300X it wins by 1.2x only at a few rows over a wide bank (Qwen3.8's
/// gate/up); hipBLASLt streams the rest at least as fast.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.dense.batch_from", constant(range(0..=4096)), default = 8, tune = search)]
pub struct DenseBatchFrom(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.dense.batch_until", constant(range(0..=4096)), default = 16, tune = search)]
pub struct DenseBatchUntil(pub u32);

/// The weights (n·k, in Mi elements) the batch engine takes, from and up to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.dense.batch_weight_mi", constant(range(0..=65_536)), default = 128, tune = search)]
pub struct DenseBatchWeightMi(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.dense.batch_weight_max_mi", constant(range(0..=65_536)), default = 192, tune = search)]
pub struct DenseBatchWeightMaxMi(pub u32);

/// The rows up to which MLA's query absorb (`q_nope · W_UK`, per head)
/// takes the native kernel (`attn/mla.hip` `mla_absorb`) over rocBLAS's
/// strided batch; 0, never.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.mla.absorb_q_until", constant(range(0..=1_048_576)), default = 0, tune = search)]
pub struct MlaAbsorbQUntil(pub u32);

/// The rows from which, and up to which, MLA's value absorb (`latent ·
/// W_UVᵀ`, per head) takes the native kernel; until 0, never.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.mla.absorb_out_from", constant(range(0..=1_048_576)), default = 0, tune = search)]
pub struct MlaAbsorbOutFrom(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.mla.absorb_out_until", constant(range(0..=1_048_576)), default = 0, tune = search)]
pub struct MlaAbsorbOutUntil(pub u32);

pub fn register(registry: &mut Registry) {
    registry
        .add::<MoeGrouped>()
        .add::<MoeLtGrouped>()
        .add::<MoeLtMlpFrom>()
        .add::<MoeLtTileDeep>()
        .add::<MoeLtTileShallow>()
        .add::<MoeLtDeepK>()
        .add::<MoeOrderedFrom>()
        .add::<MoeMxfp4Grouped>()
        .add::<MoeMxfp4Wmma>()
        .add::<DecodeChunkFloorTokens>()
        .add::<GemvSplitkMaxN>()
        .add::<QmmGemmFrom>()
        .add::<QmmLongRows>()
        .add::<QmmSplitWarpsPerSm>()
        .add::<QmmSplitThinMax>()
        .add::<QmmSplitWideMax>()
        .add::<PrefillCtasPerSm>()
        .add::<DecodeStaticMaxRequests>()
        .add::<GemmSkinnyRows>()
        .add::<GemmSkinnyQuadFrom>()
        .add::<GemmSkinnyDenseRows>()
        .add::<GemmSkinnyBatchMinN>()
        .add::<MoeGroupAt>()
        .add::<MoeMxfp4GroupedReuse>()
        .add::<MoeAffineGrouped>()
        .add::<MoeAffineWmma>()
        .add::<MoeAffineGroupedFrom>()
        .add::<MoeBatch>()
        .add::<MoeBatchFrom>()
        .add::<MoeBatchUntil>()
        .add::<MoeDenseBatchUntil>()
        .add::<MoeDenseWideFrom>()
        .add::<DenseBatchFrom>()
        .add::<DenseBatchUntil>()
        .add::<DenseBatchWeightMi>()
        .add::<DenseBatchWeightMaxMi>()
        .add::<MlaAbsorbQUntil>()
        .add::<MlaAbsorbOutFrom>()
        .add::<MlaAbsorbOutUntil>();
}

/// The kernel decisions one load runs under, read once from its profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tuning {
    pub moe_grouped: bool,
    pub moe_lt_grouped: bool,
    pub moe_lt_mlp_from: u32,
    pub moe_lt_tile_deep: u32,
    pub moe_lt_tile_shallow: u32,
    pub moe_lt_deep_k: u32,
    pub moe_ordered_from: u32,
    pub mxfp4_grouped: bool,
    pub mxfp4_wmma: bool,
    pub decode_chunk_floor_tokens: u32,
    pub gemv_splitk_max_n: u32,
    pub qmm_gemm_from: u32,
    pub qmm_long_rows: u32,
    pub qmm_split_warps_per_sm: u32,
    pub qmm_split_thin_max: u32,
    pub qmm_split_wide_max: u32,
    pub prefill_ctas_per_sm: u32,
    pub decode_static_max_requests: u32,
    pub gemm_skinny_rows: u32,
    pub gemm_skinny_quad_from: u32,
    pub gemm_skinny_dense_rows: u32,
    pub gemm_skinny_batch_min_n: u32,
    pub moe_group_at: u32,
    pub mxfp4_grouped_reuse: u32,
    pub affine_grouped: bool,
    pub affine_wmma: bool,
    pub affine_grouped_from: u32,
    pub moe_batch: bool,
    pub moe_batch_from: u32,
    pub moe_batch_until: u32,
    pub moe_dense_batch_until: u32,
    pub moe_dense_wide_from: u32,
    pub dense_batch_from: u32,
    pub dense_batch_until: u32,
    pub dense_batch_weight_mi: u32,
    pub dense_batch_weight_max_mi: u32,
    pub mla_absorb_q_until: u32,
    pub mla_absorb_out_from: u32,
    pub mla_absorb_out_until: u32,
    /// DeepSeek's fused prefill pass and bounded replay, CUDA's decisions:
    /// never on ROCm.
    pub dsv4_prefill_from: u32,
    pub dsv4_replay: bool,
}

impl Tuning {
    /// The registry defaults; a test holds the two equal.
    pub const DEFAULT: Tuning = Tuning {
        moe_grouped: true,
        moe_lt_grouped: false,
        moe_lt_mlp_from: 0,
        moe_lt_tile_deep: 256_192_064,
        moe_lt_tile_shallow: 256_160_032,
        moe_lt_deep_k: 2048,
        moe_ordered_from: 0,
        mxfp4_grouped: true,
        mxfp4_wmma: true,
        decode_chunk_floor_tokens: 64,
        gemv_splitk_max_n: 4096,
        qmm_gemm_from: 128,
        qmm_long_rows: 512,
        qmm_split_warps_per_sm: 58,
        qmm_split_thin_max: 32,
        qmm_split_wide_max: 16,
        prefill_ctas_per_sm: 2,
        decode_static_max_requests: 512,
        gemm_skinny_rows: 64,
        gemm_skinny_quad_from: 3,
        gemm_skinny_dense_rows: 4,
        gemm_skinny_batch_min_n: 65536,
        moe_group_at: 4,
        mxfp4_grouped_reuse: 2,
        affine_grouped: true,
        affine_wmma: true,
        affine_grouped_from: 16,
        moe_batch: true,
        moe_batch_from: 2,
        moe_batch_until: 32,
        moe_dense_batch_until: 160,
        moe_dense_wide_from: 26,
        dense_batch_from: 8,
        dense_batch_until: 16,
        dense_batch_weight_mi: 128,
        dense_batch_weight_max_mi: 192,
        mla_absorb_q_until: 0,
        mla_absorb_out_from: 0,
        mla_absorb_out_until: 0,
        dsv4_prefill_from: 0,
        dsv4_replay: false,
    };

    #[must_use]
    pub fn of(profile: &Profile) -> Tuning {
        Tuning {
            moe_grouped: profile.on::<MoeGrouped>(&Anywhere),
            moe_lt_grouped: profile.on::<MoeLtGrouped>(&Anywhere),
            moe_lt_mlp_from: profile.get::<MoeLtMlpFrom>().0,
            moe_lt_tile_deep: profile.get::<MoeLtTileDeep>().0,
            moe_lt_tile_shallow: profile.get::<MoeLtTileShallow>().0,
            moe_lt_deep_k: profile.get::<MoeLtDeepK>().0,
            moe_ordered_from: profile.get::<MoeOrderedFrom>().0,
            mxfp4_grouped: profile.on::<MoeMxfp4Grouped>(&Anywhere),
            mxfp4_wmma: profile.on::<MoeMxfp4Wmma>(&Anywhere),
            decode_chunk_floor_tokens: profile.get::<DecodeChunkFloorTokens>().0,
            gemv_splitk_max_n: profile.get::<GemvSplitkMaxN>().0,
            qmm_gemm_from: profile.get::<QmmGemmFrom>().0,
            qmm_long_rows: profile.get::<QmmLongRows>().0,
            qmm_split_warps_per_sm: profile.get::<QmmSplitWarpsPerSm>().0,
            qmm_split_thin_max: profile.get::<QmmSplitThinMax>().0,
            qmm_split_wide_max: profile.get::<QmmSplitWideMax>().0,
            prefill_ctas_per_sm: profile.get::<PrefillCtasPerSm>().0,
            decode_static_max_requests: profile.get::<DecodeStaticMaxRequests>().0,
            gemm_skinny_rows: profile.get::<GemmSkinnyRows>().0,
            gemm_skinny_quad_from: profile.get::<GemmSkinnyQuadFrom>().0,
            gemm_skinny_dense_rows: profile.get::<GemmSkinnyDenseRows>().0,
            gemm_skinny_batch_min_n: profile.get::<GemmSkinnyBatchMinN>().0,
            moe_group_at: profile.get::<MoeGroupAt>().0,
            mxfp4_grouped_reuse: profile.get::<MoeMxfp4GroupedReuse>().0,
            affine_grouped: profile.on::<MoeAffineGrouped>(&Anywhere),
            affine_wmma: profile.on::<MoeAffineWmma>(&Anywhere),
            affine_grouped_from: profile.get::<MoeAffineGroupedFrom>().0,
            moe_batch: profile.on::<MoeBatch>(&Anywhere),
            moe_batch_from: profile.get::<MoeBatchFrom>().0,
            moe_batch_until: profile.get::<MoeBatchUntil>().0,
            moe_dense_batch_until: profile.get::<MoeDenseBatchUntil>().0,
            moe_dense_wide_from: profile.get::<MoeDenseWideFrom>().0,
            dense_batch_from: profile.get::<DenseBatchFrom>().0,
            dense_batch_until: profile.get::<DenseBatchUntil>().0,
            dense_batch_weight_mi: profile.get::<DenseBatchWeightMi>().0,
            dense_batch_weight_max_mi: profile.get::<DenseBatchWeightMaxMi>().0,
            mla_absorb_q_until: profile.get::<MlaAbsorbQUntil>().0,
            mla_absorb_out_from: profile.get::<MlaAbsorbOutFrom>().0,
            mla_absorb_out_until: profile.get::<MlaAbsorbOutUntil>().0,
            dsv4_prefill_from: 0,
            dsv4_replay: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_tuning_is_the_registry_defaults() {
        let mut r = Registry::new("cuda", &[], &["rows"]);
        register(&mut r);
        assert_eq!(Tuning::of(&r.defaults()), Tuning::DEFAULT);
    }
}
