//! The hardware-dependent decisions the CUDA kernels read, and [`Tuning`]:
//! the values one load runs under.

use profile::{Anywhere, Decision, Profile, Registry};

/// Whether routed matmuls over bf16 or f16 banks take the grouped GEMM once
/// an expert's routes pay for it; off, every route reads its expert alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.grouped", flag, default = true, tune = search)]
pub struct MoeGrouped;

/// The same for mxfp4 banks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.mxfp4_grouped", flag, default = true, tune = search)]
pub struct MoeMxfp4Grouped;

/// Whether the grouped mxfp4 kernel multiplies bf16 rows with tensor cores
/// (WMMA) rather than the scalar grouped kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.mxfp4_wmma", flag, default = true, tune = search)]
pub struct MoeMxfp4Wmma;

/// Whether routed matmuls (bf16 banks, and 4-bit ones against bf16
/// activations) run on the grouped tcgen05 GEMM where the device has it.
/// Off but where a chip's profile turns it on (B200's): its plans assume
/// B200's shared memory and SM count.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.tc5", flag, default = false, tune = search)]
pub struct MoeTc5;

/// The fewest tokens a fire routes on the tcgen05 GEMM; narrower fires keep
/// their GEMV or grouped kernel (unless their routes pass the GEMV's grid).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.tc5_from", constant(range(1..=65_536)), default = 16, tune = search)]
pub struct MoeTc5From(pub u32);

/// The most routes a routed mxfp4 select (bf16 activations) runs on the
/// register-dequant MMA kernel with; 0 (never) but where a chip's profile
/// says (B200's).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.mxfp4_mma_until", constant(range(0..=65_536)), default = 0, tune = search)]
pub struct MoeMxfp4MmaUntil(pub u32);

/// The fewest routes a routed mxfp4 select runs on the register-dequant MMA
/// kernel with (up to `kernel.moe.mxfp4_mma_until`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.mxfp4_mma_from", constant(range(1..=65_536)), default = 64, tune = search)]
pub struct MoeMxfp4MmaFrom(pub u32);

/// The fewest rows a dense gate-up projection lands its GLU from the
/// tcgen05 GEMM's epilogue at; fewer keep the skinny kernel or the GEMM and
/// the GLU apart. 65536 (never) but where a chip's profile says (B200's).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.gemm.tc5_glu_from", constant(range(1..=65_536)), default = 65_536, tune = search)]
pub struct GemmTc5GluFrom(pub u32);

/// Whether a small decode batch still splits its keys across blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.decode.split_small", flag, default = false, tune = search)]
pub struct DecodeSplitSmall;

/// Whether a sliding-window decode splits its keys across blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.decode.split_window", flag, default = false, tune = search)]
pub struct DecodeSplitWindow;

/// Whether a split decode chunk holds at least 128 keys, as the scheduler
/// once floored it, rather than one page.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.decode.chunk_floor_128", flag, default = false, tune = search)]
pub struct DecodeChunkFloor128;

/// How a bf16 GEMV (one activation row) is laid out: `wide` (an 8-warp
/// split-K block up to `kernel.gemv.splitk_max_n` columns, 4 rows x 4 warps
/// past it), or `narrow` (4 warps x 2 for both), which Blackwell prefers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.gemv.bf16", default = "wide", tune = search)]
pub enum GemvBf16 {
    Wide,
    Narrow,
}

/// The widest bf16 GEMV that splits its contraction across warps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.gemv.splitk_max_n", constant(range(0..=1_048_576)), default = 4096, tune = search)]
pub struct GemvSplitkMaxN(pub u32);

/// How MLA attention runs: the FA2 kernel at the widest shared-memory arm
/// that fits, or the naive kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.mla.path", default = "fa2", tune = search)]
pub enum MlaPath {
    Fa2,
    Naive,
}

/// The fewest rows a quantised projection takes the tiled GEMM for (and,
/// for a held bank, the bf16 decode into cuBLAS); fewer take the GEMV.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.qmm.gemm_from", constant(range(1..=4096)), default = 16, tune = search)]
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

/// The cuBLASLt workspace, in MiB: more unlocks split-K and stream-K
/// algorithms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.gemm.workspace_mb", constant(set(4, 8, 16, 32, 64)), default = 8, tune = search)]
pub struct GemmWorkspaceMb(pub u32);

/// How many cuBLASLt heuristic candidates a dense GEMM shape is timed over.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.gemm.heuristics", constant(range(1..=32)), default = 8, tune = search)]
pub struct GemmHeuristics(pub u32);

/// The most rows the fused skinny bf16 GEMM (GEGLU, soft-capped heads)
/// serves; past it cuBLAS does. Its tile holds 64; 0 never takes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.gemm.skinny_rows", constant(range(0..=64)), default = 64, tune = search)]
pub struct GemmSkinnyRows(pub u32);

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

/// Whether a 128-wide gated delta prefill runs the two-launch tensor-core
/// scan (its scan block holds about 123 KiB of shared memory) rather than
/// the three wmma launches; a card whose blocks cannot opt into that much
/// keeps the three whatever this says.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.ssm.gdn_fused_scan", flag, default = false, tune = search)]
pub struct SsmGdnFusedScan;

pub fn register(registry: &mut Registry) {
    registry
        .add::<MoeGrouped>()
        .add::<MoeMxfp4Grouped>()
        .add::<MoeMxfp4Wmma>()
        .add::<DecodeSplitSmall>()
        .add::<DecodeSplitWindow>()
        .add::<DecodeChunkFloor128>()
        .add::<GemvBf16>()
        .add::<GemvSplitkMaxN>()
        .add::<MlaPath>()
        .add::<QmmGemmFrom>()
        .add::<QmmLongRows>()
        .add::<QmmSplitWarpsPerSm>()
        .add::<QmmSplitThinMax>()
        .add::<QmmSplitWideMax>()
        .add::<PrefillCtasPerSm>()
        .add::<DecodeStaticMaxRequests>()
        .add::<GemmWorkspaceMb>()
        .add::<GemmHeuristics>()
        .add::<GemmSkinnyRows>()
        .add::<MoeGroupAt>()
        .add::<MoeMxfp4GroupedReuse>()
        .add::<MoeAffineGrouped>()
        .add::<MoeAffineWmma>()
        .add::<MoeAffineGroupedFrom>()
        .add::<MoeTc5>()
        .add::<MoeTc5From>()
        .add::<GemmTc5GluFrom>()
        .add::<SsmGdnFusedScan>()
        .add::<MoeMxfp4MmaUntil>()
        .add::<MoeMxfp4MmaFrom>();
}

/// The kernel decisions one load runs under, read once from its profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tuning {
    pub moe_grouped: bool,
    pub mxfp4_grouped: bool,
    pub mxfp4_wmma: bool,
    pub decode_split_small: bool,
    pub decode_split_window: bool,
    pub decode_chunk_floor_128: bool,
    /// `kernel.gemv.bf16 = narrow`.
    pub gemv_narrow: bool,
    pub gemv_splitk_max_n: u32,
    /// `kernel.mla.path = naive`.
    pub mla_naive: bool,
    pub qmm_gemm_from: u32,
    pub qmm_long_rows: u32,
    pub qmm_split_warps_per_sm: u32,
    pub qmm_split_thin_max: u32,
    pub qmm_split_wide_max: u32,
    pub prefill_ctas_per_sm: u32,
    pub decode_static_max_requests: u32,
    pub gemm_workspace_mb: u32,
    pub gemm_heuristics: u32,
    pub gemm_skinny_rows: u32,
    pub moe_group_at: u32,
    pub mxfp4_grouped_reuse: u32,
    pub affine_grouped: bool,
    pub affine_wmma: bool,
    pub affine_grouped_from: u32,
    pub moe_tc5: bool,
    pub moe_tc5_from: u32,
    pub gemm_tc5_glu_from: u32,
    pub ssm_gdn_fused_scan: bool,
    pub mxfp4_mma_until: u32,
    pub mxfp4_mma_from: u32,
}

impl Tuning {
    /// The registry defaults; a test holds the two equal.
    pub const DEFAULT: Tuning = Tuning {
        moe_grouped: true,
        mxfp4_grouped: true,
        mxfp4_wmma: true,
        decode_split_small: false,
        decode_split_window: false,
        decode_chunk_floor_128: false,
        gemv_narrow: false,
        gemv_splitk_max_n: 4096,
        mla_naive: false,
        qmm_gemm_from: 16,
        qmm_long_rows: 512,
        qmm_split_warps_per_sm: 58,
        qmm_split_thin_max: 32,
        qmm_split_wide_max: 16,
        prefill_ctas_per_sm: 2,
        decode_static_max_requests: 512,
        gemm_workspace_mb: 8,
        gemm_heuristics: 8,
        gemm_skinny_rows: 64,
        moe_group_at: 4,
        mxfp4_grouped_reuse: 2,
        affine_grouped: true,
        affine_wmma: true,
        affine_grouped_from: 16,
        moe_tc5: false,
        moe_tc5_from: 16,
        gemm_tc5_glu_from: 65_536,
        ssm_gdn_fused_scan: false,
        mxfp4_mma_until: 0,
        mxfp4_mma_from: 64,
    };

    #[must_use]
    pub fn of(profile: &Profile) -> Tuning {
        Tuning {
            moe_grouped: profile.on::<MoeGrouped>(&Anywhere),
            mxfp4_grouped: profile.on::<MoeMxfp4Grouped>(&Anywhere),
            mxfp4_wmma: profile.on::<MoeMxfp4Wmma>(&Anywhere),
            decode_split_small: profile.on::<DecodeSplitSmall>(&Anywhere),
            decode_split_window: profile.on::<DecodeSplitWindow>(&Anywhere),
            decode_chunk_floor_128: profile.on::<DecodeChunkFloor128>(&Anywhere),
            gemv_narrow: *profile.get::<GemvBf16>() == GemvBf16::Narrow,
            gemv_splitk_max_n: profile.get::<GemvSplitkMaxN>().0,
            mla_naive: *profile.get::<MlaPath>() == MlaPath::Naive,
            qmm_gemm_from: profile.get::<QmmGemmFrom>().0,
            qmm_long_rows: profile.get::<QmmLongRows>().0,
            qmm_split_warps_per_sm: profile.get::<QmmSplitWarpsPerSm>().0,
            qmm_split_thin_max: profile.get::<QmmSplitThinMax>().0,
            qmm_split_wide_max: profile.get::<QmmSplitWideMax>().0,
            prefill_ctas_per_sm: profile.get::<PrefillCtasPerSm>().0,
            decode_static_max_requests: profile.get::<DecodeStaticMaxRequests>().0,
            gemm_workspace_mb: profile.get::<GemmWorkspaceMb>().0,
            gemm_heuristics: profile.get::<GemmHeuristics>().0,
            gemm_skinny_rows: profile.get::<GemmSkinnyRows>().0,
            moe_group_at: profile.get::<MoeGroupAt>().0,
            mxfp4_grouped_reuse: profile.get::<MoeMxfp4GroupedReuse>().0,
            affine_grouped: profile.on::<MoeAffineGrouped>(&Anywhere),
            affine_wmma: profile.on::<MoeAffineWmma>(&Anywhere),
            affine_grouped_from: profile.get::<MoeAffineGroupedFrom>().0,
            moe_tc5: profile.on::<MoeTc5>(&Anywhere),
            moe_tc5_from: profile.get::<MoeTc5From>().0,
            gemm_tc5_glu_from: profile.get::<GemmTc5GluFrom>().0,
            ssm_gdn_fused_scan: profile.on::<SsmGdnFusedScan>(&Anywhere),
            mxfp4_mma_until: profile.get::<MoeMxfp4MmaUntil>().0,
            mxfp4_mma_from: profile.get::<MoeMxfp4MmaFrom>().0,
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
