//! The hardware-dependent decisions the HIP kernels read, and [`Tuning`]:
//! the values one load runs under.

use profile::{Decision, Profile, Registry};

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

/// The rows a dense bf16 projection gives the turned kernel (whole 512-byte
/// weight row runs turned through LDS into the matrix core's order), from
/// and up to (at most 16); from 0, never.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.gemm.turn_from", constant(range(0..=16)), default = 0, tune = search)]
pub struct GemmTurnFrom(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.gemm.turn_until", constant(range(0..=16)), default = 16, tune = search)]
pub struct GemmTurnUntil(pub u32);

/// The most rows the turned kernel takes over a bank at least
/// `kernel.gemm.skinny_batch_min_n` wide (an lm head), up to 32.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.gemm.turn_wide_until", constant(range(0..=32)), default = 16, tune = search)]
pub struct GemmTurnWideUntil(pub u32);

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

/// The padded rows from which an lm head over a body's readouts multiplies
/// only the live ones (a device word): hipBLASLt's grouped kernels, M read
/// on the device, one launch a range of live rows (0: never).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.gemm.live_from", constant(range(0..=65_536)), default = 0, tune = search)]
pub struct GemmLiveFrom(pub u32);

/// The grouped tiles (`MT0·10⁶ + MT1·10³ + depth`) of the live lm head's
/// three launches; each takes the live rows up to its MT1 the one before
/// did not (the last, the rest).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.gemm.live_tile_small", constant(range(0..=4_000_000_000)), default = 192_032_128, tune = search)]
pub struct GemmLiveTileSmall(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.gemm.live_tile_mid", constant(range(0..=4_000_000_000)), default = 256_128_064, tune = search)]
pub struct GemmLiveTileMid(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.gemm.live_tile_large", constant(range(0..=4_000_000_000)), default = 176_256_064, tune = search)]
pub struct GemmLiveTileLarge(pub u32);

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

/// The rows up to which a dense `e4m3tile128` projection takes the FP8
/// decode tile (`dense_fp8_mfma`, at most 64), past them the prefill tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.fp8.decode_until", constant(range(1..=64)), default = 64, tune = search)]
pub struct Fp8DecodeUntil(pub u32);

/// The rows up to which those prefill tiles are 64 x 128 (a wave a 16-row
/// strip, three waves a SIMD) rather than 64 or 128 x 256; 0, never.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.fp8.narrow_until", constant(range(0..=1_048_576)), default = 0, tune = search)]
pub struct Fp8NarrowUntil(pub u32);

/// The fewest 16-column state slices (requests x value heads x v_dim / 16)
/// from which the gated-delta chunk scan walks 32-column slices, a chunk's
/// W and k̂ᵀ read for twice the columns; 0, never.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.ssm.gdn_wide_from", constant(range(0..=1_048_576)), default = 0, tune = search)]
pub struct SsmGdnWideFrom(pub u32);

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
        .add::<GemmSkinnyRows>()
        .add::<GemmSkinnyQuadFrom>()
        .add::<GemmSkinnyDenseRows>()
        .add::<GemmSkinnyBatchMinN>()
        .add::<GemmTurnFrom>()
        .add::<GemmTurnUntil>()
        .add::<GemmTurnWideUntil>()
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
        .add::<MlaAbsorbOutUntil>()
        .add::<Fp8DecodeUntil>()
        .add::<Fp8NarrowUntil>()
        .add::<SsmGdnWideFrom>()
        .add::<GemmLiveFrom>()
        .add::<GemmLiveTileSmall>()
        .add::<GemmLiveTileMid>()
        .add::<GemmLiveTileLarge>();
}

/// The kernel decisions one load runs under, read once from its profile.
/// [`Tuning`]'s decision fields, each with its registry default and its
/// read off a profile (`on D` a flag decision, `get D` a constant one's
/// value), as CUDA's: each has an accessor of its name that records the
/// decision as read on the profile it came from, so a decision counts as
/// read only where a dispatch consults it for the loaded model's shapes, and
/// `pie tune` times it only on models it can move.
macro_rules! tuning {
    ($($field:ident: $ty:ty = $default:expr, $how:ident $decision:ident;)*) => {
        /// The kernel decisions one load runs under, resolved once from its
        /// profile; dispatch reads them through the accessors, which record
        /// the read (a field read directly records nothing: for tests and
        /// for code that sizes rather than dispatches).
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct Tuning {
            $(pub $field: $ty,)*
            /// The live-tile GEMM's three tiles (`kernel.gemm.live_tile_*`).
            pub gemm_live_tiles: [u32; 3],
            /// The device's measured rates (the profile's `[rates]`, 0 where
            /// unmeasured): what a dispatch adds inside a recorded body, in ns, and
            /// device memory's copy rate, in GB/s. A route that trades a launch for
            /// redundant reads prices both with them.
            pub launch_ns: u32,
            pub hbm_gbps: u32,
            /// The reads the last-level caches serve, in GB/s (`[rates]
            /// cached_gbps`; 0 unmeasured): a normed skinny product's redundant
            /// reads of its rows, which every block reads, are priced at it.
            pub cached_gbps: u32,
            /// DeepSeek's fused prefill pass and bounded replay, CUDA's decisions:
            /// never on ROCm.
            pub compressed_prefill_from: u32,
            pub compressed_replay: bool,
            /// The profile's read flags, one a decision: `None` for values
            /// built by hand.
            pub reads: Reads,
        }

        #[allow(non_camel_case_types)]
        enum Field { $($field,)* gemm_live_tile_small, gemm_live_tile_mid, gemm_live_tile_large }

        impl Tuning {
            /// The registry defaults; a test holds the two equal.
            pub const DEFAULT: Tuning = Tuning {
                $($field: $default,)*
                gemm_live_tiles: [192_032_128, 256_128_064, 176_256_064],
                launch_ns: 0,
                hbm_gbps: 0,
                cached_gbps: 0,
                compressed_prefill_from: 0,
                compressed_replay: false,
                reads: Reads(None),
            };

            /// `profile`'s values, unread until a dispatch consults one.
            #[must_use]
            pub fn of(profile: &Profile) -> Tuning {
                let marks = vec![
                    $(profile.mark::<$decision>(),)*
                    profile.mark::<GemmLiveTileSmall>(),
                    profile.mark::<GemmLiveTileMid>(),
                    profile.mark::<GemmLiveTileLarge>(),
                ];
                Tuning {
                    $($field: tuning!(@peek profile, $how $decision),)*
                    gemm_live_tiles: [
                        tuning!(@peek profile, get GemmLiveTileSmall),
                        tuning!(@peek profile, get GemmLiveTileMid),
                        tuning!(@peek profile, get GemmLiveTileLarge),
                    ],
                    launch_ns: profile
                        .rates()
                        .launch_ns
                        .map_or(0, |ns| u32::try_from(ns).unwrap_or(u32::MAX)),
                    hbm_gbps: profile.rates().hbm_gbps.map_or(0, |gbps| gbps as u32),
                    cached_gbps: profile.rates().cached_gbps.map_or(0, |gbps| gbps as u32),
                    compressed_prefill_from: 0,
                    compressed_replay: false,
                    reads: Reads(Some(Box::leak(marks.into_boxed_slice()))),
                }
            }

            $(
                #[doc = concat!("`", stringify!($field), "`, recorded as read.")]
                #[must_use]
                #[inline]
                pub fn $field(&self) -> $ty {
                    self.reads.raise(Field::$field as usize);
                    self.$field
                }
            )*

            /// `gemm_live_tiles`, its three decisions recorded as read.
            #[must_use]
            #[inline]
            pub fn gemm_live_tiles(&self) -> [u32; 3] {
                self.reads.raise(Field::gemm_live_tile_small as usize);
                self.reads.raise(Field::gemm_live_tile_mid as usize);
                self.reads.raise(Field::gemm_live_tile_large as usize);
                self.gemm_live_tiles
            }

            /// DeepSeek's fused prefill pass: never on ROCm.
            #[must_use]
            pub const fn compressed_prefill_from(&self) -> u32 {
                self.compressed_prefill_from
            }

            /// DeepSeek's bounded replay: never on ROCm.
            #[must_use]
            pub const fn compressed_replay(&self) -> bool {
                self.compressed_replay
            }
        }
    };
    (@peek $p:ident, on $d:ident) => { $p.peek::<$d>().is_some() };
    (@peek $p:ident, get $d:ident) => { $p.peek::<$d>().expect("a constant is total").0 };
}

tuning! {
    moe_grouped: bool = true, on MoeGrouped;
    moe_lt_grouped: bool = false, on MoeLtGrouped;
    moe_lt_mlp_from: u32 = 0, get MoeLtMlpFrom;
    moe_lt_tile_deep: u32 = 256_192_064, get MoeLtTileDeep;
    moe_lt_tile_shallow: u32 = 256_160_032, get MoeLtTileShallow;
    moe_lt_deep_k: u32 = 2048, get MoeLtDeepK;
    moe_ordered_from: u32 = 0, get MoeOrderedFrom;
    mxfp4_grouped: bool = true, on MoeMxfp4Grouped;
    mxfp4_wmma: bool = true, on MoeMxfp4Wmma;
    decode_chunk_floor_tokens: u32 = 64, get DecodeChunkFloorTokens;
    gemv_splitk_max_n: u32 = 4096, get GemvSplitkMaxN;
    qmm_gemm_from: u32 = 128, get QmmGemmFrom;
    qmm_long_rows: u32 = 512, get QmmLongRows;
    qmm_split_warps_per_sm: u32 = 58, get QmmSplitWarpsPerSm;
    qmm_split_thin_max: u32 = 32, get QmmSplitThinMax;
    qmm_split_wide_max: u32 = 16, get QmmSplitWideMax;
    prefill_ctas_per_sm: u32 = 2, get PrefillCtasPerSm;
    gemm_skinny_rows: u32 = 64, get GemmSkinnyRows;
    gemm_skinny_quad_from: u32 = 3, get GemmSkinnyQuadFrom;
    gemm_skinny_dense_rows: u32 = 4, get GemmSkinnyDenseRows;
    gemm_skinny_batch_min_n: u32 = 65536, get GemmSkinnyBatchMinN;
    gemm_turn_from: u32 = 0, get GemmTurnFrom;
    gemm_turn_until: u32 = 16, get GemmTurnUntil;
    gemm_turn_wide_until: u32 = 16, get GemmTurnWideUntil;
    moe_group_at: u32 = 4, get MoeGroupAt;
    mxfp4_grouped_reuse: u32 = 2, get MoeMxfp4GroupedReuse;
    affine_grouped: bool = true, on MoeAffineGrouped;
    affine_wmma: bool = true, on MoeAffineWmma;
    affine_grouped_from: u32 = 16, get MoeAffineGroupedFrom;
    moe_batch: bool = true, on MoeBatch;
    moe_batch_from: u32 = 2, get MoeBatchFrom;
    moe_batch_until: u32 = 32, get MoeBatchUntil;
    moe_dense_batch_until: u32 = 160, get MoeDenseBatchUntil;
    moe_dense_wide_from: u32 = 26, get MoeDenseWideFrom;
    dense_batch_from: u32 = 8, get DenseBatchFrom;
    dense_batch_until: u32 = 16, get DenseBatchUntil;
    dense_batch_weight_mi: u32 = 128, get DenseBatchWeightMi;
    dense_batch_weight_max_mi: u32 = 192, get DenseBatchWeightMaxMi;
    mla_absorb_q_until: u32 = 0, get MlaAbsorbQUntil;
    mla_absorb_out_from: u32 = 0, get MlaAbsorbOutFrom;
    mla_absorb_out_until: u32 = 0, get MlaAbsorbOutUntil;
    fp8_decode_until: u32 = 64, get Fp8DecodeUntil;
    fp8_narrow_until: u32 = 0, get Fp8NarrowUntil;
    ssm_gdn_wide_from: u32 = 0, get SsmGdnWideFrom;
    gemm_live_from: u32 = 0, get GemmLiveFrom;
}

/// Where a [`Tuning`]'s reads land: its profile's read flags, one a
/// decision, held for the life of the process (a load's tuning is copied
/// onto every stream it opens). Equal whatever it holds: two tunings with
/// the same values are the same tuning.
#[derive(Clone, Copy, Default)]
pub struct Reads(Option<&'static [profile::Mark]>);

impl Reads {
    #[inline]
    fn raise(self, field: usize) {
        if let Some(marks) = self.0 {
            marks[field].raise();
        }
    }
}

impl PartialEq for Reads {
    fn eq(&self, _: &Reads) -> bool {
        true
    }
}

impl Eq for Reads {}

impl core::fmt::Debug for Reads {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("Reads")
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
