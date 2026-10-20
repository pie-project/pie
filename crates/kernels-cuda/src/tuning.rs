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
#[decision(id = "kernel.moe.mxfp4_mma_until", constant(range(0..=16_777_216)), default = 0, tune = search)]
pub struct MoeMxfp4MmaUntil(pub u32);

/// The routes a routed NVFP4 select of bf16 activations (W4A16) runs on the
/// register-dequant MMA kernel from and up to (each expert's rows
/// dequantized in registers on mma.sync); never (until 0) but where a
/// chip's profile says (B200's).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.nvfp4_mma_until", constant(range(0..=65_536)), default = 0, tune = search)]
pub struct MoeNvfp4MmaUntil(pub u32);

/// The fewest routes of [`MoeNvfp4MmaUntil`]'s span.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.nvfp4_mma_from", constant(range(1..=65_536)), default = 1, tune = search)]
pub struct MoeNvfp4MmaFrom(pub u32);

/// The most output rows (a launch's tokens) a split tcgen05 GEMM sums in the
/// tile's last split to land (a per-tile count, the same bits as the
/// finalize launch it replaces); wider launches keep the finalize, whose
/// many blocks share the sum one split's CTA would walk alone. 0 never.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.gemm.tc5_split_tickets", constant(range(0..=65536)), default = 0, tune = search)]
pub struct GemmTc5SplitTickets(pub u32);

/// The most rows (2..=8) a dense bf16 product may time the few-rows kernel
/// for, one launch streaming the weight's rows, beside cuBLAS's algorithms
/// (a decode's narrow projections run split along K in two launches there).
/// 0 never tries it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.gemm.rows_m", constant(range(0..=8)), default = 0, tune = search)]
pub struct GemmRowsM(pub u32);

/// Programmatic dependent launch: a kernel that opts in (it waits on its
/// predecessor in-kernel before reading what that one wrote) launches while
/// the stream's previous kernel finishes, its launch latency hidden. Off but
/// where a chip's profile turns it on (sm_90 and later only).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.pdl", flag, default = false, tune = search)]
pub struct KernelPdl;

/// The P tiles a dense tcgen05 GEMM's raster band walks Q together over:
/// a wave shares both operands' tiles in L2. 8 but where a chip's profile
/// says.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.gemm.tc5_raster", constant(range(1..=64)), default = 8, tune = search)]
pub struct GemmTc5Raster(pub u32);

/// What a K-splitting cuBLASLt algorithm's own reduce launch costs a
/// captured step past its lone timing, in ns, charged to it (and to
/// `cublasGemmEx`, which may split) when a dense bf16 product's tactic is
/// timed; split algorithms reduced in place (one launch) join the timing.
/// 0 (charge nothing) but where a chip's profile says (B200's).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.gemm.split_launch_ns", constant(range(0..=100_000)), default = 0, tune = search)]
pub struct GemmSplitLaunchNs(pub u32);

/// The fewest rows a dense bf16 product charges a K-splitting algorithm
/// its reduce launch at (`kernel.gemm.split_launch_ns`); fewer are timed as
/// they run alone. 0 charges every product.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.gemm.split_launch_from", constant(range(0..=65536)), default = 0, tune = search)]
pub struct GemmSplitLaunchFrom(pub u32);

/// The fewest tokens a W4A4 routed MLP keeps its rows by expert from the
/// gate-up select to the weighted sum with (its GLU landing the down leg's
/// NVFP4 operand from the gate-up epilogue); never (65 536 past any fire)
/// but where a chip's profile says (B200's).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.w4a4_fused_from", constant(range(1..=65_536)), default = 65_536, tune = search)]
pub struct MoeW4a4FusedFrom(pub u32);

/// The fewest rows the fused qkv norm, rotation and KV write takes a block a
/// row (a warp a head below): 64 but where a chip's profile says (B200's,
/// where 64 decode rows' blocks leave most SMs idle).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.attn.qkv_rows_from", constant(range(1..=65_536)), default = 64, tune = search)]
pub struct AttnQkvRowsFrom(pub u32);

/// Whether a narrow W4A4 routed MLP lays its routes out and quantizes their
/// rows in one launch. Off but where a chip's profile turns it on (B200's).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.w4a4_fused_layout", flag, default = false, tune = search)]
pub struct MoeW4a4FusedLayout;

/// Whether a KDA prefill walks its chunks on the tensor cores (a chunk's
/// update directions in parallel, then the state chunk by chunk) instead of
/// a row at a time. Off but where a chip's profile turns it on (B200's): it
/// asks a block for 219 KiB of shared memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.ssm.kda_chunked", flag, default = false, tune = search)]
pub struct SsmKdaChunked;

/// Whether a KDA decode step reads its packed rows itself (a block a row,
/// head and slice of the state's value rows normalizing its query and key
/// and forming its gates) instead of staging them in two launches first.
/// Off but where a chip's profile turns it on (B200's).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.ssm.kda_step_fused", flag, default = false, tune = search)]
pub struct SsmKdaStepFused;

/// The fewest tokens a W4A8 routed MLP (MXFP4 banks against e4m3
/// activations) keeps its rows by expert from the gate-up select to the
/// weighted sum with (its GLU landing the down leg's e4m3 operand from the
/// gate-up epilogue); never (65 536 past any fire) but where a chip's
/// profile says (B200's).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.w4a8_fused_from", constant(range(1..=65_536)), default = 65_536, tune = search)]
pub struct MoeW4a8FusedFrom(pub u32);

/// The prefill rows from which a latent attention's prefill runs expanded
/// (its latent rows through `kv_b` into a key and value head a query head,
/// then a dense prefill at those widths, as vLLM's MLA prefill runs) in
/// place of its absorbed scoring; never (65 536 past any fire) but where a
/// chip's profile says (B200's). Below it the expansion of every cached row
/// costs more than the absorbed scoring saves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.mla.expanded_from", constant(range(1..=65_536)), default = 65_536, tune = search)]
pub struct MlaExpandedFrom(pub u32);

/// The rows up to which a 128x128-block FP8 projection runs as a GEMV over
/// bf16 activations (`fp8_block_gemv`, the codes streamed once) rather than
/// quantizing them for the tcgen05 GEMM; none (0) but where a chip's profile
/// says (B200's).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.gemm.fp8_gemv_until", constant(range(0..=64)), default = 0, tune = search)]
pub struct GemmFp8GemvUntil(pub u32);

/// Whether a block-FP8 dense projection read at bf16 activations (no
/// tcgen05) keeps a bf16 copy widened once at load beside its codes: a
/// batch's rows run cuBLAS on it, a decode's few the codes (half the bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.gemm.fp8_shadow", flag, default = false, tune = search)]
pub struct GemmFp8Shadow;

/// The most rows a dense fp8b128 projection runs on the mma.sync fp8
/// kernel (`moe::dense_fp8_shape` says where it leads, in which CTA
/// shape), its rows quantized in the same launch. 0: never.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.gemm.fp8_mma_gemv_until", constant(range(0..=8)), default = 0, tune = search)]
pub struct GemmFp8MmaGemvUntil(pub u32);

/// Whether AttnRes's blend and the RMS norm of its row run as one pass
/// (`res_blend_norm`, 512 threads a row, the row in registers). Off but
/// where a chip's profile turns it on (B200's).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.attn_res.fused_norm", flag, default = false, tune = search)]
pub struct AttnResFusedNorm;

/// Whether a dense projection's bias lands from the cuBLASLt epilogue of the
/// algorithm its shape chose, instead of a separate add. Off but where a
/// chip's profile turns it on (B200's).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.gemm.bias_epilogue", flag, default = false, tune = search)]
pub struct GemmBiasEpilogue;

/// The fewest routes a routed mxfp4 select runs on the register-dequant MMA
/// kernel with (up to `kernel.moe.mxfp4_mma_until`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.mxfp4_mma_from", constant(range(1..=65_536)), default = 64, tune = search)]
pub struct MoeMxfp4MmaFrom(pub u32);

/// The fewest routes a routed bf16 select runs on the graph-safe grouped
/// mma.sync kernel with (`moe_select_bf16_mma`), rather than the per-route
/// GEMV. Bf16 experts have no tcgen05 or cuBLAS grouped path under a
/// captured graph, so this is their only fast route. 0 never.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.bf16_mma_from", constant(range(0..=65_536)), default = 0, tune = search)]
pub struct MoeBf16MmaFrom(pub u32);

/// The most routes the bf16 grouped mma.sync kernel takes (0 never).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.bf16_mma_until", constant(range(0..=65_536)), default = 0, tune = search)]
pub struct MoeBf16MmaUntil(pub u32);

/// Whether the register-dequant mxfp4 select turns its codes into f16 with
/// the e2m1 converts of the arch-specific Blackwells (`linear/moe_w4_cvt.cuh`)
/// rather than bf16 by integer ops. Off but where a chip's profile says.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.mxfp4_cvt", flag, default = false, tune = search)]
pub struct MoeMxfp4Cvt;

/// The most routes an e2m1-convert select (`kernel.moe.mxfp4_cvt`) splits
/// each work item's K across a cluster of two CTAs for, their sums met in
/// the cluster's shared memory: a fire that few leaves most SMs idle. 0:
/// never.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.mxfp4_cvt_ksplit_until", constant(range(0..=65_536)), default = 0, tune = search)]
pub struct MoeMxfp4CvtKsplitUntil(pub u32);

/// A grouped bf16 projection on routes that send each group to its own
/// plane (DeepSeek-V4's `wo_a`, dequantized at import from the checkpoint's
/// fp8b128) requantized at load to fp8b128, each value exactly the
/// checkpoint's, and run as vLLM runs it: W8A8 against the rows in e4m3
/// 1x128 blocks (a decode's few rows on mma.sync, wider fires on the
/// grouped tcgen05 GEMM). Off: the bf16 plane on cuBLAS.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.gemm.fp8_o_down", flag, default = false, tune = search)]
pub struct GemmFp8ODown;

/// The most routes a W4A8 select (an MXFP4 bank against e4m3
/// activations) reads its activations back to bf16 for and runs on the
/// e2m1-convert register-dequant kernel (`kernel.moe.mxfp4_cvt`), from
/// `kernel.moe.mxfp4_a8_from`; wider ones, and all where 0, keep the
/// grouped tcgen05 GEMM.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.mxfp4_a8_until", constant(range(0..=65_536)), default = 0, tune = search)]
pub struct MoeMxfp4A8Until(pub u32);

/// The fewest routes a W4A8 select runs on the e2m1-convert
/// register-dequant kernel (up to `kernel.moe.mxfp4_a8_until`); fewer keep
/// the grouped tcgen05 GEMM. Apart from the bf16-activation selects'
/// `kernel.moe.mxfp4_mma_from`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.mxfp4_a8_from", constant(range(0..=65_536)), default = 0, tune = search)]
pub struct MoeMxfp4A8From(pub u32);

/// The fewest rows a projection cut into two row halves (Qwen3.5's qkv|z)
/// lands as two GEMMs off the weight's two slices, where
/// `kernel.elemwise.rows_vec` is on; fewer run the GEMM and the cut.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.gemm.split_rows_from", constant(range(1..=65_536)), default = 256, tune = search)]
pub struct GemmSplitRowsFrom(pub u32);

/// The most rows Qwen3.5's attention prologue (the q/gate cut, the two
/// per-head norms, the partial rotation and the cache write) lands in one
/// launch at; more run the five launches. 0 (never) but where a chip's
/// profile says.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.elemwise.qg_prologue_until", constant(range(0..=65_536)), default = 0, tune = search)]
pub struct QgPrologueUntil(pub u32);

/// The most rows DeepSeek-V4's attention (the window, the compressed rows,
/// their fold and the sinks) lands in one launch at; more run the four
/// launches. 0 (never) but where a chip's profile says.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.dsv4_decode_until", constant(range(0..=65_536)), default = 0, tune = search)]
pub struct Dsv4DecodeUntil(pub u32);

/// Whether a selected (sparse) attention runs on the tensor cores, a block a
/// row and kv head, rather than the scalar walk. Off reads every key the
/// scalar way.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.selected_mma", flag, default = false, tune = search)]
pub struct SdpaSelectedMma;

/// Whether a compressed-pool attention (`attention.pool_lse`, and its
/// selected form) runs on the tensor cores, a block a row's 16 heads, rather
/// than the scalar walk a block a row and head. Off reads the pool the
/// scalar way.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.pool_mma", flag, default = false, tune = search)]
pub struct SdpaPoolMma;

/// The fewest rows DeepSeek-V4's attention (`attention.pool_attention`: the
/// window, the pool, their merge and the sinks) runs as one tcgen05 pass,
/// a block a row (`attn/dsv4_sparse.cuh`, sm_100a: Q in TMEM, whole 64-key
/// tiles, V the tile K brought) rather than its four ops. 0 never does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.dsv4_prefill_from", constant(range(0..=1_048_576)), default = 0, tune = search)]
pub struct SdpaDsv4PrefillFrom(pub u32);

/// Whether DeepSeek-V4.1's Causal Encoder-Decoder prefills by bounded
/// replay, as its release describes: a long prompt's leading rows run the
/// encoder alone (and the decoder's sources, its pool and index keys), the
/// decoder replaying only the prompt's last window, its windows held above
/// each sequence's replay floor (`attention.pool_attention` always on the
/// fused kernels, which read the floor). Off runs every row through every
/// layer, as vLLM does. It changes what the model computes, so no search
/// turns it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.dsv4_replay", flag, default = false, tune = never)]
pub struct SdpaDsv4Replay;

/// The fewest rows DeepSeek-V4.1's attention over its FP4 cache (an MXFP8
/// window over NVFP4 entries) takes the tcgen05 pass a block a row, which
/// widens each row's keys once for all 64 heads as its tiles land, rather
/// than the decode launch, whose four blocks a row widen them each. 0 never
/// does below `kernel.sdpa.dsv4_prefill_from`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.dsv4_fp4_pass_from", constant(range(0..=1_048_576)), default = 0, tune = search)]
pub struct SdpaDsv4Fp4PassFrom(pub u32);

/// The most rows a grouped projection on an fp8 plane (`kernel.gemm.fp8_o_down`)
/// runs group by group on the tcgen05 GEMM against the plane as it is, past
/// the mma.sync kernel's eight; more widen the plane to bf16 for cuBLAS's
/// strided batch. 0 always widens.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.gemm.fp8_grouped_tc5_until", constant(range(0..=1_048_576)), default = 0, tune = search)]
pub struct GemmFp8GroupedTc5Until(pub u32);

/// Whether a sparse MLA's selected attention (64 absorbed heads over a
/// 512-wide bf16 latent and a 64-wide rotated part or none: GLM-5.3's and
/// GLM-5.3-Flash's DSA) runs a block a row on tcgen05
/// (`attn/mla_sparse.cuh`, sm_100a); off, the warp-a-key form serves it as
/// before.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.mla_sparse", flag, default = false, tune = search)]
pub struct SdpaMlaSparse;

/// Whether the tcgen05 sparse MLA pass reads a row that sees no more keys
/// than its selection holds (a decode at a short context, a prefill's rows
/// before the top-k) as dense: every key it sees in order, its selection
/// unread, as its index (which ranks all of them) selected. Off reads every
/// row's selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.mla_dense_short", flag, default = false, tune = search)]
pub struct SdpaMlaDenseShort;

/// Whether a pooled index (`attention.index_topk`) scores its blocks on the
/// tensor cores rather than a thread a block. Off scores the scalar way.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.index_mma", flag, default = false, tune = search)]
pub struct SdpaIndexMma;

/// Whether a prefill's pooled index (512 rows and up, 64 heads of 128)
/// scores on tcgen05 (`attn/index_tc5.cuh`: two rows' heads an MMA's 128
/// lanes against tiles of 128 blocks), then ranks in a launch of its own;
/// off, the mma.sync scorer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.index_tc5", flag, default = false, tune = search)]
pub struct SdpaIndexTc5;

/// Whether a prefill's hyper-connection pre-mix norm lands its row's e4m3
/// 1x128 operand beside it, for the fp8 GEMMs that read it next, in place of
/// their quantize launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.elemwise.norm_emit", flag, default = false, tune = search)]
pub struct ElemwiseNormEmit;

/// Whether a pooled index (`attention.index_topk`) finds its best blocks by
/// a radix select (four histogram passes, then a block scan) rather than
/// forty bisection passes and a one-thread walk. Off bisects.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.index_radix", flag, default = false, tune = search)]
pub struct SdpaIndexRadix;

/// The widest dense bf16 product (in columns) the GEMM tuner may also time
/// on the narrow kernel (a block a row, every column's sum in registers);
/// 0 never offers it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.gemm.narrow_n", constant(range(0..=8)), default = 0, tune = search)]
pub struct GemmNarrowN(pub u32);

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

/// Whether a 128-wide gated delta prefill runs the one-launch tcgen05 walk
/// (`attn/gdn_tc5.cuh`, sm_100a), the state in TMEM; off, the scan above
/// serves it as before.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.ssm.gdn_tc5", flag, default = false, tune = search)]
pub struct SsmGdnTc5;

/// Whether the row kernels a prefill streams run their bandwidth-shaped
/// forms: a 128- or 256-wide RMS norm a warp a row (per head or whole), a
/// scale-free per-head norm of whole 256-element runs a warp a head, the
/// q/gate cut and sigmoid gate 16 bytes a lane, the rotation of a head's
/// tail four pairs a thread, and DeepSeek-V4's pool gather in one pass over
/// 16-byte runs; off, the block-a-row and element-a-thread forms serve them
/// as before.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.elemwise.rows_vec", flag, default = false, tune = search)]
pub struct ElemwiseRowsVec;

/// Whether the hyper-connection kernels run their tiled and 16-byte forms:
/// the mix projection a block 64 rows of every column over shared-memory
/// tiles, the fold and the gates' pre-mix eight columns a thread, and a
/// scale-free per-head RMS norm a warp a head; off, the forms before them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.elemwise.hc_tile", flag, default = false, tune = search)]
pub struct ElemwiseHcTile;

/// Whether a prefill's hyper-connection mix projection (the stream norm and
/// the projection in one pass) runs on the tensor cores, the f32 weights as
/// two bf16 halves; off, the f32 tile form.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.elemwise.hc_mma", flag, default = false, tune = search)]
pub struct ElemwiseHcMma;

/// Whether the chunked causal conv's 16-byte row-tiled form runs four
/// channels a thread under a 64-register bound (the same sums in the same
/// order); off, eight a thread as before.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.ssm.conv_vec4", flag, default = false, tune = search)]
pub struct SsmConvVec4;

/// Whether a gated delta walk (chunked or a decode step) whose one reader is
/// its gated norm lands its output in bf16 (and the norm reads it so); off,
/// f32, as before.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.ssm.gdn_out_bf16", flag, default = false, tune = search)]
pub struct SsmGdnOutBf16;

/// The most blocks an SM seats of the fused gated delta decode step: 4 but
/// where a chip's profile says (A100's 3: its persistent warps stream the
/// state 7% faster).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.ssm.gdn_step_blocks", constant(set(1, 2, 3, 4)), default = 4, tune = search)]
pub struct SsmGdnStepBlocks(pub u32);

/// Whether a prefill the tcgen05 FMHA (`attn/tc5_fmha.cuh`, sm_100a) can
/// serve runs on it; off, FA2 serves every prefill, as before it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.prefill.tc5", flag, default = false, tune = search)]
pub struct PrefillTc5;

/// Whether a sliding 256-wide prefill (Gemma-4's) on the tcgen05 FMHA runs
/// the 128-key-tile kernel (`tc5_fmha_prefill_wide`) rather than the 64-key
/// one; a 256-wide prefill without a window runs it either way.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.prefill.wide_sliding", flag, default = false, tune = search)]
pub struct PrefillWideSliding;

/// The fewest query heads to a KV head at which a single-token row reads
/// the cache through the prefill kernel (tensor cores over the group's
/// rows) instead of the decode kernel; 0 never does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.decode.grouped_prefill_from", constant(range(0..=64)), default = 0, tune = search)]
pub struct DecodeGroupedPrefillFrom(pub u32);

/// Whether a wide group's single-token rows that also land their
/// log-sum-exp (gpt-oss's, for its sinks) take the prefill kernel too,
/// where their plan has no sliding window.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.decode.grouped_prefill_lse", flag, default = false, tune = search)]
pub struct DecodeGroupedPrefillLse;

/// Whether those log-sum-exp rows take the prefill kernel under a sliding
/// window too (MiMo's sink layers: a wide group the decode kernel walks on
/// CUDA cores).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.decode.grouped_prefill_window", flag, default = false, tune = search)]
pub struct DecodeGroupedPrefillWindow;

/// Whether the FA2 prefill kernel (a decode's grouped rows too) reads a
/// pool's live widths out of its wider heads (MiMo's 192-wide keys and
/// 128-wide values stored 256 wide), not the padding. Off but where a chip's
/// profile turns it on (A100's).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.fa2_live", flag, default = false, tune = search)]
pub struct Fa2Live;

/// Whether a paged decode of a group of at most eight query heads reads its
/// cache on tensor cores (`attn/xqa.cuh`, the group's rows one MMA tile)
/// rather than the FA2 decode kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.decode.xqa", flag, default = false, tune = search)]
pub struct DecodeXqa;

/// The fewest requests a 64-wide decode takes the tensor-core kernel at;
/// narrower, the FA2 decode reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.decode.xqa_from_64", constant(range(0..=65_536)), default = 0, tune = search)]
pub struct DecodeXqaFrom64(pub u32);

/// The widest sliding window (keys a query row sees) the tensor-core decode
/// reads whole, a block a (request, kv head) landing its rows and sinks
/// itself, with no split to fold; 0 splits every window as the plan sizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.decode.xqa_whole_window", constant(range(0..=65_536)), default = 0, tune = search)]
pub struct DecodeXqaWholeWindow(pub u32);

/// The most blocks a cluster splits an unwindowed tensor-core decode row's
/// keys across, folding them through distributed shared memory in place of
/// the plan's split and its fold launch; 0 leaves the split to the plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.decode.xqa_cluster", constant(set(0, 1, 2, 4, 8)), default = 0, tune = search)]
pub struct DecodeXqaCluster(pub u32);

/// Whether a split attention's partial states fold a warp a (row, head)
/// rather than through the staged persistent merge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.merge_warp", flag, default = false, tune = search)]
pub struct SdpaMergeWarp;

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
        .add::<GemmSplitRowsFrom>()
        .add::<QgPrologueUntil>()
        .add::<Dsv4DecodeUntil>()
        .add::<SsmGdnFusedScan>()
        .add::<SsmGdnTc5>()
        .add::<ElemwiseRowsVec>()
        .add::<ElemwiseHcTile>()
        .add::<ElemwiseHcMma>()
        .add::<SsmConvVec4>()
        .add::<SsmGdnOutBf16>()
        .add::<SsmGdnStepBlocks>()
        .add::<SdpaMergeWarp>()
        .add::<MoeMxfp4MmaUntil>()
        .add::<MoeMxfp4MmaFrom>()
        .add::<MoeBf16MmaFrom>()
        .add::<MoeBf16MmaUntil>()
        .add::<MoeMxfp4Cvt>()
        .add::<MoeMxfp4CvtKsplitUntil>()
        .add::<GemmFp8ODown>()
        .add::<MoeMxfp4A8Until>()
        .add::<MoeMxfp4A8From>()
        .add::<GemmBiasEpilogue>()
        .add::<MoeW4a4FusedFrom>()
        .add::<MoeW4a8FusedFrom>()
        .add::<MlaExpandedFrom>()
        .add::<GemmFp8GemvUntil>()
        .add::<GemmFp8Shadow>()
        .add::<GemmFp8MmaGemvUntil>()
        .add::<AttnResFusedNorm>()
        .add::<SsmKdaChunked>()
        .add::<MoeW4a4FusedLayout>()
        .add::<AttnQkvRowsFrom>()
        .add::<SsmKdaStepFused>()
        .add::<GemmSplitLaunchNs>()
        .add::<GemmSplitLaunchFrom>()
        .add::<GemmTc5Raster>()
        .add::<KernelPdl>()
        .add::<GemmRowsM>()
        .add::<GemmTc5SplitTickets>()
        .add::<MoeNvfp4MmaUntil>()
        .add::<MoeNvfp4MmaFrom>()
        .add::<PrefillTc5>()
        .add::<PrefillWideSliding>()
        .add::<DecodeGroupedPrefillFrom>()
        .add::<DecodeGroupedPrefillLse>()
        .add::<DecodeGroupedPrefillWindow>()
        .add::<Fa2Live>()
        .add::<DecodeXqa>()
        .add::<DecodeXqaFrom64>()
        .add::<DecodeXqaWholeWindow>()
        .add::<DecodeXqaCluster>()
        .add::<SdpaSelectedMma>()
        .add::<SdpaIndexMma>()
        .add::<SdpaIndexTc5>()
        .add::<ElemwiseNormEmit>()
        .add::<SdpaIndexRadix>()
        .add::<SdpaPoolMma>()
        .add::<SdpaDsv4PrefillFrom>()
        .add::<SdpaDsv4Replay>()
        .add::<SdpaDsv4Fp4PassFrom>()
        .add::<GemmFp8GroupedTc5Until>()
        .add::<SdpaMlaSparse>()
        .add::<SdpaMlaDenseShort>()
        .add::<GemmNarrowN>();
}

/// [`Tuning`]'s fields, each with its registry default and its read off a
/// profile: `on D` a flag decision, `get D` a constant one's value, `is
/// D::V` whether a choice decision picks `V`.
macro_rules! tuning {
    ($($(#[$doc:meta])* $field:ident: $ty:ty = $default:expr, $how:ident $($decision:ident)::+;)*) => {
        /// The kernel decisions one load runs under, read once from its profile.
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct Tuning {
            $($(#[$doc])* pub $field: $ty,)*
        }

        impl Tuning {
            /// The registry defaults; a test holds the two equal.
            pub const DEFAULT: Tuning = Tuning { $($field: $default,)* };

            #[must_use]
            pub fn of(profile: &Profile) -> Tuning {
                Tuning { $($field: tuning!(@read profile, $how $($decision)::+),)* }
            }
        }
    };
    (@read $p:ident, on $d:ident) => { $p.on::<$d>(&Anywhere) };
    (@read $p:ident, get $d:ident) => { $p.get::<$d>().0 };
    (@read $p:ident, is $d:ident :: $v:ident) => { *$p.get::<$d>() == $d::$v };
}

tuning! {
    moe_grouped: bool = true, on MoeGrouped;
    mxfp4_grouped: bool = true, on MoeMxfp4Grouped;
    mxfp4_wmma: bool = true, on MoeMxfp4Wmma;
    decode_split_small: bool = false, on DecodeSplitSmall;
    decode_split_window: bool = false, on DecodeSplitWindow;
    decode_chunk_floor_128: bool = false, on DecodeChunkFloor128;
    /// `kernel.gemv.bf16 = narrow`.
    gemv_narrow: bool = false, is GemvBf16::Narrow;
    gemv_splitk_max_n: u32 = 4096, get GemvSplitkMaxN;
    /// `kernel.mla.path = naive`.
    mla_naive: bool = false, is MlaPath::Naive;
    qmm_gemm_from: u32 = 16, get QmmGemmFrom;
    qmm_long_rows: u32 = 512, get QmmLongRows;
    qmm_split_warps_per_sm: u32 = 58, get QmmSplitWarpsPerSm;
    qmm_split_thin_max: u32 = 32, get QmmSplitThinMax;
    qmm_split_wide_max: u32 = 16, get QmmSplitWideMax;
    prefill_ctas_per_sm: u32 = 2, get PrefillCtasPerSm;
    decode_static_max_requests: u32 = 512, get DecodeStaticMaxRequests;
    gemm_workspace_mb: u32 = 8, get GemmWorkspaceMb;
    gemm_heuristics: u32 = 8, get GemmHeuristics;
    gemm_skinny_rows: u32 = 64, get GemmSkinnyRows;
    moe_group_at: u32 = 4, get MoeGroupAt;
    mxfp4_grouped_reuse: u32 = 2, get MoeMxfp4GroupedReuse;
    affine_grouped: bool = true, on MoeAffineGrouped;
    affine_wmma: bool = true, on MoeAffineWmma;
    affine_grouped_from: u32 = 16, get MoeAffineGroupedFrom;
    moe_tc5: bool = false, on MoeTc5;
    moe_tc5_from: u32 = 16, get MoeTc5From;
    gemm_split_rows_from: u32 = 256, get GemmSplitRowsFrom;
    qg_prologue_until: u32 = 0, get QgPrologueUntil;
    dsv4_decode_until: u32 = 0, get Dsv4DecodeUntil;
    ssm_gdn_fused_scan: bool = false, on SsmGdnFusedScan;
    ssm_gdn_tc5: bool = false, on SsmGdnTc5;
    rows_vec: bool = false, on ElemwiseRowsVec;
    hc_tile: bool = false, on ElemwiseHcTile;
    hc_mma: bool = false, on ElemwiseHcMma;
    ssm_conv_vec4: bool = false, on SsmConvVec4;
    ssm_gdn_out_bf16: bool = false, on SsmGdnOutBf16;
    ssm_gdn_step_blocks: u32 = 4, get SsmGdnStepBlocks;
    merge_warp: bool = false, on SdpaMergeWarp;
    mxfp4_mma_until: u32 = 0, get MoeMxfp4MmaUntil;
    mxfp4_mma_from: u32 = 64, get MoeMxfp4MmaFrom;
    bf16_mma_from: u32 = 0, get MoeBf16MmaFrom;
    bf16_mma_until: u32 = 0, get MoeBf16MmaUntil;
    mxfp4_cvt: bool = false, on MoeMxfp4Cvt;
    mxfp4_cvt_ksplit_until: u32 = 0, get MoeMxfp4CvtKsplitUntil;
    fp8_o_down: bool = false, on GemmFp8ODown;
    mxfp4_a8_until: u32 = 0, get MoeMxfp4A8Until;
    mxfp4_a8_from: u32 = 0, get MoeMxfp4A8From;
    gemm_bias_epilogue: bool = false, on GemmBiasEpilogue;
    moe_w4a4_fused_from: u32 = 65_536, get MoeW4a4FusedFrom;
    moe_w4a8_fused_from: u32 = 65_536, get MoeW4a8FusedFrom;
    mla_expanded_from: u32 = 65_536, get MlaExpandedFrom;
    gemm_fp8_gemv_until: u32 = 0, get GemmFp8GemvUntil;
    gemm_fp8_shadow: bool = false, on GemmFp8Shadow;
    gemm_fp8_mma_gemv_until: u32 = 0, get GemmFp8MmaGemvUntil;
    attn_res_fused_norm: bool = false, on AttnResFusedNorm;
    ssm_kda_chunked: bool = false, on SsmKdaChunked;
    moe_w4a4_fused_layout: bool = false, on MoeW4a4FusedLayout;
    qkv_rows_from: u32 = 64, get AttnQkvRowsFrom;
    ssm_kda_step_fused: bool = false, on SsmKdaStepFused;
    gemm_split_launch_ns: u32 = 0, get GemmSplitLaunchNs;
    gemm_split_launch_from: u32 = 0, get GemmSplitLaunchFrom;
    gemm_tc5_raster: u32 = 8, get GemmTc5Raster;
    pdl: bool = false, on KernelPdl;
    gemm_rows_m: u32 = 0, get GemmRowsM;
    gemm_tc5_split_tickets: u32 = 0, get GemmTc5SplitTickets;
    nvfp4_mma_from: u32 = 1, get MoeNvfp4MmaFrom;
    nvfp4_mma_until: u32 = 0, get MoeNvfp4MmaUntil;
    prefill_tc5: bool = false, on PrefillTc5;
    prefill_wide_sliding: bool = false, on PrefillWideSliding;
    decode_grouped_prefill_from: u32 = 0, get DecodeGroupedPrefillFrom;
    decode_grouped_prefill_lse: bool = false, on DecodeGroupedPrefillLse;
    decode_grouped_prefill_window: bool = false, on DecodeGroupedPrefillWindow;
    fa2_live: bool = false, on Fa2Live;
    decode_xqa: bool = false, on DecodeXqa;
    decode_xqa_from_64: u32 = 0, get DecodeXqaFrom64;
    decode_xqa_whole_window: u32 = 0, get DecodeXqaWholeWindow;
    decode_xqa_cluster: u32 = 0, get DecodeXqaCluster;
    selected_mma: bool = false, on SdpaSelectedMma;
    index_mma: bool = false, on SdpaIndexMma;
    index_tc5: bool = false, on SdpaIndexTc5;
    norm_emit: bool = false, on ElemwiseNormEmit;
    index_radix: bool = false, on SdpaIndexRadix;
    pool_mma: bool = false, on SdpaPoolMma;
    dsv4_prefill_from: u32 = 0, get SdpaDsv4PrefillFrom;
    dsv4_replay: bool = false, on SdpaDsv4Replay;
    dsv4_fp4_pass_from: u32 = 0, get SdpaDsv4Fp4PassFrom;
    gemm_fp8_grouped_tc5_until: u32 = 0, get GemmFp8GroupedTc5Until;
    mla_sparse: bool = false, on SdpaMlaSparse;
    mla_dense_short: bool = false, on SdpaMlaDenseShort;
    gemm_narrow_n: u32 = 0, get GemmNarrowN;
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
