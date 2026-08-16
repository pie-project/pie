//===-- fa4.cuh ---------------------------------------------------*- CUDA -*-===//
//
// FlashAttention-4 forward, written out as CUDA.
//
// # What this is
//
// FlashAttention-4 ships as a CuTe-DSL program: Python that emits MLIR that
// emits PTX. On an SM100 part its kernel is a `tcgen05` pipeline with a
// tensor-memory accumulator and five specialised warp roles. On an SM120 part
// it is none of those things, because SM120 has none of those instructions,
// and upstream says so by construction: `flash_fwd_sm120.py` is nineteen lines
// that subclass `FlashAttentionForwardSm80` and lower the shared-memory
// ceiling to 99 KB.
//
//     class FlashAttentionForwardSm120(FlashAttentionForwardSm80):
//         def __init__(self, *args, **kwargs):
//             super().__init__(*args, **kwargs)
//             self.arch = Arch.sm_80
//
// So the thing to reimplement, for this hardware, is precisely defined: the
// SM80 lattice — `mma.sync.aligned.m16n8k16`, `cp.async` staging through
// XOR-swizzled shared memory, `ldmatrix` for the operands and `ldmatrix.trans`
// for V — driven by FA4's online-softmax recurrence. That is what this file
// is. It is not a port of the DSL; it is the same algorithm with the same
// tile geometry, written against FlashInfer's PTX wrappers instead of CuTe's.
//
// **Verified, not assumed.** `tcgen05.alloc` and `wgmma.fence` were both put
// through `ptxas` for `sm_120a` before this file was written:
//
//     error : Instruction 'tcgen05.alloc' not supported on .target 'sm_120a'
//     error : Instruction 'wgmma.fence' not supported on .target 'sm_120a'
//
// while `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`,
// `cp.async.bulk.tensor.2d`, `ldmatrix.x4.trans`, `stmatrix.x4`,
// `mbarrier.try_wait.parity` and `ex2.approx.f32` all assemble. The absence of
// the first two is the whole reason this file looks like FA2 and not like the
// FA4 paper's pipeline diagram.
//
// # Why no CUTLASS
//
// This crate is a JIT: a kernel is text in the binary and NVRTC turns it into
// a cubin at run time (see `src/source.rs`). CUTLASS is not carried and could
// not be — it is megabytes of headers per `includeNames[]` entry, and CuTe's
// template depth is a compile-time cost paid on every cold start. FlashInfer's
// `mma.cuh` / `cp_async.cuh` / `permuted_smem.cuh` are already carried
// (`source.rs` `UPSTREAM`) and are thin: each is a named wrapper around one
// PTX instruction. Everything below composes those.
//
// # Geometry, and where it comes from
//
// The tile choice is upstream's, read off `interface.py` rather than guessed:
//
//     if arch // 10 == 12:
//         if head_dim <= 64: fwd_cfg = FwdConfig(128, 128, True, True)
//         else:              fwd_cfg = FwdConfig(128, 64,  True, True)
//     if arch // 10 in [8, 12]: num_threads = 128
//     ... num_stages=1, Q_in_regs=False
//
// 128 threads is four warps, split along M so that a row of the score matrix
// lives entirely inside one warp — softmax then needs no cross-warp traffic,
// only a reduction across the four lanes of a quad. `num_stages=1` is what
// makes 99 KB work: Q(128x128) + K(64x128) + V(64x128) is 64 KB at head_dim
// 128, and a second K/V stage would not fit beside it.
//
// # The one register-layout coincidence the whole kernel rests on
//
// `acc_S` comes out of the QK MMA in C layout and goes into the PV MMA as the
// A operand. Those two layouts are, for a 16x16 tile, *the same layout*:
//
//     C reg pair (0,1) -> row r0,   cols 2q, 2q+1      A reg 0 -> row r0,   cols 2q, 2q+1
//     C reg pair (2,3) -> row r0+8, cols 2q, 2q+1      A reg 1 -> row r0+8, cols 2q, 2q+1
//     C reg pair (4,5) -> row r0,   cols 2q+8, 2q+9    A reg 2 -> row r0,   cols 2q+8, 2q+9
//     C reg pair (6,7) -> row r0+8, cols 2q+8, 2q+9    A reg 3 -> row r0+8, cols 2q+8, 2q+9
//
// (r0 = lane/4, q = lane%4.) So P = exp(S - m) needs no shuffle and no round
// trip through shared memory: convert eight floats to eight bf16 in place and
// reinterpret the four registers as an A fragment. FlashInfer relies on the
// same identity (`prefill.cuh:1645`, passing `(uint32_t*)s_frag_f16`), and
// CuTe spells it `reshape_acc_to_frgA`. If it were not true this kernel would
// need an smem bounce per iteration and would not reach parity.
//
// # Numerics
//
// Transcribed from `softmax.py::Softmax.online_softmax`, not reinvented:
// exponentials are base-2 with the scale folded in (`scale_log2 = scale *
// log2 e`), the running max is stored unsanitised and only the *copy* used in
// the exponent is clamped away from -inf, the correction factor is
// `exp2((m_prev - m_cur) * scale_log2)`, and the row sum is reduced across the
// quad once at the end rather than every iteration. `finalize` divides by the
// row sum with a reciprocal approximation and folds a zero/NaN row to 1.
//
// Note that the JIT compiles with `--fmad=false --prec-div=true
// --prec-sqrt=true` (`src/jit/nvrtc.rs:526`), so the FFMA contraction a
// hand-written `a*b+c` would get under `nvcc` does not happen here. The
// exponent is written as a single `__fmaf_rn` for that reason, and `exp2` and
// the reciprocal go through `flashinfer::math`'s inline PTX, which the flag
// cannot reach.
//
// WHERE THE REMAINING FEW PERCENT ARE
// -----------------------------------
//
// Measured against FA4's own CuteDSL kernel on the same device, over eight
// realistic prefill shapes: geomean 1.011x, causal uniformly at or above
// parity, and non-causal the weaker column at 0.92x (d=64) and 0.95x (d=128).
// The residual is worth stating precisely, because the obvious explanations
// are all wrong and someone will otherwise spend a day rediscovering that.
//
// It is NOT the tile. `interface.py:620-627` picks 128x128 with 128 threads
// for d<=64 on SM120, which is exactly the geometry below, and it reaches
// 140 TFLOP/s on the shape where this reaches 130.
//
// It is NOT the pipeline depth. `interface.py:1113` constructs the SM120
// forward with `num_stages=1` and `Q_in_regs=False` — the single-buffered
// mainloop this file has, not a deeper one.
//
// It is NOT `intra_wg_overlap`. The flag is computed for every architecture
// and consumed by `flash_fwd_sm90.py` alone; the SM80 kernel SM120 subclasses
// never reads it, so there is no GEMM/softmax overlap upstream to copy.
//
// It is NOT `pack_gqa` either, which is the interesting one, because that WAS
// the outstanding hypothesis and it is implemented here now. Packing is
// upstream's remaining structural difference and it does not close this gap:
// on long prefill it saves no M tiles at all, and folding the group makes a
// causal block's key bound round up over a coarser grid, so it measures 0.95x
// on the very shapes it was supposed to fix. What it is instead is a 3.4x on
// BATCHED DECODE, which the eight-shape prefill benchmark simply did not
// contain. See `PackedRow` for the numbers and the rule the host picks by.
//
// The three above were each eliminated by reading upstream and the fourth by
// building it, so whatever is left is not visible in `interface.py`. What it
// is instead is visible in the kernel's own clock. Instrumented as described
// under WHAT IT ACTUALLY IS below, a prefill block spends its time like this:
//
//                            d128 causal   d128 full   d64 causal
//     staging K and V             4.8%        4.0%        4.8%
//     QK GEMM                    44.6%       44.8%       38.9%
//     PV GEMM                    41.4%       41.7%       36.3%
//     exp2 and the row sum        6.4%        6.6%       14.7%
//     mask, row max, rescale,
//       cast to bf16              2.9%        2.9%        5.3%
//
// Eighty-six percent of a d=128 prefill is the two GEMMs, and the rest is not
// obviously removable: the exponential is one `ex2.approx` per score and the
// online softmax needs it. Closing 3% against FlashInfer would mean deleting
// most of the remaining 14%, and there is nothing there to delete.
//
// The d=64 column is the interesting one, because it is where this kernel is
// FURTHEST behind FlashInfer — 0.94x and 0.95x, against 0.96x-0.99x at d=128 —
// and it is also where the softmax more than doubles as a share of the block,
// from 6.4% to 14.7%. That is arithmetic, not a bug: halving the head dimension
// halves the MMA work behind every score, and the exponentials that were hidden
// behind it are not. So the residual is not one gap but two, it tracks the head
// dimension, and the part of it that grows is the part that cannot be tuned
// away inside this algorithm.
//
// A fifth was eliminated by sweeping rather than reading, once the decode tile
// had shown that the geometry is worth varying per launch. If prefill were
// leaving a few percent on a badly chosen tile, some other tile would find it.
// At d=64, TFLOP/s over the six geometries that fit:
//
//                            128x128  128x128  128x64  128x64  64x128  256x128
//                              w4       w8       w4      w8      w4      w8
//     b1 sq4096 hq32 causal    115.6    114.1   114.5   110.6   106.3   106.0
//     b2 sq4096 hq8  causal     99.7    100.9   104.6   101.1   103.8    91.8
//     b1 sq4096 hq32 full      130.4    117.7   124.8   113.9   113.3   117.6
//
// The shipped geometry wins two of the three outright. The row it loses is
// group_size 1, where a narrower N tile raises occupancy and buys 5% — but the
// same change costs 4% on the row below it, so there is no rule here, only a
// per-shape fit, and a plan axis fitted to one benchmark row is how a kernel
// acquires constants nobody can later justify. The remaining prefill percent
// is not a tile choice.
//
//
// WHERE THIS STANDS AGAINST THE KERNEL pie ACTUALLY RUNS
// -------------------------------------------------------
//
// Parity with FA4 is the goal this was written to, but it is not the number
// that decides whether the kernel belongs in pie's attention path, because
// pie's path is FlashInfer's. So, measured against FlashInfer on the same
// device, both dense so neither pays for paging:
//
//     prefill, eight shapes, geomean            0.966x
//     b1   sk4096 hq32 hk8 d128 decode          0.83x
//     b8   sk4096 hq32 hk8 d128 decode          1.02x
//     b32  sk4096 hq32 hk8 d128 decode          0.99x
//     b64  sk2048 hq32 hk8 d128 decode          0.98x
//     b128 sk1024 hq32 hk8 d64  decode          0.99x
//
// The decode column used to read 0.12x at batch 1 and 0.88x-1.01x elsewhere.
// Three things closed it, and they are one decision rather than three, taken
// per launch by `plan` in `fa4.rs`:
//
//   pack the group    divides the KV traffic by `group_size`, and empties the
//                     machine, because it divides the block count too
//   split the keys    fills the machine back up by giving each block a RANGE
//                     of key tiles, and leaves the traffic where it was
//   halve the M tile  stops a decode paying for 128 rows of padding when it
//                     has four real ones
//
// The first two are only useful together: at batch 1 either alone leaves a
// 4096-key decode at 138us, and both together reach 23us. The third is what
// takes it from 23us to 19us, and it stops there because 64 rows is where the
// memory wall is — 32 and 16 measure no better. `combine` below is the fold
// that makes splitting legal: a split block holds an attention over its own
// key range, normalised by its own denominator, and only the fold knows the
// real one.
//
// On the batch-1 decode this is 6.5x faster than FA4's own CuteDSL kernel,
// which does not split there either.
//
// What remains between this and FlashInfer is a few percent on prefill, the
// batch-1 decode, and paging. This kernel is dense, so it still cannot be
// dropped into a serving path that stores KV in pages; that plumbing, not more
// tuning, is what a real integration would be.
//
//
// THE SHAPES A SERVER ACTUALLY SENDS
//
// The two benchmarks above are square prefill and square decode, which is not
// what a server sees. A request arrives as a CHUNK of new tokens against a long
// existing cache, speculative decode arrives as a handful of query rows against
// the same, and context lengths are not 4096. Fifteen such shapes against
// FlashInfer, all d=128 bf16:
//
//     chunked prefill    s512/8192, s1024/16384, s512/8192 b4,
//                        s2048/32768 MQA              0.97x - 1.01x
//     speculative decode s4/4096 b8, b32; s8/8192 b8;
//                        s2/2048 b64                  0.97x - 1.04x
//     long decode        s1/32768, s1/16384 b4,
//                        s1/16384 b16 MQA             0.99x - 1.00x
//     wide GQA           s4096 hq64, s1/8192 hq64     0.99x
//     long decode b1     s1/16384 hq32 hk8            0.73x
//     decode b32 MQA     s1/4096 hq16 hk1             0.85x
//
// So the kernel is at parity across the regimes, and the exception is narrow
// and specific: ONE STREAM against a cache that fits in L2. Both outliers have
// a 67 MB working set, both stay resident across the timing loop, and in both
// FlashInfer sustains about 1.8 TB/s where this reaches 1.3.
//
// That gap is not any of the things it looks like, and each was eliminated by
// measurement rather than by reading, because there is no profiler here:
//
// It is NOT the split count. Swept 4..32 at sk 8192/16384/32768; the plan's
// choice is the measured optimum in every case, and the curve is flat around
// it. Under-splitting is visible (s=4 doubles the time) and so is over-
// splitting (s=16 crosses the 82-SM wave boundary); the plan sits between.
//
// It is NOT tile padding, though this is the closest call. A 32-row tile does
// help these shapes, by 6-9%, and is never worse anywhere it was measured.
// It is not shipped: 6% on one benchmark row costs eight more instantiations
// and a third value on the tile axis, which is the same fitted-constant trade
// refused above for prefill's N tile. If the rest of this gap ever closes, it
// becomes worth revisiting.
//
// It is NOT occupancy. The 64-row tile at d=128 takes 80 KB of shared memory,
// so one block per SM; a 64-wide N tile takes 48 KB and fits two. Two fit, and
// measure the same, because halving the tile doubles the iterations.
//
// It is NOT the thread count, which is the answer that looked most certain and
// is the reason to write this down. A NAIVE streaming read at this geometry —
// 80 blocks of 128 threads over 67 MB — reaches only 416 GB/s, and rises to
// 1990 GB/s at 164 blocks of 512. Read alone, that says the kernel is starved
// of threads and wants warps split over N instead of M. It is wrong. Repeat
// the measurement with cp.async staging through shared memory, which is what
// this kernel actually does and which gives each thread sixteen outstanding
// 16-byte requests, and 80 blocks of 128 threads reach 2332 GB/s — with 256
// threads changing nothing. The bandwidth is already there at this geometry.
//
// It is NOT the KV layout. With hq32 hk8 d128 a key row is 256 contiguous
// bytes on a 2048-byte stride, so the walk is a gather and not a stream. Same
// probe at that stride: 2555 GB/s, identical to contiguous.
//
// It is not the depth of this mainloop either, which is where the reasoning
// above pointed and which was therefore built rather than argued. The loop
// holds one transfer in flight: K lands, a barrier, V is issued and overlapped
// with the QK GEMM, V lands, a barrier, the next K is issued and overlapped
// with PV. Two stages fit only under a 64-wide N tile — 16 KB of Q plus four
// 16 KB buffers is 80 KB against the SM's 100 KB, where `TILE_N = 128` would
// ask for 144 KB — and with both K and V of the next tile in flight the block
// never waits on a transfer it has not issued. It is correct on every harness
// shape and it measures 48us against one stage's 44us: the stage does pay
// against its own baseline (50us at the same narrow tile), but the narrow tile
// costs more than the stage returns, and the two cannot be combined.
//
//
// WHAT IT ACTUALLY IS, AND HOW TO SEE IT WITHOUT A PROFILER
//
// Every hypothesis above is about memory, and all of them are wrong, because
// the premise is wrong. `ncu` is unavailable here — the container drops
// `CAP_SYS_ADMIN` and `CAP_PERFMON` — but a kernel can time itself. Take a
// copy of this file, bracket each phase of the mainloop with `clock64()`, and
// have `tid == 0` accumulate the deltas into a `__device__` array; seven
// probes and an atomic per tile. Bracket `clock64()` with `asm volatile("" :::
// "memory")` or the compiler will sink work across the probe and hand you a
// GEMM that costs nothing; the first version of this said exactly that, and
// the tell was a phase reading 0.0%. Done properly the total agrees with the
// wall clock to within about 6% on both shapes, which is what makes it worth
// believing. Where a decode block's time goes, against prefill's:
//
//                            b1 sk16384 decode     s4096 prefill
//     staging K and V                5.8%              4.8%
//     QK GEMM                       40.0%             44.6%
//     PV GEMM                       39.0%             41.4%
//     exp2 and the row sum          10.5%              6.4%
//     mask, row max, rescale, cast   4.6%              2.9%
//
// Under six percent of the decode kernel goes anywhere near memory, and that
// figure INCLUDES issuing the copies, not just waiting on them. It was never
// bandwidth-bound, and the 1.3 TB/s it "achieves" is simply how much traffic
// its arithmetic happens to need per unit time. Note also how little the two
// columns differ: decode is not running a different kernel badly, it is
// running the same kernel on four real rows out of sixty-four.
//
// What binds it is warp count, and this is visible by turning the knob. Hold
// the tile at 64x128 and vary the warps over M — total block work is constant,
// only the number of issuing warps changes:
//
//     b1 sk16384 decode      4 warps   2 warps   1 warp
//     time                      44us      94us    393us
//
// Time is inversely proportional to warps, which is what an issue-limited
// pipeline looks like: one warp cannot keep the tensor cores fed, and four
// barely can. A 64-row tile admits exactly four warps, because warps divide M
// and an `m16n8k16` fragment needs 16 rows, so this is the ceiling the small
// tile buys its 2x with.
//
// Going the other way confirms it and prices the only remaining fix. At
// 128x128 with eight warps the per-warp work per tile is identical and the
// block does twice as much of it; if warps were free that would take the same
// time, and it takes 1.67x. So eight issuing warps deliver about 1.2x the
// throughput of four, not 2x. The way to eight warps on a 64-row tile is to
// split them over N instead of M — four warps on each half of the key tile,
// folded through a shared-memory reduction like the one `combine` already does
// across blocks. That is a large change to the most delicate code here for a
// measured 1.2x on one decode regime, and it is not taken.
//
// The honest conclusion is that FlashInfer is not winning this shape with a
// better version of this kernel. A 64-row MMA tile holding four real query
// rows does sixteen times the arithmetic the answer needs, and no amount of
// tuning inside the SM80 lattice removes that; a decode path built on FMA dot
// products rather than tensor cores would not pay it at all. Matching that
// means writing a second kernel, not fixing this one.
//
// None of this blocks anything today: nothing calls this kernel yet, so a
// quarter of one decode regime buys nothing until the paging plumbing exists.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "flashinfer/cp_async.cuh"
#include "flashinfer/math.cuh"
#include "flashinfer/mma.cuh"
#include "flashinfer/permuted_smem.cuh"
#include "flashinfer/vec_dtypes.cuh"

namespace pie {
namespace attn {
namespace fa4 {

using ::flashinfer::b128_t;
using ::flashinfer::smem_t;
using ::flashinfer::SwizzleMode;
using ::flashinfer::upcast_size;
using ::flashinfer::cp_async::PrefetchMode;
using ::flashinfer::cp_async::SharedMemFillMode;
using ::flashinfer::mma::MMAMode;

/// Everything the kernel needs, passed by value in one block.
///
/// Strides are in ELEMENTS, not bytes, and name the axis they step over:
/// `q_stride_s` moves one query position, `q_stride_h` one head. The layout
/// this was written against is bshd — `(batch, seqlen, head, dim)` — which is
/// what `flash_attn_func` takes, but nothing here assumes contiguity beyond
/// the head dimension itself, which must be contiguous because the loads are
/// 128-bit.
struct Params {
  const void* q;
  const void* k;
  const void* v;
  void* o;
  /// Optional: log-sum-exp, one float per (batch, head, query). Skipped when null.
  float* lse;

  int32_t q_stride_b, q_stride_s, q_stride_h;
  int32_t k_stride_b, k_stride_s, k_stride_h;
  int32_t v_stride_b, v_stride_s, v_stride_h;
  int32_t o_stride_b, o_stride_s, o_stride_h;
  int32_t lse_stride_b, lse_stride_h;

  int32_t seqlen_q;
  int32_t seqlen_k;
  /// Query heads per key/value head. 1 is plain MHA.
  int32_t group_size;

  /// Split-KV scratch, and how many blocks share one output row.
  ///
  /// With `num_splits > 1` a block covers a CONTIGUOUS RANGE of key tiles
  /// rather than all of them, and writes its normalised O and its log-sum-exp
  /// to these rather than to `o`/`lse`. `combine` then folds the splits. Both
  /// are `[split][row][...]` with `row = (batch * heads_q + head) * seqlen_q +
  /// position`, which is the layout that makes the combine's reads of one row
  /// contiguous in `d` and its walk over splits a constant stride.
  float* o_partial;
  float* lse_partial;
  int32_t num_splits;
  /// `heads_q`, needed only to index the partials by row.
  int32_t heads_q;

  /// `softmax_scale * log2(e)`, folded once on the host so the inner loop
  /// spends one FMA instead of two multiplies.
  float scale_log2;
};

__device__ __forceinline__ int32_t ceil_div(int32_t a, int32_t b) { return (a + b - 1) / b; }

/// `-inf`, without the `INFINITY` macro NVRTC does not define.
///
/// This file was authored against `nvcc`, where `<cmath>` supplies `INFINITY`;
/// the crate compiles it through NVRTC, which answers 0 of 31 standard headers
/// and ships no CUDA ones, so `-INFINITY` is undefined here. `__int_as_float(
/// 0xff800000)` is the same bit pattern with no macro at all, and it is the
/// exact substitution `attention_naive.cuh` and the prelude's `neg_inf()`
/// already make for this reason. A negative sign in front is not wanted: this
/// value is already `-inf`.
__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }

/// The compile-time shape of one instantiation.
///
/// Everything here is derived, so a caller states four numbers and cannot
/// state an inconsistent fifth.
template <typename DTypeIn_, uint32_t HEAD_DIM_, uint32_t TILE_M_, uint32_t TILE_N_, bool CAUSAL_,
          uint32_t NUM_WARPS_ = 8, bool PACKED_ = false>
struct Traits {
  using DTypeIn = DTypeIn_;
  static constexpr uint32_t HEAD_DIM = HEAD_DIM_;
  static constexpr uint32_t TILE_M = TILE_M_;
  static constexpr uint32_t TILE_N = TILE_N_;
  static constexpr bool CAUSAL = CAUSAL_;

  /// Whether the M tile indexes `(position, head)` pairs of one KV group
  /// rather than query positions. See [`PackedRow`] for what that buys and
  /// what it costs; the host picks between the two per launch.
  ///
  /// Compile-time and not a runtime flag because the unpacked case has to cost
  /// NOTHING. Packing's row mapping is two integer divisions and a
  /// `[NUM_MMA_M][2]` position/head table, and at `d=64` with four warps this
  /// kernel is already at 255 registers and spilling — measured, those extra
  /// registers cost 4% on a shape where packing does no work at all. As a
  /// `constexpr` the divisor folds to one, the table folds to the row index
  /// and the block's single head, and the unpacked instantiation is
  /// instruction-for-instruction what it was before packing existed.
  static constexpr bool PACKED = PACKED_;

  /// Warps split the M tile, never the N tile: a warp owning whole score rows
  /// is what keeps the online softmax a quad shuffle with no cross-warp
  /// reduction and no shared-memory rendezvous. More warps therefore means
  /// fewer m16 fragments each, which is the lever on register pressure --
  /// `acc_o` alone is `NUM_MMA_M * NUM_MMA_D * 8` floats.
  static constexpr uint32_t NUM_WARPS = NUM_WARPS_;
  static constexpr uint32_t NUM_THREADS = NUM_WARPS * 32;

  /// Rows of the Q tile owned by one warp. A score row never crosses a warp,
  /// which is what keeps the softmax reduction inside a quad.
  static constexpr uint32_t WARP_M = TILE_M / NUM_WARPS;
  static constexpr uint32_t NUM_MMA_M = WARP_M / 16;
  static constexpr uint32_t NUM_MMA_N = TILE_N / 16;
  static constexpr uint32_t NUM_MMA_D = HEAD_DIM / 16;

  /// Row stride in 128-bit units: one row of a head is `HEAD_DIM` elements of
  /// two bytes, so `HEAD_DIM / 8` of them.
  static constexpr uint32_t STRIDE = HEAD_DIM / 8;
  /// 128-byte swizzle needs a row that is a multiple of 128 bytes.
  static constexpr SwizzleMode SWIZZLE = SwizzleMode::k128B;

  /// Threads cooperating on one row during a global->shared copy, and the
  /// rows that covers per pass.
  static constexpr uint32_t COPY_THREADS_PER_ROW = STRIDE;
  static constexpr uint32_t COPY_ROWS_PER_PASS = NUM_THREADS / COPY_THREADS_PER_ROW;

  static constexpr uint32_t Q_ELEMS = TILE_M * HEAD_DIM;
  static constexpr uint32_t K_ELEMS = TILE_N * HEAD_DIM;
  static constexpr uint32_t V_ELEMS = TILE_N * HEAD_DIM;
  /// O is written through Q's shared memory once the last QK product has been
  /// read, exactly as `flash_fwd.py` does (`sO = make_tensor(sQ.iterator, …)`).
  static constexpr uint32_t SMEM_ELEMS = Q_ELEMS + K_ELEMS + V_ELEMS;
  static constexpr uint32_t SMEM_BYTES = SMEM_ELEMS * sizeof(DTypeIn);

  static_assert(HEAD_DIM % 64 == 0, "128B swizzle wants a 128-byte row");
  static_assert(TILE_M % NUM_WARPS == 0, "M tile must split over the warps");
  static_assert(WARP_M % 16 == 0, "each warp needs whole m16 fragments");
  static_assert(TILE_N % 16 == 0, "N tile must be whole n16 fragments");
  static_assert(NUM_THREADS % COPY_THREADS_PER_ROW == 0, "copy must tile the row evenly");
  static_assert(TILE_M % COPY_ROWS_PER_PASS == 0, "copy must tile the M tile evenly");
  static_assert(TILE_N % COPY_ROWS_PER_PASS == 0, "copy must tile the N tile evenly");
};

/// The shared-memory size a launch must request, echoed into the cubin so the
/// host arithmetic can be cross-checked against the compiler's rather than
/// trusted. `fa2.cuh` exports the same kind of witness for its paged traits.
template <class T>
__device__ unsigned smem_bytes = T::SMEM_BYTES;

/// Stage a `rows x HEAD_DIM` tile from global into swizzled shared memory.
///
/// One 128-bit `cp.async` per thread per pass. `FILL` decides what an
/// out-of-range row leaves behind: K and V zero-fill, because a masked score
/// multiplies V and `0 * NaN` is NaN, while Q may hold garbage — its rows are
/// never stored.
template <class T, SharedMemFillMode FILL, uint32_t ROWS>
__device__ __forceinline__ void stage_tile(smem_t<T::SWIZZLE>* dst, const typename T::DTypeIn* src,
                                           int32_t row_base, int32_t row_limit, int32_t src_stride,
                                           uint32_t tid) {
  constexpr uint32_t PER_ROW = T::COPY_THREADS_PER_ROW;
  constexpr uint32_t PER_PASS = T::COPY_ROWS_PER_PASS;
  const uint32_t r = tid / PER_ROW;
  const uint32_t c = tid % PER_ROW;
  const uint32_t elem_col = c * upcast_size<typename T::DTypeIn>();

#pragma unroll
  for (uint32_t pass = 0; pass < ROWS / PER_PASS; ++pass) {
    const uint32_t row = pass * PER_PASS + r;
    const int32_t global_row = row_base + static_cast<int32_t>(row);
    const uint32_t offset = smem_t<T::SWIZZLE>::template get_permuted_offset<T::STRIDE>(row, c);
    dst->template load_128b_async<FILL>(offset, src + global_row * src_stride + elem_col,
                                        global_row < row_limit);
  }
}

/// Where an M row lives in Q, O and the log-sum-exp, packed or not.
///
/// Unpacked (`PACKED == false`), the M tile indexes query POSITIONS and
/// `blockIdx.y` is a query head, which is what this kernel did originally:
/// `pos(pm) == pm`, `head(pm) == slot`, and the key head is `slot /
/// group_size`.
///
/// Packed, the M tile indexes `(position, head)` pairs of ONE KV group —
/// upstream's `pack_gqa` (`interface.py:539-540`) — so `blockIdx.y` is a key
/// head, a block covers `TILE_M / group_size` positions across all
/// `group_size` query heads behind that key head, and each K and V tile is
/// read once for the group instead of once per head.
///
/// Head varies FASTEST, so the `group_size` rows of one position are adjacent.
/// Two things follow, and both are why this order and not the other one:
///
/// 1. The causal mask is constant across those rows, so a block still spans
///    only `TILE_M / group_size` positions and its `n_block_max` bound is
///    computed from those, not from the row index.
/// 2. In `bshd` those rows are physically contiguous, since head is the axis
///    next to the contiguous one, so the staging copy gets a longer run than
///    the unpacked kernel's stride-`q_stride_s` walk rather than a gather.
///
/// # When packing is worth it, measured
///
/// Packing does not reduce arithmetic; it reduces the number of M TILES, and
/// only when the unpacked tiles were mostly padding. Unpacked costs
/// `heads_q * ceil(seqlen_q / TILE_M)` tiles and packed costs
/// `heads_kv * ceil(seqlen_q * group_size / TILE_M)`, so with
/// `seqlen_q >= TILE_M` and a whole number of tiles the two are EQUAL — and
/// then packing is a small loss, because a causal block spanning fewer
/// positions rounds its key bound up over a coarser grid.
///
/// The measurements, this kernel against itself on the same device:
///
/// | shape                              | unpacked | packed | packed/unpacked |
/// |------------------------------------|----------|--------|-----------------|
/// | b1 s4096 hq32 hk8 d64 causal        | 109.5T   | 104.1T | 0.95x           |
/// | b1 s4096 hq32 hk8 d128 full         | 127.9T   | 126.6T | 0.99x           |
/// | b8 sq1 sk4096 hq32 hk8 d128 causal  | 620 us   | 181 us | 3.4x            |
/// | b32 sq1 sk4096 hq32 hk8 d128 causal | 2221 us  | 816 us | 2.7x            |
///
/// So it is neither an optimisation nor a pessimisation in general: it is the
/// right shape for decode and the wrong one for long prefill, and the host
/// picks per launch on the tile count. Shipping only the packed kernel — which
/// is what upstream does, defaulting it on for every GQA call — would have
/// cost 5% on prefill; shipping only the unpacked one costs 3.4x on batched
/// decode, which is most of what a server does.
template <class T>
struct PackedRow {
  /// Query heads per key head. One means MHA, and packing is then the
  /// identity mapping whichever instantiation is running.
  int32_t group_size;
  /// Rows past the last real query position, which the copies predicate on.
  int32_t seqlen_q;
  /// `blockIdx.y`: a key head when packed, a query head when not.
  int32_t slot;

  /// Rows of one position packed together — `group_size`, or 1 unpacked.
  __device__ __forceinline__ int32_t pack() const { return T::PACKED ? group_size : 1; }

  /// The query position of M row `pm`.
  __device__ __forceinline__ int32_t pos(int32_t pm) const {
    return T::PACKED ? pm / group_size : pm;
  }

  /// The query head of M row `pm`.
  __device__ __forceinline__ int32_t head(int32_t pm) const {
    return T::PACKED ? slot * group_size + pm % group_size : slot;
  }

  /// The key head this block reads.
  __device__ __forceinline__ int32_t kv_head() const {
    return T::PACKED ? slot : slot / group_size;
  }
};

/// Stage the Q tile, whose rows are `(position, head)` pairs.
///
/// Separate from [`stage_tile`] rather than a mode of it: K and V are indexed
/// by a key position and nothing else, they are staged once per mainloop
/// iteration, and giving that path a row-mapping indirection it never uses
/// would be paying the hot loop for the cold one's generality.
template <class T>
__device__ __forceinline__ void stage_q_tile(smem_t<T::SWIZZLE>* dst,
                                             const typename T::DTypeIn* src, int32_t row_base,
                                             const PackedRow<T>& map, int32_t stride_s,
                                             int32_t stride_h, uint32_t tid) {
  constexpr uint32_t PER_ROW = T::COPY_THREADS_PER_ROW;
  constexpr uint32_t PER_PASS = T::COPY_ROWS_PER_PASS;
  const uint32_t r = tid / PER_ROW;
  const uint32_t c = tid % PER_ROW;
  const uint32_t elem_col = c * upcast_size<typename T::DTypeIn>();

#pragma unroll
  for (uint32_t pass = 0; pass < T::TILE_M / PER_PASS; ++pass) {
    const uint32_t row = pass * PER_PASS + r;
    const int32_t pm = row_base + static_cast<int32_t>(row);
    const int32_t position = map.pos(pm);
    const bool ok = position < map.seqlen_q;
    // The address is only formed when the row is real; an out-of-range row
    // reads element zero with the copy predicated off, which never faults.
    const int32_t global = ok ? position * stride_s + map.head(pm) * stride_h : 0;
    const uint32_t offset = smem_t<T::SWIZZLE>::template get_permuted_offset<T::STRIDE>(row, c);
    dst->template load_128b_async<SharedMemFillMode::kNoFill>(offset, src + global + elem_col, ok);
  }
}

/// S = Q K^T for one n-tile, accumulating in registers.
///
/// The `mma_d == 0` pass initialises the accumulator with `MMAMode::kInit`
/// rather than a separate zeroing loop, which is worth 8 registers' worth of
/// `mov` per fragment.
template <class T>
__device__ __forceinline__ void compute_qk(smem_t<T::SWIZZLE>* q_smem, smem_t<T::SWIZZLE>* k_smem,
                                           uint32_t lane, uint32_t warp,
                                           float (*acc_s)[T::NUM_MMA_N][8]) {
  using DTypeIn = typename T::DTypeIn;
  uint32_t a_frag[T::NUM_MMA_M][4], b_frag[4];
  // A and B want *different* ldmatrix address maps, because `ldmatrix.x4`
  // assigns its four output registers by lane octet while the MMA's A and B
  // fragments disagree about which octet holds which 8x8 quadrant. A wants the
  // quadrants in (row, col) order; B wants them in (col, row) order — see
  // flashinfer/attention/prefill.cuh:2124 (Q) against :2175 (K), which differ
  // for exactly this reason. Getting this wrong is silent: every value is a
  // real dot product, just of the wrong pair of vectors.
  const uint32_t a_row = lane % 16;
  const uint32_t a_col = lane / 16;
  const uint32_t b_row = 8 * (lane / 16) + lane % 8;
  const uint32_t b_col = (lane % 16) / 8;

#pragma unroll
  for (uint32_t mma_d = 0; mma_d < T::NUM_MMA_D; ++mma_d) {
#pragma unroll
    for (uint32_t mma_m = 0; mma_m < T::NUM_MMA_M; ++mma_m) {
      const uint32_t row = warp * T::WARP_M + mma_m * 16 + a_row;
      const uint32_t off =
          smem_t<T::SWIZZLE>::template get_permuted_offset<T::STRIDE>(row, mma_d * 2 + a_col);
      q_smem->ldmatrix_m8n8x4(off, a_frag[mma_m]);
    }
#pragma unroll
    for (uint32_t mma_n = 0; mma_n < T::NUM_MMA_N; ++mma_n) {
      const uint32_t off = smem_t<T::SWIZZLE>::template get_permuted_offset<T::STRIDE>(
          mma_n * 16 + b_row, mma_d * 2 + b_col);
      k_smem->ldmatrix_m8n8x4(off, b_frag);
#pragma unroll
      for (uint32_t mma_m = 0; mma_m < T::NUM_MMA_M; ++mma_m) {
        if (mma_d == 0) {
          ::flashinfer::mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeIn, MMAMode::kInit>(
              acc_s[mma_m][mma_n], a_frag[mma_m], b_frag);
        } else {
          ::flashinfer::mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeIn>(
              acc_s[mma_m][mma_n], a_frag[mma_m], b_frag);
        }
      }
    }
  }
}

/// O += P V, with V read transposed straight out of shared memory.
///
/// `p_frag` is the *same storage* as the score accumulator, reinterpreted —
/// see the layout note in this file's header.
template <class T>
__device__ __forceinline__ void compute_pv(smem_t<T::SWIZZLE>* v_smem, uint32_t lane,
                                           const typename T::DTypeIn (*p_frag)[T::NUM_MMA_N][8],
                                           float (*acc_o)[T::NUM_MMA_D][8]) {
  using DTypeIn = typename T::DTypeIn;
  const uint32_t row_in_frag = lane % 16;
  const uint32_t col_half = lane / 16;

#pragma unroll
  for (uint32_t mma_n = 0; mma_n < T::NUM_MMA_N; ++mma_n) {
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < T::NUM_MMA_D; ++mma_d) {
      uint32_t b_frag[4];
      const uint32_t off = smem_t<T::SWIZZLE>::template get_permuted_offset<T::STRIDE>(
          mma_n * 16 + row_in_frag, mma_d * 2 + col_half);
      v_smem->ldmatrix_m8n8x4_trans(off, b_frag);
#pragma unroll
      for (uint32_t mma_m = 0; mma_m < T::NUM_MMA_M; ++mma_m) {
        ::flashinfer::mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeIn>(
            acc_o[mma_m][mma_d], (uint32_t*)p_frag[mma_m][mma_n], b_frag);
      }
    }
  }
}

/// The query row a given accumulator register belongs to, within the M tile.
///
/// `half` is 0 for registers {0,1,4,5} and 1 for {2,3,6,7}; the C layout puts
/// the second group eight rows down.
template <class T>
__device__ __forceinline__ uint32_t acc_row(uint32_t warp, uint32_t mma_m, uint32_t lane,
                                            uint32_t half) {
  return warp * T::WARP_M + mma_m * 16 + lane / 4 + 8 * half;
}

/// The key column a given register belongs to, within the N tile.
__device__ __forceinline__ uint32_t acc_col(uint32_t mma_n, uint32_t lane, uint32_t reg) {
  return mma_n * 16 + 8 * (reg / 4) + 2 * (lane % 4) + (reg % 2);
}

/// Reduce across the four lanes of a quad, which is where one score row lives.
__device__ __forceinline__ float quad_max(float x) {
  x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, 1));
  x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, 2));
  return x;
}

__device__ __forceinline__ float quad_sum(float x) {
  x += __shfl_xor_sync(0xffffffff, x, 1);
  x += __shfl_xor_sync(0xffffffff, x, 2);
  return x;
}

/// The entry point, with the operands flat.
///
/// `Params` is a device-side convenience, not the ABI. A `__global__` taking
/// an aggregate by value would need a Rust mirror whose size, alignment and
/// every field offset are asserted against a measured layout; twenty-two
/// scalars need none of that, because each one is checked by the typecheck TU
/// on its own. The struct is rebuilt below so the body still reads against
/// `flash_fwd.py`'s names.
template <class T>
__global__ __launch_bounds__(T::NUM_THREADS) void kernel(
    const void* q, const void* k, const void* v, void* o, float* lse, int32_t q_stride_b,
    int32_t q_stride_s, int32_t q_stride_h, int32_t k_stride_b, int32_t k_stride_s,
    int32_t k_stride_h, int32_t v_stride_b, int32_t v_stride_s, int32_t v_stride_h,
    int32_t o_stride_b, int32_t o_stride_s, int32_t o_stride_h, int32_t lse_stride_b,
    int32_t lse_stride_h, int32_t seqlen_q, int32_t seqlen_k, int32_t group_size,
    float* o_partial, float* lse_partial, int32_t num_splits, int32_t heads_q,
    float scale_log2) {
  Params p;
  p.q = q;
  p.k = k;
  p.v = v;
  p.o = o;
  p.lse = lse;
  p.q_stride_b = q_stride_b;
  p.q_stride_s = q_stride_s;
  p.q_stride_h = q_stride_h;
  p.k_stride_b = k_stride_b;
  p.k_stride_s = k_stride_s;
  p.k_stride_h = k_stride_h;
  p.v_stride_b = v_stride_b;
  p.v_stride_s = v_stride_s;
  p.v_stride_h = v_stride_h;
  p.o_stride_b = o_stride_b;
  p.o_stride_s = o_stride_s;
  p.o_stride_h = o_stride_h;
  p.lse_stride_b = lse_stride_b;
  p.lse_stride_h = lse_stride_h;
  p.seqlen_q = seqlen_q;
  p.seqlen_k = seqlen_k;
  p.group_size = group_size;
  p.o_partial = o_partial;
  p.lse_partial = lse_partial;
  p.num_splits = num_splits;
  p.heads_q = heads_q;
  p.scale_log2 = scale_log2;

  using DTypeIn = typename T::DTypeIn;

  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % 32;
  const uint32_t warp = tid / 32;
  const int32_t m_block = static_cast<int32_t>(blockIdx.x);
  /// `blockIdx.z` carries the split as the FAST axis of the batch.
  ///
  /// Splits of one batch row read the same K and V, so making the split
  /// adjacent puts them on nearby SMs and lets them share what they pull into
  /// L2. With `num_splits == 1` this is `blockIdx.z` and a division the
  /// compiler folds against the constant 1.
  const int32_t split = static_cast<int32_t>(blockIdx.z) % p.num_splits;
  const int32_t batch = static_cast<int32_t>(blockIdx.z) / p.num_splits;
  const PackedRow<T> map{p.group_size, p.seqlen_q, static_cast<int32_t>(blockIdx.y)};
  const int32_t kv_head = map.kv_head();

  extern __shared__ uint8_t smem_raw[];
  DTypeIn* smem_q = reinterpret_cast<DTypeIn*>(smem_raw);
  DTypeIn* smem_k = smem_q + T::Q_ELEMS;
  DTypeIn* smem_v = smem_k + T::K_ELEMS;
  smem_t<T::SWIZZLE> q_smem(smem_q), k_smem(smem_k), v_smem(smem_v);

  const DTypeIn* q_base = reinterpret_cast<const DTypeIn*>(p.q) + batch * p.q_stride_b;
  const DTypeIn* k_base =
      reinterpret_cast<const DTypeIn*>(p.k) + batch * p.k_stride_b + kv_head * p.k_stride_h;
  const DTypeIn* v_base =
      reinterpret_cast<const DTypeIn*>(p.v) + batch * p.v_stride_b + kv_head * p.v_stride_h;

  const int32_t q_row_base = m_block * static_cast<int32_t>(T::TILE_M);
  /// Under causality, query `i` may see key `j` when `j <= i + (k - q)`. The
  /// offset is what makes a short query against a long cache line up with the
  /// END of the cache, which is what every decode-time caller means.
  const int32_t causal_offset = p.seqlen_k - p.seqlen_q;

  /// The first and last query POSITIONS this tile covers.
  ///
  /// Not the first and last rows: packing puts `group_size` heads of one
  /// position in adjacent rows, so a `TILE_M`-row tile spans
  /// `TILE_M / group_size` positions. Both bounds below are about the mask,
  /// which is a function of position alone, and using the row index here
  /// would over-estimate the diagonal by a factor of `group_size` — correct,
  /// but it would throw away exactly the K tiles packing is meant to save.
  const int32_t pos_first = map.pos(q_row_base);
  const int32_t pos_last_raw = map.pos(q_row_base + static_cast<int32_t>(T::TILE_M) - 1);
  const int32_t pos_last = pos_last_raw < p.seqlen_q - 1 ? pos_last_raw : p.seqlen_q - 1;

  int32_t n_block_max = ceil_div(p.seqlen_k, static_cast<int32_t>(T::TILE_N));
  if (T::CAUSAL) {
    const int32_t last_key = pos_last + causal_offset;
    const int32_t bound = ceil_div(last_key + 1, static_cast<int32_t>(T::TILE_N));
    n_block_max = bound < n_block_max ? bound : n_block_max;
  }

  // The key tiles THIS block owns. Without splitting that is all of them; with
  // splitting it is an even chunk, and a split past the end contributes an
  // empty range whose epilogue still has to write a zero row and a -inf
  // log-sum-exp so the combine has something well-defined to read.
  const int32_t n_chunk = ceil_div(n_block_max, p.num_splits);
  const int32_t n_block_lo = split * n_chunk;
  const int32_t n_block_hi = n_block_max < n_block_lo + n_chunk ? n_block_max : n_block_lo + n_chunk;

  float acc_o[T::NUM_MMA_M][T::NUM_MMA_D][8];
  float row_max[T::NUM_MMA_M][2];
  float row_sum[T::NUM_MMA_M][2];
#pragma unroll
  for (uint32_t mm = 0; mm < T::NUM_MMA_M; ++mm) {
#pragma unroll
    for (uint32_t md = 0; md < T::NUM_MMA_D; ++md) {
#pragma unroll
      for (uint32_t r = 0; r < 8; ++r) acc_o[mm][md][r] = 0.f;
    }
    row_max[mm][0] = neg_inf();
    row_max[mm][1] = neg_inf();
    row_sum[mm][0] = 0.f;
    row_sum[mm][1] = 0.f;
  }

  // The position and head each accumulator row belongs to, computed once.
  //
  // `acc_row` depends on the warp, the lane and the fragment — never on the
  // key block — so these are loop invariants, and hoisting them is what keeps
  // packing's two integer divisions out of a mask that would otherwise do
  // `NUM_MMA_M * NUM_MMA_N * 8` of them per key tile.
  int32_t row_pos[T::NUM_MMA_M][2];
  int32_t row_head[T::NUM_MMA_M][2];
#pragma unroll
  for (uint32_t mm = 0; mm < T::NUM_MMA_M; ++mm) {
#pragma unroll
    for (uint32_t h = 0; h < 2; ++h) {
      const int32_t pm = q_row_base + static_cast<int32_t>(acc_row<T>(warp, mm, lane, h));
      row_pos[mm][h] = map.pos(pm);
      row_head[mm][h] = map.head(pm);
    }
  }

  // A tile with nothing to attend to still has to write its zeros and its
  // -inf log-sum-exp, so this is a jump to the epilogue and not a return.
  if (n_block_hi > n_block_lo) {
    stage_q_tile<T>(&q_smem, q_base, q_row_base, map, p.q_stride_s, p.q_stride_h, tid);
    ::flashinfer::cp_async::commit_group();

    int32_t n_block = n_block_hi - 1;
    stage_tile<T, SharedMemFillMode::kFillZero, T::TILE_N>(
        &k_smem, k_base, n_block * static_cast<int32_t>(T::TILE_N), p.seqlen_k, p.k_stride_s, tid);
    ::flashinfer::cp_async::commit_group();

    // Wait for Q only: one group (K's) is still allowed to be outstanding.
    ::flashinfer::cp_async::wait_group<1>();
    __syncthreads();

    for (; n_block >= n_block_lo; --n_block) {
      const int32_t k_col_base = n_block * static_cast<int32_t>(T::TILE_N);

      // K has landed. The barrier also certifies that every warp is done
      // reading V from the previous iteration, which is what makes it safe to
      // overwrite V below with a single buffer.
      ::flashinfer::cp_async::wait_group<0>();
      __syncthreads();

      stage_tile<T, SharedMemFillMode::kFillZero, T::TILE_N>(&v_smem, v_base, k_col_base,
                                                             p.seqlen_k, p.v_stride_s, tid);
      ::flashinfer::cp_async::commit_group();

      float acc_s[T::NUM_MMA_M][T::NUM_MMA_N][8];
      compute_qk<T>(&q_smem, &k_smem, lane, warp, acc_s);

      // V has landed, and every warp is past its use of K.
      ::flashinfer::cp_async::wait_group<0>();
      __syncthreads();

      if (n_block > n_block_lo) {
        stage_tile<T, SharedMemFillMode::kFillZero, T::TILE_N>(
            &k_smem, k_base, k_col_base - static_cast<int32_t>(T::TILE_N), p.seqlen_k,
            p.k_stride_s, tid);
      }
      ::flashinfer::cp_async::commit_group();

      // Masking is skipped wholesale on interior tiles. The predicate is
      // uniform across the block, so this is one branch, not a divergence.
      const bool ragged = k_col_base + static_cast<int32_t>(T::TILE_N) > p.seqlen_k;
      const bool diagonal =
          T::CAUSAL &&
          (k_col_base + static_cast<int32_t>(T::TILE_N) - 1 > pos_first + causal_offset);
      if (ragged || diagonal) {
#pragma unroll
        for (uint32_t mm = 0; mm < T::NUM_MMA_M; ++mm) {
#pragma unroll
          for (uint32_t mn = 0; mn < T::NUM_MMA_N; ++mn) {
#pragma unroll
            for (uint32_t r = 0; r < 8; ++r) {
              const int32_t qi = row_pos[mm][(r % 4) / 2];
              const int32_t kj = k_col_base + static_cast<int32_t>(acc_col(mn, lane, r));
              const bool ok = kj < p.seqlen_k && (!T::CAUSAL || kj <= qi + causal_offset);
              if (!ok) acc_s[mm][mn][r] = neg_inf();
            }
          }
        }
      }

      // Online softmax, transcribed from `softmax.py::online_softmax`.
      DTypeIn p_frag[T::NUM_MMA_M][T::NUM_MMA_N][8];
#pragma unroll
      for (uint32_t mm = 0; mm < T::NUM_MMA_M; ++mm) {
        float scale[2];
#pragma unroll
        for (uint32_t h = 0; h < 2; ++h) {
          float local = neg_inf();
#pragma unroll
          for (uint32_t mn = 0; mn < T::NUM_MMA_N; ++mn) {
#pragma unroll
            for (uint32_t j = 0; j < 4; ++j) {
              // Registers of row-half `h`: {2h, 2h+1, 4+2h, 5+2h}.
              const uint32_t r = (j / 2) * 4 + 2 * h + (j % 2);
              local = fmaxf(local, acc_s[mm][mn][r]);
            }
          }
          const float prev = row_max[mm][h];
          const float cur = fmaxf(prev, quad_max(local));
          row_max[mm][h] = cur;
          // Only the copy that reaches an exponent is clamped; the stored max
          // stays -inf so a fully-masked row is still recognisable in
          // `finalize`.
          const float safe = cur == neg_inf() ? 0.f : cur;
          scale[h] = ::flashinfer::math::ptx_exp2((prev - safe) * p.scale_log2);
          row_sum[mm][h] *= scale[h];
          const float bias = -safe * p.scale_log2;
          float acc = 0.f;
#pragma unroll
          for (uint32_t mn = 0; mn < T::NUM_MMA_N; ++mn) {
#pragma unroll
            for (uint32_t j = 0; j < 4; ++j) {
              const uint32_t r = (j / 2) * 4 + 2 * h + (j % 2);
              const float e =
                  ::flashinfer::math::ptx_exp2(__fmaf_rn(acc_s[mm][mn][r], p.scale_log2, bias));
              acc_s[mm][mn][r] = e;
              acc += e;
            }
          }
          row_sum[mm][h] += acc;
        }
        // Rescale O by the same two factors before it is added to.
#pragma unroll
        for (uint32_t md = 0; md < T::NUM_MMA_D; ++md) {
#pragma unroll
          for (uint32_t r = 0; r < 8; ++r) acc_o[mm][md][r] *= scale[(r % 4) / 2];
        }
#pragma unroll
        for (uint32_t mn = 0; mn < T::NUM_MMA_N; ++mn) {
          ::flashinfer::vec_cast<DTypeIn, float>::template cast<8>(p_frag[mm][mn],
                                                                   acc_s[mm][mn]);
        }
      }

      compute_pv<T>(&v_smem, lane, p_frag, acc_o);
    }
  }

  // Epilogue. The row sum was only ever accumulated per lane; this is the one
  // place it has to be made whole across the quad.
  float o_scale[T::NUM_MMA_M][2];
#pragma unroll
  for (uint32_t mm = 0; mm < T::NUM_MMA_M; ++mm) {
#pragma unroll
    for (uint32_t h = 0; h < 2; ++h) {
      const float s = quad_sum(row_sum[mm][h]);
      row_sum[mm][h] = s;
      const bool degenerate = s == 0.f || s != s;
      o_scale[mm][h] = ::flashinfer::math::ptx_rcp(degenerate ? 1.f : s);
    }
#pragma unroll
    for (uint32_t md = 0; md < T::NUM_MMA_D; ++md) {
#pragma unroll
      for (uint32_t r = 0; r < 8; ++r) acc_o[mm][md][r] *= o_scale[mm][(r % 4) / 2];
    }
  }

  // Split-KV blocks stop here: they own a range of key tiles, not all of them,
  // so what they hold is a PARTIAL attention — correct for its own range and
  // normalised within it. `combine` reweights the ranges against each other
  // through their log-sum-exps, which is the only place the true denominator
  // exists. Written from registers as f32 rather than through Q's shared
  // memory as bf16: the combine is a sum of these, and rounding each addend to
  // 8 mantissa bits before summing would cost more than the store bandwidth
  // saved on what is, at decode, a handful of rows.
  if (p.num_splits > 1) {
#pragma unroll
    for (uint32_t mm = 0; mm < T::NUM_MMA_M; ++mm) {
#pragma unroll
      for (uint32_t h = 0; h < 2; ++h) {
        const int32_t qi = row_pos[mm][h];
        if (qi >= p.seqlen_q) continue;
        const int64_t prow =
            (static_cast<int64_t>(batch) * p.heads_q + row_head[mm][h]) * p.seqlen_q + qi;
        const int64_t slot = prow * p.num_splits + split;
        if (lane % 4 == 0) {
          const float s = row_sum[mm][h];
          const bool degenerate = s == 0.f || s != s;
          p.lse_partial[slot] =
              degenerate ? neg_inf()
                         : (row_max[mm][h] * p.scale_log2 + ::flashinfer::math::ptx_log2(s)) *
                               0.6931471805599453f;
        }
        float* row = p.o_partial + slot * T::HEAD_DIM;
#pragma unroll
        for (uint32_t md = 0; md < T::NUM_MMA_D; ++md) {
#pragma unroll
          for (uint32_t half = 0; half < 2; ++half) {
            // The fragment register holding row-half `h`, column group `half`:
            // the same {2h, 2h+1, 4+2h, 5+2h} map the softmax uses, read as
            // the two adjacent columns one lane owns.
            const uint32_t r = half * 4 + 2 * h;
            const uint32_t col = md * 16 + half * 8 + (lane % 4) * 2;
            *reinterpret_cast<float2*>(row + col) =
                make_float2(acc_o[mm][md][r], acc_o[mm][md][r + 1]);
          }
        }
      }
    }
    return;
  }

  if (p.lse != nullptr) {
#pragma unroll
    for (uint32_t mm = 0; mm < T::NUM_MMA_M; ++mm) {
#pragma unroll
      for (uint32_t h = 0; h < 2; ++h) {
        // One lane of each quad owns the row. The head is per-row under
        // packing, so the base pointer is formed here rather than once.
        if (lane % 4 == 0) {
          const int32_t qi = row_pos[mm][h];
          if (qi < p.seqlen_q) {
            float* lse = p.lse + batch * p.lse_stride_b + row_head[mm][h] * p.lse_stride_h;
            const float s = row_sum[mm][h];
            const bool degenerate = s == 0.f || s != s;
            lse[qi] = degenerate ? neg_inf()
                                 : (row_max[mm][h] * p.scale_log2 +
                                    ::flashinfer::math::ptx_log2(s)) *
                                       0.6931471805599453f;
          }
        }
      }
    }
  }

  // O goes out through Q's shared memory: the register layout stores four
  // bytes at a time, and global memory would rather have sixteen.
  __syncthreads();
  DTypeIn* smem_o = smem_q;
  smem_t<T::SWIZZLE> o_smem(smem_o);
#pragma unroll
  for (uint32_t mm = 0; mm < T::NUM_MMA_M; ++mm) {
#pragma unroll
    for (uint32_t md = 0; md < T::NUM_MMA_D; ++md) {
#pragma unroll
      for (uint32_t pair = 0; pair < 4; ++pair) {
        const uint32_t r = pair * 2;
        const uint32_t h = (r % 4) / 2;
        const uint32_t row = acc_row<T>(warp, mm, lane, h);
        const uint32_t b128_col = md * 2 + (r / 4);
        const uint32_t off =
            smem_t<T::SWIZZLE>::template get_permuted_offset<T::STRIDE>(row, b128_col);
        DTypeIn two[2];
        ::flashinfer::vec_cast<DTypeIn, float>::template cast<2>(two, &acc_o[mm][md][r]);
        reinterpret_cast<uint32_t*>(o_smem.base + off)[lane % 4] =
            *reinterpret_cast<const uint32_t*>(two);
      }
    }
  }
  __syncthreads();

  DTypeIn* o_base = reinterpret_cast<DTypeIn*>(p.o) + batch * p.o_stride_b;
  {
    constexpr uint32_t PER_ROW = T::COPY_THREADS_PER_ROW;
    constexpr uint32_t PER_PASS = T::COPY_ROWS_PER_PASS;
    const uint32_t rr = tid / PER_ROW;
    const uint32_t cc = tid % PER_ROW;
    const uint32_t elem_col = cc * upcast_size<DTypeIn>();
#pragma unroll
    for (uint32_t pass = 0; pass < T::TILE_M / PER_PASS; ++pass) {
      const uint32_t row = pass * PER_PASS + rr;
      const int32_t pm = q_row_base + static_cast<int32_t>(row);
      const int32_t position = map.pos(pm);
      if (position < p.seqlen_q) {
        const uint32_t off =
            smem_t<T::SWIZZLE>::template get_permuted_offset<T::STRIDE>(row, cc);
        o_smem.store_128b(off,
                          o_base + position * p.o_stride_s + map.head(pm) * p.o_stride_h +
                              elem_col);
      }
    }
  }
}

/// Fold the splits of one output row back into one attention.
///
/// A split-KV block computed attention over its own range of keys and
/// normalised by its own denominator. Range `s` therefore holds
/// `O_s = sum_{j in s} p_j v_j / d_s` alongside `lse_s = log d_s` (plus its
/// max, which `lse` already carries), and the answer is
/// `sum_s d_s O_s / sum_s d_s`. Weighting by `exp(lse_s - max_s lse_s)`
/// rather than by `d_s` is the same ratio with the largest weight pinned at
/// one, which is what keeps a long range from overflowing the sum.
///
/// A range that saw nothing has `lse_s = -inf` and so weight exactly zero,
/// and a row where EVERY range saw nothing — the fully-masked leading rows of
/// a bottom-right causal mask — comes out zero with `lse = -inf`, which is the
/// same definition the unsplit epilogue writes.
///
/// One block per output row, one thread per channel, so `HEAD_DIM` threads.
/// The split loop is read twice, once for the weights and once for the sum;
/// the second pass hits L1 for everything the first pass just touched.
template <class T>
__global__ __launch_bounds__(T::HEAD_DIM) void combine(
    const float* o_partial, const float* lse_partial, void* o, float* lse, int32_t num_splits,
    int32_t heads_q, int32_t seqlen_q, int32_t o_stride_b, int32_t o_stride_s, int32_t o_stride_h,
    int32_t lse_stride_b, int32_t lse_stride_h) {
  using DTypeIn = typename T::DTypeIn;
  const int64_t prow = blockIdx.x;
  const int32_t qi = static_cast<int32_t>(prow % seqlen_q);
  const int32_t rest = static_cast<int32_t>(prow / seqlen_q);
  const int32_t head = rest % heads_q;
  const int32_t batch = rest / heads_q;
  const uint32_t d = threadIdx.x;

  const float* lp = lse_partial + prow * num_splits;
  float m = neg_inf();
  for (int32_t s = 0; s < num_splits; ++s) m = fmaxf(m, lp[s]);

  float acc = 0.f;
  float denom = 0.f;
  if (m != neg_inf()) {
    const float* op = o_partial + prow * num_splits * T::HEAD_DIM + d;
    for (int32_t s = 0; s < num_splits; ++s) {
      const float w = ::flashinfer::math::ptx_exp2((lp[s] - m) * 1.4426950408889634f);
      denom += w;
      acc = __fmaf_rn(w, op[s * T::HEAD_DIM], acc);
    }
    acc *= ::flashinfer::math::ptx_rcp(denom);
  }

  DTypeIn out;
  ::flashinfer::vec_cast<DTypeIn, float>::template cast<1>(&out, &acc);
  reinterpret_cast<DTypeIn*>(o)[batch * o_stride_b + qi * o_stride_s + head * o_stride_h + d] = out;

  if (lse != nullptr && d == 0) {
    lse[batch * lse_stride_b + head * lse_stride_h + qi] =
        m == neg_inf() ? neg_inf()
                       : m + ::flashinfer::math::ptx_log2(denom) * 0.6931471805599453f;
  }
}

}  // namespace fa4
}  // namespace attn
}  // namespace pie
