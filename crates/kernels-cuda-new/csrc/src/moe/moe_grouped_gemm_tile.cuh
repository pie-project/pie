//===-- moe_grouped_gemm_tile.cuh - the short-K grouped GEMM, in CuTile ---===//
//
// The CuTile twin of `moe_grouped_gemm.cuh`: one `__tile_global__`, no
// launcher, no `wmma`, and no CUTLASS. It runs, it is exact, and **it is
// faster than either kernel pie fires for MoE decode today.** It is still
// carried as TEXT and declared no `Unit`, for a reason that is now purely
// about toolchain floors rather than about the kernel — §"Where this
// stands".
//
// # READ THIS FIRST: an earlier version of this header said the opposite
//
// Every "it loses" in the history of this file was measured on a kernel
// crippled by a workaround that should never have existed. The workaround
// carried bf16 through as `unsigned short` and widened it to fp32 inside
// the tile, because 16-bit float tiles appeared not to compile. They
// compile. The compiler was fine; the HEADERS were the wrong version, and
// the whole story is `.wiki/driver/cutile-16bit-header-trap.cu`.
//
// Widening to fp32 doubled the operand register footprint and turned off
// the bf16 tensor-core path:
//
//     16x64x32     widened  REG 224      native bf16  REG 92
//     64x64x32     widened  REG 255*     native bf16  REG 160
//                           * pinned, i.e. spilling
//
// So the "register cliff at kTileM=64" that this file reported as a
// property of the tile compiler was a property of the workaround. With
// native `__nv_bfloat16` operands and an fp32 accumulator — the tensor
// core's own shape, and what `cuda_tile.h` constrains `mma` to — the cliff
// is gone and so is every conclusion drawn from it.
//
// **This is the eleventh correction in this spike and by far the largest.**
// The rule it earns is the same one the previous ten earned, one level
// meaner: a workaround is a measurement hazard. Every number taken while
// one is in place is a number about the workaround.
//
// # Where it stands now, measured
//
// L40S sm_89, decode census: 318 aligned blocks, 106 live, 256 experts,
// N and K as MoE fires them. bf16 in, bf16 out, fp32 accumulate. Every
// entry below is exact — worst relative error 0 against an fp64 reference,
// at every shape and every tiling, not "within tolerance".
//
// At `kTileM = 16`, which is what `moe_align_decode` emits and the only
// height the WMMA twin accepts (`M == kFrag`):
//
//                        gate_up            down
//                        N=512 K=2048       N=2048 K=256
//     cutile (best)      0.324 ms           0.149 ms
//     wmma twin          0.858              0.214
//     cuBLAS, captured   0.972              0.449
//     cuBLAS, ideal      0.327              0.177
//
//     best tiling        32 x 128           32 x 64
//     registers          174                84
//
// **The tile kernel is the fastest of the three at both shapes** — faster
// even than the cuBLAS ideal, which is unattainable. The margin over the
// WMMA twin is 2.65x at gate_up and 1.44x at down; over cuBLAS as pie can
// actually call it, 3.00x and 3.01x.
//
// The two cuBLAS rows are the honest part and need stating plainly.
// `cuBLAS, ideal` is a batched call over the 106 LIVE blocks only —
// unattainable, because the batch count is a host argument baked into a
// captured graph and must therefore be the worst case. `cuBLAS, captured`
// is that worst case: all 318 blocks, including the 212 that are padding.
// `moe_grouped_gemm.hpp` reaches the same conclusion from the other end,
// measuring the batched path at 9.2 ms of a 41 ms step.
//
// So the fair reading is two-sided, and both sides matter:
//
//   * against what pie can actually run, this kernel wins by 2.4-2.8x;
//   * against an ideal that skips padding for free, it is within 6-7%.
//
// The second is the more interesting number. It says the early exit is
// nearly free and the kernel is near what a dense library achieves on the
// live work alone — which is exactly what a grouped GEMM is for.
//
// At `kTileM = 64`, `moe_dispatch.hpp`'s `kMoeAlignedBlockMax`:
//
//     gate_up   0.659 ms      down   0.320 ms      (64 x 64, REG 218)
//
// No spills. The previous version of this file measured 1.361 ms and 255
// registers here and called it a collapse; there is no collapse. These are
// NOT comparable to the wmma column above — the bench re-derives which
// blocks are live from the block height, so the two heights present
// different expert censuses and different weight traffic. They are here to
// show the cliff is gone, not to win a race.
//
// # Verified, on an L40S
//
//     compile     nvcc --tilecubin to a real cubin with SASS and a
//                 STT_FUNC entry; nvrtc 13.3.33 rc=0 on the same source
//     launch      grid (N / kTileN, max_blocks), loaded and run on this
//                 box's driver
//     numerics    worst relative error 0 over every configuration swept
//                 above, against an fp64 reference
//     early exit  padding blocks leave poison bytes untouched, so the skip
//                 is a measurement and not a claim
//     stability   the two headline numbers repeat to the millisecond's
//                 third digit over five runs (0.349 x5; 0.185, 0.185,
//                 0.185, 0.186, 0.182)
//
// Exact rather than close, and that is a property of the arithmetic and not
// of luck: bf16 operands enter the tensor core unwidened and the
// accumulate is fp32, which is what the WMMA twin does and what the fp64
// reference rounds to.
//
// # The 16-bit trap, named so it is not fallen into twice
//
// `cuda::tiles` accepts `__half` and `__nv_bfloat16`. If yours does not,
// the runtime headers are older than the tile compiler.
//
// CUDA 13.3's tile frontend injects `-D__NV_TL_BUILTIN__=__tile_builtin__`
// and 13.3's `cuda_bf16.h` / `cuda_fp16.h` / `cuda_fp8.h` / `cuda_tf32.h`
// carry that marker on the struct:
//
//     13.0   struct                   __CUDA_ALIGN__(2) __nv_bfloat16 {...}
//     13.3   struct __NV_TL_BUILTIN__ __CUDA_ALIGN__(2) __nv_bfloat16 {...}
//
// Without the marker the type is an ordinary two-byte aggregate, tiles of
// it lower as `tile<2xi8>`, and tile codegen dies with `"Unexpected element
// type in tile!"`. Adding the attribute by hand to a 13.0 header is the
// entire fix — a one-token A/B that pins the cause.
//
// This is easy to hit because the CUDA toolchain arrives as independently
// versioned pip wheels: `nvidia-cuda-nvcc` and `nvidia-cuda-nvrtc` can be
// 13.3 while `nvidia-cuda-runtime`, which owns these headers, is 13.0. No
// version check fires. The cheap detector is that `cuda_tf32.h` ships only
// in 13.3+; `tests/vendor_manifest.rs` gates on it.
//
// One consequence is structural rather than incidental: `cuda::tiles`
// constrains tile elements to the scalar types it knows, so this tree's own
// `device::bf16` is refused — `template constraint not satisfied` — even
// carrying `__tile_builtin__`. A tile kernel must name NVIDIA's
// `__nv_bfloat16`. That collides with `csrc/src/cuda_bf16.h`, which aliases
// that same name to `device::bf16` so FlashInfer stays byte-identical to
// upstream, so the two can never share a translation unit and NVIDIA's
// include directory must precede the tree's for this file. It includes
// nothing that wants the adapter, which is what keeps that tractable.
//
// Two earlier explanations were wrong and are recorded so they are not
// reached for again: that tile codegen keys on NVIDIA's canonical
// declaration (it keys on the marker), and that the float-to-bf16 narrowing
// was at fault (it was not). Both survived a round of probing because the
// harness matched the literal string `SUCCESS` inside its own label
// `compile rc=%d (0=SUCCESS)`, so every run reported success. **A probe
// that cannot fail has not measured anything.**
//
// # Where this stands, and what it is waiting for
//
// Not the kernel, not the driver, and not NVIDIA. Three pip wheels and one
// environment variable.
//
//   * **NVRTC 13.3.** The crate loads the system `libnvrtc`, 13.0.88 here,
//     whose `cuda_tile.h` is 60 lines and declares `print`.
//     `tests/units.rs` compiles every declared unit with that compiler, so
//     a unit added today fails the gate for the whole crate. NVRTC 13.3 is
//     a wheel and compiles this file clean.
//   * **13.3 runtime headers**, per the trap above. Also a wheel.
//   * **`tileiras` on PATH, with `CUDA_ROOT` set.** Also a wheel, 95 MB,
//     binary only — no library form, so pie shells out.
//
// **The JIT path works on this box's 13.0 driver.** Measured end to end,
// with a bf16 tile `mma` in the kernel:
//
//     1. nvrtc compiled
//     2. tile IR extracted, 6,314 bytes
//     3. tileiras assembled                       rc=0
//     4. cuModuleLoadData                         SUCCESS
//     5. cuModuleGetFunction                      SUCCESS
//     6. cuLaunchKernel                           SUCCESS
//     7. cuCtxSynchronize                         SUCCESS
//     8. numerics                                 EXACT, 0/1024 mismatches
//
// NVRTC hands back pure Tile IR — `.note.nv.tkinfo`, no `.text` — and this
// driver loads that image without assembling it, which is why
// `cuModuleGetFunction` answers `NOT_FOUND` if you stop there. Running
// `nvrtcGetTileIR` output through `tileiras` yourself closes the gap. Two
// earlier versions of this file were wrong about this: the first said the
// driver rejects tile images (it loads them), the second said standalone
// `tileiras` cannot assemble NVRTC output (it can).
//
// **`tileiras` requires `CUDA_ROOT` and does not say so.** Without it every
// input fails with a bare `error: failed to compile Tile IR program` —
// including nvcc's OWN `.tilebc`, byte-identical, same argv, same cwd, same
// binary. That symmetry is what located it: when the vendor's tool succeeds
// on an input and you fail on the same bytes, the difference is the
// environment. §23.18 has the bisect.
//
// Cold JIT latency here is 0.62-0.71 s end to end, of which `tileiras` is
// 0.18 s — a `PIE_HOME/cache` cost, paid once, in the same range as the
// NVRTC compiles the crate already does.
//
// So `families/moe.rs` declares no unit for this file only until those
// wheels are packaged. The unit is three lines and its options are already
// known: `-std=c++20 -enable-tile -default-device`, all three load-bearing
// — `cuda_tile.h` `#error`s below C++20, tile annotations are silently
// IGNORED with only a warning without `-enable-tile`, and the header's own
// unannotated helpers are host functions to NVRTC without
// `-default-device`. They belong on `Unit::opts`, which is what that field
// is for.
//
// # It scales with rows, and the win widens
//
// gate_up N=512 K=2048 at the tuned 32x128, against the WMMA twin, which is
// the kernel pie fires for MoE decode:
//
//     rows       tile       wmma       ratio
//      5,088    0.350 ms   0.928 ms   2.65x
//     10,176    0.770      2.292      2.98x
//     20,352    1.372      4.413      3.22x
//     40,704    2.744      8.541      3.11x
//
// So the decode result is not a small-shape artifact. With the widening
// workaround in place this same sweep read 1.25x to 1.52x; the shape of the
// curve survived the rewrite and its magnitude did not.
//
// # Tuning, which is per-shape and worth doing
//
// `kTileN` and `kTileK` are the whole tuning surface and the spread is
// wide — at gate_up, 0.349 ms to 1.112 ms across the sweep. The two
// production shapes want different tilings:
//
//     gate_up  N=512  K=2048     32 x 128     0.349 ms   REG 150
//     down     N=2048 K=256      32 x 32      0.185      REG 76
//
// The defaults below are 64 x 32, which is neither. They are left alone
// deliberately: this file has no `Unit`, so a default is documentation
// rather than a decision, and baking in one shape's winner would make the
// other shape's number look like the kernel's ceiling. Whoever rows this
// should set them per row, which is what `Unit::opts` is for.
//
// Direction of the sweep, for whoever retunes on another part: at gate_up
// (large K) time falls monotonically with `kTileK` to 128 and rises at 256;
// at down (small K) it is flat past 32. `kTileN = 32` wins at both, which
// is not obvious and is worth re-checking on a part with a different
// register file.
//
// # Reproducing any of the above
//
// Three pip wheels and no install, plus the runtime headers that are the
// point of the trap section:
//
//     nvidia-cuda-nvcc==13.3.73        nvcc --enable-tile --tilecubin
//     nvidia-cuda-nvrtc==13.3.33       libnvrtc that speaks tile
//     nvidia-cuda-tileiras==13.3.36    the Tile IR assembler nvcc shells
//                                      out to; must be on PATH or nvcc
//                                      reports only `tileiras: not found`
//     nvidia-cuda-runtime==13.3.29     cuda_bf16.h WITH the marker
//
// The 13.3 runtime include directory must come FIRST, before this tree's
// `csrc/src`, or `#include <cuda_bf16.h>` finds the adapter and the build
// dies on a redefinition of `__nv_bfloat16`.
//
//     nvcc -tilecubin -arch=sm_89 -std=c++20 --tile-only \
//          -I <13.3 runtime>/include -I <nvcc>/include \
//          -I <nvcc>/include/crt -I crates/kernels-cuda-new/csrc/src \
//          -DPIE_TILE_M=16 -DPIE_TILE_N=32 -DPIE_TILE_K=128 ...
//
// # Why it is worth keeping warm
//
// The kernel is ~90 lines of code and no launcher. It expresses a grouped
// GEMM with a data-dependent early exit — the thing cuBLAS structurally
// cannot do under graph capture and the thing CUTLASS needs a host-side
// `Params` expansion to do — as a plain kernel with an `if` and a `return`.
// That is the property `new-horizon.md` wants from the whole redesign:
// launch logic in Rust, kernels that are just kernels.
//
// It is not yet a case for replacing anything. The MoE decode path is
// guarded (`moe_grouped_gemm_bf16_supported` demands `M == kFrag` and
// `K <= kShortK = 512`), the CUTLASS island fuses permute + fc1 + act + fc2
// and is a different weight class, and none of that was re-measured against
// this version. What changed is that the kernel is no longer disqualified
// on speed, which is the only reason it was disqualified before.
//
// # The island, re-raced — the GEMM gap closed, the fusion gap did not
//
// `flashinfer_cutlass_moe_bf16` fuses permute + fc1 + activation + fc2 +
// finalise. Timed directly rather than quoted (`island_bench.cu` links it
// out of `libpie_kernels_cuda.a`), on §20.8's shape — 318 tokens, hidden
// 2048, inter 256, 256 experts, top-k 8, so 2,544 routed rows and 805 MB of
// weights. Both censuses repeat to the third digit.
//
//                                      island       two tile GEMMs
//                                      (5 stages)   (2 stages)
//     256 experts, 16 rows each        1.241 ms     1.328 ms   kTileM=16
//     106 experts, 24 rows each        0.581        0.933      kTileM=32
//
// §20.8 recorded the island at 1.9x faster than two tile GEMMs while doing
// strictly more work. It is now within 7% of them at the first census, and
// on weight streaming they have converged — 649 GB/s against 640 and 548.
//
// At the DECODE census it is still 1.6x faster, and that gap widens as the
// expert count falls, because fewer experts over the same rows is more rows
// per expert, which is exactly the reuse a fused batched pass converts into
// bandwidth. Held at 106 experts the island streams 333 MB of live weights
// at 573 GB/s where these GEMMs manage ~358.
//
// That is a property of this SOURCE, not of the tile compiler: the kernel
// reads `W[e]` once per block and round-trips the intermediate through HBM,
// where the island reads each expert once and keeps fc1's output resident.
//
// **The fused version was written, and it is slower still.** One
// `__tile_global__` doing fc1 + swiglu + fc2, one expert block per CUDA
// block owning the whole fc2 output panel, intermediate never stored:
// correct to 0.42% — 2^-8, the bf16 rounding floor — and 1.778 ms against
// the unfused pair's 0.933 and the island's 0.581.
//
// The cause is shared memory, not registers. The tile compiler stages
// `partition_view::load` operands through shared, and a fused kernel's
// working set saturates it:
//
//     unfused grouped GEMM    REG 150    SHARED 16,384
//     fused kernel            REG 233+   SHARED 92,168 - 99,336
//
// which is ONE block per SM out of a 100 KB budget, and 106-318 blocks each
// alone on an SM cannot hide HBM latency. It is pinned there: sweeping
// `FNK`, `FN2` and `FM` never drops below 92 KB, and two structurally
// different versions (a `cat`/`extract` intermediate, and four chunks kept
// separate so neither appears) differ by 27% and not by a factor.
// `cuda::tiles` exposes no shared-memory scratch, no `insert`, and no
// occupancy control, so there is no third version to try.
//
// So: wherever pie fires a grouped GEMM as a GEMM this kernel wins;
// wherever pie fires the island, the island still does, and fusion is not
// the way in. `.wiki/driver/new-horizon.md` §23.17 has the sweep.
//
// # Where else in this tree CuTile applies, surveyed
//
// The win below is not "CuTile is fast". It is three properties holding at
// once, and each has been measured by removing it:
//
//   1. **tensor-core shaped** — a real `mma`;
//   2. **ragged, with a data-dependent early exit** — what cuBLAS
//      structurally cannot do under graph capture, since the batch count is
//      a host argument baked into the graph and must be the worst case;
//   3. **unfused** — one output per block, so shared stays at 16 KB.
//
//     ragged tensor-core GEMM   this file            2.65x / 1.44x FASTER
//     row reduction             rmsnorm_tile.cuh     1.51x / 1.59x FASTER
//     small argmax + softmax    topk_softmax_tile    1.28x FASTER at decode,
//                                                    1.19x slower at 2k rows
//     elementwise, L2-resident  swiglu_tile.cuh      1.53x FASTER
//     elementwise, HBM-bound    swiglu_tile.cuh      4% slower, both at
//                                                    ~77% of peak
//     fused multi-stage         moe_fused_tile.cuh   1.5x slower at 106
//                                                    blocks, 2.3x FASTER at
//                                                    1,696 -- the grid, not
//                                                    the fusion
//     scan / cross-lane         --                   not expressible
//
// The `topk_softmax_tile` row is the strongest of these, because its
// opponent is not a first draft: `moe/topk_softmax.cuh` had already measured
// its own block form at 4.39 us against a 0.54 us floor and rewritten it
// into an all-`__shfl_xor` warp form. CuTile beats THAT at decode, with
// identical expert indices.
//
// Two earlier versions of this section were wrong and in opposite
// directions. The first claimed a CuTile reduction is 15-30% slower; it
// measured a naively-written kernel, and in NVIDIA's idiom the same
// reduction is 1.5x FASTER. The second concluded from that bad number that
// only one kernel here was worth a tile alternative. It is not.
//
// What survives as a real boundary is the ROOFLINE, which is the elementwise
// row above: at 25 MB the tile swiglu runs at 3,273 GB/s against 2,135, and
// at 805 MB both sit at ~77% of the L40S's ~864 GB/s peak and the tile
// version is 4% behind. **No programming model makes a kernel at the memory
// roofline faster.** So the wins in the reduction and elementwise classes
// are latency and occupancy wins, and they appear where the data is cached
// — which at decode is everywhere.
//
// cuBLAS wins where cuBLAS is not handicapped. That one has not moved.
//
// "Fusing costs" HAS moved, and it was the third wrong explanation in this
// file's history. It was measured at one grid -- 106 blocks on a 142-SM
// part, under one block per SM -- where nothing is fusion-bound because
// everything is latency-bound. Past 212 blocks the fused kernel wins, and at
// 1,696 it is 2.3x ahead. `moe_fused_tile.cuh` carries the sweep.
//
// Also excluded, and not as a matter of taste: `cuda::tiles` has `sum`,
// `prod`, `reduce_min/max`, `transpose`, `broadcast`, `iota`, `permute`,
// `select`, the elementwise math and atomics — and **no scan, and no
// cross-lane escape hatch** (zero `__shfl`, `__syncthreads` or cooperative
// groups anywhere in `cuda_tile.h`). That rules out `ssm/*` (recurrences),
// `moe_dispatch` and `layout/*` (CSR, prefix sums, compaction),
// `sample/argmax` (sorting networks) and `quant/*` (PRMT bit manipulation,
// and M=1 GEMVs that are weight-bandwidth-bound so tensor cores do not
// apply).
//
// `attn/attention_naive*.cuh` IS the right shape — Q@K^T, softmax, @V — and
// should still not move: those are the REFERENCE kernels, deliberately
// untiled with fp32 accumulators, and their value is being trustworthy
// enough that flashinfer can be checked against them.
//
// So the set of kernels that have a worthwhile CuTile ALTERNATIVE is not one
// file. Each of these is an addition beside the incumbent, with a
// `*_tile_preferred` predicate stating when to fire it; none of them
// replaces anything, and the incumbents remain the fallback for every
// toolchain that cannot compile a tile kernel -- which today is every
// toolchain this crate loads. Measured or structurally clear:
//
//     moe/moe_grouped_gemm.cuh    167 lines   2.65x / 1.44x
//     norm/rmsnorm.cuh            747         1.51x / 1.59x
//     moe/topk_softmax.cuh        575         1.28x at decode
//     mlp/swiglu.cuh              685         1.53x cached, ~par at roofline
//     norm/elementwise.cuh         81         elementwise class
//     norm/add_bias.cuh           107         elementwise class
//
// ~2,400 lines of C++ that now have a ~310-line alternative beside them,
// before `attn/` and `vision/` are re-examined at all. The C++ stays: an
// alternative that cannot be selected on a machine without NVRTC 13.3 is
// not an alternative, it is a removal.
//
// The STRUCTURAL exclusions are the durable half and they stand: no scan and
// no cross-lane escape hatch in `cuda_tile.h` rules out `ssm/*` (3,342 lines
// of recurrence), `moe_dispatch` and `layout/*` (CSR, prefix sums,
// compaction), `sample/argmax` (sorting networks) and `quant/*` (PRMT bit
// manipulation, and M=1 GEMVs that are weight-bandwidth-bound).
// `attn/attention_naive*` is still the reference flashinfer is checked
// against and should still not move. §23.21 has the re-run.
//
// # What has NOT been re-measured since the rewrite
//
// Stated so nobody quotes stale numbers as this file's own:
//
//   * anything at a part other than an L40S. The fused result in particular
//     is a shared-capacity finding, so a part with a larger shared budget
//     per SM changes its arithmetic and nothing else about it;
//   * fp8 and the quantised routes, untouched here as in §20.
//
// # The header collision, which is real and independent of all the above
//
// `cuda_tile.h` forward-declares `struct __nv_bfloat16;` at global scope.
// This tree's `csrc/src/cuda_bf16.h` answers the same name with
// `using __nv_bfloat16 = device::bf16;`. A struct declaration and a type
// alias cannot share a name in one translation unit, so `cuda_bf16.h`,
// `pie_mma.cuh` and this file can never enter one unit. This file includes
// neither, which is what keeps that from mattering.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "pie_device.cuh"

#include <cuda_bf16.h>
#include <crt/cuda_tile.h>

namespace pie_cuda_driver::kernels::moe::device {

namespace ct = ::cuda::tiles;

using ::pie_cuda_driver::kernels::device::bf16;
using ::pie_cuda_driver::kernels::device::i32;


/// Rows of C one block covers — the aligned block's row count, as
/// `moe_align_decode` builds it. Matches `moe_grouped_gemm.cuh`'s `kFrag`.
/// Overridable so the same source can serve a taller block than the decode
/// aligner's, which is what a prefill-shaped fire wants. 16 is the decode
/// value and the WMMA twin's `kFrag`.
#ifndef PIE_TILE_M
#define PIE_TILE_M 16
#endif
constexpr int kTileM = PIE_TILE_M;
/// Columns of C one block covers.
///
/// 64 so the grid is the WMMA twin's exactly — `(N / 64, max_blocks)` — which
/// is what lets a bench swap one kernel for the other without touching its
/// launch. 32 and 16 measure the same to within 0.008 ms (see the sweep in
/// this file's header), so the tie is broken by launch compatibility rather
/// than by the clock.
#ifndef PIE_TILE_N
#define PIE_TILE_N 64
#endif
constexpr int kTileN = PIE_TILE_N;

/// The K step of the mainloop, and the one number that matters.
///
/// 32 is not a preference, it is a cliff: 16 and 64 both measure ~1.1-1.4 ms
/// against 32's 0.687, on a sweep where nothing else changed. Sixteen bf16
/// elements are 32 bytes and thirty-two are 64 bytes — half a sector and a
/// full one — so the shape that wins is the one whose innermost read is a
/// whole transaction. That is the same finding the WMMA twin's seventh and
/// last optimisation records in its own words ("kChunk=64 so each row read
/// is a full cache line"), reached here by changing one number.
///
/// Both are overridable so the sweep that found this is repeatable.
#ifndef PIE_TILE_K
#define PIE_TILE_K 32
#endif
constexpr int kTileK = PIE_TILE_K;

/// Whether this ALTERNATIVE should be preferred to `moe_grouped_gemm.cuh`.
///
/// At the shapes pie fires, yes -- it is the fastest of the three
/// implementations at both, including against a cuBLAS batched call that
/// only ran the live blocks, which no captured graph can schedule:
///
///                        gate_up N=512 K=2048   down N=2048 K=256
///     this file              0.324 ms               0.149 ms
///     moe_grouped_gemm       0.858                  0.214
///     cuBLAS, captured       0.972                  0.449
///     cuBLAS, ideal          0.327                  0.177
///
/// Unconditional, and that is swept rather than assumed. Across four orders
/// of magnitude of grid, at gate_up N=512 K=2048, it is ahead everywhere:
///
///     blocks       tile        wmma        ratio
///          1     0.005 ms     0.017 ms     3.40x
///         16     0.013        0.018        1.38x
///        128     0.388        1.149        2.96x
///        318     0.324        0.861        2.66x
///      1,272     1.287        4.070        3.16x
///      5,088     5.127       16.697        3.26x
///
/// Worst relative error 0 at every one. The dip in the middle is the grid
/// crossing the SM count from below; it never crosses 1.0x.
///
/// The divisibility conditions are the ones the kernel's `static_assert`s
/// state; they are repeated here so a caller can ask before instantiating.
///
/// This does NOT say the WMMA twin should be deleted. It is the fallback for
/// every toolchain that cannot compile a tile kernel, which today is every
/// toolchain this crate actually loads --
/// see "Where this stands" above.
constexpr bool moe_grouped_gemm_tile_preferred(int N, int K)
{
    return N > 0 && K > 0 && (N % kTileN) == 0 && (K % kTileK) == 0;
}

/// `C[b] = A[b] @ W[ids[b]]^T` over `kTileM`-row blocks, skipping padding.
///
/// Grid is `(N / kTileN, max_blocks)`, exactly the WMMA twin's, so a bench
/// can swap one for the other without touching its launch.
///
/// * `a`    `[max_blocks * kTileM, K]` row-major
/// * `w`    `[num_experts, N, K]` row-major
/// * `c`    `[max_blocks * kTileM, N]` row-major
/// * `ids`  `[max_blocks]`, `< 0` marks padding
///
/// `N % kTileN == 0` and `K % kTileK == 0` are the caller's to guarantee;
/// the support predicate states them.
///
/// # Every operand crosses as `__nv_bfloat16`
///
/// NVIDIA's type, from NVIDIA's header, because `cuda::tiles` constrains
/// tile elements to the scalar types it knows: a two-byte struct of this
/// tree's own — `device::bf16`, say — is refused as `template constraint
/// not satisfied` whether or not it carries `__tile_builtin__`. So the row
/// names the prelude's `bf16` through `T` and the two meet at the
/// `reinterpret_cast` below, which is the same boundary the WMMA twin's
/// fragments cross for the same reason.
///
/// This requires CUDA **13.3 or newer runtime headers**. Older ones declare
/// `__nv_bfloat16` without the `__NV_TL_BUILTIN__` marker the tile frontend
/// keys on, and every 16-bit tile then dies inside tile codegen with
/// `"Unexpected element type in tile!"` —
/// `.wiki/driver/cutile-16bit-header-trap.cu` is that whole story.
/// The kernel's N and K are TEMPLATE parameters, not arguments.
///
/// They are model constants — gate_up is N=512 K=2048, down is N=2048 K=256
/// — and a JIT instantiates per shape, which is what `Unit::opts` is for.
/// The cost of spelling them `dynamic_extent` instead was measured and it is
/// not small: 0.349 ms against 0.324 at gate_up and 0.185 against 0.149 at
/// down, because static extents let the tile compiler compute every address
/// at compile time and stage far more deeply. Its shared-memory footprint
/// goes from 16 KB to 96 KB doing so, and is FASTER for it — the budget
/// being used is not the same thing as occupancy collapsing.
template <class T, int N, int K>
__tile_global__ void moe_grouped_gemm_tile(
    const T* __restrict__ a_in,
    const T* __restrict__ w_in,
    T* __restrict__ c_in,
    const i32* __restrict__ ids)
{
    static_assert(sizeof(T) == 2, "bf16 only, as the WMMA twin is");
    static_assert(N % kTileN == 0, "the caller guarantees this; state it");
    static_assert(K % kTileK == 0, "likewise");

    using elem = __nv_bfloat16;

    // The rows are 16-byte aligned by construction — every buffer here comes
    // from the driver's arena — and saying so lets the compiler use the wide
    // loads. NVIDIA's own `matmul.cuh` opens the same way.
    const auto* a = ct::assume_aligned<16>(reinterpret_cast<const elem*>(a_in));
    const auto* w = ct::assume_aligned<16>(reinterpret_cast<const elem*>(w_in));
    auto* c = ct::assume_aligned<16>(reinterpret_cast<elem*>(c_in));

    const int n_blk = static_cast<int>(ct::bid().x);
    const int b = static_cast<int>(ct::bid().y);

    const int e = ids[b];
    if (e < 0) return;  // padding block: the whole point of this kernel

    const elem* a_blk = a + static_cast<long long>(b) * kTileM * K;
    elem* c_blk = c + static_cast<long long>(b) * kTileM * N;
    const elem* w_e = w + static_cast<long long>(e) * N * K;

    // A: [kTileM, K] row-major.
    auto pA = ct::partition_view{
        ct::tensor_span{a_blk, ct::extents<unsigned, kTileM, K>{}},
        ct::shape<kTileM, kTileK>{}};
    // W^T: [K, N] as a COLUMN-major view of W's [N, K] row-major bytes — the
    // same aliasing the WMMA twin gets from a `col_major` b-fragment at
    // `ld = K`, stated in the type instead of in a fragment argument.
    auto pW = ct::partition_view{
        ct::tensor_span<const elem, ct::extents<unsigned, K, N>, ct::layout_left>{
            w_e, ct::extents<unsigned, K, N>{}},
        ct::shape<kTileK, kTileN>{}};
    // C: [kTileM, N] row-major.
    auto pC = ct::partition_view{
        ct::tensor_span{c_blk, ct::extents<unsigned, kTileM, N>{}},
        ct::shape<kTileM, kTileN>{}};

    auto acc = ct::zeros<ct::tile<float, ct::shape<kTileM, kTileN>>>();

    // Native bf16 `mma`: bf16 operands, fp32 accumulator, which is the tensor
    // core's own shape and `cuda_tile.h`'s stated constraint
    // (`is_bfloat16_v<Ele> && is_float32_v<Acc>`). The operands are NOT
    // widened first — doing that cost 224 registers here against 92, and 255
    // with spills against 160 at `kTileM = 64`.
    //
    // `irange` over a CONSTEXPR trip count, not `K / kTileK` read at run
    // time. A `latency=1` hint on the two loads was tried and measures
    // identically to three digits: an `mma` loop is already the shape the
    // scheduler recognises. It is the loop BOUND that had to be static.
    constexpr int kTiles = K / kTileK;
    for (auto k : ct::irange(0, kTiles)) {
        acc = ct::mma(pA.load(0, k), pW.load(k, n_blk), acc);
    }

    pC.store(ct::element_cast<elem>(acc), 0, n_blk);
}

}  // namespace pie_cuda_driver::kernels::moe::device
