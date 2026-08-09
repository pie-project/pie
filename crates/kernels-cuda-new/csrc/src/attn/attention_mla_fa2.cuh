//===-- attention_mla_fa2.cuh -------------------------------------*- CUDA -*-===//
//
// The FA2 MLA unit's NVRTC root: every type its two name expressions need, and
// NOTHING ELSE.
//
// # What this file is, and what it deliberately is not
//
// `csrc/src/attn/fa2.cuh` beside it is the pattern and its header is the
// design; this is the same shape for a second upstream kernel. It is the
// `#include` list plus four `using` declarations, one alias template and one
// `__device__` echo. It holds no `__global__` of ours, no launcher, no
// `<<<>>>` and no host function -- the `__global__` it exists to instantiate
// is upstream's `flashinfer::mla::BatchMLAPagedAttentionKernel` (`mla.cuh:879`)
// and the launch arithmetic that used to surround it is
// `kernels_cuda_new::x::attn::mla_fa2` in Rust.
//
// The shape being replaced is `crates/kernels-cuda/csrc/src/attn/attention_mla.cu`,
// whose two `<<<>>>` -- reached through `flashinfer::mla::BatchMLAPagedAttention`
// and `attention_mla_naive.cuh` -- are the last nvcc-compiled launches in the
// workspace. That file is host program throughout: zero `__global__`, zero
// `__device__`, zero `<<<>>>` of its own.
//
// # Why the aliases exist at all
//
// A row names its instantiation as ONE string handed to
// `nvrtcAddNameExpression`. `BatchMLAPagedAttentionKernel` takes **two**
// template arguments and not one --
//
//   template <typename KTraits, typename Params>            // mla.cuh:879
//   __global__ __launch_bounds__(KTraits::NUM_THREADS)
//   void BatchMLAPagedAttentionKernel(const __grid_constant__ Params params);
//
// -- and `KTraits` is an eleven-parameter `KernelTraits` (`mla.cuh:78-81`) of
// which SEVEN do not vary across this unit: the four types, and the three
// head-dimension constants MLA fixes at 512/64/64. Four vary, and all four are
// numbers or `bool`s the host derives. `Traits` below is that expression with
// the seven filled in, so a row is `<Traits<causal, stages, shard, tile_kv>,
// Params>` rather than a paragraph of C++.
//
// # §3.2's hazard is live here: there are TWO `KernelTraits` in the closure
//
// `prefill.cuh:159` declares `flashinfer::KernelTraits` whose first parameter
// is a `MaskMode`; `mla.cuh:81` declares `flashinfer::mla::KernelTraits` whose
// first parameter is `bool CAUSAL_`. `mla.cuh:1124` spells it unqualified and
// enclosing-namespace lookup picks the right one -- a transcription writing
// the QUALIFIED name has to know which, and both are nameable. Every mention
// below is `::flashinfer::mla::KernelTraits`, fully qualified, deliberately.
// The two are not distinguishable by arity either: fifteen against eleven is a
// substitution failure and not a diagnostic that names the confusion.
//
// # The three residency facts, which no launch path can check
//
// `runtime::module::fire_ex` takes a grid and a `cooperative` flag and has no
// way to see any of these. A caller that gets one wrong gets a HANG, not an
// error, so they are stated where the kernel is declared rather than only
// where it is launched (`x::attn::mla_fa2` repeats them beside the Rust).
//
//  1. **The grid is `num_sm` blocks and is resident by construction.** It is
//     not an occupancy query. `scheduler.cuh:1607-1608` sets
//     `num_blks_x = cluster_size` (1 or 2) and `num_blks_y = num_sm /
//     cluster_size`, so the product is exactly `num_sm` and every block of a
//     cooperative launch is co-resident because the PLAN was built that way.
//     `driver-cuda/src/fire/flashinfer_decode.rs:1860-1885` claims otherwise
//     and is wrong; the plan is `plan::mla`'s `Schedule`, already Rust.
//  2. **`sizeof(KTraits::SharedStorage)` is exactly the arm's own threshold
//     literal** -- 221 696, 147 968, 92 672, measured under NVRTC 13.0 for all
//     three arms, align 16, and `CAUSAL` changes none of them. That is why
//     `DISPATCH_SMEM_CONFIG` (`mla.cuh:1100-1120`) can compare a DEVICE
//     property against those literals and then size the launch with a
//     `sizeof`: the two agree by upstream's construction and by nothing that
//     checks. `smem_bytes_mla` below is the compiler's own answer, exported so
//     a reader can compare rather than trust -- see the echo's own comment.
//  3. **The launch must be cooperative.** `mla.cuh:1061` calls `grid.sync()`
//     between the two stages, `mla.cuh:1132` launches through
//     `cudaLaunchCooperativeKernel`, and pie's `cooperative_groups.h` shim
//     resolves `this_grid().sync()` to NVIDIA's own
//     `cudaCGGetIntrinsicHandle(scopeGrid)` / `cudaCGSynchronize(handle, 0)`
//     pair. A NON-cooperative launch of this kernel is not an error at any
//     layer; it is a deadlock in the second stage.
//
//     That pair lowers to `.extern .func` and **whether `cuModuleLoadData`
//     resolves it is UNMEASURED** -- the shim carries the marker
//     `CG_SYNC_RESOLUTION_UNMEASURED` and that is the line to read first if
//     MLA's second stage fails at module load. `libcudadevrt.a` contains no
//     `cudaCG*`, and it contains no `cudaGraphSetConditional` either, which is
//     the case where the driver resolved it anyway.
//
// # What this unit needs from NVRTC, and what it does not
//
// `--device-as-default-execution-space`, and it is **measured, not assumed**.
// Without it this root is rejected sixteen times: `csrc/shim/type_traits:253`,
// seven sites in `cascade.cuh` and eight in `prefill.cuh`, all
// *"A function without execution space annotations ... is considered a host
// function"*. `mla.cuh:33` includes `prefill.cuh`, which is the same closure
// `families::fa2` compiles and the same flag it passes (`families/fa2.rs:302`),
// so this is that entry's third instance rather than a new class. The flag is
// per-unit and not global for the reason `unit::Unit::options` gives: it would
// silently compile OUR unannotated host helpers onto the device.
//
// It needs `Headers::LibraryAndVendor` for `fa2.cuh`'s reason -- the whole
// vendored FlashInfer closure -- and `csrc/src/unit.rs`'s `DEMANDS` says so
// under the exact key `attn/attention_mla_fa2`. Exact and not a prefix:
// there is one unit and a rename should be a compile error.
//
// It needs no toolkit include path. With the flag, the whole root compiles
// clean against `csrc/{shim,vendor,src}` alone, and all six kernel name
// expressions plus all three `&`-prefixed echoes lower (`compute_89`,
// 2.3 MB of PTX, six `.entry`).
//
// # Every name below arrives from `carried.rs`, not from a search path
//
// This crate's NVRTC passes NO `-I`. `fa2.cuh`'s header is the full account;
// the consequence here is that `"attn/flashinfer/attention/mla.cuh"` resolves
// because `csrc/src/attn/flashinfer/attention/mla.cuh` is carried under
// exactly that name, and that it is QUOTED, so
// `tests/layers.rs::every_include_reachable_from_a_unit_resolves` — which
// walks quoted includes only — now covers it. It did not while the spelling
// was `<flashinfer/attention/mla.cuh>`.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cstdint>
#include <cuda_bf16.h>

#include "attn/flashinfer/attention/mla.cuh"
#include "attn/flashinfer/attention/mla_params.cuh"

namespace pie_cuda_driver::kernels::attn::mla_fa2 {

// ── The four types that do not vary ─────────────────────────────────────────
//
// bf16 throughout, on the Q, the KV cache and the output alike. The FP8 KV
// path exists upstream and is reached by giving `DTypeKV` an FP8 type and the
// two per-tensor scales in `MLAParams` a value other than 1.0; nothing here
// instantiates it, and `x::attn::mla_fa2::pack` writes both scales as 1.0
// EXPLICITLY rather than relying on `mla_params.cuh`'s default member
// initialiser, because a Rust struct literal has no such thing and a zeroed
// pair scales every value to zero. §3.2's two-formats-one-width hazard is
// live in this family and this is where it would land.
using DTypeQ = __nv_bfloat16;
using DTypeKV = __nv_bfloat16;
using DTypeO = __nv_bfloat16;
using IdType = std::int32_t;

/// The by-value aggregate the kernel's single parameter is.
///
/// `MLAParams<DTypeQ, DTypeKV, DTypeO, IdType>` -- 288 bytes, align 8, as
/// NVRTC lays it out with pie's `cuda::fast_mod_div`. **It is 248 under nvcc
/// with CCCL's**, because `flashinfer::uint_fastdiv` wraps a class whose
/// interior the two spell differently at the same `sizeof`; `csrc/shim/cuda/cmath:265-273`
/// states the rule and `x::attn::mla_params::UintFastdiv::new` carries the
/// warning. Nothing may fill this struct on one path and launch it on the
/// other.
using Params = ::flashinfer::MLAParams<DTypeQ, DTypeKV, DTypeO, IdType>;

// ── The three head dimensions MLA fixes ─────────────────────────────────────
//
// `attention_mla.cu` instantiates `BatchMLAPagedAttention<MASK, 512, 64>` and
// nothing else: 512 is DeepSeek's `kv_lora_rank`, 64 its `qk_rope_head_dim`.
// `CTA_TILE_Q` is 64 at every call site (`mla.cuh:1125`, written as a comment
// argument there rather than a name), and it is a TILE and not a head
// dimension -- it is here because it is the fourth constant that never varies.
inline constexpr std::uint32_t HEAD_DIM_CKV = 512;
inline constexpr std::uint32_t HEAD_DIM_KPE = 64;
inline constexpr std::uint32_t CTA_TILE_Q = 64;

/// `flashinfer::mla::KernelTraits` with the seven invariant arguments filled.
///
/// The four that remain are `DISPATCH_SMEM_CONFIG`'s three (`mla.cuh:1100-1120`)
/// plus the mask, in `KernelTraits`' own parameter order:
///
/// ```text
/// CAUSAL_, NUM_STAGES_, QK_SHARD_, HEAD_DIM_CKV_, HEAD_DIM_KPE_,
/// CTA_TILE_Q_, CTA_TILE_KV_, DTypeQ_, DTypeKV_, DTypeO_, IdType_
/// ```
///
/// `CAUSAL` is `MASK_MODE == MaskMode::kCausal` and nothing else: the launcher
/// refuses `kCustom` with `cudaErrorNotSupported` before it forms a traits
/// type, so the mask is a `bool` in this unit and a three-valued enum nowhere.
///
/// `NUM_THREADS` is 256 for every instantiation -- `KernelTraits` fixes it,
/// and `mla.cuh:1128` spells the same 256 as `dim3(32, 4, 2)`. The block is
/// not a free parameter and `x::attn::mla_fa2::BLOCK` states it once.
template <bool CAUSAL, std::uint32_t NUM_STAGES, bool QK_SHARD, std::uint32_t CTA_TILE_KV>
using Traits =
    ::flashinfer::mla::KernelTraits<CAUSAL, NUM_STAGES, QK_SHARD, HEAD_DIM_CKV, HEAD_DIM_KPE,
                                   CTA_TILE_Q, CTA_TILE_KV, DTypeQ, DTypeKV, DTypeO, IdType>;

/// The compiler's own `sizeof(KTraits::SharedStorage)`, exported so the three
/// threshold literals can be compared against it rather than trusted.
///
/// `fa2.cuh`'s `smem_bytes_paged` is the precedent and its header carries the
/// full argument, including the one that surprises: this is a name expression
/// **only with a leading `&`**. `nvrtcAddNameExpression` refuses
/// `smem_bytes_mla<KT>` and accepts `&smem_bytes_mla<KT>`, because a
/// function's name is its address and a variable's is not.
///
/// The comparison this makes possible is narrower here than there, and more
/// important. `fa2::PrefillGeometry::smem_bytes` RE-DERIVES a layout in Rust,
/// so its echo catches an arithmetic error. Here the Rust states three
/// LITERALS copied out of `DISPATCH_SMEM_CONFIG`'s own comparisons, and the
/// thing that can go wrong is upstream changing `SharedStorage` without
/// changing the thresholds -- at which point the launch is sized correctly by
/// the `sizeof` and the ARM IS CHOSEN WRONG, silently, on a device whose smem
/// falls between the old literal and the new size.
template <class KTraits>
__device__ unsigned smem_bytes_mla =
    static_cast<unsigned>(sizeof(typename KTraits::SharedStorage));

}  // namespace pie_cuda_driver::kernels::attn::mla_fa2
