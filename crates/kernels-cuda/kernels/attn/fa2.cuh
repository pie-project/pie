//===-- fa2.cuh ---------------------------------------------------*- CUDA -*-===//
//
// The FA2 lattice's NVRTC root: every type a row's name expression needs, and
// NOTHING ELSE.
//
// # What this file is, and what it deliberately is not
//
// It is the `#include` list plus a dozen `using` declarations. It holds no
// `__global__` of ours, no launcher, no `<<<>>>`, no host function -- the
// `__global__`s it exists to instantiate are upstream's
// (`flashinfer::BatchDecodeWithPagedKVCacheKernel`,
// `flashinfer::BatchPrefillWithPagedKVCacheKernel`) and the launch arithmetic
// that used to surround them is `kernels_cuda::fa2` in Rust.
//
// That split is the whole of north-star §5 step 8. `attention_flashinfer_common.cuh`
// is the shape being replaced: the same includes, plus
// `AttnHd<HEAD_DIM>` whose members are HOST launchers -- `<vector>`,
// `<stdexcept>`, `cudaFuncSetAttribute`, `cudaLaunchKernel`. NVRTC compiles
// device text; the host half went to Rust rather than through the compiler.
//
// **That file is no longer in the tree.** It sat beside this one until it
// left `csrc/` for `kernels-cuda/spec/`, because a host header in a directory
// that is carried WHOLE is 45 KB of `std::vector` in every `includeNames[]`
// array, and it has since been deleted with that directory. So the
// `attention_flashinfer_common.cuh:NNN` citations below resolve to nothing.
// That is the same bargain `kernels_cuda::attn::fa2`'s module doc spells out
// for the Rust half of this port: a line number is provenance, not the
// answer, and every citation below has the thing it points at written out
// beside it.
//
// # Why the aliases exist at all
//
// A row names its instantiation as ONE string handed to
// `nvrtcAddNameExpression`. Upstream's decode kernel takes nine template
// arguments and its prefill kernel takes a `KernelTraits` pack of fifteen, and
// seven of the decode nine are integers the HOST computes from head_dim and
// the GQA group (`decode.cuh:761-770`). Those integers are spelled by the row,
// because they are what varies across the lattice; the two TYPES that do not
// vary -- the attention variant and the params struct -- are spelled once
// here, so a row is a tuple of numbers and a variant name rather than a
// paragraph of C++.
//
// # The `__device__` echoes, and the one number a Rust port cannot own
//
// `BatchPrefillWithPagedKVCacheDispatched` sizes its dynamic shared memory as
// `sizeof(typename KTraits::SharedStoragePaged)` (`prefill.cuh:4300`) -- a C++
// STRUCT LAYOUT over a union of three alternatives and five trailing
// `alignas(16)` arrays (`prefill.cuh:98-147`). `fa2::PrefillGeometry::smem_bytes`
// re-derives it in Rust, field by field, and a re-derived layout is the one
// kind of constant that can be wrong without anything saying so: too small and
// the kernel reads past its buffer, too large and it silently loses a
// block per SM.
//
// So the compiler's own answer is exported beside it. `smem_bytes_paged<KT>`
// is a `__device__` variable template, one per instantiation, initialised to
// the `sizeof` NVRTC computed. It is a name expression, **but only with a
// leading `&`**: `nvrtcAddNameExpression` refuses `smem_bytes_paged<KT>` and
// accepts `&smem_bytes_paged<KT>`, with
//
//   __nv_name_map(4): error: Name expression must form address of a
//     __global__ function or the address of a __device__/__constant__ variable
//
// in between. Measured on this box against `libnvrtc.so.13`; a plain
// `__constant__ unsigned` fails and recovers identically, so this is not about
// variable templates. A function's name is its address and a variable's is
// not, which is C++ rather than an NVRTC quirk, and it went unnoticed until
// now because every name expression this crate had was a `__global__`.
// `kernels_cuda::fa2::PrefillGeometry::ECHO_TEMPLATE` carries the `&`.
//
// With that, `nvrtcGetLoweredName` mangles it, `cuModuleGetGlobal` finds it,
// and four bytes of D2H at module-load time turn the Rust derivation from an
// assertion into a comparison. Whoever wires the read compares and refuses,
// rather than trusting either side.
//
// One point has been compared by hand already. For
// `PagedTraits<kCausal, 64, 1, 4, 8, 8, 4, 1, VariantFull>` -- head dim 128,
// `CTA_TILE_Q` 64, `NUM_MMA_KV` 4 -- NVRTC emitted `49232`, and
// `PrefillGeometry::shared_storage_paged` returns 49232. The interesting half
// is the 80: five trailing `alignas(16)` members at 16 bytes each, four of
// them one-element placeholders whose element widths are 1, 1, 2 and 8. An
// arithmetic that read the widths gets 49184, is wrong by 48 bytes, and fails
// at the shared-memory cap with nothing naming the cause.
//
// Decode needs no echo: its `smem_size` is an arithmetic expression in the
// launcher (`decode.cuh:771-775`), not a `sizeof`, so the Rust is a
// transcription of visible arithmetic and there is nothing hidden to compare
// against.
//
//===----------------------------------------------------------------------===//
#pragma once

// ── Every name below arrives from `src/source.rs`, not from a search path ───
//
// This crate's NVRTC passes NO `-I` and reads nothing from disk. Headers are
// handed to `nvrtcCreateProgram` as `includeNames[]` from a set listed in
// `src/source.rs`, and NVRTC matches those names against the literal string in
// the directive — so the two angle includes here resolve because
// `shim/cuda_bf16.h` is carried as `cuda_bf16.h`, and the six quoted ones
// resolve because `kernels/flashinfer/attention/decode.cuh` is carried
// as `attn/flashinfer/attention/decode.cuh` — its path relative to `kernels/`.
// The trees are rooted to make that fall out; `SHIM`'s doc comment is the
// design.
//
// The bracket style is not decoration. `source.rs::quoted_includes` parses
// QUOTED directives only, so `reachable()` and the include-resolution test
// follow the six below and skip the two above; the internalised closure is
// quoted precisely so it is inside that graph. It was `<flashinfer/...>` while
// it lived in `csrc/vendor/`, which put it outside.
//
// This unit therefore demands `Headers::LibraryAndUpstream` (`unit.rs`'s
// `DEMANDS`, keyed `attn/fa2_*`), which is `ALL_HEADERS` — `SHIM + LIBRARY +
// UPSTREAM`, the 1.1 MB internalised closure included. A unit that forgot to
// say so would get `DEVICE_HEADERS` and fail on line one of `decode.cuh`.
//
// **The reachability test does not cover this file's angle includes.**
// `tests/layers.rs::every_include_reachable_from_a_unit_resolves` walks QUOTED
// includes only, so an angle include of a header that is not carried passes it
// and fails at the first fire on a GPU box with *"could not open source file"*
// — a diagnostic that names the include rather than the omission. Only the
// last line below is behind that gate. The other seven are behind the
// directory, which is why nothing here may be added by editing a list.
//
// The FA2 lattice was derived with a hand-run `libnvrtc` probe that passed
// `-I csrc/{src,shim,vendor} -I /usr/local/cuda/include` and
// `--gpu-architecture=compute_89`. That probe is a faithful SIMULATION of the
// fire and not the fire: the real compile resolves the same names from the
// carried set, targets `sm_XY` so NVRTC emits SASS rather than PTX the driver
// would JIT a second time at load, and adds `--fmad=false --prec-div=true
// --prec-sqrt=true`, which the probe did not. Those three change numerics, not
// whether this text compiles, so the probe's clean runs stand — but no output
// of either build has ever been compared against the other.
//
// Nothing here needs a toolkit include path. If a future edit reaches for one,
// the precedent is `supergraph.cu`'s: hand-declare the builtin under
// `#ifdef __CUDACC_RTC__` with the toolkit header under `#else`, rather than
// putting `/usr/local/cuda/include` on the launch path.

#include <cuda_bf16.h>
#include <cstdint>

#include "flashinfer/attention/decode.cuh"
#include "flashinfer/attention/default_decode_params.cuh"
#include "flashinfer/attention/default_prefill_params.cuh"
#include "flashinfer/attention/mask.cuh"
#include "flashinfer/attention/prefill.cuh"
#include "flashinfer/attention/variants.cuh"

#include "attn/attention_score_capture.cuh"

namespace pie::attn::fa2 {

// The element types, unchanged from `attention_flashinfer_common.cuh:67-70`.
// pie's KV cache is bf16 by the time FA2 sees it -- every quantised scheme is
// dequantised into `kv_layer.{k,v}_bf16_pages` first, which is what
// `attn::dequant_kv_cache_layer_to_bf16_active` is for -- so there is one
// element type here and no dtype axis in the lattice.
using DTypeQ = __nv_bfloat16;
using DTypeKV = __nv_bfloat16;
using DTypeO = __nv_bfloat16;
using IdType = std::int32_t;

// `attention_flashinfer_common.cuh:72`. pie applies RoPE before attention, so
// the kernel never rotates; `kNone` is the only positional mode in the
// lattice, and it is what makes `KernelTraits::USE_SHARED_ROPE_FREQ` false and
// `SharedStoragePaged` the plain storage rather than the rope-freq wrapper
// (`prefill.cuh:229-230`, `:295-298`).
inline constexpr auto POS_ENC = ::flashinfer::PosEncodingMode::kNone;

// ── The six variants, verbatim from `attention_flashinfer_common.cuh:76-120` ──
//
// Each is a `DefaultAttention<use_custom_mask, use_sliding_window,
// use_logits_soft_cap, use_alibi>`. The comments there are the argument for
// each combination and are not repeated; what matters here is that these are
// the SIX names a row may put in its ninth template slot, and that no seventh
// exists -- alibi is never instantiated because no pie call site sets
// `maybe_alibi_slopes`.

/// Causal + sliding window. `window_left = -1` makes the window predicate
/// trivially true, so this also serves full causal attention.
using VariantWindow = ::flashinfer::DefaultAttention<false, true, false, false>;
/// `VariantWindow` plus per-element `logits_soft_cap` (gemma-2's 50).
using VariantWindowSoftcap = ::flashinfer::DefaultAttention<false, true, true, false>;
/// No window predicate at all -- the full-attention variant.
using VariantFull = ::flashinfer::DefaultAttention<false, false, false, false>;
/// `VariantFull` plus soft cap.
using VariantFullSoftcap = ::flashinfer::DefaultAttention<false, false, true, false>;
/// Caller-supplied mask bitmap AND the sliding window, which is one
/// predicate each and both of them ANDed in `LogitsMask`
/// (`flashinfer/attention/variants.cuh`). `attention.masked` states a window
/// beside its mask -- gemma's 35 sliding layers state 512 on it -- and the two
/// are not one fact spelled twice: a mask says which pairs the CALLER admits
/// and a window says how far back the TEXT reads, so a reading that dropped
/// either would be plausible and wrong. `VariantWindow`'s note above is what
/// makes one instantiation enough for both readings: `window_left = -1` turns
/// the window predicate into `kv_idx + qo_len >= qo_idx`, which every pair
/// satisfies, so the 7 global layers that state window 0 run this arm and
/// attend exactly what the mask admits.
using VariantCustom = ::flashinfer::DefaultAttention<true, true, false, false>;
/// `VariantCustom` plus soft cap.
using VariantCustomSoftcap = ::flashinfer::DefaultAttention<true, true, true, false>;

// ── The params structs ──────────────────────────────────────────────────────

using DecodeParams = ::flashinfer::BatchDecodeParams<DTypeQ, DTypeKV, DTypeO, IdType>;
using PrefillParams = ::flashinfer::BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeO, IdType>;

// ── The two capture variants and their params ───────────────────────────────
//
// `PieScoreCapture` / `PieScoreCaptureWindow` are OURS
// (`attention_score_capture.cuh`), wrapping a stock `DefaultAttention` so the
// attention output is unchanged and only the side channel is new. They are
// template arguments, which is precisely why §44 could not give the two
// capture dispatches a table row under the AOT build: the sink is compiled
// INTO the instantiation. Under a JIT that is no longer an obstacle -- it is
// four more rows.
//
// Neither soft-cap nor sliding-window is instantiated with a capture, and the
// refusal is the dispatch's (`fa2::DecodeCapture::launch`), not a gap here:
// soft-cap records `cap*tanh(s/cap)` rather than the pre-softmax score H2O and
// TOVA are defined over, and a sliding window masks in `LogitsMask`, which
// runs after `LogitsTransform`.
using CaptureWindow = PieScoreCapture<VariantWindow>;
using CaptureFull = PieScoreCapture<VariantFull>;
using CapturePrefill = PieScoreCaptureWindow<VariantFull>;

using DecodeCaptureParams = PieScoreParams<DecodeParams, IdType>;
using PrefillCaptureParams = PieScoreWindowParams<PrefillParams, IdType>;

// ── The prefill traits pack ─────────────────────────────────────────────────
//
// `BatchPrefillWithPagedKVCacheKernel` takes ONE template argument, a
// `KernelTraits`, and the launcher builds it at `prefill.cuh:4285-4288` out of
// the eight numbers it derived plus five types. `PagedTraits` is that
// expression with the five invariant types filled in, so a row states the
// eight numbers and a variant -- exactly what `fa2::PrefillGeometry` computes.
//
// `DTypeQKAccum` is `float` and not `half`: `prefill.cuh:4208-4210` selects
// `half` only when `USE_FP16_QK_REDUCTION && is_same_v<DTypeQ, half>`, and
// pie's DTypeQ is bf16, so the reduction flag `attention_flashinfer_common.cuh:764`
// passes as `true` has no effect on the accumulator. It is passed through
// anyway, because it also gates `IsInvalid()`'s rope clause.
template <::flashinfer::MaskMode MASK, std::uint32_t CTA_TILE_Q, std::uint32_t NUM_MMA_Q,
          std::uint32_t NUM_MMA_KV, std::uint32_t NUM_MMA_D_QK, std::uint32_t NUM_MMA_D_VO,
          std::uint32_t NUM_WARPS_Q, std::uint32_t NUM_WARPS_KV, class Variant,
          class Params = PrefillParams>
using PagedTraits =
    ::flashinfer::KernelTraits<MASK, CTA_TILE_Q, NUM_MMA_Q, NUM_MMA_KV, NUM_MMA_D_QK, NUM_MMA_D_VO,
                              NUM_WARPS_Q, NUM_WARPS_KV, POS_ENC, DTypeQ, DTypeKV, DTypeO, float,
                              IdType, Variant>;

/// The compiler's own `sizeof(KTraits::SharedStoragePaged)`, exported so the
/// Rust derivation can be compared against it rather than trusted.
///
/// See the file header for why this exists and what reads it. `__device__`
/// rather than `__constant__` for the reason NVRTC's own documentation gives:
/// both are addressable through `cuModuleGetGlobal`, and `__device__` needs no
/// bank.
template <class KTraits>
__device__ unsigned smem_bytes_paged =
    static_cast<unsigned>(sizeof(typename KTraits::SharedStoragePaged));

}  // namespace pie::attn::fa2
