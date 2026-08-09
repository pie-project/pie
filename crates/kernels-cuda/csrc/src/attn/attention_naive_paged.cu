// THE ENUM MIRRORS, AND NOTHING ELSE. This translation unit holds no
// launcher and no launch.
//
// The device text -- `naive_paged_attn<BLOCK>`, `naive_paged_decode<BLOCK>`,
// the five helpers they call and the two enum mirrors -- moved to
// `crates/kernels-cuda-new/csrc/src/attn/attention_naive_paged.cuh`, which
// this file includes.
//
// THE THREE LAUNCHERS ARE DELETED. `attn::attention_naive_paged` is named in
// `kernels_cuda_new::device::JIT_DISPATCHED`, so `abi::emit_c_shim` skips its
// row and no `pie_k_attn_attention_naive_paged` is generated; the shim entry
// was the whole consumer set. `attention_naive_paged_bf16` had no row at all
// and was called only by the sibling `attention_naive_paged` overload that
// dequantised first, so the two died together, and with them this file's
// `#include "attn/kv_paged.hpp"` -- the last archive caller of
// `dequant_kv_cache_layer_to_bf16_active` outside `driver-cuda/csrc/`.
//
// WHAT WENT WITH THEM, recorded because a rule does not say it:
//
//   * `dim3 grid(num_requests, total_tokens, num_q_heads)` -- `total_tokens`
//     standing in for the largest single-request `qo_len`, which the host
//     does not have, with the kernel early-exiting when
//     `qo_off >= qo_hi - qo_lo`. `LaunchRule::PagedScores` states exactly
//     this rectangle, which is why the row could be routed at all.
//   * `smem = (head_dim + BLOCK) * sizeof(float)`, stated by the same rule.
//   * `check_head_dim_supported`, which THREW for `head_dim` outside
//     `[1, kMaxHeadDim]`. A throw cannot cross the ABI, and the JIT path has
//     no equivalent: `acc[]` in the kernel is sized
//     `(kMaxHeadDim + BLOCK - 1) / BLOCK`, so a larger `head_dim` overruns it
//     rather than being diagnosed. That is a REFUSAL with no home, and
//     `.wiki/driver/new-horizon.md` §56 is where it is written down.
//
// WHY THE FILE STAYS. The `static_assert`s below are the only place the host
// enums (`KvCacheScheme`, `DType`) and the device mirrors NVRTC reads
// (`device::KvScheme`, `device::KvDType`) are compared. `mxfp4_marlin.cuh`
// keeps its mirror in step with a comment, and a comment is one careless
// renumbering away from decoding fp8 pages as int8. Deleting the launchers
// does not make the mirrors agree, so the check outlives them --
// `attn/attention_xqa.cu` is the precedent for a `.cu` that survives its last
// `<<<>>>`.

#include <cstdint>

#include <cuda_runtime.h>

#include "attn/attention_naive_paged.cuh"

#include "attn/kv_cache_view.hpp"
#include "tensor.hpp"


namespace pie_cuda_driver::kernels::attn {

namespace {

// The one block width the two kernels are instantiated at, kept because the
// `static_assert` below is about it: `acc[]` is sized against `BLOCK` and the
// two must agree or the halving reduction folds through shared memory the
// launch never wrote. `LaunchRule::PagedScores` states the same 128.
constexpr int BLOCK = 128;
constexpr int MAX_HEAD_DIM = device::kMaxHeadDim;
static_assert(MAX_HEAD_DIM == BLOCK * 8,
              "acc[] in the kernels is sized (kMaxHeadDim + BLOCK - 1) / BLOCK "
              "and was written as a literal 8 at BLOCK = 128");

// The mirrors in `attention_naive_paged.cuh` exist because the host enums live
// in headers that pull `<cstdint>`, which NVRTC cannot answer. This is the one
// translation unit that sees both spellings, so it is where they are CHECKED.
// `mxfp4_marlin.cuh` keeps its mirror in step with a comment; a comment is one
// careless renumbering away from decoding fp8 pages as int8.
#define PIE_SCHEME_MIRRORS_HOST(name)                                       \
    static_assert(static_cast<std::uint8_t>(KvCacheScheme::name) ==         \
                      static_cast<std::uint8_t>(device::KvScheme::name),    \
                  "device::KvScheme::" #name " drifted from KvCacheScheme")
PIE_SCHEME_MIRRORS_HOST(Native);
PIE_SCHEME_MIRRORS_HOST(Fp8PerTensor);
PIE_SCHEME_MIRRORS_HOST(Int8PerTokenHead);
PIE_SCHEME_MIRRORS_HOST(Fp8PerTokenHead);
PIE_SCHEME_MIRRORS_HOST(Fp4Block);
#undef PIE_SCHEME_MIRRORS_HOST

#define PIE_DTYPE_MIRRORS_HOST(name)                                        \
    static_assert(static_cast<std::uint8_t>(DType::name) ==                 \
                      static_cast<std::uint8_t>(device::KvDType::name),     \
                  "device::KvDType::" #name " drifted from DType")
PIE_DTYPE_MIRRORS_HOST(BF16);
PIE_DTYPE_MIRRORS_HOST(FP16);
PIE_DTYPE_MIRRORS_HOST(FP32);
PIE_DTYPE_MIRRORS_HOST(INT8);
PIE_DTYPE_MIRRORS_HOST(INT32);
PIE_DTYPE_MIRRORS_HOST(INT64);
PIE_DTYPE_MIRRORS_HOST(UINT8);
PIE_DTYPE_MIRRORS_HOST(FP8_E4M3);
PIE_DTYPE_MIRRORS_HOST(FP8_E5M2);
PIE_DTYPE_MIRRORS_HOST(INT4_PACKED);
#undef PIE_DTYPE_MIRRORS_HOST

}  // namespace

}  // namespace pie_cuda_driver::kernels::attn
