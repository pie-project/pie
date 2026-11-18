/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef PIE_FA2_ATTENTION_VARIANTS_CUH_
#define PIE_FA2_ATTENTION_VARIANTS_CUH_
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

#include "attn/fa2/math.cuh"
#include "attn/fa2/utils.cuh"
#include "attn/fa2/variant_helper.cuh"

namespace pie::attn::fa2 {

DEFINE_HAS_MEMBER(maybe_mask_indptr)
DEFINE_HAS_MEMBER(maybe_mask_origin)

template <bool use_custom_mask, bool use_sliding_window, bool use_logits_soft_cap, bool use_alibi>
struct DefaultAttention : AttentionVariantBase {
  static constexpr bool use_softmax = true;
  // PIE: a decode row sees only the window's keys, nothing else masks them.
  static constexpr bool skips_decode_window = use_sliding_window && !use_custom_mask;

  uint8_t* custom_mask_ptr;
  // PIE: the request's committed boundary: keys below it are read under the
  // model's own bounds and no mask bit; the plane's column `c` is key
  // `mask_origin + c`. `mask_wide`: its rows read both ways (no causal term).
  uint32_t mask_origin;
  bool mask_wide;
  uint32_t qo_len, kv_len;
  uint32_t window_left;
  float sm_scale_log2;
  float soft_cap_pre_tanh_scale;

  template <typename Params>
  __device__ __host__ DefaultAttention(const Params& params, uint32_t batch_idx,
                                       uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    if constexpr (use_logits_soft_cap) {
      soft_cap_pre_tanh_scale = params.sm_scale * math::ptx_rcp(params.logits_soft_cap);
      sm_scale_log2 = math::log2e * params.logits_soft_cap;
    } else {
      if constexpr (use_alibi) {
        sm_scale_log2 = math::log2e;
      } else {
        sm_scale_log2 = params.sm_scale * math::log2e;
      }
    }
    mask_origin = 0;
    mask_wide = true;
    if constexpr (use_custom_mask) {
      if constexpr (has_maybe_mask_origin_v<Params>) {
        if (params.maybe_mask_origin != nullptr) {
          const uint32_t word = params.maybe_mask_origin[batch_idx];
          mask_origin = word & 0x7fffffffu;
          mask_wide = (word >> 31) != 0;
        }
      }
      if constexpr (has_maybe_mask_indptr_v<Params>) {
        // PIE: a negative offset states no mask (a group none of whose
        // rows states one): its boundary is past every key.
        const int64_t at = params.maybe_mask_indptr[batch_idx];
        custom_mask_ptr = params.maybe_custom_mask + (at < 0 ? 0 : at);
        if (at < 0) {
          mask_origin = 0x7fffffffu;
        }
      } else {
        custom_mask_ptr = params.maybe_custom_mask;
      }
    }
    window_left = (params.window_left >= 0) ? params.window_left : kv_len;
  }

  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    if constexpr (use_alibi) {
      logits = logits * params.sm_scale +
               params.maybe_alibi_slopes[qo_head_idx] * float(int(kv_idx) - int(qo_idx));
    }
    if constexpr (use_logits_soft_cap) {
      logits = float(math::tanh(logits * soft_cap_pre_tanh_scale));
    }
    return logits;
  })

  REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    bool mask = true;
    if constexpr (use_custom_mask) {
      if (qo_idx >= qo_len || kv_idx >= kv_len) {
        mask = false;
      } else if (!mask_wide && kv_idx + qo_len > kv_len + qo_idx) {
        // PIE: a row reads no key after its own (the cache row), whatever its bits.
        mask = false;
      } else if (kv_idx >= mask_origin) {
        // PIE: past the committed boundary the plane speaks, a row of
        // `kv_len - mask_origin` bits; below it every key is read.
        const uint64_t width = kv_len - mask_origin;
        const uint64_t offset = static_cast<uint64_t>(qo_idx) * width + (kv_idx - mask_origin);
        mask &= ((custom_mask_ptr[offset / 8] >> (offset % 8)) & 1);
      }
    }
    if constexpr (use_sliding_window) {
      mask &= (kv_idx + qo_len + window_left >= kv_len + qo_idx);
    }
    return mask;
  })
};

};

#endif
