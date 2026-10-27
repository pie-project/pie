/*
 * Copyright (c) 2023 by FlashInfer team.
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
#ifndef PIE_FA2_LAYOUT_CUH_
#define PIE_FA2_LAYOUT_CUH_

#include <cstdint>

namespace pie::attn::fa2 {

enum class QKVLayout {

  kNHD = 0U,

  kHND = 1U,
};

struct tensor_info_t {
  uint32_t qo_len;
  uint32_t kv_len;
  uint32_t num_qo_heads;
  uint32_t num_kv_heads;
  uint32_t q_stride_n;
  uint32_t q_stride_h;
  uint32_t kv_stride_n;
  uint32_t kv_stride_h;
  uint32_t head_dim;
  __host__ __device__ __forceinline__ tensor_info_t(uint32_t qo_len, uint32_t kv_len,
                                                    uint32_t num_qo_heads, uint32_t num_kv_heads,
                                                    uint32_t q_stride_n, uint32_t q_stride_h,
                                                    uint32_t kv_stride_n, uint32_t kv_stride_h,
                                                    uint32_t head_dim)
      : qo_len(qo_len),
        kv_len(kv_len),
        num_qo_heads(num_qo_heads),
        num_kv_heads(num_kv_heads),
        q_stride_n(q_stride_n),
        q_stride_h(q_stride_h),
        kv_stride_n(kv_stride_n),
        kv_stride_h(kv_stride_h),
        head_dim(head_dim) {}

  __host__ __device__ __forceinline__ tensor_info_t(uint32_t qo_len, uint32_t kv_len,
                                                    uint32_t num_qo_heads, uint32_t num_kv_heads,
                                                    QKVLayout kv_layout, uint32_t head_dim)
      : qo_len(qo_len),
        kv_len(kv_len),
        num_qo_heads(num_qo_heads),
        num_kv_heads(num_kv_heads),
        head_dim(head_dim) {
    q_stride_n = num_qo_heads * head_dim;
    q_stride_h = head_dim;
    kv_stride_n = (kv_layout == QKVLayout::kNHD) ? num_kv_heads * head_dim : head_dim;
    kv_stride_h = (kv_layout == QKVLayout::kNHD) ? head_dim : kv_len * head_dim;
  }

};

}
#endif
