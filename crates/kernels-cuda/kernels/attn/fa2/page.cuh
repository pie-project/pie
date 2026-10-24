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
#ifndef PIE_FA2_PAGE_CUH_
#define PIE_FA2_PAGE_CUH_


#include "attn/fa2/fastdiv.cuh"
#include "attn/fa2/layout.cuh"
#include "attn/fa2/utils.cuh"
#include "attn/fa2/vec_dtypes.cuh"

namespace pie::attn::fa2 {






template <typename DType, typename IdType>
struct paged_kv_t {
  uint_fastdiv page_size;
  uint32_t num_heads;
  uint32_t head_dim;
  uint32_t batch_size;
  uint32_t stride_page;
  uint32_t stride_n;
  uint32_t stride_h;

  DType* k_data;
  DType* v_data;
  IdType* indices;

  IdType* indptr;

  IdType* last_page_len;

  IdType* rope_pos_offset;



  __host__ __device__ __forceinline__ paged_kv_t()
      : num_heads(0),
        page_size(),
        head_dim(0),
        batch_size(0),
        stride_page(0),
        stride_n(0),
        stride_h(0),
        k_data(nullptr),
        v_data(nullptr),
        indices(nullptr),
        indptr(nullptr),
        last_page_len(nullptr),
        rope_pos_offset(nullptr) {}






























  __host__ __device__ __forceinline__ uint32_t get_length(uint32_t batch_idx) const {
    if (indptr[batch_idx + 1] == indptr[batch_idx]) {
      return 0;
    }
    return (indptr[batch_idx + 1] - indptr[batch_idx] - 1) * page_size + last_page_len[batch_idx];
  }







  __host__ __device__ __forceinline__ size_t get_elem_offset(size_t page_idx, size_t head_idx,
                                                             size_t entry_idx,
                                                             size_t feat_idx) const {
    return page_idx * stride_page + head_idx * stride_h + entry_idx * stride_n + feat_idx;
  }






  __host__ __device__ __forceinline__ size_t get_elem_offset_in_page(size_t head_idx,
                                                                     size_t entry_idx,
                                                                     size_t feat_idx) const {
    return head_idx * stride_h + entry_idx * stride_n + feat_idx;
  }

  __device__ __forceinline__ DType* get_k_ptr(IdType page_iter, uint32_t head_idx,
                                              uint32_t entry_idx, uint32_t feat_idx) const {
    return k_data + get_elem_offset(__ldg(indices + page_iter), head_idx, entry_idx, feat_idx);
  }

  __device__ __forceinline__ size_t protective_get_kv_offset(IdType page_iter, uint32_t head_idx,
                                                             uint32_t entry_idx, uint32_t feat_idx,
                                                             IdType last_indptr) const {
    if (page_iter < last_indptr) {
      return get_elem_offset(__ldg(indices + page_iter), head_idx, entry_idx, feat_idx);
    } else {
      return 0;
    }
  }

  __device__ __forceinline__ DType* protective_get_k_ptr(IdType page_iter, uint32_t head_idx,
                                                         uint32_t entry_idx, uint32_t feat_idx,
                                                         IdType last_indptr) const {
    return k_data + protective_get_kv_offset(page_iter, head_idx, entry_idx, feat_idx, last_indptr);
  }

  __device__ __forceinline__ DType* get_v_ptr(IdType page_iter, uint32_t head_idx,
                                              uint32_t entry_idx, uint32_t feat_idx) const {
    return v_data + get_elem_offset(__ldg(indices + page_iter), head_idx, entry_idx, feat_idx);
  }

  __device__ __forceinline__ DType* protective_get_v_ptr(IdType page_iter, uint32_t head_idx,
                                                         uint32_t entry_idx, uint32_t feat_idx,
                                                         IdType last_indptr) const {
    return v_data + protective_get_kv_offset(page_iter, head_idx, entry_idx, feat_idx, last_indptr);
  }
};

template <typename DType>
__device__ __forceinline__ float nvfp4_append_to_float(DType value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float nvfp4_append_to_float<nv_half>(nv_half value) {
  return __half2float(value);
}

template <>
__device__ __forceinline__ float nvfp4_append_to_float<nv_bfloat16>(nv_bfloat16 value) {
  return __bfloat162float(value);
}

__device__ __forceinline__ uint8_t nvfp4_append_quantize_e2m1(float value) {
  const uint8_t sign = signbit(value) ? 0x8 : 0x0;
  const float mag = fabsf(value);
  uint8_t code;
  if (!(mag > 0.25f)) {
    code = 0;
  } else if (mag < 0.75f) {
    code = 1;
  } else if (mag <= 1.25f) {
    code = 2;
  } else if (mag < 1.75f) {
    code = 3;
  } else if (mag <= 2.5f) {
    code = 4;
  } else if (mag < 3.5f) {
    code = 5;
  } else if (mag <= 5.0f) {
    code = 6;
  } else {
    code = 7;
  }
  return sign | code;
}

template <typename DType>
__device__ __forceinline__ void nvfp4_append_quantize_block(
    const DType* __restrict__ input, const float global_scale, const size_t input_base,
    const uint32_t dim_base, uint8_t* __restrict__ packed_out, uint8_t* __restrict__ sf_out) {
  float values[16];
  float amax = 0.0f;
#pragma unroll
  for (uint32_t i = 0; i < 16; ++i) {
    const float value = nvfp4_append_to_float(input[input_base + dim_base + i]);
    values[i] = value;
    amax = fmaxf(amax, fabsf(value));
  }

  float sf_value = 0.0f;
  if (amax > 0.0f && global_scale > 0.0f) {
    sf_value = amax / (6.0f * global_scale);
  }
  __nv_fp8_e4m3 sf_fp8 = __nv_fp8_e4m3(sf_value);
  *sf_out = sf_fp8.__x;

  const float sf_rounded = static_cast<float>(sf_fp8);
  const float output_scale = (amax > 0.0f && sf_rounded > 0.0f && global_scale > 0.0f)
                                 ? (1.0f / (sf_rounded * global_scale))
                                 : 0.0f;

#pragma unroll
  for (uint32_t i = 0; i < 8; ++i) {
    const uint8_t lo = nvfp4_append_quantize_e2m1(values[i * 2] * output_scale);
    const uint8_t hi = nvfp4_append_quantize_e2m1(values[i * 2 + 1] * output_scale);
    packed_out[i] = lo | (hi << 4);
  }
}

__device__ __forceinline__ bool nvfp4_append_is_positive_finite_scale(float scale) {
  return isfinite(scale) && scale > 0.0f;
}










template <typename DType, typename IdType>
struct paged_kv_mla_t {
  uint_fastdiv page_size;
  uint32_t head_dim_ckv;
  uint32_t head_dim_kpe;
  uint32_t batch_size;
  uint32_t stride_page_ckv;
  uint32_t stride_page_kpe;
  uint32_t stride_n_ckv;
  uint32_t stride_n_kpe;

  DType* ckv_data;
  DType* kpe_data;
  IdType* indices;

  IdType* indptr;

  IdType* last_page_len;

  IdType* rope_pos_offset;



  __host__ __device__ __forceinline__ paged_kv_mla_t()
      : head_dim_ckv(0),
        head_dim_kpe(0),
        batch_size(0),
        stride_page_ckv(0),
        stride_page_kpe(0),
        stride_n_ckv(0),
        stride_n_kpe(0),
        ckv_data(nullptr),
        kpe_data(nullptr),
        indices(nullptr),
        indptr(nullptr),
        last_page_len(nullptr),
        rope_pos_offset(nullptr) {}





























  __host__ __device__ __forceinline__ uint32_t get_length(uint32_t batch_idx) const {
    if (indptr[batch_idx + 1] == indptr[batch_idx]) {
      return 0;
    }
    return (indptr[batch_idx + 1] - indptr[batch_idx] - 1) * page_size + last_page_len[batch_idx];
  }

  __host__ __device__ __forceinline__ size_t get_elem_offset_ckv(size_t page_idx, size_t entry_idx,
                                                                 size_t feat_idx) const {
    return page_idx * stride_page_ckv + entry_idx * stride_n_ckv + feat_idx;
  }

  __device__ __forceinline__ size_t protective_get_offset_ckv(IdType page_iter, uint32_t entry_idx,
                                                              uint32_t feat_idx,
                                                              IdType last_indptr) const {
    if (page_iter < last_indptr) {
      return get_elem_offset_ckv(__ldg(indices + page_iter), entry_idx, feat_idx);
    } else {
      return 0;
    }
  }

  __host__ __device__ __forceinline__ size_t get_elem_offset_kpe(size_t page_idx, size_t entry_idx,
                                                                 size_t feat_idx) const {
    return page_idx * stride_page_kpe + entry_idx * stride_n_kpe + feat_idx;
  }

  __device__ __forceinline__ size_t protective_get_offset_kpe(IdType page_iter, uint32_t entry_idx,
                                                              uint32_t feat_idx,
                                                              IdType last_indptr) const {
    if (page_iter < last_indptr) {
      return get_elem_offset_kpe(__ldg(indices + page_iter), entry_idx, feat_idx);
    } else {
      return 0;
    }
  }

  __device__ __forceinline__ DType* get_ckv_ptr(size_t page_idx, size_t entry_idx,
                                                size_t feat_idx) const {
    return ckv_data + get_elem_offset_ckv(__ldg(indices + page_idx), entry_idx, feat_idx);
  }

  __device__ __forceinline__ DType* get_kpe_ptr(size_t page_idx, size_t entry_idx,
                                                size_t feat_idx) const {
    return kpe_data + get_elem_offset_kpe(__ldg(indices + page_idx), entry_idx, feat_idx);
  }
};

}

#endif
