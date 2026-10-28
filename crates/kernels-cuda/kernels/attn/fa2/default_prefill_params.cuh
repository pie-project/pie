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
#ifndef PIE_FA2_PREFILL_PARAMS_CUH_
#define PIE_FA2_PREFILL_PARAMS_CUH_

#include <cuda_runtime.h>

#include <cstdint>

#include "attn/fa2/page.cuh"

namespace pie::attn::fa2 {

template <typename DTypeQ_, typename DTypeKV_, typename DTypeO_, typename IdType_>
struct BatchPrefillRaggedParams {
  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  using IdType = IdType_;

  DTypeQ* q;
  DTypeKV* k;
  DTypeKV* v;
  uint8_t* maybe_custom_mask;
  IdType* q_indptr;
  IdType* kv_indptr;
  IdType* maybe_mask_indptr;
  IdType* maybe_q_rope_offset;
  IdType* maybe_k_rope_offset;
  DTypeO* o;
  float* lse;
  float* maybe_alibi_slopes;
  uint_fastdiv group_size;
  uint32_t num_qo_heads;
  uint32_t num_kv_heads;
  uint32_t q_stride_n;
  uint32_t q_stride_h;
  uint32_t k_stride_n;
  uint32_t k_stride_h;
  uint32_t v_stride_n;
  uint32_t v_stride_h;
  uint32_t k_sf_stride_page;
  uint32_t k_sf_stride_n;
  uint32_t k_sf_stride_h;
  uint32_t v_sf_stride_page;
  uint32_t v_sf_stride_n;
  uint32_t v_sf_stride_h;
  int32_t window_left;
  float logits_soft_cap;
  float sm_scale;
  float rope_rcp_scale;
  float rope_rcp_theta;

  IdType* request_indices;
  IdType* qo_tile_indices;
  IdType* kv_tile_indices;
  IdType* merge_indptr;
  IdType* o_indptr;
  IdType* kv_chunk_size_ptr;
  bool* block_valid_mask;
  uint32_t max_total_num_rows;
  uint32_t* total_num_rows;
  uint32_t padded_batch_size;
  bool partition_kv;
  uint32_t* maybe_prefix_len_ptr;
  uint16_t* maybe_token_pos_in_items_ptr;
  uint32_t token_pos_in_items_len;
  uint16_t* maybe_max_item_len_ptr;

  __host__ __device__ __forceinline__ uint32_t get_qo_len(uint32_t batch_idx) const {
    return q_indptr[batch_idx + 1] - q_indptr[batch_idx];
  }

  __host__ __device__ __forceinline__ uint32_t get_kv_len(uint32_t batch_idx) const {
    return kv_indptr[batch_idx + 1] - kv_indptr[batch_idx];
  }
};

template <typename DTypeQ_, typename DTypeKV_, typename DTypeO_, typename IdType_>
struct BatchPrefillPagedParams {
  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  using IdType = IdType_;

  DTypeQ* q;
  paged_kv_t<DTypeKV, IdType> paged_kv;
  uint8_t* maybe_custom_mask;
  IdType* q_indptr;
  IdType* maybe_mask_indptr;
  IdType* maybe_q_rope_offset;
  DTypeO* o;
  float* lse;
  float* maybe_alibi_slopes;
  uint_fastdiv group_size;
  uint32_t num_qo_heads;
  IdType q_stride_n;
  IdType q_stride_h;
  uint32_t k_sf_stride_page;
  uint32_t k_sf_stride_n;
  uint32_t k_sf_stride_h;
  uint32_t v_sf_stride_page;
  uint32_t v_sf_stride_n;
  uint32_t v_sf_stride_h;
  int32_t window_left;
  float logits_soft_cap;
  float sm_scale;
  float rope_rcp_scale;
  float rope_rcp_theta;

  IdType* request_indices;
  IdType* qo_tile_indices;
  IdType* kv_tile_indices;
  IdType* merge_indptr;
  IdType* o_indptr;
  bool* block_valid_mask;
  IdType* kv_chunk_size_ptr;
  uint32_t max_total_num_rows;
  uint32_t* total_num_rows;
  uint32_t padded_batch_size;
  bool partition_kv;
  uint32_t* maybe_prefix_len_ptr;
  uint16_t* maybe_token_pos_in_items_ptr;
  uint32_t token_pos_in_items_len;
  uint16_t* maybe_max_item_len_ptr;

  __host__ __device__ __forceinline__ uint32_t get_qo_len(uint32_t batch_idx) const {
    return q_indptr[batch_idx + 1] - q_indptr[batch_idx];
  }

  __host__ __device__ __forceinline__ uint32_t get_kv_len(uint32_t batch_idx) const {
    return paged_kv.get_length(batch_idx);
  }
};

}

#endif
