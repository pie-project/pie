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
#ifndef FLASHINFER_PAGE_CUH_
#define FLASHINFER_PAGE_CUH_

// PIE: REMOVED -- host-only `<driver_types.h>` and `<vector>`. 3 lines of host C++, guarded out
// of every NVRTC compile before it was removed, so removing it changes no compile. This marker
// is one a strip does NOT undo; see MODIFICATIONS.

// PIE: `exception.h` is deleted, and this include with it. That file was pure
// host C++ -- `flashinfer::Error` derives from `std::exception` and every macro
// in it builds its message in a `std::ostringstream` -- so its entire body was
// already inside `#ifndef __CUDACC_RTC__` and NVRTC has always seen an empty
// file through this line. `src/plan/error.rs` owns the refusals now. This is
// the one marker in the tree that a strip does NOT undo: restoring upstream
// means restoring the deleted file too. See MODIFICATIONS.
#include "fastdiv.cuh"
#include "layout.cuh"
#include "utils.cuh"
#include "vec_dtypes.cuh"

namespace flashinfer {

/*!
 * \brief Paged key-value cache
 * \tparam layout The layout of last 3 dimensions in KV-Cache.
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 */
template <typename DType, typename IdType>
struct paged_kv_t {
  uint_fastdiv page_size;
  uint32_t num_heads;
  uint32_t head_dim;
  uint32_t batch_size;
  uint32_t stride_page;
  uint32_t stride_n;
  uint32_t stride_h;

  // Internal layout:
  // [max_num_pages, num_heads, page_size, head_dim] if layout == HND
  // [max_num_pages, page_size, num_heads, head_dim] if layout == NHD
  DType* k_data;
  DType* v_data;
  IdType* indices;

  // [batch_size + 1] The page indptr array, with the first element 0, the last element nnz_pages
  IdType* indptr;
  // [batch_size] The offset of the last page for each request in the batch
  IdType* last_page_len;
  // [batch_size] The start position of each request in the batch.
  IdType* rope_pos_offset;

  /*!
   * \brief Construct an empty paged key-value cache
   */
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

  /*!
   * \brief Construct a paged key-value cache
   * \param num_heads The number of heads
   * \param page_size The size of each page
   * \param head_dim The dimension of each head
   * \param batch_size The batch size
   * \param layout The layout of last 3 dimensions in KV-Cache.
   * \param k_data The start pointer of key cache, k_cache should be contiguous
   * \param v_data The start pointer of value cache, v_cache should be contiguous
   * \param indices The page indices array
   * \param indptr The page indptr array
   * \param last_page_len The offset of the last page for each request in the batch
   * \param rope_pos_offset The start position of each request in the batch.
   */
// PIE: REMOVED -- a `__host__` `paged_kv_t` constructor. 18 lines of host C++, guarded out of
// every NVRTC compile before it was removed, so removing it changes no compile. This marker is
// one a strip does NOT undo; see MODIFICATIONS.

  /*!
   * \brief Construct a paged key-value cache with custom kv-cache strides
   * \param num_heads The number of heads
   * \param page_size The size of each page
   * \param head_dim The dimension of each head
   * \param batch_size The batch size
   * \param layout The layout of last 3 dimensions in KV-Cache.
   * \param k_data The start pointer of key cache, k_cache doesn't have to be contiguous
   * \param v_data The start pointer of value cache, v_cache doesn't have to be contiguous
   * \param kv_strides custom strides of each dimensions of k_data and v_data
   * \param indices The page indices array
   * \param indptr The page indptr array
   * \param last_page_len The offset of the last page for each request in the batch
   * \param rope_pos_offset The start position of each request in the batch.
   */
// PIE: REMOVED -- the second `__host__` `paged_kv_t` constructor. 19 lines of host C++, guarded
// out of every NVRTC compile before it was removed, so removing it changes no compile. This
// marker is one a strip does NOT undo; see MODIFICATIONS.

  __host__ __device__ __forceinline__ uint32_t get_length(uint32_t batch_idx) const {
    if (indptr[batch_idx + 1] == indptr[batch_idx]) {
      return 0;
    }
    return (indptr[batch_idx + 1] - indptr[batch_idx] - 1) * page_size + last_page_len[batch_idx];
  }

  /*!
   * \brief Compute the offset of element in the allocated buffer.
   * \param page_idx The page index
   * \param head_idx The head index
   * \param entry_idx The page entry index
   * \param feat_idx The feature index
   */
  __host__ __device__ __forceinline__ size_t get_elem_offset(size_t page_idx, size_t head_idx,
                                                             size_t entry_idx,
                                                             size_t feat_idx) const {
    return page_idx * stride_page + head_idx * stride_h + entry_idx * stride_n + feat_idx;
  }

  /*!
   * \brief Compute the offset of element inside the page.
   * \param head_idx The head index
   * \param entry_idx The page entry index
   * \param feat_idx The feature index
   */
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

/*!
 * \brief CUDA kernel to append new keys/values to the paged key-value cache in the decode phase
 * \tparam head_dim The dimension of each head
 * \tparam vec_size The vector size used in the kernel
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 * \param paged_kv The paged key-value cache
 * \param key The key to be appended
 * \param value The value to be appended
 */
template <uint32_t head_dim, uint32_t vec_size, typename DType, typename IdType>
__global__ void AppendPagedKVCacheDecodeKernel(paged_kv_t<DType, IdType> paged_kv,
                                               DType* __restrict__ key, DType* __restrict__ value) {
  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t num_heads = paged_kv.num_heads;
  uint32_t batch_idx = blockIdx.x;
  uint32_t head_idx = ty;

  uint32_t seq_len =
      (paged_kv.indptr[batch_idx + 1] - paged_kv.indptr[batch_idx] - 1) * paged_kv.page_size +
      paged_kv.last_page_len[batch_idx];

  uint32_t page_iter = paged_kv.indptr[batch_idx] + (seq_len - 1) / paged_kv.page_size;
  uint32_t entry_idx = (seq_len - 1) % paged_kv.page_size;

  DType* k_ptr = paged_kv.get_k_ptr(page_iter, head_idx, entry_idx, tx * vec_size);
  DType* v_ptr = paged_kv.get_v_ptr(page_iter, head_idx, entry_idx, tx * vec_size);
  vec_t<DType, vec_size>::memcpy(
      k_ptr, key + (batch_idx * num_heads + head_idx) * head_dim + tx * vec_size);

  vec_t<DType, vec_size>::memcpy(
      v_ptr, value + (batch_idx * num_heads + head_idx) * head_dim + tx * vec_size);
}

/*!
 * \brief CUDA kernel to append new keys/values to the paged key-value cache in the prefill phase
 * \tparam head_dim The dimension of each head
 * \tparam vec_size The vector size used in the kernel
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 * \param paged_kv The paged key-value cache
 * \param key The key to be appended
 * \param value The value to be appended
 * \param batch_indices The batch indices of elements to be appended
 * \param positions The positions of elements to be appended
 */
template <uint32_t head_dim, uint32_t vec_size, typename DType, typename IdType>
__global__ void AppendPagedKVCacheKernel(paged_kv_t<DType, IdType> paged_kv,
                                         DType* __restrict__ append_key,
                                         DType* __restrict__ append_value,
                                         IdType* __restrict__ batch_indices,
                                         IdType* __restrict__ positions, uint32_t nnz,
                                         size_t append_k_stride_n, size_t append_k_stride_h,
                                         size_t append_v_stride_n, size_t append_v_stride_h) {
  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t num_heads = paged_kv.num_heads;
  uint32_t head_idx = ty;
  uint32_t cta_id = blockIdx.x;
  uint32_t num_ctas = gridDim.x;

#pragma unroll 4
  for (uint32_t i = cta_id; i < nnz; i += num_ctas) {
    uint32_t page_iter, entry_idx;
    paged_kv.page_size.divmod(paged_kv.indptr[batch_indices[i]] * paged_kv.page_size + positions[i],
                              page_iter, entry_idx);
    DType* k_ptr = paged_kv.get_k_ptr(page_iter, head_idx, entry_idx, tx * vec_size);
    DType* v_ptr = paged_kv.get_v_ptr(page_iter, head_idx, entry_idx, tx * vec_size);
    vec_t<DType, vec_size>::memcpy(
        k_ptr, append_key + i * append_k_stride_n + head_idx * append_k_stride_h + tx * vec_size);
    vec_t<DType, vec_size>::memcpy(
        v_ptr, append_value + i * append_v_stride_n + head_idx * append_v_stride_h + tx * vec_size);
  }
}

/*!
 * \brief Append new keys/values to the paged key-value cache in the decode phase
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 * \param paged_kv The paged key-value cache
 * \param key The key to be appended
 * \param value The value to be appended
 * \param stream The CUDA stream to execute kernels.
 * \return status Indicates whether CUDA calls are successful
 */
// PIE: REMOVED -- `AppendPagedKVCacheDecode` and `AppendPagedKVCache`, host launchers. 67 lines
// of host C++, guarded out of every NVRTC compile before it was removed, so removing it changes
// no compile. This marker is one a strip does NOT undo; see MODIFICATIONS.

template <uint32_t HEAD_DIM, typename DType, typename IdType>
__global__ void NVFP4QuantizeAppendPagedKVCacheKernel(
    paged_kv_t<uint8_t, IdType> paged_kv, const DType* __restrict__ append_key,
    const DType* __restrict__ append_value, const IdType* __restrict__ batch_indices,
    const IdType* __restrict__ positions, uint32_t nnz, size_t append_k_stride_n,
    size_t append_k_stride_h, size_t append_v_stride_n, size_t append_v_stride_h,
    uint8_t* __restrict__ k_scale_cache, uint8_t* __restrict__ v_scale_cache,
    size_t k_sf_stride_page, size_t k_sf_stride_n, size_t k_sf_stride_h, size_t v_sf_stride_page,
    size_t v_sf_stride_n, size_t v_sf_stride_h, float k_scale, float v_scale) {
  constexpr uint32_t SF_VEC_SIZE = 16;
  constexpr uint32_t PACKED_PER_SF = SF_VEC_SIZE / 2;
  constexpr uint32_t NUM_SF_BLOCKS = HEAD_DIM / SF_VEC_SIZE;
  static_assert(HEAD_DIM % SF_VEC_SIZE == 0);

  const uint32_t token_idx = blockIdx.x;
  const uint32_t head_idx = blockIdx.y;
  if (token_idx >= nnz) return;

  const IdType batch_idx = batch_indices[token_idx];
  const IdType pos = positions[token_idx];
  uint32_t page_iter, entry_idx;
  paged_kv.page_size.divmod(paged_kv.indptr[batch_idx] * paged_kv.page_size + pos, page_iter,
                            entry_idx);
  const IdType page_idx = paged_kv.indices[page_iter];

  const size_t append_k_base =
      static_cast<size_t>(token_idx) * append_k_stride_n + head_idx * append_k_stride_h;
  const size_t append_v_base =
      static_cast<size_t>(token_idx) * append_v_stride_n + head_idx * append_v_stride_h;
  uint8_t* k_out = paged_kv.get_k_ptr(page_iter, head_idx, entry_idx, 0);
  uint8_t* v_out = paged_kv.get_v_ptr(page_iter, head_idx, entry_idx, 0);
  uint8_t* k_sf_out = k_scale_cache + static_cast<size_t>(page_idx) * k_sf_stride_page +
                      entry_idx * k_sf_stride_n + head_idx * k_sf_stride_h;
  uint8_t* v_sf_out = v_scale_cache + static_cast<size_t>(page_idx) * v_sf_stride_page +
                      entry_idx * v_sf_stride_n + head_idx * v_sf_stride_h;

  for (uint32_t sf_idx = threadIdx.x; sf_idx < NUM_SF_BLOCKS * 2; sf_idx += blockDim.x) {
    const bool is_v = sf_idx >= NUM_SF_BLOCKS;
    const uint32_t block_idx = is_v ? sf_idx - NUM_SF_BLOCKS : sf_idx;
    const uint32_t dim_base = block_idx * SF_VEC_SIZE;
    const uint32_t packed_base = block_idx * PACKED_PER_SF;
    if (is_v) {
      nvfp4_append_quantize_block(append_value, v_scale, append_v_base, dim_base,
                                  v_out + packed_base, v_sf_out + block_idx);
    } else {
      nvfp4_append_quantize_block(append_key, k_scale, append_k_base, dim_base, k_out + packed_base,
                                  k_sf_out + block_idx);
    }
  }
}

template <uint32_t HEAD_DIM, typename DType, typename IdType>
__global__ void NVFP4QuantizeAppendPagedKVCacheWithSlotMappingKernel(
    const DType* __restrict__ append_key, const DType* __restrict__ append_value,
    const IdType* __restrict__ slot_mapping, uint32_t nnz, uint32_t num_heads, uint32_t page_size,
    uint32_t max_num_pages, size_t append_k_stride_n, size_t append_k_stride_h,
    size_t append_v_stride_n, size_t append_v_stride_h, uint8_t* __restrict__ paged_k_cache,
    uint8_t* __restrict__ paged_v_cache, uint8_t* __restrict__ k_scale_cache,
    uint8_t* __restrict__ v_scale_cache, size_t k_stride_page, size_t k_stride_n, size_t k_stride_h,
    size_t v_stride_page, size_t v_stride_n, size_t v_stride_h, size_t k_sf_stride_page,
    size_t k_sf_stride_n, size_t k_sf_stride_h, size_t v_sf_stride_page, size_t v_sf_stride_n,
    size_t v_sf_stride_h, const float* __restrict__ k_scale_ptr,
    const float* __restrict__ v_scale_ptr) {
  constexpr uint32_t SF_VEC_SIZE = 16;
  constexpr uint32_t PACKED_PER_SF = SF_VEC_SIZE / 2;
  constexpr uint32_t NUM_SF_BLOCKS = HEAD_DIM / SF_VEC_SIZE;
  static_assert(HEAD_DIM % SF_VEC_SIZE == 0);

  const uint32_t token_idx = blockIdx.x;
  const uint32_t head_idx = blockIdx.y;
  if (token_idx >= nnz) return;

  const float k_scale = __ldg(k_scale_ptr);
  const float v_scale = __ldg(v_scale_ptr);
  if (!(nvfp4_append_is_positive_finite_scale(k_scale) &&
        nvfp4_append_is_positive_finite_scale(v_scale))) {
    asm volatile("trap;");
    return;
  }

  const IdType slot = slot_mapping[token_idx];
  if (slot < 0 || static_cast<size_t>(slot) >= static_cast<size_t>(max_num_pages) * page_size) {
    return;
  }

  const size_t page_idx = static_cast<size_t>(slot) / page_size;
  const size_t entry_idx = static_cast<size_t>(slot) % page_size;
  const size_t append_k_base =
      static_cast<size_t>(token_idx) * append_k_stride_n + head_idx * append_k_stride_h;
  const size_t append_v_base =
      static_cast<size_t>(token_idx) * append_v_stride_n + head_idx * append_v_stride_h;

  uint8_t* k_out =
      paged_k_cache + page_idx * k_stride_page + entry_idx * k_stride_n + head_idx * k_stride_h;
  uint8_t* v_out =
      paged_v_cache + page_idx * v_stride_page + entry_idx * v_stride_n + head_idx * v_stride_h;
  uint8_t* k_sf_out = k_scale_cache + page_idx * k_sf_stride_page + entry_idx * k_sf_stride_n +
                      head_idx * k_sf_stride_h;
  uint8_t* v_sf_out = v_scale_cache + page_idx * v_sf_stride_page + entry_idx * v_sf_stride_n +
                      head_idx * v_sf_stride_h;

  for (uint32_t sf_idx = threadIdx.x; sf_idx < NUM_SF_BLOCKS * 2; sf_idx += blockDim.x) {
    const bool is_v = sf_idx >= NUM_SF_BLOCKS;
    const uint32_t block_idx = is_v ? sf_idx - NUM_SF_BLOCKS : sf_idx;
    const uint32_t dim_base = block_idx * SF_VEC_SIZE;
    const uint32_t packed_base = block_idx * PACKED_PER_SF;
    if (is_v) {
      nvfp4_append_quantize_block(append_value, v_scale, append_v_base, dim_base,
                                  v_out + packed_base, v_sf_out + block_idx);
    } else {
      nvfp4_append_quantize_block(append_key, k_scale, append_k_base, dim_base, k_out + packed_base,
                                  k_sf_out + block_idx);
    }
  }
}

// PIE: REMOVED -- `NVFP4QuantizeAppendPagedKVCache` and its `WithSlotMapping` twin, host
// launchers. 89 lines of host C++, guarded out of every NVRTC compile before it was removed, so
// removing it changes no compile. This marker is one a strip does NOT undo; see MODIFICATIONS.

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

  // Internal layout:
  // [max_num_pages, page_size, head_dim]
  DType* ckv_data;
  DType* kpe_data;
  IdType* indices;

  // [batch_size + 1] The page indptr array, with the first element 0, the last element nnz_pages
  IdType* indptr;
  // [batch_size] The offset of the last page for each request in the batch
  IdType* last_page_len;
  // [batch_size] The start position of each request in the batch.
  IdType* rope_pos_offset;

  /*!
   * \brief Construct an empty paged key-value cache
   */
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

  /*!
   * \brief Construct a paged mla kv cache
   * \param page_size The size of each page
   * \param head_dim_compressed_kv The dimension of compressed-kv
   * \param head_dim_kpe The dimension of k-pe
   * \param batch_size The batch size
   * \param compressed_kv_data The start pointer of compressed-kv cache, cache should be contiguous
   * \param kpe_data The start pointer of k-pe cache, cache should be contiguous
   * \param indices The page indices array
   * \param indptr The page indptr array
   * \param last_page_len The offset of the last page for each request in the batch
   * \param rope_pos_offset The start position of each request in the batch.
   */
// PIE: REMOVED -- a `__host__` `paged_kv_mla_t` constructor. 20 lines of host C++, guarded out
// of every NVRTC compile before it was removed, so removing it changes no compile. This marker
// is one a strip does NOT undo; see MODIFICATIONS.

  /*!
   * \brief Construct a paged key-value cache with custom kv-cache strides
   * \param page_size The size of each page
   * \param head_dim_compressed_kv The dimension of compressed-kv
   * \param head_dim_kpe The dimension of k-pe
   * \param batch_size The batch size
   * \param compressed_kv_data The start pointer of compressed-kv cache, cache should be contiguous
   * \param compressed_kv_strides custom strides of each dimensions of compressed-kv cache
   * \param kpe_data The start pointer of k-pe cache, cache should be contiguous
   * \param kpe_strides custom strides of each dimensions of k-pe cache
   * \param indices The page indices array
   * \param indptr The page indptr array
   * \param last_page_len The offset of the last page for each request in the batch
   * \param rope_pos_offset The start position of each request in the batch.
   */
// PIE: REMOVED -- the second `__host__` `paged_kv_mla_t` constructor. 22 lines of host C++,
// guarded out of every NVRTC compile before it was removed, so removing it changes no compile.
// This marker is one a strip does NOT undo; see MODIFICATIONS.

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

template <uint32_t head_dim_ckv, uint32_t head_dim_kpe, uint32_t vec_size, typename DType,
          typename IdType>
__global__ void AppendPagedKVMlaCacheKernel(paged_kv_mla_t<DType, IdType> paged_kv_mla,
                                            DType* __restrict__ append_ckv,
                                            DType* __restrict__ append_kpe,
                                            IdType* __restrict__ batch_indices,
                                            IdType* __restrict__ positions, uint32_t nnz,
                                            size_t append_ckv_stride_n,
                                            size_t append_kpe_stride_n) {
  uint32_t tx = threadIdx.x;
  uint32_t cta_id = blockIdx.x;
  uint32_t num_ctas = gridDim.x;

#pragma unroll 4
  for (uint32_t i = cta_id; i < nnz; i += num_ctas) {
    uint32_t page_iter, entry_idx;
    paged_kv_mla.page_size.divmod(
        paged_kv_mla.indptr[batch_indices[i]] * paged_kv_mla.page_size + positions[i], page_iter,
        entry_idx);
    DType* ckv_ptr = paged_kv_mla.get_ckv_ptr(page_iter, entry_idx, tx * vec_size);
    vec_t<DType, vec_size>::memcpy(ckv_ptr, append_ckv + i * append_ckv_stride_n + tx * vec_size);

    if (tx * vec_size < head_dim_kpe) {
      DType* kpe_ptr = paged_kv_mla.get_kpe_ptr(page_iter, entry_idx, tx * vec_size);
      vec_t<DType, vec_size>::memcpy(kpe_ptr, append_kpe + i * append_kpe_stride_n + tx * vec_size);
    }
  }
}

// PIE: REMOVED -- `AppendPagedKVMlaCache`, a host launcher. 39 lines of host C++, guarded out
// of every NVRTC compile before it was removed, so removing it changes no compile. This marker
// is one a strip does NOT undo; see MODIFICATIONS.

}  // namespace flashinfer

#endif  // FLAHSINFER_PAGE_CUH_
