














#ifndef FLASHINFER_POS_ENC_CUH_
#define FLASHINFER_POS_ENC_CUH_

#include <cstdint>

#include <type_traits>

#include "layout.cuh"
#include "math.cuh"
#include "page.cuh"
#include "utils.cuh"
#include "vec_dtypes.cuh"

namespace flashinfer {

struct RopeQuantizeAppendPagedKVCacheParams {
  uint32_t nnz;
  uint32_t num_qo_heads;
  uint32_t num_kv_heads;
  uint32_t rope_dim;
  uint32_t no_rope_dim;
  size_t q_rope_in_stride_n;
  size_t q_rope_in_stride_h;
  size_t q_nope_in_stride_n;
  size_t q_nope_in_stride_h;
  size_t q_rope_out_stride_n;
  size_t q_rope_out_stride_h;
  size_t q_nope_out_stride_n;
  size_t q_nope_out_stride_h;
  size_t k_rope_in_stride;
  size_t k_rope_in_stride_h;
  size_t k_nope_in_stride;
  size_t k_nope_in_stride_h;
  size_t v_in_stride;
  size_t v_in_stride_h;
  float quant_scale_q;
  float quant_scale_kv;
};




enum class PosEncodingMode {

  kNone = 0U,

  kRoPELlama = 1U,

  kALiBi = 2U
};





__device__ __forceinline__ float get_alibi_slope(uint32_t head_idx, uint32_t num_heads) {
  int n = math::ptx_exp2((int)math::ptx_log2(num_heads));
  return head_idx < n ? math::ptx_exp2(-8. * float(head_idx + 1) / float(n))
                      : math::ptx_exp2(-4. * float((head_idx + 1 - n) * 2 - 1) / float(n));
}











template <uint32_t vec_size, uint32_t bdx, typename T>
__device__ __forceinline__ vec_t<float, vec_size> vec_apply_llama_rope(
    const T* x, const vec_t<float, vec_size>& freq, int32_t offset,
    const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<float, vec_size> permuted_vec, vec;
  vec.cast_load(x + threadIdx.x * vec_size);

  if (threadIdx.x * vec_size < rotary_dim) {
    permuted_vec.cast_load(x + ((threadIdx.x * vec_size < rotary_dim / 2)
                                    ? threadIdx.x * vec_size + rotary_dim / 2
                                    : threadIdx.x * vec_size - rotary_dim / 2));
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      float embed = float(offset) * freq[i];
      float cos, sin;
      __sincosf(embed, &sin, &cos);
      vec[i] =
          vec[i] * cos +
          ((threadIdx.x * vec_size < rotary_dim / 2) ? -permuted_vec[i] : permuted_vec[i]) * sin;
    }
  }
  return vec;
}

template <uint32_t vec_size, uint32_t bdx, typename T>
__device__ __forceinline__ vec_t<float, vec_size> vec_apply_llama_rope_cos_sin(
    const T* x, const vec_t<float, vec_size>& cos, const vec_t<float, vec_size>& sin,
    const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<float, vec_size> permuted_vec, vec;
  vec.cast_load(x + threadIdx.x * vec_size);

  if (threadIdx.x * vec_size < rotary_dim) {
    permuted_vec.cast_load(x + ((threadIdx.x * vec_size < rotary_dim / 2)
                                    ? threadIdx.x * vec_size + rotary_dim / 2
                                    : threadIdx.x * vec_size - rotary_dim / 2));
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      vec[i] =
          vec[i] * cos[i] +
          ((threadIdx.x * vec_size < rotary_dim / 2) ? -permuted_vec[i] : permuted_vec[i]) * sin[i];
    }
  }
  return vec;
}











template <uint32_t vec_size, uint32_t bdx, typename T>
__device__ __forceinline__ vec_t<float, vec_size> vec_apply_llama_rope_interleave(
    const T* x, const vec_t<float, vec_size>& freq, int32_t offset,
    const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<float, vec_size> vec, vec_before;
  vec.cast_load(x + threadIdx.x * vec_size);

  if (threadIdx.x * vec_size < rotary_dim) {
    vec_before = vec;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      float embed = float(offset) * freq[i];
      float cos, sin;
      __sincosf(embed, &sin, &cos);
      vec[i] = vec[i] * cos + ((i % 2 == 0) ? -vec_before[i ^ 1] : vec_before[i ^ 1]) * sin;
    }
  }
  return vec;
}

template <uint32_t vec_size, uint32_t bdx, typename T>
__device__ __forceinline__ vec_t<float, vec_size> vec_apply_llama_rope_cos_sin_interleave(
    const T* x, const vec_t<float, vec_size>& cos, const vec_t<float, vec_size>& sin,
    const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<float, vec_size> vec, vec_before;
  vec.cast_load(x + threadIdx.x * vec_size);

  if (threadIdx.x * vec_size < rotary_dim) {
    vec_before = vec;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      vec[i] = vec[i] * cos[i] + ((i % 2 == 0) ? -vec_before[i ^ 1] : vec_before[i ^ 1]) * sin[i];
    }
  }
  return vec;
}



















template <uint32_t vec_size, uint32_t bdx, typename T>
__device__ __forceinline__ vec_t<float, vec_size>
vec_apply_llama_rope_cos_sin_interleave_reuse_half(const T* x, const vec_t<float, vec_size>& cos,
                                                   const vec_t<float, vec_size>& sin,
                                                   const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<float, vec_size> vec, vec_before;
  vec.cast_load(x + threadIdx.x * vec_size);

  if (threadIdx.x * vec_size < rotary_dim) {
    vec_before = vec;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {

      vec[i] = vec[i] * cos[i / 2] +
               ((i % 2 == 0) ? -vec_before[i ^ 1] : vec_before[i ^ 1]) * sin[i / 2];
    }
  }
  return vec;
}














template <typename DType, typename QuantType, uint32_t vec_size>
__device__ __forceinline__ void scale_store_partial_chunk(const DType* in_ptr, QuantType* out_ptr,
                                                          uint32_t lane_elem_offset,
                                                          uint32_t chunk_valid, float scale) {
  if (chunk_valid == 0 || lane_elem_offset >= chunk_valid) {
    return;
  }
  vec_t<float, vec_size> vec;
  if (lane_elem_offset + vec_size <= chunk_valid) {
    vec.cast_load(in_ptr + lane_elem_offset);
  } else {
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      uint32_t elem_idx = lane_elem_offset + i;
      if (elem_idx < chunk_valid) {
        vec_t<float, 1> tmp;
        tmp.cast_load(in_ptr + elem_idx);
        vec[i] = tmp[0];
      } else {
        vec[i] = 0.f;
      }
    }
  }
#pragma unroll
  for (uint32_t i = 0; i < vec_size; ++i) {
    vec[i] = vec[i] * scale;
  }
  if (lane_elem_offset + vec_size <= chunk_valid) {
    vec.cast_store(out_ptr + lane_elem_offset);
  } else {
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      uint32_t elem_idx = lane_elem_offset + i;
      if (elem_idx < chunk_valid) {
        vec_t<float, 1> tmp;
        tmp[0] = vec[i];
        tmp.cast_store(out_ptr + elem_idx);
      }
    }
  }
}

template <bool interleave, uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType,
          typename IdType>
__global__ void BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel(
    DType* q, DType* k, DType* q_rope, DType* k_rope, float* __restrict__ cos_sin_cache,
    IdType* __restrict__ pos_ids, uint32_t nnz, uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t rotary_dim, size_t q_stride_n, size_t q_stride_h, size_t k_stride_n, size_t k_stride_h,
    size_t q_rope_stride_n, size_t q_rope_stride_h, size_t k_rope_stride_n,
    size_t k_rope_stride_h) {
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  uint32_t by = blockIdx.y;
  const uint32_t bdy = blockDim.y;

  vec_t<float, vec_size> cos, sin;
  if (bx * bdy + ty < nnz) {
    const uint32_t idx = bx * bdy + ty;
    const IdType pos = pos_ids[idx];

    const int half_rotary_dim = rotary_dim / 2;

    if (tx * vec_size < rotary_dim) {
      int sin_offset = rotary_dim / 2;
      int vec_idx;
      if constexpr (interleave) {
        vec_idx = (tx * vec_size) / 2;
      } else {
        vec_idx = (tx * vec_size) % half_rotary_dim;
      }
      cos.load(cos_sin_cache + (pos * rotary_dim) + vec_idx);
      sin.load(cos_sin_cache + (pos * rotary_dim) + (sin_offset + vec_idx));
    }

    if (by < num_qo_heads) {
      uint32_t qo_head_idx = by;
      DType* q_ptr = q + get_elem_offset_impl(idx, qo_head_idx, 0, q_stride_n, q_stride_h);
      DType* q_rope_ptr =
          q_rope + get_elem_offset_impl(idx, qo_head_idx, 0, q_rope_stride_n, q_rope_stride_h);
      vec_t<float, vec_size> q_vec;
      if constexpr (interleave) {
        q_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(q_ptr, cos, sin,
                                                                                  rotary_dim);
      } else {
        q_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      }
      q_vec.cast_store(q_rope_ptr + tx * vec_size);
    } else {
      uint32_t kv_head_idx = by - num_qo_heads;
      DType* k_ptr = k + get_elem_offset_impl(idx, kv_head_idx, 0, k_stride_n, k_stride_h);
      DType* k_rope_ptr =
          k_rope + get_elem_offset_impl(idx, kv_head_idx, 0, k_rope_stride_n, k_rope_stride_h);
      vec_t<float, vec_size> k_vec;
      if constexpr (interleave) {
        k_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(k_ptr, cos, sin,
                                                                                  rotary_dim);
      } else {
        k_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      }
      k_vec.cast_store(k_rope_ptr + tx * vec_size);
    }
  }
}

template <bool interleave, uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType,
          typename IdType>
__global__ void BatchQKApplyRotaryPosIdsCosSinCacheKernel(
    DType* q, DType* k, DType* q_rope, DType* k_rope, float* __restrict__ cos_sin_cache,
    IdType* __restrict__ pos_ids, uint32_t nnz, uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t rotary_dim, size_t q_stride_n, size_t q_stride_h, size_t k_stride_n, size_t k_stride_h,
    size_t q_rope_stride_n, size_t q_rope_stride_h, size_t k_rope_stride_n,
    size_t k_rope_stride_h) {
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  const uint32_t bdy = blockDim.y;

  vec_t<float, vec_size> cos, sin;
  if (bx * bdy + ty < nnz) {
    const uint32_t idx = bx * bdy + ty;
    const IdType pos = pos_ids[idx];
    const int half_rotary_dim = rotary_dim / 2;

    if (tx * vec_size < rotary_dim) {
      int sin_offset = rotary_dim / 2;
      int vec_idx;
      if constexpr (interleave) {
        vec_idx = (tx * vec_size) / 2;
      } else {
        vec_idx = (tx * vec_size) % half_rotary_dim;
      }
      cos.load(cos_sin_cache + (pos * rotary_dim) + vec_idx);
      sin.load(cos_sin_cache + (pos * rotary_dim) + (sin_offset + vec_idx));
    }

#pragma unroll 1
    for (uint32_t qo_head_idx = 0; qo_head_idx < num_qo_heads; ++qo_head_idx) {
      DType* q_ptr = q + get_elem_offset_impl(idx, qo_head_idx, 0, q_stride_n, q_stride_h);
      DType* q_rope_ptr =
          q_rope + get_elem_offset_impl(idx, qo_head_idx, 0, q_rope_stride_n, q_rope_stride_h);
      vec_t<float, vec_size> q_vec;
      if constexpr (interleave) {
        q_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(q_ptr, cos, sin,
                                                                                  rotary_dim);
      } else {
        q_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      }
      q_vec.cast_store(q_rope_ptr + tx * vec_size);
    }

#pragma unroll 1
    for (uint32_t kv_head_idx = 0; kv_head_idx < num_kv_heads; ++kv_head_idx) {
      DType* k_ptr = k + get_elem_offset_impl(idx, kv_head_idx, 0, k_stride_n, k_stride_h);
      DType* k_rope_ptr =
          k_rope + get_elem_offset_impl(idx, kv_head_idx, 0, k_rope_stride_n, k_rope_stride_h);
      vec_t<float, vec_size> k_vec;
      if constexpr (interleave) {
        k_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(k_ptr, cos, sin,
                                                                                  rotary_dim);
      } else {
        k_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      }
      k_vec.cast_store(k_rope_ptr + tx * vec_size);
    }
  }
}

template <bool interleave, uint32_t vec_size, uint32_t bdx, typename DType, typename IdType,
          typename QuantType>
__global__ void RopeQuantizeKernel(
    DType* q_rope_in, DType* k_rope_in, DType* q_nope_in, DType* k_nope_in, QuantType* q_rope_out,
    QuantType* k_rope_out, QuantType* q_nope_out, QuantType* k_nope_out,
    float* __restrict__ cos_sin_cache, IdType* __restrict__ pos_ids, uint32_t nnz,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t rope_dim, uint32_t no_rope_dim,
    size_t q_rope_in_stride_n, size_t q_rope_in_stride_h, size_t q_nope_in_stride_n,
    size_t q_nope_in_stride_h, size_t q_rope_out_stride_n, size_t q_rope_out_stride_h,
    size_t q_nope_out_stride_n, size_t q_nope_out_stride_h, size_t k_rope_in_stride,
    size_t k_rope_in_stride_h, size_t k_nope_in_stride, size_t k_nope_in_stride_h,
    size_t k_rope_out_stride, size_t k_rope_out_stride_h, size_t k_nope_out_stride,
    size_t k_nope_out_stride_h, float quant_scale_q, float quant_scale_kv) {
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  uint32_t by = blockIdx.y;
  uint32_t bdy = blockDim.y;

  uint32_t rope_chunk_size = rope_dim;
  uint32_t rope_chunks = (rope_dim + rope_chunk_size - 1) / rope_chunk_size;
  uint32_t no_rope_chunks = (no_rope_dim + rope_chunk_size - 1) / rope_chunk_size;

  uint32_t q_rope_end = num_qo_heads * rope_chunks;
  uint32_t k_rope_end = q_rope_end + num_kv_heads * rope_chunks;
  uint32_t k_nope_end = k_rope_end + num_kv_heads * no_rope_chunks;

  vec_t<float, vec_size> cos, sin;
  if (bx * bdy + ty < nnz) {
    const uint32_t idx = bx * bdy + ty;
    const IdType pos = pos_ids[idx];

    const int half_rope_dim = rope_dim / 2;

    if ((tx * vec_size < rope_dim) && (by < k_rope_end)) {
      int sin_offset = rope_dim / 2;
      int vec_idx;
      if constexpr (interleave) {
        vec_idx = (tx * vec_size) / 2;
      } else {
        vec_idx = (tx * vec_size) % half_rope_dim;
      }
      cos.load(cos_sin_cache + (pos * rope_dim) + vec_idx);
      sin.load(cos_sin_cache + (pos * rope_dim) + (sin_offset + vec_idx));
    }

    if (by < q_rope_end) {

      uint32_t q_head_idx = by / rope_chunks;
      uint32_t rope_chunk_idx = by % rope_chunks;
      uint32_t elem_offset = rope_chunk_idx * rope_chunk_size;

      DType* q_rope_in_ptr =
          q_rope_in + get_elem_offset_impl(idx, q_head_idx, elem_offset, q_rope_in_stride_n,
                                           q_rope_in_stride_h);
      QuantType* q_rope_out_ptr =
          q_rope_out + get_elem_offset_impl(idx, q_head_idx, elem_offset, q_rope_out_stride_n,
                                            q_rope_out_stride_h);

      vec_t<float, vec_size> q_rope_vec;
      if constexpr (interleave) {
        q_rope_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(
            q_rope_in_ptr, cos, sin, rope_dim);
      } else {
        q_rope_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(q_rope_in_ptr, cos, sin, rope_dim);
      }
#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        q_rope_vec[i] = q_rope_vec[i] * quant_scale_q;
      }
      q_rope_vec.cast_store(q_rope_out_ptr + tx * vec_size);

    } else if (by < k_rope_end) {

      uint32_t k_head_idx = (by - q_rope_end) / rope_chunks;
      uint32_t rope_chunk_idx = (by - q_rope_end) % rope_chunks;
      uint32_t elem_offset = rope_chunk_idx * rope_chunk_size;

      DType* k_rope_in_ptr = k_rope_in + get_elem_offset_impl(idx, k_head_idx, elem_offset,
                                                              k_rope_in_stride, k_rope_in_stride_h);
      QuantType* k_rope_out_ptr =
          k_rope_out + get_elem_offset_impl(idx, k_head_idx, elem_offset, k_rope_out_stride,
                                            k_rope_out_stride_h);

      vec_t<float, vec_size> k_rope_vec;
      if constexpr (interleave) {
        k_rope_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(
            k_rope_in_ptr, cos, sin, rope_dim);
      } else {
        k_rope_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(k_rope_in_ptr, cos, sin, rope_dim);
      }
#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        k_rope_vec[i] = k_rope_vec[i] * quant_scale_kv;
      }
      k_rope_vec.cast_store(k_rope_out_ptr + tx * vec_size);

    } else if (by < k_nope_end) {

      uint32_t k_head_idx = (by - k_rope_end) / no_rope_chunks;
      uint32_t nope_chunk_idx = (by - k_rope_end) % no_rope_chunks;
      uint32_t elem_offset = nope_chunk_idx * rope_chunk_size;

      DType* k_nope_in_ptr = k_nope_in + get_elem_offset_impl(idx, k_head_idx, elem_offset,
                                                              k_nope_in_stride, k_nope_in_stride_h);
      QuantType* k_nope_out_ptr =
          k_nope_out + get_elem_offset_impl(idx, k_head_idx, elem_offset, k_nope_out_stride,
                                            k_nope_out_stride_h);

      uint32_t chunk_valid =
          (elem_offset < no_rope_dim) ? min(rope_chunk_size, no_rope_dim - elem_offset) : 0u;
      uint32_t lane_elem_offset = tx * vec_size;

      scale_store_partial_chunk<DType, QuantType, vec_size>(
          k_nope_in_ptr, k_nope_out_ptr, lane_elem_offset, chunk_valid, quant_scale_kv);

    } else {

      uint32_t q_head_idx = (by - k_nope_end) / no_rope_chunks;
      uint32_t nope_chunk_idx = (by - k_nope_end) % no_rope_chunks;
      uint32_t elem_offset = nope_chunk_idx * rope_chunk_size;

      DType* q_nope_in_ptr =
          q_nope_in + get_elem_offset_impl(idx, q_head_idx, elem_offset, q_nope_in_stride_n,
                                           q_nope_in_stride_h);
      QuantType* q_nope_out_ptr =
          q_nope_out + get_elem_offset_impl(idx, q_head_idx, elem_offset, q_nope_out_stride_n,
                                            q_nope_out_stride_h);

      uint32_t chunk_valid =
          (elem_offset < no_rope_dim) ? min(rope_chunk_size, no_rope_dim - elem_offset) : 0u;
      uint32_t lane_elem_offset = tx * vec_size;

      scale_store_partial_chunk<DType, QuantType, vec_size>(
          q_nope_in_ptr, q_nope_out_ptr, lane_elem_offset, chunk_valid, quant_scale_q);
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <bool interleave, uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType,
          typename IdType>
__global__ void BatchQKApplyRotaryPosIdsHeadParallelismKernel(
    DType* q, DType* k, DType* q_rope, DType* k_rope, IdType* __restrict__ pos_ids, uint32_t nnz,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t rotary_dim, size_t q_stride_n,
    size_t q_stride_h, size_t k_stride_n, size_t k_stride_h, size_t q_rope_stride_n,
    size_t q_rope_stride_h, size_t k_rope_stride_n, size_t k_rope_stride_h, float smooth_a,
    float smooth_b, float rope_rcp_scale, float rope_rcp_theta) {

  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  uint32_t by = blockIdx.y;
  const uint32_t bdy = blockDim.y;
  vec_t<float, vec_size> freq;
  if (tx * vec_size < rotary_dim) {
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      if constexpr (interleave) {
        freq[i] = __powf(rope_rcp_theta, float(2 * ((tx * vec_size + i) / 2)) / float(rotary_dim));
      } else {
        freq[i] = __powf(rope_rcp_theta,
                         float(2 * ((tx * vec_size + i) % (rotary_dim / 2))) / float(rotary_dim));
      }

      float smooth = freq[i] * smooth_a + smooth_b;
      smooth = max(0.0f, min(1.0f, smooth));
      freq[i] = (1 - smooth) * (freq[i] * rope_rcp_scale) + smooth * freq[i];
    }
  }

  vec_t<float, vec_size> cos, sin;

  if (bx * bdy + ty < nnz) {
    const uint32_t idx = bx * bdy + ty;
    const IdType pos = pos_ids[idx];

    if (tx * vec_size < rotary_dim) {
#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        float embed = float(pos) * freq[i];
        __sincosf(embed, &sin[i], &cos[i]);
      }
    }

    if (by < num_qo_heads) {
      uint32_t qo_head_idx = by;
      DType* q_ptr = q + get_elem_offset_impl(idx, qo_head_idx, 0, q_stride_n, q_stride_h);
      DType* q_rope_ptr =
          q_rope + get_elem_offset_impl(idx, qo_head_idx, 0, q_rope_stride_n, q_rope_stride_h);
      vec_t<float, vec_size> q_vec;
      if constexpr (interleave) {
        q_vec = vec_apply_llama_rope_cos_sin_interleave<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      } else {
        q_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      }
      q_vec.cast_store(q_rope_ptr + tx * vec_size);
    } else {
      uint32_t kv_head_idx = by - num_qo_heads;
      DType* k_ptr = k + get_elem_offset_impl(idx, kv_head_idx, 0, k_stride_n, k_stride_h);
      DType* k_rope_ptr =
          k_rope + get_elem_offset_impl(idx, kv_head_idx, 0, k_rope_stride_n, k_rope_stride_h);
      vec_t<float, vec_size> k_vec;
      if constexpr (interleave) {
        k_vec = vec_apply_llama_rope_cos_sin_interleave<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      } else {
        k_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      }
      k_vec.cast_store(k_rope_ptr + tx * vec_size);
    }
  }
}

template <bool interleave, uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType,
          typename IdType>
__global__ void BatchQKApplyRotaryPosIdsKernel(
    DType* q, DType* k, DType* q_rope, DType* k_rope, IdType* __restrict__ pos_ids, uint32_t nnz,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t rotary_dim, size_t q_stride_n,
    size_t q_stride_h, size_t k_stride_n, size_t k_stride_h, size_t q_rope_stride_n,
    size_t q_rope_stride_h, size_t k_rope_stride_n, size_t k_rope_stride_h, float smooth_a,
    float smooth_b, float rope_rcp_scale, float rope_rcp_theta) {

  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  const uint32_t bdy = blockDim.y;
  vec_t<float, vec_size> freq;
  if (tx * vec_size < rotary_dim) {
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      if constexpr (interleave) {
        freq[i] = __powf(rope_rcp_theta, float(2 * ((tx * vec_size + i) / 2)) / float(rotary_dim));
      } else {
        freq[i] = __powf(rope_rcp_theta,
                         float(2 * ((tx * vec_size + i) % (rotary_dim / 2))) / float(rotary_dim));
      }

      float smooth = freq[i] * smooth_a + smooth_b;
      smooth = max(0.0f, min(1.0f, smooth));
      freq[i] = (1 - smooth) * (freq[i] * rope_rcp_scale) + smooth * freq[i];
    }
  }

  vec_t<float, vec_size> cos, sin;

  if (bx * bdy + ty < nnz) {
    const uint32_t idx = bx * bdy + ty;
    const IdType pos = pos_ids[idx];

    if (tx * vec_size < rotary_dim) {
#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        float embed = float(pos) * freq[i];
        __sincosf(embed, &sin[i], &cos[i]);
      }
    }

#pragma unroll 1
    for (uint32_t qo_head_idx = 0; qo_head_idx < num_qo_heads; ++qo_head_idx) {
      DType* q_ptr = q + get_elem_offset_impl(idx, qo_head_idx, 0, q_stride_n, q_stride_h);
      DType* q_rope_ptr =
          q_rope + get_elem_offset_impl(idx, qo_head_idx, 0, q_rope_stride_n, q_rope_stride_h);
      vec_t<float, vec_size> q_vec;
      if constexpr (interleave) {
        q_vec = vec_apply_llama_rope_cos_sin_interleave<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      } else {
        q_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      }
      q_vec.cast_store(q_rope_ptr + tx * vec_size);
    }

#pragma unroll 1
    for (uint32_t kv_head_idx = 0; kv_head_idx < num_kv_heads; ++kv_head_idx) {
      DType* k_ptr = k + get_elem_offset_impl(idx, kv_head_idx, 0, k_stride_n, k_stride_h);
      DType* k_rope_ptr =
          k_rope + get_elem_offset_impl(idx, kv_head_idx, 0, k_rope_stride_n, k_rope_stride_h);
      vec_t<float, vec_size> k_vec;
      if constexpr (interleave) {
        k_vec = vec_apply_llama_rope_cos_sin_interleave<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      } else {
        k_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      }
      k_vec.cast_store(k_rope_ptr + tx * vec_size);
    }
  }
}

template <bool interleave, uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType,
          typename IdType>
__global__ void BatchQKApplyRotaryKernel(
    DType* q, DType* k, DType* q_rope, DType* k_rope, IdType* __restrict__ indptr,
    IdType* __restrict__ offsets, uint32_t batch_size, uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t rotary_dim, size_t q_stride_n, size_t q_stride_h, size_t k_stride_n, size_t k_stride_h,
    size_t q_rope_stride_n, size_t q_rope_stride_h, size_t k_rope_stride_n, size_t k_rope_stride_h,
    float smooth_a, float smooth_b, float rope_rcp_scale, float rope_rcp_theta) {
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  const uint32_t bdy = blockDim.y;
  vec_t<float, vec_size> freq;
  if (tx * vec_size < rotary_dim) {
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      if constexpr (interleave) {
        freq[i] = __powf(rope_rcp_theta, float(2 * ((tx * vec_size + i) / 2)) / float(rotary_dim));
      } else {
        freq[i] = __powf(rope_rcp_theta,
                         float(2 * ((tx * vec_size + i) % (rotary_dim / 2))) / float(rotary_dim));
      }

      float smooth = freq[i] * smooth_a + smooth_b;
      smooth = max(0.0f, min(1.0f, smooth));
      freq[i] = (1 - smooth) * (freq[i] * rope_rcp_scale) + smooth * freq[i];
    }
  }

  if (bx < batch_size * num_qo_heads) {

    const uint32_t batch_idx = bx / num_qo_heads;
    const uint32_t qo_head_idx = bx % num_qo_heads;
    const uint32_t seq_len = indptr[batch_idx + 1] - indptr[batch_idx];
    const uint32_t offset = offsets[batch_idx];
#pragma unroll 2
    for (uint32_t i = 0; i < (seq_len + bdy - 1) / bdy; ++i) {
      vec_t<float, vec_size> q_vec;
      if (i * bdy + ty < seq_len) {
        DType* q_ptr = q + get_elem_offset_impl(indptr[batch_idx] + i * bdy + ty, qo_head_idx, 0,
                                                q_stride_n, q_stride_h);
        DType* q_rope_ptr =
            q_rope + get_elem_offset_impl(indptr[batch_idx] + i * bdy + ty, qo_head_idx, 0,
                                          q_rope_stride_n, q_rope_stride_h);
        if constexpr (interleave) {
          q_vec = vec_apply_llama_rope_interleave<vec_size, bdx>(q_ptr, freq, offset + i * bdy + ty,
                                                                 rotary_dim);
        } else {
          q_vec =
              vec_apply_llama_rope<vec_size, bdx>(q_ptr, freq, offset + i * bdy + ty, rotary_dim);
        }
        q_vec.cast_store(q_rope_ptr + tx * vec_size);
      }
    }
  } else {

    uint32_t batch_idx = (bx - batch_size * num_qo_heads) / num_kv_heads;
    uint32_t kv_head_idx = (bx - batch_size * num_qo_heads) % num_kv_heads;
    const uint32_t seq_len = indptr[batch_idx + 1] - indptr[batch_idx];
    const uint32_t offset = offsets[batch_idx];
#pragma unroll 2
    for (uint32_t i = 0; i < (seq_len + bdy - 1) / bdy; ++i) {
      vec_t<float, vec_size> k_vec;
      if (i * bdy + ty < seq_len) {
        DType* k_ptr = k + get_elem_offset_impl(indptr[batch_idx] + i * bdy + ty, kv_head_idx, 0,
                                                k_stride_n, k_stride_h);
        DType* k_rope_ptr =
            k_rope + get_elem_offset_impl(indptr[batch_idx] + i * bdy + ty, kv_head_idx, 0,
                                          k_rope_stride_n, k_rope_stride_h);
        if constexpr (interleave) {
          k_vec = vec_apply_llama_rope_interleave<vec_size, bdx>(k_ptr, freq, offset + i * bdy + ty,
                                                                 rotary_dim);
        } else {
          k_vec =
              vec_apply_llama_rope<vec_size, bdx>(k_ptr, freq, offset + i * bdy + ty, rotary_dim);
        }
        k_vec.cast_store(k_rope_ptr + tx * vec_size);
      }
    }
  }
}






template <bool interleave, uint32_t vec_size, uint32_t bdx, typename DType, typename RoPEIdType,
          typename PagedKVIdType, typename QuantType, typename CacheT>
__global__ void RopeQuantizeAppendPagedKVCacheKernel(
    DType* q_rope_in, DType* k_rope_in, DType* q_nope_in, DType* k_nope_in, DType* v_in,
    QuantType* q_rope_out, QuantType* q_nope_out, CacheT paged_kv_like,
    PagedKVIdType* __restrict__ batch_indices, PagedKVIdType* __restrict__ positions,
    float* __restrict__ cos_sin_cache, RoPEIdType* __restrict__ pos_ids,
    const RopeQuantizeAppendPagedKVCacheParams params) {
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  uint32_t by = blockIdx.y;
  uint32_t bdy = blockDim.y;

  const uint32_t nnz = params.nnz;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const uint32_t num_kv_heads = params.num_kv_heads;
  const uint32_t rope_dim = params.rope_dim;
  const uint32_t no_rope_dim = params.no_rope_dim;
  const size_t q_rope_in_stride_n = params.q_rope_in_stride_n;
  const size_t q_rope_in_stride_h = params.q_rope_in_stride_h;
  const size_t q_nope_in_stride_n = params.q_nope_in_stride_n;
  const size_t q_nope_in_stride_h = params.q_nope_in_stride_h;
  const size_t q_rope_out_stride_n = params.q_rope_out_stride_n;
  const size_t q_rope_out_stride_h = params.q_rope_out_stride_h;
  const size_t q_nope_out_stride_n = params.q_nope_out_stride_n;
  const size_t q_nope_out_stride_h = params.q_nope_out_stride_h;
  const size_t k_rope_in_stride = params.k_rope_in_stride;
  const size_t k_rope_in_stride_h = params.k_rope_in_stride_h;
  const size_t k_nope_in_stride = params.k_nope_in_stride;
  const size_t k_nope_in_stride_h = params.k_nope_in_stride_h;
  const size_t v_in_stride = params.v_in_stride;
  const size_t v_in_stride_h = params.v_in_stride_h;
  const float quant_scale_q = params.quant_scale_q;
  const float quant_scale_kv = params.quant_scale_kv;

  uint32_t rope_chunk_size = rope_dim;
  uint32_t rope_chunks = (rope_dim + rope_chunk_size - 1) / rope_chunk_size;
  uint32_t no_rope_chunks = (no_rope_dim + rope_chunk_size - 1) / rope_chunk_size;

  uint32_t q_rope_end = num_qo_heads * rope_chunks;

  uint32_t k_rope_end = q_rope_end + num_kv_heads * rope_chunks;
  uint32_t k_nope_end = k_rope_end + num_kv_heads * no_rope_chunks;

  constexpr bool IS_MLA = std::is_same<CacheT, paged_kv_mla_t<QuantType, PagedKVIdType>>::value;

  vec_t<float, vec_size> cos, sin;
  if (bx * bdy + ty < nnz) {
    const uint32_t idx = bx * bdy + ty;
    const RoPEIdType pos = pos_ids[idx];

    if (batch_indices[idx] >= 0) {

      uint32_t page_iter, entry_idx;
      paged_kv_like.page_size.divmod(
          paged_kv_like.indptr[batch_indices[idx]] * paged_kv_like.page_size + positions[idx],
          page_iter, entry_idx);

      const int half_rope_dim = rope_dim / 2;

      if ((tx * vec_size < rope_dim) && (by < k_rope_end)) {
        int sin_offset = rope_dim / 2;
        int vec_idx;
        if constexpr (interleave) {
          vec_idx = (tx * vec_size) / 2;
        } else {
          vec_idx = (tx * vec_size) % half_rope_dim;
        }
        cos.load(cos_sin_cache + (pos * rope_dim) + vec_idx);
        sin.load(cos_sin_cache + (pos * rope_dim) + (sin_offset + vec_idx));
      }

      if (by < q_rope_end) {

        uint32_t q_head_idx = by / rope_chunks;
        uint32_t rope_chunk_idx = by % rope_chunks;
        uint32_t elem_offset = rope_chunk_idx * rope_chunk_size;

        DType* q_rope_in_ptr =
            q_rope_in + get_elem_offset_impl(idx, q_head_idx, elem_offset, q_rope_in_stride_n,
                                             q_rope_in_stride_h);
        QuantType* q_rope_out_ptr =
            q_rope_out + get_elem_offset_impl(idx, q_head_idx, elem_offset, q_rope_out_stride_n,
                                              q_rope_out_stride_h);

        vec_t<float, vec_size> q_rope_vec;
        if constexpr (interleave) {
          q_rope_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(
              q_rope_in_ptr, cos, sin, rope_dim);
        } else {
          q_rope_vec =
              vec_apply_llama_rope_cos_sin<vec_size, bdx>(q_rope_in_ptr, cos, sin, rope_dim);
        }
#pragma unroll
        for (uint32_t i = 0; i < vec_size; ++i) {
          q_rope_vec[i] = q_rope_vec[i] * quant_scale_q;
        }
        q_rope_vec.cast_store(q_rope_out_ptr + tx * vec_size);

      } else if (by < k_rope_end) {

        uint32_t k_head_idx = (by - q_rope_end) / rope_chunks;
        uint32_t rope_chunk_idx = (by - q_rope_end) % rope_chunks;
        uint32_t elem_offset = rope_chunk_idx * rope_chunk_size;

        DType* k_rope_in_ptr;
        if constexpr (IS_MLA) {

          k_rope_in_ptr = k_rope_in + idx * k_rope_in_stride + elem_offset;
        } else {

          k_rope_in_ptr = k_rope_in + get_elem_offset_impl(idx, k_head_idx, elem_offset,
                                                           k_rope_in_stride, k_rope_in_stride_h);
        }

        vec_t<float, vec_size> k_rope_vec;
        if constexpr (interleave) {
          k_rope_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(
              k_rope_in_ptr, cos, sin, rope_dim);
        } else {
          k_rope_vec =
              vec_apply_llama_rope_cos_sin<vec_size, bdx>(k_rope_in_ptr, cos, sin, rope_dim);
        }
#pragma unroll
        for (uint32_t i = 0; i < vec_size; ++i) {
          k_rope_vec[i] = k_rope_vec[i] * quant_scale_kv;
        }

        if constexpr (IS_MLA) {
          QuantType* kpe_ptr =
              paged_kv_like.get_kpe_ptr(page_iter, entry_idx, elem_offset + tx * vec_size);
          k_rope_vec.cast_store(kpe_ptr);
        } else {
          QuantType* k_ptr =
              paged_kv_like.get_k_ptr(page_iter, k_head_idx, entry_idx, tx * vec_size);
          k_rope_vec.cast_store(k_ptr);
        }

      } else if (by < k_nope_end) {

        uint32_t k_head_idx = (by - k_rope_end) / no_rope_chunks;
        uint32_t nope_chunk_idx = (by - k_rope_end) % no_rope_chunks;
        uint32_t elem_offset = nope_chunk_idx * rope_chunk_size;

        DType* k_nope_in_ptr;
        if constexpr (IS_MLA) {
          k_nope_in_ptr = k_nope_in + idx * k_nope_in_stride + elem_offset;
        } else {
          k_nope_in_ptr = k_nope_in + get_elem_offset_impl(idx, k_head_idx, elem_offset,
                                                           k_nope_in_stride, k_nope_in_stride_h);
        }

        vec_t<float, vec_size> k_nope_vec;
        k_nope_vec.cast_load(k_nope_in_ptr + tx * vec_size);
#pragma unroll
        for (uint32_t i = 0; i < vec_size; ++i) {
          k_nope_vec[i] = k_nope_vec[i] * quant_scale_kv;
        }

        if constexpr (IS_MLA) {
          QuantType* ckv_ptr =
              paged_kv_like.get_ckv_ptr(page_iter, entry_idx, elem_offset + tx * vec_size);
          k_nope_vec.cast_store(ckv_ptr);
        } else {
          QuantType* k_ptr = paged_kv_like.get_k_ptr(page_iter, k_head_idx, entry_idx,
                                                     rope_dim + elem_offset + tx * vec_size);
          k_nope_vec.cast_store(k_ptr);
        }

      } else if (by < k_nope_end + (IS_MLA ? 0u : num_kv_heads)) {

        if constexpr (!IS_MLA) {
          uint32_t kv_head_idx = by - k_nope_end;
          DType* v_in_ptr =
              v_in + get_elem_offset_impl(idx, kv_head_idx, 0, v_in_stride, v_in_stride_h);

          uint32_t head_dim_total = rope_dim + no_rope_dim;
          uint32_t v_chunks = (head_dim_total + rope_chunk_size - 1) / rope_chunk_size;
#pragma unroll 1
          for (uint32_t j = 0; j < v_chunks; ++j) {
            uint32_t v_elem_offset = j * rope_chunk_size;
            if (v_elem_offset + tx * vec_size < head_dim_total) {
              vec_t<float, vec_size> v_vec;
              v_vec.cast_load(v_in_ptr + v_elem_offset + tx * vec_size);
#pragma unroll
              for (uint32_t i = 0; i < vec_size; ++i) {
                v_vec[i] = v_vec[i] * quant_scale_kv;
              }
              QuantType* v_ptr = paged_kv_like.get_v_ptr(page_iter, kv_head_idx, entry_idx,
                                                         v_elem_offset + tx * vec_size);
              v_vec.cast_store(v_ptr);
            }
          }
        }

      } else {

        uint32_t q_nope_start = k_nope_end + (IS_MLA ? 0u : num_kv_heads);
        uint32_t q_head_idx = (by - q_nope_start) / no_rope_chunks;
        uint32_t nope_chunk_idx = (by - q_nope_start) % no_rope_chunks;
        uint32_t elem_offset = nope_chunk_idx * rope_chunk_size;

        DType* q_nope_in_ptr =
            q_nope_in + get_elem_offset_impl(idx, q_head_idx, elem_offset, q_nope_in_stride_n,
                                             q_nope_in_stride_h);
        QuantType* q_nope_out_ptr =
            q_nope_out + get_elem_offset_impl(idx, q_head_idx, elem_offset, q_nope_out_stride_n,
                                              q_nope_out_stride_h);

        vec_t<float, vec_size> q_nope_vec;
        q_nope_vec.cast_load(q_nope_in_ptr + tx * vec_size);
#pragma unroll
        for (uint32_t i = 0; i < vec_size; ++i) {
          q_nope_vec[i] = q_nope_vec[i] * quant_scale_q;
        }
        q_nope_vec.cast_store(q_nope_out_ptr + tx * vec_size);
      }
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}


}

#endif
