/*!
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
#ifndef PIE_FA2_CASCADE_CUH_
#define PIE_FA2_CASCADE_CUH_

#include "attn/fa2/cp_async.cuh"
#include "attn/fa2/math.cuh"
#include "attn/fa2/utils.cuh"
#include "attn/fa2/state.cuh"

namespace pie::attn::fa2 {

using cp_async::PrefetchMode;
using cp_async::SharedMemFillMode;















template <uint32_t bdx, uint32_t bdy, uint32_t vec_size, typename DTypeIn>
__device__ __forceinline__ void threadblock_sync_state(state_t<vec_size>& st, DTypeIn* v_smem,
                                                       float* s_smem,
                                                       const uint32_t tx = threadIdx.x,
                                                       const uint32_t ty = threadIdx.y) {
  constexpr uint32_t head_dim = vec_size * bdx;
  st.o.cast_store(v_smem + ty * head_dim + tx * vec_size);
  s_smem[ty] = st.get_lse();
  st.init();
  __syncthreads();

#pragma unroll
  for (uint32_t iter = 0; iter < bdy; ++iter) {
    float s = s_smem[iter];
    vec_t<float, vec_size> v;
    v.cast_load(v_smem + iter * head_dim + tx * vec_size);
    st.merge(v, s, 1);
  }
}

template <uint32_t bdx, uint32_t bdy, uint32_t vec_size, typename DTypeIn>
__device__ __forceinline__ void warp_sync_state(state_t<vec_size>& st, DTypeIn* v_smem,
                                                float* s_smem, const uint32_t tx = threadIdx.x,
                                                const uint32_t ty = threadIdx.y) {
  constexpr uint32_t head_dim = vec_size * bdx;
  st.o.cast_store(v_smem + ty * head_dim + tx * vec_size);
  s_smem[ty] = st.get_lse();
  st.init();
  __syncwarp();

#pragma unroll
  for (uint32_t iter = 0; iter < bdy; ++iter) {
    float s = s_smem[iter];
    vec_t<float, vec_size> v;
    v.cast_load(v_smem + iter * head_dim + tx * vec_size);
    st.merge(v, s, 1);
  }
}

template <uint32_t bdx, uint32_t bdy, uint32_t vec_size, typename DTypeIn>
__device__ __forceinline__ void threadblock_sum(vec_t<float, vec_size>& v, DTypeIn* v_smem) {
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t head_dim = vec_size * bdx;
  v.cast_store(v_smem + ty * head_dim + tx * vec_size);
  v.fill(DTypeIn(0.f));
  __syncthreads();

#pragma unroll
  for (uint32_t iter = 0; iter < bdy; ++iter) {
    vec_t<float, vec_size> v_iter;
    v_iter.cast_load(v_smem + iter * head_dim + tx * vec_size);
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      v[i] += v_iter[i];
    }
  }
}

template <uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t num_smem_stages, typename DTypeIn,
          typename DTypeO, typename IdType>
__global__ void PersistentVariableLengthMergeStatesKernel(
    DTypeIn* __restrict__ V, float* __restrict__ S, IdType* indptr, DTypeO* __restrict__ v_merged,
    float* __restrict__ s_merged, uint32_t max_seq_len, uint32_t* __restrict__ seq_len_ptr,
    uint32_t num_heads, const uint32_t* __restrict__ win) {
  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t cta_id = blockIdx.x;
  uint32_t num_ctas = gridDim.x;

  // pie: the staged lengths are read only once the producer is done
  // (`kernel.pdl`).
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  const uint32_t staged = seq_len_ptr ? *seq_len_ptr : max_seq_len;
  const uint32_t seq_len = (win != nullptr && win[0] < staged) ? win[0] : staged;
  const uint32_t row_base = win != nullptr ? win[1] : 0;
  uint32_t num_iters = ceil_div(seq_len * num_heads, num_ctas);
  constexpr uint32_t vec_bits = sizeof(DTypeIn) * vec_size * 8;
  constexpr uint32_t head_dim = vec_size * bdx;
  extern __shared__ uint8_t smem[];
  DTypeIn* v_smem = (DTypeIn*)smem;
  float* s_smem = (float*)(smem + num_smem_stages * bdy * head_dim * sizeof(DTypeIn));

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

#pragma unroll 1
  for (uint32_t i = cta_id; i < seq_len * num_heads; i += num_ctas) {

    __syncthreads();

    uint32_t pos = i / num_heads;
    uint32_t head_idx = i % num_heads;
    state_t<vec_size> st;
    const uint32_t num_index_sets = indptr[pos + 1] - indptr[pos];

    if (num_index_sets == 0) {
      vec_t<DTypeO, vec_size> v;
      v.fill(DTypeO(0.f));
      v.store(v_merged + ((row_base + pos) * num_heads + head_idx) * head_dim + tx * vec_size);
      if (s_merged != nullptr) {
        s_merged[(row_base + pos) * num_heads + head_idx] = -math::inf;
      }
      continue;
    }

    if (num_index_sets == 1) {
      vec_t<DTypeO, vec_size> v;
      v.cast_load(V + (indptr[pos] * num_heads + head_idx) * head_dim + tx * vec_size);
      v.store(v_merged + ((row_base + pos) * num_heads + head_idx) * head_dim + tx * vec_size);
      if (s_merged != nullptr) {
        s_merged[(row_base + pos) * num_heads + head_idx] = S[indptr[pos] * num_heads + head_idx];
      }
      continue;
    }

#pragma unroll
    for (uint32_t iter = 0; iter < num_smem_stages; ++iter) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          v_smem + (iter * bdy + ty) * head_dim + tx * vec_size,
          V + ((indptr[pos] + (iter * bdy + ty)) * num_heads + head_idx) * head_dim + tx * vec_size,
          (iter * bdy + ty) < num_index_sets);
      cp_async::commit_group();
    }
#pragma unroll 4
    for (uint32_t iter = 0; iter < ceil_div(num_index_sets, bdy); ++iter) {
      if (iter % bdx == 0) {
        s_smem[ty * bdx + tx] =
            iter * bdy + (ty * bdx + tx) < num_index_sets
                ? S[(indptr[pos] + (iter * bdy + ty * bdx + tx)) * num_heads + head_idx]
                : 0.f;
        __syncthreads();
      }
      cp_async::wait_group<num_smem_stages - 1>();
      __syncthreads();
      vec_t<float, vec_size> v;
      v.cast_load(v_smem + ((iter % num_smem_stages) * bdy + ty) * head_dim + tx * vec_size);
      if (iter * bdy + ty < num_index_sets) {
        float s = s_smem[(iter % bdx) * bdy + ty];
        st.merge(v, s, 1);
      }
      __syncthreads();
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          v_smem + ((iter % num_smem_stages) * bdy + ty) * head_dim + tx * vec_size,
          V +
              ((indptr[pos] + ((iter + num_smem_stages) * bdy + ty)) * num_heads + head_idx) *
                  head_dim +
              tx * vec_size,
          (iter + num_smem_stages) * bdy + ty < num_index_sets);
      cp_async::commit_group();
    }
    cp_async::wait_group<0>();
    __syncthreads();

    st.normalize();
    threadblock_sync_state<bdx, bdy, vec_size>(st, v_smem, s_smem);
    st.normalize();

    st.o.cast_store(v_merged + ((row_base + pos) * num_heads + head_idx) * head_dim +
                    tx * vec_size);
    if (s_merged != nullptr) {
      s_merged[(row_base + pos) * num_heads + head_idx] = st.get_lse();
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

}

#endif
