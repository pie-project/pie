

#ifndef FLASHINFER_DECODE_CUH_
#define FLASHINFER_DECODE_CUH_
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "../cp_async.cuh"
#include "../math.cuh"
#include "../pos_enc.cuh"
#include "../utils.cuh"
#include "../vec_dtypes.cuh"
#include "cascade.cuh"
#include "state.cuh"

namespace flashinfer {

DEFINE_HAS_MEMBER(decode_maybe_q_rope_offset)

namespace cg = cooperative_groups;
using cp_async::PrefetchMode;
using cp_async::SharedMemFillMode;

namespace {

template <PosEncodingMode pos_encoding_mode, uint32_t vec_size, uint32_t bdx, uint32_t tile_size,
          typename AttentionVariant, typename Params, typename T>
__device__ __forceinline__ void compute_qk(
    const Params& params, AttentionVariant variant, const uint32_t batch_idx, const T* smem,
    const vec_t<float, vec_size>& q_vec, const vec_t<float, vec_size>& freq, uint32_t kv_idx_base,
    uint32_t iter_base, uint32_t iter_bound, uint32_t qo_head_idx, uint32_t kv_head_idx, float* s,
    state_t<vec_size>& st, const uint32_t tx, const uint32_t ty, const uint32_t tz) {
  float m_prev = st.m;
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    vec_t<float, vec_size> k_vec;
    if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {

      k_vec = vec_apply_llama_rope<vec_size, bdx>(smem + j * bdx * vec_size, freq,
                                                  kv_idx_base + tz * tile_size + j);
    } else {

      k_vec.cast_load(smem + (j * bdx + tx) * vec_size);
    }
    s[j] = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      s[j] += q_vec[i] * k_vec[i];
    }
#pragma unroll
    for (uint32_t offset = bdx / 2; offset > 0; offset /= 2) {
      s[j] += math::shfl_xor_sync(s[j], offset);
    }
    const uint32_t pos = kv_idx_base + tz * tile_size + j;
    s[j] = variant.LogitsTransform(params, s[j], batch_idx, 0, pos,
                                   qo_head_idx, kv_head_idx);
    if constexpr (variant.use_softmax) {
      s[j] *= variant.sm_scale_log2;
    }

    bool mask = variant.LogitsMask(params, batch_idx, 0, pos, qo_head_idx,
                                   kv_head_idx);
    s[j] = (iter_base + tz * tile_size + j < iter_bound && mask) ? s[j] : -math::inf;
    st.m = max(st.m, s[j]);
  }

  if constexpr (variant.use_softmax) {
    float o_scale = math::ptx_exp2(m_prev - st.m);
    st.d *= o_scale;
#pragma unroll
    for (uint32_t j = 0; j < tile_size; ++j) {
      s[j] = math::ptx_exp2(s[j] - st.m);
      st.d += s[j];
    }
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      st.o[i] = st.o[i] * o_scale;
    }
  }
}

template <uint32_t vec_size, uint32_t bdx, uint32_t tile_size, typename T>
__device__ __forceinline__ void update_local_state(const T* smem, const float* s,
                                                   uint32_t compute_stage_idx,
                                                   state_t<vec_size>& st, uint32_t tx) {
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    vec_t<float, vec_size> v_vec;
    v_vec.cast_load(smem + (j * bdx + tx) * vec_size);
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      st.o[i] = st.o[i] + s[j] * v_vec[i];
    }
  }
}

template <uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, typename AttentionVariant>
__device__ __forceinline__ void sync_state(AttentionVariant variant, state_t<vec_size>& st,
                                           float* smem, float* smem_md, const uint32_t tx,
                                           const uint32_t ty, const uint32_t tz) {
  if constexpr (bdz > 1) {
    constexpr uint32_t head_dim = bdx * vec_size;
    auto block = cg::this_thread_block();
    st.o.store(smem + (tz * bdy + ty) * head_dim + tx * vec_size);
    if constexpr (variant.use_softmax) {
      smem_md[(tz * bdy + ty) * 2] = st.m;
      smem_md[(tz * bdy + ty) * 2 + 1] = st.d;
      block.sync();
      st.init();
#pragma unroll
      for (uint32_t j = 0; j < bdz; ++j) {
        float mz = smem_md[(j * bdy + ty) * 2], dz = smem_md[(j * bdy + ty) * 2 + 1];
        vec_t<float, vec_size> oz;
        oz.load(smem + (j * bdy + ty) * head_dim + tx * vec_size);
        st.merge(oz, mz, dz);
      }
    } else {
      block.sync();
      st.init();
#pragma unroll
      for (uint32_t j = 0; j < bdz; ++j) {
        vec_t<float, vec_size> oz;
        oz.load(smem + (j * bdy + ty) * head_dim + tx * vec_size);
#pragma unroll
        for (uint32_t i = 0; i < vec_size; ++i) {
          st.o[i] += oz[i];
        }
      }
    }
  }
}

}

template <PosEncodingMode pos_encoding_mode, uint32_t num_stages_smem, uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, typename AttentionVariant,
          typename Params>
__global__ void SingleDecodeWithKVCacheKernel(const __grid_constant__ Params params) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  const DTypeQ* q = params.q;
  const DTypeKV* k = params.k;
  const DTypeKV* v = params.v;
  const uint32_t q_stride_n = params.q_stride_n;
  const uint32_t q_stride_h = params.q_stride_h;
  const uint32_t kv_stride_n = params.kv_stride_n;
  const uint32_t kv_stride_h = params.kv_stride_h;
  DTypeO* o = params.o;
  float* lse = params.lse;
  uint32_t kv_chunk_size = params.kv_chunk_size;

  auto block = cg::this_thread_block();
  auto grid = cg::this_grid();

  constexpr uint32_t head_dim = bdx * vec_size;
  uint32_t kv_head_idx = blockIdx.y;
  uint32_t qo_head_idx = kv_head_idx * bdy + threadIdx.y;
  uint32_t kv_chunk_idx = blockIdx.x;
  uint32_t num_qo_heads = params.num_qo_heads;

  extern __shared__ uint8_t smem[];
  AttentionVariant variant(params, 0, smem);
  const uint32_t seq_len = variant.kv_len;
  DTypeKV* k_smem = (DTypeKV*)smem;
  DTypeKV* v_smem = (DTypeKV*)(smem + num_stages_smem * bdy * tile_size_per_bdx * bdz * head_dim *
                                          sizeof(DTypeKV));
  float* smem_md = (float*)(smem + 2 * num_stages_smem * bdy * tile_size_per_bdx * bdz * head_dim *
                                       sizeof(DTypeKV));

  uint32_t tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  vec_t<float, vec_size> q_vec;
  vec_t<float, vec_size> freq;
  if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
    const float rope_rcp_scale = params.rope_rcp_scale;
    const float rope_rcp_theta = params.rope_rcp_theta;

#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      freq[i] = rope_rcp_scale *
                __powf(rope_rcp_theta,
                       float(2 * ((tx * vec_size + i) % (head_dim / 2))) / float(head_dim));
    }

    q_vec = vec_apply_llama_rope<vec_size, bdx>(q + qo_head_idx * q_stride_h, freq, seq_len - 1);
  } else {

    q_vec.cast_load(q + qo_head_idx * q_stride_h + tx * vec_size);
  }
  block.sync();

  uint32_t chunk_start = kv_chunk_idx * kv_chunk_size;
  kv_chunk_size = min(kv_chunk_size, seq_len - chunk_start);
  uint32_t chunk_end = chunk_start + kv_chunk_size;

  uint32_t producer_kv_idx_base = chunk_start;
  constexpr uint32_t vec_bits = sizeof(DTypeKV) * vec_size * 8;
#pragma unroll
  for (uint32_t iter = 0; iter < num_stages_smem; ++iter) {
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          k + (producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j) * kv_stride_n +
              kv_head_idx * kv_stride_h + tx * vec_size,
          producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j < chunk_end);
    }
    cp_async::commit_group();
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
          v_smem + (((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          v + (producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j) * kv_stride_n +
              kv_head_idx * kv_stride_h + tx * vec_size,
          producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j < chunk_end);
    }
    cp_async::commit_group();
    producer_kv_idx_base += bdy * bdz * tile_size_per_bdx;
  }

  uint32_t consumer_kv_idx_base = chunk_start, stage_idx = 0;
  state_t<vec_size> st_local;
  float s[bdy * tile_size_per_bdx];

#pragma unroll 2
  for (uint32_t iter = 0; iter < ceil_div(kv_chunk_size, tile_size_per_bdx * bdy * bdz); ++iter) {

    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    compute_qk<pos_encoding_mode, vec_size, bdx, bdy * tile_size_per_bdx>(
        params, variant, 0,
        k_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, q_vec, freq,
        consumer_kv_idx_base, iter * bdy * tile_size_per_bdx * bdz, kv_chunk_size, qo_head_idx,
        kv_head_idx, s, st_local, tx, ty, tz);
    block.sync();

    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          k + (producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j) * kv_stride_n +
              kv_head_idx * kv_stride_h + tx * vec_size,
          producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j < chunk_end);
    }
    cp_async::commit_group();

    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    update_local_state<vec_size, bdx, bdy * tile_size_per_bdx>(
        v_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, s, stage_idx,
        st_local, tx);
    block.sync();

    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
          v_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          v + (producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j) * kv_stride_n +
              kv_head_idx * kv_stride_h + tx * vec_size,
          producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j < chunk_end);
    }
    cp_async::commit_group();

    stage_idx = (stage_idx + 1) % num_stages_smem;
    producer_kv_idx_base += tile_size_per_bdx * bdy * bdz;
    consumer_kv_idx_base += tile_size_per_bdx * bdy * bdz;
  }
  cp_async::wait_group<0>();
  block.sync();

  sync_state<vec_size, bdx, bdy, bdz>(variant, st_local, reinterpret_cast<float*>(smem), smem_md,
                                      tx, ty, tz);
#pragma unroll
  for (size_t i = 0; i < vec_size; ++i) {
    st_local.o[i] = variant.OutputTransform(params, st_local.o[i], 0, 0,
                                            qo_head_idx, st_local.m, st_local.d, 1.0f);
  }

  st_local.o.cast_store(o + (kv_chunk_idx * num_qo_heads + qo_head_idx) * head_dim + tx * vec_size);
  if (lse != nullptr) {
    lse[kv_chunk_idx * num_qo_heads + qo_head_idx] = st_local.get_lse();
  }
}

template <PosEncodingMode POS_ENCODING_MODE, uint32_t num_stages_smem, uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, typename AttentionVariant,
          typename Params>
__device__ __inline__ void BatchDecodeWithPagedKVCacheDevice(const Params& params, uint8_t smem[],
                                                             const uint32_t bx = blockIdx.x,
                                                             const uint32_t by = blockIdx.y,
                                                             const uint32_t tx = threadIdx.x,
                                                             const uint32_t ty = threadIdx.y,
                                                             const uint32_t tz = threadIdx.z) {
  auto block = cg::this_thread_block();
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  const DTypeQ* q = params.q;
  DTypeO* o = params.o;
  float* lse = params.lse;
  const auto paged_kv = params.paged_kv;
  const bool* block_valid_mask = params.block_valid_mask;
  const uint32_t padded_batch_size = params.padded_batch_size;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const bool partition_kv = params.partition_kv;

  constexpr uint32_t head_dim = bdx * vec_size;
  const uint32_t batch_idx = params.request_indices[bx];
  const uint32_t kv_tile_idx = params.kv_tile_indices[bx];
  const uint32_t kv_head_idx = by;
  const uint32_t qo_head_idx = kv_head_idx * bdy + ty;

  if (block_valid_mask && !block_valid_mask[bx]) return;
  const uint32_t kv_chunk_size = *(params.kv_chunk_size_ptr);
  const uint32_t kv_len = paged_kv.get_length(batch_idx);
  const uint32_t max_chunk_size = partition_kv ? kv_chunk_size : kv_len;
  const uint32_t chunk_start = partition_kv ? kv_tile_idx * max_chunk_size : 0;
  const uint32_t chunk_end =
      partition_kv ? min((kv_tile_idx + 1) * max_chunk_size, kv_len) : kv_len;
  if (partition_kv && chunk_start >= kv_len) {
    if (tz == 0) {
      state_t<vec_size> empty;
      empty.o.cast_store(o + (bx * num_qo_heads + qo_head_idx) * head_dim + tx * vec_size);
      if (lse != nullptr) {
        lse[bx * num_qo_heads + qo_head_idx] = empty.get_lse();
      }
    }
    return;
  }
  const uint32_t chunk_size = chunk_end - chunk_start;

  AttentionVariant variant(params, batch_idx, smem);
  DTypeKV* k_smem = (DTypeKV*)smem;
  DTypeKV* v_smem = (DTypeKV*)(smem + num_stages_smem * tile_size_per_bdx * bdy * bdz * head_dim *
                                          sizeof(DTypeKV));
  size_t* kv_offset_smem = (size_t*)(smem + 2 * num_stages_smem * tile_size_per_bdx * bdy * bdz *
                                                head_dim * sizeof(DTypeKV));
  float* smem_md = (float*)(smem + 2 * num_stages_smem * tile_size_per_bdx * bdy * bdz * head_dim *
                                       sizeof(DTypeKV));

  vec_t<float, vec_size> q_vec;
  vec_t<float, vec_size> freq;
  const uint32_t q_stride_n = params.q_stride_n;
  const uint32_t q_stride_h = params.q_stride_h;
  if constexpr (POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
    const IdType* q_rope_offset = nullptr;
    if constexpr (has_decode_maybe_q_rope_offset_v<Params>) {
      q_rope_offset = params.decode_maybe_q_rope_offset;
    }
    int32_t q_rope_offset_val = q_rope_offset == nullptr ? (kv_len - 1) : q_rope_offset[batch_idx];
    const float rope_rcp_scale = params.rope_rcp_scale;
    const float rope_rcp_theta = params.rope_rcp_theta;

#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      freq[i] = rope_rcp_scale *
                __powf(rope_rcp_theta,
                       float(2 * ((tx * vec_size + i) % (head_dim / 2))) / float(head_dim));
    }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif

    q_vec = vec_apply_llama_rope<vec_size, bdx>(
        q + params.q_indptr[batch_idx] * q_stride_n + qo_head_idx * q_stride_h, freq,
        q_rope_offset_val);
  } else {

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif

    q_vec.cast_load(q + params.q_indptr[batch_idx] * q_stride_n + qo_head_idx * q_stride_h +
                    tx * vec_size);
  }

  uint32_t stage_idx = 0;
  constexpr uint32_t vec_bits = sizeof(DTypeKV) * vec_size * 8;
  const IdType last_indptr = paged_kv.indptr[paged_kv.batch_size];

  static_assert(num_stages_smem <= bdx);
  uint32_t packed_page_iter_base = paged_kv.indptr[batch_idx] * paged_kv.page_size + chunk_start;
#pragma unroll
  for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
    uint32_t q, r;
    paged_kv.page_size.divmod(packed_page_iter_base + ((j * bdz + tz) * bdy + ty) * bdx + tx, q, r);
    kv_offset_smem[((j * bdz + tz) * bdy + ty) * bdx + tx] =
        paged_kv.protective_get_kv_offset(q, kv_head_idx, r, 0, last_indptr);
  }
  block.sync();

  size_t kv_offset[tile_size_per_bdx];
#pragma unroll
  for (uint32_t iter = 0; iter < num_stages_smem; ++iter) {
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      kv_offset[j] =
          kv_offset_smem[((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j] + tx * vec_size;
    }
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          paged_kv.k_data + kv_offset[j],
          ((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
    }
    cp_async::commit_group();
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
          v_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          paged_kv.v_data + kv_offset[j],
          ((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
    }
    cp_async::commit_group();
    stage_idx = (stage_idx + 1) % num_stages_smem;
  }

  state_t<vec_size> st;
  float s[bdy * tile_size_per_bdx];

#pragma unroll 2
  for (uint32_t iter = 0; iter < ceil_div(chunk_size, tile_size_per_bdx * bdy * bdz); ++iter) {
    if ((iter + num_stages_smem) % bdx == 0) {
#pragma unroll
      for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
        uint32_t q, r;
        paged_kv.page_size.divmod(
            packed_page_iter_base + ((iter + num_stages_smem) * tile_size_per_bdx * bdy * bdz +
                                     ((j * bdz + tz) * bdy + ty) * bdx + tx),
            q, r);
        kv_offset_smem[((j * bdz + tz) * bdy + ty) * bdx + tx] =
            paged_kv.protective_get_kv_offset(q, kv_head_idx, r, 0, last_indptr);
      }
    }

    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    compute_qk<POS_ENCODING_MODE, vec_size, bdx, bdy * tile_size_per_bdx>(
        params, variant, batch_idx,
        k_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, q_vec, freq,
        (paged_kv.rope_pos_offset == nullptr ? 0 : paged_kv.rope_pos_offset[batch_idx]) +
            chunk_start + iter * tile_size_per_bdx * bdy * bdz,
        iter * tile_size_per_bdx * bdy * bdz, chunk_size, qo_head_idx, kv_head_idx, s, st, tx, ty,
        tz);
    block.sync();

#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      kv_offset[j] = kv_offset_smem[((((iter + num_stages_smem) % bdx) * bdz + tz) * bdy + ty) *
                                        tile_size_per_bdx +
                                    j] +
                     tx * vec_size;
    }

#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          paged_kv.k_data + kv_offset[j],
          (((iter + num_stages_smem) * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
    }
    cp_async::commit_group();

    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    update_local_state<vec_size, bdx, bdy * tile_size_per_bdx>(
        v_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, s, stage_idx, st, tx);
    block.sync();

#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
          v_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          paged_kv.v_data + kv_offset[j],
          (((iter + num_stages_smem) * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
    }
    cp_async::commit_group();
    stage_idx = (stage_idx + 1) % num_stages_smem;
  }
  cp_async::wait_group<0>();
  block.sync();

  sync_state<vec_size, bdx, bdy, bdz>(variant, st, reinterpret_cast<float*>(smem), smem_md, tx, ty,
                                      tz);
#pragma unroll
  for (size_t i = 0; i < vec_size; ++i) {
    st.o[i] = variant.OutputTransform(params, st.o[i], bx, 0, qo_head_idx, st.m, st.d,
                                      1.0f);
  }

  if (tz == 0) {
    st.o.cast_store(o + (bx * num_qo_heads + qo_head_idx) * head_dim + tx * vec_size);

    if (lse != nullptr) {
      lse[bx * num_qo_heads + qo_head_idx] = st.get_lse();
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <PosEncodingMode POS_ENCODING_MODE, uint32_t num_stages_smem, uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, typename AttentionVariant,
          typename Params>
__global__ void BatchDecodeWithPagedKVCacheKernel(const __grid_constant__ Params params) {
  extern __shared__ uint8_t smem[];
  BatchDecodeWithPagedKVCacheDevice<POS_ENCODING_MODE, num_stages_smem, tile_size_per_bdx, vec_size,
                                    bdx, bdy, bdz, AttentionVariant>(params, smem);
}

constexpr uint32_t get_heuristic_num_threads(uint32_t group_size, uint32_t sizeof_dtype) {
  if (group_size == 8U) {
    if (sizeof_dtype == 1U) {
      return 256U;
    } else {
      return 512U;
    }
  } else {
    return 128U;
  }
}

template <uint32_t vec_size_ckv, uint32_t vec_size_kpe, uint32_t bdx, uint32_t tile_size,
          typename AttentionVariant, typename Params, typename T>
__device__ __forceinline__ void compute_qk_and_update_local_stat_mla(
    const Params& params, AttentionVariant variant, const uint32_t batch_idx, const T* ckv_smem,
    const vec_t<float, vec_size_ckv>& q_nope_vec, const T* kpe_smem,
    const vec_t<float, vec_size_kpe>& q_pe_vec, const vec_t<float, vec_size_kpe>& freq,
    uint32_t kv_idx_base, uint32_t iter_base, uint32_t iter_bound, state_t<vec_size_ckv>& st) {
  uint32_t tx = threadIdx.x, tz = threadIdx.z;
  constexpr uint32_t head_dim_ckv = bdx * vec_size_ckv;
  constexpr uint32_t head_dim_kpe = bdx * vec_size_kpe;
  float s[tile_size];
  float m_prev = st.m;
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    vec_t<float, vec_size_ckv> ckv_vec;
    ckv_vec.cast_load(ckv_smem + j * head_dim_ckv + tx * vec_size_ckv);

    vec_t<float, vec_size_kpe> kpe_vec;
    kpe_vec.cast_load(kpe_smem + j * head_dim_kpe + tx * vec_size_kpe);

    s[j] = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < vec_size_ckv; ++i) {
      s[j] += q_nope_vec[i] * ckv_vec[i];
    }
#pragma unroll
    for (uint32_t i = 0; i < vec_size_kpe; ++i) {
      s[j] += q_pe_vec[i] * kpe_vec[i];
    }
    s[j] *= params.sm_scale;
#pragma unroll
    for (uint32_t offset = bdx / 2; offset > 0; offset /= 2) {
      s[j] += math::shfl_xor_sync(s[j], offset);
    }
    s[j] = (iter_base + tz * tile_size + j < iter_bound) ? s[j] : -math::inf;
    st.m = max(st.m, s[j]);
  }

  float o_scale = math::ptx_exp2(m_prev - st.m);
  st.d *= o_scale;
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    s[j] = math::ptx_exp2(s[j] - st.m);
    st.d += s[j];
  }
#pragma unroll
  for (uint32_t i = 0; i < vec_size_ckv; ++i) {
    st.o[i] = st.o[i] * o_scale;
  }

#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    vec_t<float, vec_size_ckv> v_vec;
    v_vec.cast_load(ckv_smem + j * head_dim_ckv + tx * vec_size_ckv);
#pragma unroll
    for (uint32_t i = 0; i < vec_size_ckv; ++i) {
      st.o[i] = st.o[i] + s[j] * v_vec[i];
    }
  }
}

template <uint32_t num_stages_smem, uint32_t vec_size_ckv, uint32_t vec_size_kpe, uint32_t bdx,
          uint32_t bdy, uint32_t bdz, uint32_t tile_size_qo_heads, typename AttentionVariant,
          typename Params>
__global__ void BatchDecodeWithPagedKVCacheKernelMLA(Params params) {
  auto block = cg::this_thread_block();
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  const DTypeQ* q_nope = params.q_nope;
  const DTypeQ* q_pe = params.q_pe;
  DTypeO* o = params.o;
  float* lse = params.lse;
  const auto& paged_kv = params.paged_kv;
  const IdType* q_rope_offset = params.q_rope_offset;
  const bool* block_valid_mask = params.block_valid_mask;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const float rope_rcp_scale = params.rope_rcp_scale;
  const float rope_rcp_theta = params.rope_rcp_theta;
  const bool partition_kv = params.partition_kv;
  params.sm_scale *= math::log2e;

  constexpr uint32_t head_dim_ckv = bdx * vec_size_ckv;
  constexpr uint32_t head_dim_kpe = bdx * vec_size_kpe;
  const uint32_t batch_idx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  const uint32_t t_offset = dim3_offset(bdy, bdx, tz, ty, tx);

  if (block_valid_mask && !block_valid_mask[batch_idx]) return;
  const uint32_t mapped_batch_idx = params.request_indices[batch_idx];

  const uint32_t orig_seq_len = paged_kv.get_length(mapped_batch_idx);
  int32_t q_rope_offset_val =
      q_rope_offset == nullptr ? (orig_seq_len - 1) : q_rope_offset[mapped_batch_idx];

  const uint32_t kv_chunk_idx_in_orig_mapped_batch = params.kv_tile_indices[batch_idx];
  const uint32_t kv_chunk_size = *(params.kv_chunk_size_ptr);
  const uint32_t cur_chunk_start =
      partition_kv ? kv_chunk_idx_in_orig_mapped_batch * kv_chunk_size : 0;
  const uint32_t cur_chunk_end =
      partition_kv ? min((kv_chunk_idx_in_orig_mapped_batch + 1) * kv_chunk_size, orig_seq_len)
                   : orig_seq_len;
  const uint32_t cur_chunk_len = cur_chunk_end - cur_chunk_start;

  uint32_t packed_page_iter_base =
      paged_kv.indptr[mapped_batch_idx] * paged_kv.page_size + cur_chunk_start;
  const IdType last_indptr = paged_kv.indptr[paged_kv.batch_size];

  constexpr uint32_t kv_iter_len = bdy * bdz;
  constexpr uint32_t compute_qk_tile = bdy;

  extern __attribute__((shared)) uint8_t smem[];
  DTypeKV* ckv_smem = (DTypeKV*)smem;
  DTypeKV* kpe_smem = (DTypeKV*)((uint8_t*)ckv_smem +
                                 num_stages_smem * kv_iter_len * head_dim_ckv * sizeof(DTypeKV));
  size_t* ckv_offset_smem = (size_t*)((uint8_t*)kpe_smem + num_stages_smem * kv_iter_len *
                                                               head_dim_kpe * sizeof(DTypeKV));
  size_t* kpe_offset_smem = (size_t*)((uint8_t*)ckv_offset_smem + bdx * bdy * bdz * sizeof(size_t));
  float* smem_md = (float*)ckv_offset_smem;

  AttentionVariant variant(params, batch_idx, smem);

  vec_t<float, vec_size_ckv> q_nope_vec[tile_size_qo_heads];
  vec_t<float, vec_size_kpe> q_pe_vec[tile_size_qo_heads];
  state_t<vec_size_ckv> st[tile_size_qo_heads];
  uint32_t qo_head_idx[tile_size_qo_heads];

  vec_t<float, vec_size_kpe> freq;

#pragma unroll
  for (uint32_t i = 0; i < vec_size_kpe; ++i) {
    freq[i] = rope_rcp_scale * __powf(rope_rcp_theta, float(2 * ((tx * vec_size_kpe + i) / 2)) /
                                                          float(head_dim_kpe));
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

#pragma unroll
  for (int i = 0; i < tile_size_qo_heads; ++i) {
    qo_head_idx[i] = dim3_offset(bdy, tile_size_qo_heads, blockIdx.y, threadIdx.y, i);
    if (qo_head_idx[i] < num_qo_heads) {
      q_nope_vec[i].cast_load(q_nope +
                              (mapped_batch_idx * num_qo_heads + qo_head_idx[i]) * head_dim_ckv +
                              tx * vec_size_ckv);
      q_pe_vec[i].cast_load(q_pe +
                            (mapped_batch_idx * num_qo_heads + qo_head_idx[i]) * head_dim_kpe +
                            tx * vec_size_kpe);
    }
  }

  uint32_t q, r;
  paged_kv.page_size.divmod(packed_page_iter_base + t_offset, q, r);
  ckv_offset_smem[t_offset] = paged_kv.protective_get_offset_ckv(q, r,  0, last_indptr);
  kpe_offset_smem[t_offset] = paged_kv.protective_get_offset_kpe(q, r,  0, last_indptr);
  block.sync();

  uint32_t stage_idx = 0;
  constexpr uint32_t vec_bits = sizeof(DTypeKV) * vec_size_ckv * 8;
  constexpr uint32_t tx_fold = vec_size_ckv / vec_size_kpe;
  static_assert(num_stages_smem <= bdx);
  size_t offset_bytes;
  bool is_valid_range;
#pragma unroll
  for (uint32_t iter = 0; iter < num_stages_smem; ++iter) {
    is_valid_range = (iter * kv_iter_len + dim2_offset(bdy, tz, ty)) < cur_chunk_len;

    offset_bytes = ckv_offset_smem[dim3_offset(bdz, bdy, iter, tz, ty)] + tx * vec_size_ckv;
    cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
        ckv_smem + (stage_idx * kv_iter_len + dim2_offset(bdy, tz, ty)) * head_dim_ckv +
            tx * vec_size_ckv,
        paged_kv.ckv_data + offset_bytes, is_valid_range);

    offset_bytes =
        kpe_offset_smem[dim3_offset(bdz, bdy, iter, tz, ty)] + tx / tx_fold * vec_size_ckv;
    cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
        kpe_smem + (stage_idx * kv_iter_len + dim2_offset(bdy, tz, ty)) * head_dim_kpe +
            tx / tx_fold * vec_size_ckv,
        paged_kv.kpe_data + offset_bytes, is_valid_range);

    cp_async::commit_group();
    stage_idx = (stage_idx + 1) % num_stages_smem;
  }

#pragma unroll
  for (uint32_t iter = 0; iter < ceil_div(cur_chunk_len, kv_iter_len); ++iter) {
    cp_async::wait_group<1 * num_stages_smem - 1>();
    block.sync();
    const int32_t kv_idx_base =
        (paged_kv.rope_pos_offset == nullptr ? 0 : paged_kv.rope_pos_offset[mapped_batch_idx]) +
        cur_chunk_start + iter * kv_iter_len;
#pragma unroll
    for (int i = 0; i < tile_size_qo_heads; ++i) {
      compute_qk_and_update_local_stat_mla<vec_size_ckv, vec_size_kpe, bdx, compute_qk_tile>(
          params, variant, mapped_batch_idx,
          ckv_smem + (stage_idx * kv_iter_len + tz * compute_qk_tile) * head_dim_ckv, q_nope_vec[i],
          kpe_smem + (stage_idx * kv_iter_len + tz * compute_qk_tile) * head_dim_kpe, q_pe_vec[i],
          freq, kv_idx_base,
           iter * kv_iter_len,  cur_chunk_len, st[i]);
    }

    if ((iter + num_stages_smem) % bdx == 0) {
      uint32_t q, r;
      paged_kv.page_size.divmod(
          packed_page_iter_base + (iter + num_stages_smem) * kv_iter_len + t_offset, q, r);
      ckv_offset_smem[t_offset] =
          paged_kv.protective_get_offset_ckv(q, r,  0, last_indptr);
      kpe_offset_smem[t_offset] =
          paged_kv.protective_get_offset_kpe(q, r,  0, last_indptr);
    }
    block.sync();

    is_valid_range =
        ((iter + num_stages_smem) * kv_iter_len + dim2_offset(bdy, tz, ty)) < cur_chunk_len;
    offset_bytes = ckv_offset_smem[dim3_offset(bdz, bdy, (iter + num_stages_smem) % bdx, tz, ty)] +
                   tx * vec_size_ckv;
    cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
        ckv_smem + (stage_idx * kv_iter_len + dim2_offset(bdy, tz, ty)) * head_dim_ckv +
            tx * vec_size_ckv,
        paged_kv.ckv_data + offset_bytes, is_valid_range);

    offset_bytes = kpe_offset_smem[dim3_offset(bdz, bdy, (iter + num_stages_smem) % bdx, tz, ty)] +
                   tx / tx_fold * vec_size_ckv;
    cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
        kpe_smem + (stage_idx * kv_iter_len + dim2_offset(bdy, tz, ty)) * head_dim_kpe +
            tx / tx_fold * vec_size_ckv,
        paged_kv.kpe_data + offset_bytes, is_valid_range);
    cp_async::commit_group();

    stage_idx = (stage_idx + 1) % num_stages_smem;
  }
  cp_async::wait_group<0>();
  block.sync();

  if (bdz != 1) {
#pragma unroll
    for (int i = 0; i < tile_size_qo_heads; ++i) {
      if (qo_head_idx[i] < num_qo_heads)
        sync_state<vec_size_ckv, bdx, bdy, bdz>(variant, st[i], (float*)smem, smem_md, tx, ty, tz);
    }
  }

  if (tz == 0) {
#pragma unroll
    for (int i = 0; i < tile_size_qo_heads; ++i) {
      if (qo_head_idx[i] < num_qo_heads) {
#pragma unroll
        for (size_t j = 0; j < vec_size_ckv; ++j) {
          st[i].o[j] = variant.OutputTransform(params, st[i].o[j], batch_idx, 0,
                                               qo_head_idx[i], st[i].m, st[i].d, 1.0f);
        }
        st[i].o.cast_store(o + (batch_idx * num_qo_heads + qo_head_idx[i]) * head_dim_ckv +
                           tx * vec_size_ckv);

        if (lse != nullptr) {
          lse[batch_idx * num_qo_heads + qo_head_idx[i]] = st[i].get_lse();
        }
      }
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

}

#endif
