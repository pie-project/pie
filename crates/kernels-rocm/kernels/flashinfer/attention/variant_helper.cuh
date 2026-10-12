














#ifndef FLASHINFER_ATTENTION_VARIANT_HELPER_H
#define FLASHINFER_ATTENTION_VARIANT_HELPER_H

#include <cuda_runtime.h>

#include <cstdint>

#include "../utils.cuh"

namespace flashinfer {

DEFINE_HAS_MEMBER(v_scale)

#define REGISTER_QUERY_TRANSFORM(params, q, ...)                                    \
  template <typename Params, typename T>                                            \
  __device__ __forceinline__ T QueryTransform(const Params& params, void* q_smem) { \
    __VA_ARGS__                                                                     \
  }

#define REGISTER_KEY_TRANSFORM(params, k, ...)                                    \
  template <typename Params, typename T>                                          \
  __device__ __forceinline__ T KeyTransform(const Params& params, void* k_smem) { \
    __VA_ARGS__                                                                   \
  }

#define REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx,          \
                                  kv_head_idx, ...)                                                \
  template <typename Params, typename T>                                                           \
  __device__ __forceinline__ T LogitsTransform(const Params& params, T logits, uint32_t batch_idx, \
                                               uint32_t qo_idx, uint32_t kv_idx,                   \
                                               uint32_t qo_head_idx, uint32_t kv_head_idx) {       \
    __VA_ARGS__                                                                                    \
  }

#define REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, ...) \
  template <typename Params>                                                                   \
  __device__ __forceinline__ bool LogitsMask(const Params& params, uint32_t batch_idx,         \
                                             uint32_t qo_idx, uint32_t kv_idx,                 \
                                             uint32_t qo_head_idx, uint32_t kv_head_idx) {     \
    __VA_ARGS__                                                                                \
  }

#define REGISTER_M_D_UPDATE(params, kv_tile_idx, qo_head_idx, m, d, scale, ...)          \
  template <typename Params, typename T_M>                                               \
  __device__ __forceinline__ void update_m_d(const Params& params, uint32_t kv_tile_idx, \
                                             uint32_t qo_head_idx, T_M& m, float& d,     \
                                             float& scale) {                             \
    __VA_ARGS__                                                                          \
  }

#define REGISTER_OUTPUT_TRANSFORM(params, output, batch_idx, qo_idx, qo_head_idx, m, d, scale,     \
                                  ...)                                                             \
  template <typename Params, typename T, typename T_M>                                             \
  __device__ __forceinline__ T OutputTransform(const Params& params, T output, uint32_t batch_idx, \
                                               uint32_t qo_idx, uint32_t qo_head_idx, T_M& m,      \
                                               float& d, float scale) {                            \
    __VA_ARGS__                                                                                    \
  }

struct AttentionVariantBase {
  constexpr static bool use_softmax = true;
  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx,
                            { return logits; })

  REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx,
                       { return true; })

  REGISTER_M_D_UPDATE(params, kv_tile_idx, qo_head_idx, m, d, scale, { return; })

  REGISTER_OUTPUT_TRANSFORM(params, output, batch_idx, qo_idx, qo_head_idx, m, d, scale, {
    float d_rcp = (m != -math::inf) ? math::ptx_rcp(d) : 0.f;
    float v_scale_val = get_v_scale(params);
    return output * d_rcp * v_scale_val;
  })

  template <typename Params>
  __device__ __forceinline__ static float get_v_scale(const Params& params) {
    if constexpr (has_v_scale_v<Params>) {
      return params.v_scale;
    } else {
      return 1.0f;
    }
  }
};

}

#endif
