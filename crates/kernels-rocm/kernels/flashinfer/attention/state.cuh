














#ifndef FLASHINFER_STATE_CUH_
#define FLASHINFER_STATE_CUH_

#include "../math.cuh"
#include "../vec_dtypes.cuh"

namespace flashinfer {




template <size_t vec_size>
struct state_t {

  vec_t<float, vec_size> o;

  float m;

  float d;

  __device__ __forceinline__ void init() {
    o.fill(0.f);
    m = -math::inf;
    d = 1.f;
  }

  __device__ __forceinline__ state_t() { init(); }

  __device__ __forceinline__ float get_lse() const { return m + math::ptx_log2(d); }






  __device__ __forceinline__ void merge(const vec_t<float, vec_size>& other_o, float other_m,
                                        float other_d) {
    float m_prev = m, d_prev = d;
    m = max(m_prev, other_m);
    d = d_prev * math::ptx_exp2(m_prev - m) + other_d * math::ptx_exp2(other_m - m);
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      o[i] = o[i] * math::ptx_exp2(m_prev - m) + other_o[i] * math::ptx_exp2(other_m - m);
    }
  }




  __device__ __forceinline__ void merge(const state_t<vec_size>& other) {
    merge(other.o, other.m, other.d);
  }

  __device__ __forceinline__ void normalize() {

#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      o[i] = __fdividef(o[i], d);
    }
  }
};

}

#endif
