














#ifndef FLASHINFER_FASTDIV_CUH_
#define FLASHINFER_FASTDIV_CUH_
#include <cstdint>
#include <cuda/cmath>

namespace flashinfer {

struct uint_fastdiv {
  __host__ __device__ uint_fastdiv() : impl_(1), d_(0) {}


  __host__ __device__ __forceinline__ operator unsigned int() const { return d_; }

  __host__ __device__ __forceinline__ void divmod(uint32_t n, uint32_t& q, uint32_t& r) const {
    q = n / impl_;
    r = n - q * d_;
  }

 private:
  cuda::fast_mod_div<uint32_t> impl_;
  uint32_t d_;
};

__host__ __device__ __forceinline__ uint32_t operator/(const uint32_t n,
                                                       const uint_fastdiv& divisor) {
  uint32_t q, r;
  divisor.divmod(n, q, r);
  return q;
}

__host__ __device__ __forceinline__ uint32_t operator%(const uint32_t n,
                                                       const uint_fastdiv& divisor) {
  uint32_t q, r;
  divisor.divmod(n, q, r);
  return r;
}

}

#endif
