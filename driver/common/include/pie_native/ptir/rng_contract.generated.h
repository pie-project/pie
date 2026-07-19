// rng_contract.generated.h — GENERATED from interface/ptir/src/rng.rs.
// DO NOT EDIT. Regenerate: PTIR_REGEN=1 cargo test -p pie-ptir --test rng_contract
#pragma once
#include <stdint.h>

#if defined(__CUDACC__)
#define PTIR_RNG_INLINE static __host__ __device__ __forceinline__
#else
#define PTIR_RNG_INLINE static inline
#endif

PTIR_RNG_INLINE uint64_t ptir_rng_splitmix64(uint64_t x) {
  x ^= x >> 27;
  x *= 0x3C79AC492BA7B653ULL;
  x ^= x >> 33;
  x *= 0x1C69B3F74AC4AE35ULL;
  x ^= x >> 27;
  return x;
}
PTIR_RNG_INLINE uint64_t ptir_rng_seed_eff(uint32_t seed) {
  return (uint64_t)seed ^ 0x00000000A5A5A5A5ULL;
}
PTIR_RNG_INLINE uint64_t ptir_rng_stream_salt(uint32_t stream) {
  return ptir_rng_splitmix64(
      (uint64_t)stream * 0x9E3779B97F4A7C15ULL);
}
PTIR_RNG_INLINE uint64_t ptir_rng_seed_eff_stream(
    uint32_t seed, uint32_t stream) {
  return ptir_rng_seed_eff(seed) ^ ptir_rng_stream_salt(stream);
}
PTIR_RNG_INLINE uint64_t ptir_rng_keyed_seed(
    uint32_t key, uint32_t counter) {
  return ptir_rng_splitmix64(
      ((uint64_t)key << 32) | (uint64_t)counter);
}
PTIR_RNG_INLINE float ptir_rng_hash_uniform(
    uint64_t seed_eff, uint32_t index) {
  const uint64_t x = seed_eff +
      0x9E3779B97F4A7C15ULL * ((uint64_t)index + 1ULL);
  const uint32_t bits =
      (uint32_t)(ptir_rng_splitmix64(x) >> 40);
  return ((float)bits + 0.5f) * (1.0f / 16777216.0f);
}

#undef PTIR_RNG_INLINE

#ifdef __cplusplus
inline constexpr char PTIR_RNG_CUDA_PREAMBLE[] = R"PTIR_RNG_CUDA(
__device__ __forceinline__ unsigned long long ptir_rng_splitmix64(unsigned long long x) {
  x ^= x >> 27;
  x *= 0x3C79AC492BA7B653ULL;
  x ^= x >> 33;
  x *= 0x1C69B3F74AC4AE35ULL;
  x ^= x >> 27;
  return x;
}
__device__ __forceinline__ unsigned long long ptir_rng_seed_eff(unsigned int seed) {
  return (unsigned long long)seed ^ 0x00000000A5A5A5A5ULL;
}
__device__ __forceinline__ unsigned long long ptir_rng_stream_salt(unsigned int stream) {
  return ptir_rng_splitmix64(
      (unsigned long long)stream * 0x9E3779B97F4A7C15ULL);
}
__device__ __forceinline__ unsigned long long ptir_rng_seed_eff_stream(
    unsigned int seed, unsigned int stream) {
  return ptir_rng_seed_eff(seed) ^ ptir_rng_stream_salt(stream);
}
__device__ __forceinline__ unsigned long long ptir_rng_keyed_seed(
    unsigned int key, unsigned int counter) {
  return ptir_rng_splitmix64(
      ((unsigned long long)key << 32) | (unsigned long long)counter);
}
__device__ __forceinline__ float ptir_rng_hash_uniform(
    unsigned long long seed_eff, unsigned int index) {
  const unsigned long long x = seed_eff +
      0x9E3779B97F4A7C15ULL * ((unsigned long long)index + 1ULL);
  const unsigned int bits =
      (unsigned int)(ptir_rng_splitmix64(x) >> 40);
  return ((float)bits + 0.5f) * (1.0f / 16777216.0f);
}

)PTIR_RNG_CUDA";
#endif
