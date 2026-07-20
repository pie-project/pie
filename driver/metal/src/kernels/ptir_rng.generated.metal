// ptir_rng.generated.metal — GENERATED from interface/ptir/src/rng.rs.
// DO NOT EDIT. Regenerate: PTIR_REGEN=1 cargo test -p pie-ptir --test rng_contract
#ifndef PIE_PTIR_RNG_GENERATED_METAL
#define PIE_PTIR_RNG_GENERATED_METAL

inline ulong ptir_rng_splitmix64(ulong x) {
  x ^= x >> 27;
  x *= 0x3C79AC492BA7B653ul;
  x ^= x >> 33;
  x *= 0x1C69B3F74AC4AE35ul;
  x ^= x >> 27;
  return x;
}
inline ulong ptir_rng_seed_eff(uint seed) {
  return ulong(seed) ^ 0x00000000A5A5A5A5ul;
}
inline ulong ptir_rng_stream_salt(uint stream) {
  return ptir_rng_splitmix64(
      ulong(stream) * 0x9E3779B97F4A7C15ul);
}
inline ulong ptir_rng_seed_eff_stream(uint seed, uint stream) {
  return ptir_rng_seed_eff(seed) ^ ptir_rng_stream_salt(stream);
}
inline ulong ptir_rng_keyed_seed(uint key, uint counter) {
  return ptir_rng_splitmix64(
      (ulong(key) << 32) | ulong(counter));
}
inline float ptir_rng_hash_uniform(ulong seed_eff, uint index) {
  const ulong x = seed_eff +
      0x9E3779B97F4A7C15ul * (ulong(index) + 1ul);
  const uint bits = uint(ptir_rng_splitmix64(x) >> 40);
  return (float(bits) + 0.5f) * (1.0f / 16777216.0f);
}

#endif
