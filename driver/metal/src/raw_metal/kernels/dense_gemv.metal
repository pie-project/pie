// Raw-Metal dense bf16 GEMV (M=1) for Phase-0 decode (GDN `gdn_in_a`/`gdn_in_b`).
//
// out[n] = sum_k W[n,k] * x[k]   with W,x in T (bf16), float accumulation.
// qwen3.5 GDN `in_proj_a` / `in_proj_b` are stored DENSE bf16 [V_h=16, hidden=1024]
// (NOT 4-bit quantized like the other projections) — outputs the per-value-head
// a/b gating projections that feed beta's GdnCore (AGate/BGate). M=1 single token.
// MLX `apply_linear` on an unquantized weight = matmul with fp32 accumulation.
//
// Launch: dispatchThreads grid=(N, 1, 1), tg=(N, 1, 1) — one thread per output row
// (N=16 is tiny; each thread does the full K=1024 dot).

#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void dense_gemv(
    const device T* w   [[buffer(0)]],   // [N, K] row-major
    const device T* x   [[buffer(1)]],   // [K]
    device T* out       [[buffer(2)]],   // [N]
    constant uint& K    [[buffer(3)]],
    constant uint& N    [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= N) return;
  const device T* wrow = w + (uint64_t)gid * K;
  float acc = 0.0f;
  for (uint k = 0; k < K; ++k) {
    acc += float(wrow[k]) * float(x[k]);
  }
  out[gid] = T(acc);
}

#define instantiate_dense_gemv(name, itype)                       \
  template [[host_name("dense_gemv_" #name)]]                     \
  [[kernel]] void dense_gemv<itype>(                              \
      const device itype*, const device itype*, device itype*,    \
      constant uint&, constant uint&, uint);

instantiate_dense_gemv(float32, float)
instantiate_dense_gemv(float16, half)
instantiate_dense_gemv(bfloat16, bfloat)
