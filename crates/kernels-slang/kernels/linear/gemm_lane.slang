#include "prim/params.slang"
#include "prim/slot16.slang"

// y = x * w^T for an f32 activation, an invocation an output: the f32
// projections (a router, a small head) whose rows are too few to tile. The
// weight is bf16 or f32 (`PIE_W_F32`), the result f32 or bf16
// (`PIE_Y_BF16`), a bf16 value a 16-bit slot. WGSL has no 16-bit storage: it
// reads a bf16 weight from words, and a bf16 result, which would tear a word
// between invocations, is SPIR-V's alone. WGSL has no 16-bit storage: it
// reads a bf16 weight from words, and a bf16 result, which would tear a word
// between invocations, is SPIR-V's alone.

[[vk::binding(0)]] StructuredBuffer<float> x;
#if defined(PIE_W_F32)
[[vk::binding(1)]] StructuredBuffer<float> w;
#define PIE_WL(i) (w[i])
#elif defined(PIE_WGSL)
[[vk::binding(1)]] StructuredBuffer<uint> w;
#define PIE_WL(i) bf16_to_f32((w[(i) >> 1u] >> (((i) & 1u) * 16u)) & 0xFFFFu)
#else
PIE_SLOTS_RO(1, w);
#define PIE_WL(i) slot_at(w, i)
#endif
#if defined(PIE_Y_BF16)
PIE_SLOTS_RW(2, y);
#define PIE_YS(i, f) slot_put(y, i, f)
#else
[[vk::binding(2)]] RWStructuredBuffer<float> y;
#define PIE_YS(i, f) y[i] = (f)
#endif

struct Params {
    int m;
    int n;
    int k;
};
PIE_PARAMS(3, Params, params);

[numthreads(64, 4, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint col = gid.x;
    uint row = gid.y;
    if (col >= uint(params.n) || row >= uint(params.m)) return;
    uint k = uint(params.k);
    float acc = 0.0;
    for (uint kk = 0u; kk < k; kk++) acc += x[row * k + kk] * PIE_WL(col * k + kk);
    PIE_YS(row * uint(params.n) + col, acc);
}

// pie:instantiate linear_gemm_lane_f32_bf16_f32
// pie:instantiate linear_gemm_lane_f32_f32_f32 PIE_W_F32=1
// pie:instantiate linear_gemm_lane_f32_bf16_bf16 PIE_Y_BF16=1 PIE_TARGET=spirv
// pie:instantiate linear_gemm_lane_f32_f32_bf16 PIE_W_F32=1 PIE_Y_BF16=1 PIE_TARGET=spirv
