#include "common/bf16.slang"


[[vk::binding(0)]] StructuredBuffer<float> x;
#ifdef PIE_W_F32
[[vk::binding(1)]] StructuredBuffer<float> w;
#define PIE_WL(v) (v)
#else
[[vk::binding(1)]] StructuredBuffer<uint16_t> w;
#define PIE_WL(v) bf16_to_f32(v)
#endif
#ifdef PIE_Y_BF16
[[vk::binding(2)]] RWStructuredBuffer<uint16_t> y;
#define PIE_YS(f) f32_to_bf16(f)
#else
[[vk::binding(2)]] RWStructuredBuffer<float> y;
#define PIE_YS(f) (f)
#endif

struct Push { int m; int n; int k; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(64, 4, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint col = gid.x;
    uint row = gid.y;
    if (col >= uint(pc.n) || row >= uint(pc.m)) return;
    uint k = uint(pc.k);
    float acc = 0.0;
    for (uint kk = 0u; kk < k; kk++) acc += x[row * k + kk] * PIE_WL(w[col * k + kk]);
    y[row * uint(pc.n) + col] = PIE_YS(acc);
}

// pie:instantiate lane_gemm_f32_bf16_f32
// pie:instantiate lane_gemm_f32_f32_f32 PIE_W_F32=1
// pie:instantiate lane_gemm_f32_bf16_bf16 PIE_Y_BF16=1
// pie:instantiate lane_gemm_f32_f32_bf16 PIE_W_F32=1 PIE_Y_BF16=1
