#include "common/bf16.slang"

#if defined(PIE_F32)
#define PIE_T float
#else
#define PIE_T PIE_ACT
#endif

[[vk::binding(0)]] StructuredBuffer<PIE_T> in_;
[[vk::binding(1)]] RWStructuredBuffer<PIE_T> out_;
#if defined(PIE_LIVE)
[[vk::binding(2)]] StructuredBuffer<int> rows;
#else
[[vk::binding(2)]] StructuredBuffer<uint> rows;
#endif
struct Push { uint width; uint count; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint c = gid.x;
    uint i = gid.y;
    if (c >= pc.width || i >= pc.count) return;
#if defined(PIE_LIVE)
    int at = rows[i];
    if (at < 0) return;
    out_[uint(at) * pc.width + c] = in_[i * pc.width + c];
#elif defined(PIE_SCATTER)
    out_[rows[i] * pc.width + c] = in_[i * pc.width + c];
#else
    out_[i * pc.width + c] = in_[rows[i] * pc.width + c];
#endif
}

// pie:instantiate row_gather_bf16
// pie:instantiate row_gather_f32 PIE_F32=1
// pie:instantiate row_scatter_bf16 PIE_SCATTER=1
// pie:instantiate row_scatter_f32 PIE_SCATTER=1 PIE_F32=1
// pie:instantiate row_scatter_live_bf16 PIE_LIVE=1
// pie:instantiate row_scatter_live_f32 PIE_LIVE=1 PIE_F32=1
