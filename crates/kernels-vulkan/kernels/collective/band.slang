#include "common/bf16.slang"

#if defined(PIE_F32)
#define PIE_T float
#define PIE_C_LOAD(x) (x)
#define PIE_C_STORE(f) (f)
#else
#define PIE_T PIE_ACT
#define PIE_C_LOAD(x) PIE_LOAD(x)
#define PIE_C_STORE(f) PIE_STORE(f)
#endif

[[vk::binding(0)]] StructuredBuffer<PIE_T> in_;
[[vk::binding(1)]] RWStructuredBuffer<PIE_T> out_;

struct Push { uint n; uint world; uint rank; uint slot; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = gid.x;
    if (i >= pc.n) {
        return;
    }
#if defined(PIE_PUBLISH)

    out_[pc.rank * pc.slot + i] = in_[i];
#elif defined(PIE_GATHER)

    for (uint r = 0u; r < pc.world; ++r) {
        out_[r * pc.n + i] = in_[r * pc.slot + i];
    }
#elif defined(PIE_SCATTER)

    float acc = 0.0;
    for (uint r = 0u; r < pc.world; ++r) {
        acc += PIE_C_LOAD(in_[r * pc.slot + pc.rank * pc.n + i]);
    }
    out_[i] = PIE_C_STORE(acc);
#else

    float acc = 0.0;
    for (uint r = 0u; r < pc.world; ++r) {
        acc += PIE_C_LOAD(in_[r * pc.slot + i]);
    }
    out_[i] = PIE_C_STORE(acc);
#endif
}

// pie:instantiate collective_publish_bf16 PIE_PUBLISH=1
// pie:instantiate collective_publish_f32 PIE_PUBLISH=1 PIE_F32=1
// pie:instantiate collective_reduce_bf16
// pie:instantiate collective_reduce_f32 PIE_F32=1
// pie:instantiate collective_gather_bf16 PIE_GATHER=1
// pie:instantiate collective_gather_f32 PIE_GATHER=1 PIE_F32=1
// pie:instantiate collective_scatter_bf16 PIE_SCATTER=1
// pie:instantiate collective_scatter_f32 PIE_SCATTER=1 PIE_F32=1
