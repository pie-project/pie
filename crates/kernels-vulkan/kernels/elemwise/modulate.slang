#include "common/bf16.slang"

#ifdef PIE_MOD_F32
#define PIE_MOD float
#define PIE_MOD_LOAD(v) (v)
#else
#define PIE_MOD uint16_t
#define PIE_MOD_LOAD(v) bf16_to_f32(v)
#endif

#ifdef PIE_GATED
[[vk::binding(0)]] StructuredBuffer<PIE_ACT> r;
[[vk::binding(1)]] StructuredBuffer<PIE_MOD> m;
[[vk::binding(2)]] StructuredBuffer<PIE_ACT> y;
[[vk::binding(3)]] StructuredBuffer<int> lane_of_row;
[[vk::binding(4)]] RWStructuredBuffer<PIE_ACT> o;
struct Push { uint width; uint has_lanes; };
#else
[[vk::binding(0)]] StructuredBuffer<PIE_ACT> x;
[[vk::binding(1)]] StructuredBuffer<PIE_MOD> m;
[[vk::binding(2)]] StructuredBuffer<int> lane_of_row;
[[vk::binding(3)]] RWStructuredBuffer<PIE_ACT> o;
struct Push { uint width; uint m_width; int form; uint has_lanes; };
#endif
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = gid.x;
    uint row = gid.y;
    if (i >= pc.width) return;
    uint at = row * pc.width + i;
    uint lane = pc.has_lanes != 0u ? uint(lane_of_row[row]) : row;
#ifdef PIE_GATED
    float v = PIE_MOD_LOAD(m[lane * pc.width + i]) * PIE_LOAD(y[at]) + PIE_LOAD(r[at]);
#else
    uint mat = lane * pc.m_width;
    float xv = PIE_LOAD(x[at]);
    float v;
    if (pc.form == 0) {
        v = xv * (1.0 + PIE_MOD_LOAD(m[mat + i])) + PIE_MOD_LOAD(m[mat + i + pc.width]);
    } else if (pc.form == 1) {
        v = xv * (1.0 + PIE_MOD_LOAD(m[mat + i]));
    } else {
        v = tanh(PIE_MOD_LOAD(m[mat + i])) * xv;
    }
#endif
    o[at] = PIE_STORE(v);
}

// pie:instantiate modulate_bf16
// pie:instantiate modulate_bf16_f32 PIE_MOD_F32=1
// pie:instantiate modulate_f32 PIE_ACT_F32=1 PIE_MOD_F32=1
// pie:instantiate gated_residual_add_bf16 PIE_GATED=1
// pie:instantiate gated_residual_add_bf16_f32 PIE_GATED=1 PIE_MOD_F32=1
// pie:instantiate gated_residual_add_f32 PIE_GATED=1 PIE_ACT_F32=1 PIE_MOD_F32=1
