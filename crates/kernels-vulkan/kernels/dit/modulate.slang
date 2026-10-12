#include "dit/types.slang"

// adaLN modulation of each row by its lane's (or its own) vectors:
// scale-shift x * (1 + s) + b, scale x * (1 + s), or tanh-gate tanh(g) * x.
// PIE_GATED instead folds a gated residual: r + g * y.

#ifdef PIE_GATED
[[vk::binding(0)]] StructuredBuffer<PIE_XT> r;
[[vk::binding(1)]] StructuredBuffer<PIE_MT> m;
[[vk::binding(2)]] StructuredBuffer<PIE_XT> y;
[[vk::binding(3)]] StructuredBuffer<int> lane_of_row;
[[vk::binding(4)]] RWStructuredBuffer<PIE_XT> o;
struct Push { uint width; uint has_lanes; };
#else
[[vk::binding(0)]] StructuredBuffer<PIE_XT> x;
[[vk::binding(1)]] StructuredBuffer<PIE_MT> m;
[[vk::binding(2)]] StructuredBuffer<int> lane_of_row;
[[vk::binding(3)]] RWStructuredBuffer<PIE_XT> o;
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
    float v = PIE_ML(m[lane * pc.width + i]) * PIE_XL(y[at]) + PIE_XL(r[at]);
#else
    uint mat = lane * pc.m_width;
    float xv = PIE_XL(x[at]);
    float v;
    if (pc.form == 0) {
        v = xv * (1.0 + PIE_ML(m[mat + i])) + PIE_ML(m[mat + i + pc.width]);
    } else if (pc.form == 1) {
        v = xv * (1.0 + PIE_ML(m[mat + i]));
    } else {
        v = tanh(PIE_ML(m[mat + i])) * xv;
    }
#endif
    o[at] = PIE_XS(v);
}

// pie:instantiate modulate_bf16
// pie:instantiate modulate_bf16_f32 PIE_M_F32=1
// pie:instantiate modulate_f32 PIE_X_F32=1 PIE_M_F32=1
// pie:instantiate gated_residual_add_bf16 PIE_GATED=1
// pie:instantiate gated_residual_add_bf16_f32 PIE_GATED=1 PIE_M_F32=1
// pie:instantiate gated_residual_add_f32 PIE_GATED=1 PIE_X_F32=1 PIE_M_F32=1
