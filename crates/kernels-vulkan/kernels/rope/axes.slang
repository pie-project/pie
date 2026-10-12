#include "common/bf16.slang"

[[vk::binding(0)]] StructuredBuffer<PIE_ACT> x;
[[vk::binding(1)]] StructuredBuffer<float> positions;
[[vk::binding(2)]] RWStructuredBuffer<PIE_ACT> o;

struct Push {
    uint d0; uint d1; uint d2; uint d3;
    float t0; float t1; float t2; float t3;
    uint axes; uint rotary_dim; uint head_dim; uint heads; int form; uint copy_tail;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

float theta_pow(float base, float e) { return exp(e * log(base)); }

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint dims[4] = { pc.d0, pc.d1, pc.d2, pc.d3 };
    float thetas[4] = { pc.t0, pc.t1, pc.t2, pc.t3 };
    uint angles = pc.rotary_dim / 2u;
    uint width = pc.heads * pc.head_dim;
    uint idx = gid.x;
    uint row = gid.y;
    uint pos = row * pc.axes;
    uint xr = row * width;

    if (idx >= pc.heads * angles) {
        if (pc.copy_tail == 0u) return;
        uint per_head = pc.head_dim - pc.rotary_dim;
        if (per_head == 0u) return;
        uint tail = idx - pc.heads * angles;
        uint head = tail / per_head;
        if (head >= pc.heads) return;
        uint i = pc.rotary_dim + tail % per_head;
        o[xr + head * pc.head_dim + i] = x[xr + head * pc.head_dim + i];
        return;
    }

    uint head = idx / angles;
    uint angle = idx % angles;
    uint xh = xr + head * pc.head_dim;
    uint lo;
    uint hi;
    float cos_v = 1.0;
    float sin_v = 0.0;

    if (pc.form == 3) {
        uint span = 0u;
        for (uint a = 0u; a < pc.axes; a++) span += dims[a];
        uint pad = (pc.heads * pc.rotary_dim - span) / 2u;
        if (idx >= pad) {
            uint slot = idx - pad;
            uint axis = slot % pc.axes;
            uint f = slot / pc.axes;
            uint ladder = dims[axis] / 2u;
            float exponent = ladder > 1u ? float(f) / float(ladder - 1u) : 0.0;
            float a = positions[pos + axis] * theta_pow(thetas[axis], exponent);
            sin_v = sin(a);
            cos_v = cos(a);
        }
        lo = angle;
        hi = angle + angles;
    } else {
        uint axis = 0u;
        uint first_angle = 0u;
        uint first_channel = 0u;
        while (axis < pc.axes && angle >= first_angle + dims[axis] / 2u) {
            first_angle += dims[axis] / 2u;
            first_channel += dims[axis];
            axis++;
        }
        if (axis >= pc.axes) return;
        uint within = angle - first_angle;
        float freq = theta_pow(thetas[axis], -2.0 * float(within) / float(dims[axis]));
        float a = positions[pos + axis] * freq;
        sin_v = sin(a);
        cos_v = cos(a);
        if (pc.form == 0) {
            lo = first_channel + 2u * within;
            hi = lo + 1u;
        } else if (pc.form == 1) {
            lo = angle;
            hi = angle + angles;
        } else {
            lo = first_channel + within;
            hi = first_channel + dims[axis] / 2u + within;
        }
    }
    float va = PIE_LOAD(x[xh + lo]);
    float vb = PIE_LOAD(x[xh + hi]);
    o[xh + lo] = PIE_STORE(va * cos_v - vb * sin_v);
    o[xh + hi] = PIE_STORE(vb * cos_v + va * sin_v);
}

// pie:instantiate rope_axes_bf16
// pie:instantiate rope_axes_f32 PIE_ACT_F32=1
