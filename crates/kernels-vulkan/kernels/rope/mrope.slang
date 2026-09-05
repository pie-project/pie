#include "common/bf16.slang"

PIE_BUFFER_RW(0, x);
[[vk::binding(1)]] StructuredBuffer<int> positions;
struct Push { float base_; int head_dim; int s0; int s1; int s2; int heads; int pairs; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int i = int(gid.x);
    if (i >= pc.pairs) return;
    uint h = gid.y;
    uint m = gid.z;
    int half_hd = pc.head_dim / 2;
    int pos_t = positions[3u * m + 0u];
    int pos_h = positions[3u * m + 1u];
    int pos_w = positions[3u * m + 2u];
    uint row = (m * uint(pc.heads) + h) * uint(pc.head_dim);
    float inv_freq;
    int axis_pos;
    uint i1;
    uint i2;
#if defined(PIE_BLOCKED)
    int total = pc.s0 + pc.s1 + pc.s2;
    int within;
    if (i < pc.s0) { axis_pos = pos_t; within = i; }
    else if (i < pc.s0 + pc.s1) { axis_pos = pos_h; within = i - pc.s0; }
    else { axis_pos = pos_w; within = i - pc.s0 - pc.s1; }
    inv_freq = exp2(-(2.0 * float(within) / float(total)) * pc.base_);
    i1 = row + uint(i);
    i2 = i1 + uint(half_hd);
#elif defined(PIE_SPLIT)
    int within;
    int before;
    int width;
    if (i < pc.s0) { axis_pos = pos_t; within = i; before = 0; width = pc.s0; }
    else if (i < pc.s0 + pc.s1) { axis_pos = pos_h; within = i - pc.s0; before = pc.s0; width = pc.s1; }
    else { axis_pos = pos_w; within = i - pc.s0 - pc.s1; before = pc.s0 + pc.s1; width = pc.s2; }
    if (width <= 0) return;
    inv_freq = exp2(-(float(within) / float(width)) * pc.base_);
    i1 = row + uint(2 * before + within);
    i2 = i1 + uint(width);
#else
    int r = i % 3;
    if (r == 1 && i < 3 * pc.s1) axis_pos = pos_h;
    else if (r == 2 && i < 3 * pc.s2) axis_pos = pos_w;
    else axis_pos = pos_t;
    inv_freq = exp2(-(2.0 * float(i) / float(pc.head_dim)) * pc.base_);
    i1 = row + uint(i);
    i2 = i1 + uint(half_hd);
#endif
    float theta = float(axis_pos) * inv_freq;
    float c = cos(theta);
    float s = sin(theta);
    float x1 = PIE_LOAD(x[i1]);
    float x2 = PIE_LOAD(x[i2]);
    x[i1] = PIE_STORE(x1 * c - x2 * s);
    x[i2] = PIE_STORE(x1 * s + x2 * c);
}

// pie:instantiate rope_mrope_interleaved_bf16
// pie:instantiate rope_mrope_blocked_bf16 PIE_BLOCKED=1
// pie:instantiate rope_mrope_split_bf16 PIE_SPLIT=1
