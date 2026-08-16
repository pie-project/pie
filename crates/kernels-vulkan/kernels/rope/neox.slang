// NEOX rotary embeddings, in place on one tensor.
// Ported from Metal's rope/neox.metal. The arithmetic and the distinction
// between geometric, frequency-table, proportional, packed-multibatch and
// strided rows stay intact; Vulkan moves scalar operands into push constants
// and uses the launch's grid extent as the rotary pair count.

#include "common/bf16.slang"

PIE_BUFFER_RW(0, x);
[[vk::binding(1)]] StructuredBuffer<int> position;
#if defined(PIE_FREQS)
[[vk::binding(2)]] StructuredBuffer<float> inv_freq;
#endif

struct Push {
    float scale;
#if defined(PIE_FREQS)
    int head_dim;
    float mscale;
#else
    float base;
    int head_dim;
#if defined(PIE_STRIDED)
    int row_pitch;
#endif
#endif
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

void rotate_pair(uint i1, uint i2, float theta, float out_scale) {
    float c = cos(theta);
    float s = sin(theta);
    float x1 = PIE_LOAD(x[i1]);
    float x2 = PIE_LOAD(x[i2]);
    x[i1] = PIE_STORE(out_scale * (x1 * c - x2 * s));
    x[i2] = PIE_STORE(out_scale * (x1 * s + x2 * c));
}

[numthreads(1, 1, 1)]
void main(uint3 group : SV_GroupID) {
    uint i = group.x;
    uint h = group.y;
    uint row = group.z;
    uint pair_half = gl_NumWorkGroups.x;
    uint n_head = gl_NumWorkGroups.y;

#if defined(PIE_DECODE)
    row = 0u;
#endif

#if defined(PIE_STRIDED)
    uint row_base = row * uint(pc.row_pitch);
#else
    uint row_base = row * n_head * uint(pc.head_dim);
#endif

#if defined(PIE_PROP)
    uint half_hd = uint(pc.head_dim) / 2u;
    float d = 2.0 * float(i) / float(pc.head_dim);
    float theta = pc.scale * float(position[row]) * exp2(-d * pc.base);
    uint i1 = row_base + h * uint(pc.head_dim) + i;
    rotate_pair(i1, i1 + half_hd, theta, 1.0);
#elif defined(PIE_FREQS)
    float theta = pc.scale * float(position[row]) * inv_freq[i];
    uint i1 = row_base + h * uint(pc.head_dim) + i;
    rotate_pair(i1, i1 + pair_half, theta, pc.mscale);
#else
    float d = float(i) / float(pair_half);
    float theta = pc.scale * float(position[row]) * exp2(-d * pc.base);
    uint i1 = row_base + h * uint(pc.head_dim) + i;
    rotate_pair(i1, i1 + pair_half, theta, 1.0);
#endif
}

// The grid IS the shape: `gl_NumWorkGroups` is how the pair count and the
// head count reach the shader, because the launch already knows both and no
// scalar carries them. Slang has no wave-model spelling for it -- it is a
// workgroup fact, not a lane fact -- so the GLSL builtin stays, as it does in
// `common/reduce.slang` and for the same reason.

// pie:instantiate neox_decode_bfloat16 PIE_DECODE=1
// pie:instantiate neox_freqs_decode_bfloat16 PIE_FREQS=1 PIE_DECODE=1
// pie:instantiate neox_freqs_mb_bfloat16 PIE_FREQS=1
// pie:instantiate neox_mb_bfloat16
// pie:instantiate neox_prop_decode_bfloat16 PIE_PROP=1 PIE_DECODE=1
// pie:instantiate neox_prop_mb_bfloat16 PIE_PROP=1
// pie:instantiate neox_strided_bfloat16 PIE_STRIDED=1
