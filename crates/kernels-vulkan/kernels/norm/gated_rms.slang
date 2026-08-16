// Gated RMSNorm for GDN: `w * rmsnorm(x) * silu(z)`.
//
// The math is the Metal kernel's GDN epilogue, including the numerically stable
// sigmoid and the raw gate-norm weight (no Gemma `1 + w`). What moved is only
// the reduction and row pitch: subgroup width is implementation-defined in
// Vulkan, and the strided scalar rides in the push block.

#include "common/bf16.slang"
#include "common/reduce.slang"
#include "norm/rms_params.slang"

#define PIE_GROUP_X 256

PIE_BUFFER_RO(0, x);
PIE_BUFFER_RO(1, z);
PIE_BUFFER_RO(2, w);
PIE_BUFFER_RW(3, out_);
PIE_PARAMS(4, GatedRmsParams, p);

#if defined(PIE_STRIDED)
struct Push { int row_pitch; };
[[vk::push_constant]] ConstantBuffer<Push> pc;
#endif

float stable_sigmoid(float v) {
    float y = 1.0 / (1.0 + exp(-abs(v)));
    return v < 0.0 ? 1.0 - y : y;
}

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint lid = local.x;
#if defined(PIE_STRIDED)
    uint base = group.z * uint(pc.row_pitch) + group.y * p.vd;
#else
    uint base = (group.z * gl_NumWorkGroups.y + group.y) * p.vd;
#endif
    uint idx = base + lid;
    // Metal launches `tg = (V_d, 1, 1)` -- the threadgroup is exactly as wide
    // as the axis, so every lane owns one channel and there is nothing to loop
    // over. A Vulkan workgroup size is fixed at compile time and cannot follow
    // `p.vd`, so this walks the axis in strides of the workgroup instead.
    //
    // Only the `lid < p.vd` half of that was ported. That is correct for the
    // NARROW direction and silently wrong for the wide one: at `v_d > 256`
    // every channel from 256 up was left out of the sum AND never written, so
    // the norm divided by a mean taken over the first 256 channels only and
    // the rest of the head kept whatever was in the buffer. `v_d` is 128 on
    // every GDN checkpoint the tree has seen, which is why nothing caught it,
    // but the row states `vd` as a runtime scalar and so promises to honour it.
    float sum_sq = 0.0;
    for (uint at = lid; at < p.vd; at += uint(PIE_GROUP_X)) {
        float v = PIE_LOAD(x[base + at]);
        sum_sq += v * v;
    }
    float inv = pie_inv_rms(sum_sq, p.vd, p.eps);
    for (uint at = lid; at < p.vd; at += uint(PIE_GROUP_X)) {
        float xi = PIE_LOAD(x[base + at]);
        float zr = PIE_LOAD(z[base + at]);
        out_[base + at] =
            PIE_STORE((xi * inv * PIE_LOAD(w[at])) * (zr * stable_sigmoid(zr)));
    }
}

// pie:instantiate gated_rms_bfloat16
// pie:instantiate gated_rms_strided_bfloat16 PIE_STRIDED=1
