// Gated RMSNorm for GDN: `w * rmsnorm(x) * silu(z)`.
//
// The math is the Metal kernel's GDN epilogue, including the numerically stable
// sigmoid and the raw gate-norm weight (no Gemma `1 + w`). What moved is only
// the reduction and row pitch: subgroup width is implementation-defined in
// Vulkan, and the row pitch rides in the push block.
//
// ALL THREE SCALARS RIDE THERE NOW, WHERE TWO OF THEM USED TO BE A STRUCT.
// `eps` and `vd` were `GatedRmsParams` on a storage binding -- Metal's
// buffer-4 layout, carried over whole through a shared norm/rms_params.slang
// header, since deleted along with the last module that included it -- so
// this module declared a descriptor for eight live bytes AND a push range
// beside it, which is the one arrangement `binding::params` cannot serve from
// a row: it asks push first, and a module answering with the right push size
// "is not also hiding a parameter buffer". `gated_rms` states them as `eps:
// Const<f32>` and `vd: Const<i32>` instead -- words 0 and 1 of the same
// statement run the struct was staged from, in the struct's own order because
// that order is the statement's -- and the block below is the only place a
// scalar reaches this shader.
//
// `Encoder::words` builds it from the arguments the BODY passes, in the order
// it passes them, so `row_pitch` is LAST: a stride folded in ahead of the pair
// would renumber what the two variants share, and the pair is what they share.

#include "common/bf16.slang"
#include "common/reduce.slang"

#define PIE_GROUP_X 256

PIE_BUFFER_RO(0, x);
PIE_BUFFER_RO(1, z);
PIE_BUFFER_RO(2, w);
PIE_BUFFER_RW(3, out_);

#if defined(PIE_STRIDED)
struct Push { float eps; uint vd; int row_pitch; };
#else
struct Push { float eps; uint vd; };
#endif
[[vk::push_constant]] ConstantBuffer<Push> pc;

float stable_sigmoid(float v) {
    float y = 1.0 / (1.0 + exp(-abs(v)));
    return v < 0.0 ? 1.0 - y : y;
}

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint lid = local.x;
#if defined(PIE_STRIDED)
    uint base = group.z * uint(pc.row_pitch) + group.y * pc.vd;
#else
    uint base = (group.z * gl_NumWorkGroups.y + group.y) * pc.vd;
#endif
    uint idx = base + lid;
    // Metal launches `tg = (V_d, 1, 1)` -- the threadgroup is exactly as wide
    // as the axis, so every lane owns one channel and there is nothing to loop
    // over. A Vulkan workgroup size is fixed at compile time and cannot follow
    // `pc.vd`, so this walks the axis in strides of the workgroup instead.
    //
    // Only the `lid < pc.vd` half of that was ported. That is correct for the
    // NARROW direction and silently wrong for the wide one: at `v_d > 256`
    // every channel from 256 up was left out of the sum AND never written, so
    // the norm divided by a mean taken over the first 256 channels only and
    // the rest of the head kept whatever was in the buffer. `v_d` is 128 on
    // every GDN checkpoint the tree has seen, which is why nothing caught it,
    // but the statement states `vd` as a runtime scalar and so promises to
    // honour it.
    float sum_sq = 0.0;
    for (uint at = lid; at < pc.vd; at += uint(PIE_GROUP_X)) {
        float v = PIE_LOAD(x[base + at]);
        sum_sq += v * v;
    }
    float inv = pie_inv_rms(sum_sq, pc.vd, pc.eps);
    for (uint at = lid; at < pc.vd; at += uint(PIE_GROUP_X)) {
        float xi = PIE_LOAD(x[base + at]);
        float zr = PIE_LOAD(z[base + at]);
        out_[base + at] =
            PIE_STORE((xi * inv * PIE_LOAD(w[at])) * (zr * stable_sigmoid(zr)));
    }
}

// pie:instantiate gated_rms_bfloat16
// pie:instantiate gated_rms_strided_bfloat16 PIE_STRIDED=1
