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
#if defined(PIE_YARN)
    float base;
    int head_dim;
    float factor;
    float low_dim;
    float high_dim;
    float mscale;
#elif defined(PIE_FREQS)
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
#elif defined(PIE_YARN)
    // THE RAMP, AND THE HOST DECIDED WHERE IT STARTS AND ENDS.
    // `kernels_vulkan::rope::ramp_bounds` computes `low_dim`/`high_dim` from
    // beta_fast/beta_slow and the checkpoint's original context, exactly as
    // `kernels-wgpu`'s does and for the same reason: they are functions of the
    // MODEL and not of the row, so computing them per invocation would be the
    // same number derived a few thousand times.
    //
    // `base` is `log2(theta)` -- the host takes the log so this can use
    // `exp2`, which is the same trade `PIE_PROP` and the plain arm already
    // make on this file.
    float dn = float(i) / float(pair_half);
    float base_freq = exp2(-dn * pc.base);
    // A degenerate ramp is possible -- `high == low` when the correction
    // bounds collapse -- and dividing by zero there would make every channel
    // NaN. The epsilon is the one `kernels-cuda` states and it is on the
    // DENOMINATOR only, so the clamp still saturates the way it should.
    float denom = (pc.high_dim == pc.low_dim)
        ? (pc.high_dim + 1e-3 - pc.low_dim)
        : (pc.high_dim - pc.low_dim);
    float ramp = clamp((float(i) - pc.low_dim) / denom, 0.0, 1.0);
    float theta = float(position[row]) * base_freq
        * ((1.0 - ramp) + ramp / pc.factor);
    uint i1 = row_base + h * uint(pc.head_dim) + i;
    rotate_pair(i1, i1 + pair_half, theta, pc.mscale);
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

// # The merge this kernel is on the wrong side of
//
// A decode step is 73% ordering (see the `hazards` doc in
// `driver-vulkan/src/device.rs` for how that was priced against the bus), so
// the only thing worth buying is a stage boundary, and only a merge that adds
// no arithmetic can afford one. Two of the eleven stages in a layer are the
// qk-norms and then the ropes, in that order, and they are the same shape:
// `rms_single_row` at a per-head axis puts one workgroup on one head, and so
// would this kernel if it were written per head instead of per pair.
//
// So a fused `rms_rope` is free by construction, which almost nothing else
// is. One workgroup owns a head. It reduces the head's `head_dim` elements
// exactly as `norm/rms.slang` does, gets `inv_rms`, and then instead of
// writing the normed row for another launch to read back, each thread takes
// some of the head_dim/2 rotary pairs, applies the gain and the scale to the
// two elements it already has, rotates them, and writes once. The norm's two
// passes over the row stay two passes. THE ROPE'S OWN READ AND WRITE
// DISAPPEAR ENTIRELY, along with the barrier between them.
//
// It is worth 0.099 ms of a 2.77 ms step, and that is measured rather than
// modelled: with the barrier before every `neox` fire suppressed -- which is
// exactly the ordering the fused kernel removes and nothing else -- the step
// goes 2.772 -> 2.674 short and 2.954 -> 2.853 long, over four runs each with
// no overlap between the spreads. 3.5 us for each of the 28, which is the
// MARGINAL price from the curve in the `hazards` doc and not the 6.5 us
// average that dropping every barrier at once implies. Merging two stages
// buys the marginal one; only overlap across the whole run buys the other.
//
// Unlike every merge refuted before it, this one does not change how much
// parallelism the launch has: the norm already runs one workgroup per head,
// and this kernel today runs head_dim/2 workgroups of ONE THREAD per head,
// so the fused form has the same threads in a better shape.
//
// What it costs to write: this file's five arms (plain, prop, freqs, decode,
// strided) have to survive the move, `kernels-metal` has to declare the
// symbol or `check_plan` refuses the text (see `model-ir/src/kernels.rs`),
// and the census signatures move in the same diff. Validate it the way
// `the_tiled_gemm_answers_the_way_the_vector_kernel_does` does -- one text
// under two fact settings, tolerance and argmax rather than equality.
//
// The stage after it, the kv append, is the same trade again: if the fused
// kernel wrote k's rotated pair straight into its page slot, and the v
// projection wrote v into its own, `kv_append_paged` would have nothing left
// to copy. That is a second boundary for no arithmetic at all.

// pie:instantiate neox_decode_bfloat16 PIE_DECODE=1
// pie:instantiate neox_freqs_decode_bfloat16 PIE_FREQS=1 PIE_DECODE=1
// pie:instantiate neox_freqs_mb_bfloat16 PIE_FREQS=1
// pie:instantiate neox_mb_bfloat16
// pie:instantiate neox_yarn_mb_bfloat16 PIE_YARN=1
// pie:instantiate neox_prop_decode_bfloat16 PIE_PROP=1 PIE_DECODE=1
// pie:instantiate neox_prop_mb_bfloat16 PIE_PROP=1
// pie:instantiate neox_strided_bfloat16 PIE_STRIDED=1
