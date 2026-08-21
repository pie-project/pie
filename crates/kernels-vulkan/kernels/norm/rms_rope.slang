// The per-head RMS norm and the NEOX rotation, in one dispatch.
//
// # Why this kernel exists
//
// A decode step on this backend is 73% ordering rather than arithmetic --
// `driver-vulkan`'s `hazards` doc prices it: 313 barriers, 6.5 us each when
// all of them go and 3.5 us when two stages merge. So the only change worth
// making to a decode is one that REMOVES A STAGE BOUNDARY, and the only
// merges that can afford one are those that add no arithmetic and lose no
// parallelism. Two of a layer's eleven stages are the q/k norms and then the
// ropes, in that order, and they are the same shape:
//
//   - `norm/rms.slang` at `PIE_HEAD_ROWS` puts ONE WORKGROUP ON ONE HEAD and
//     reduces `axis_size = head_dim` elements across `PIE_GROUP_X` threads.
//   - `rope/neox.slang` puts `head_dim/2` workgroups of ONE THREAD on the
//     same head.
//
// Same total threads, one of them in a far worse shape. Fusing them is free
// by construction, which is why this merge was chosen over the four others
// that were priced and refused (see the memo at the end of `neox.slang` and
// `qmv.slang`'s two refutations).
//
// Measured before it was written, by suppressing the barrier in front of
// every `neox` fire and nothing else: the step goes 2.772 -> 2.674 short and
// 2.954 -> 2.853 long, four runs each with no overlap between the spreads.
// **0.099 ms**, which is 3.5 us for each of the 28 layers.
//
// # What the fusion actually removes
//
// The norm's two passes over the row stay two passes -- a reduction needs
// them. What disappears is the ROPE'S OWN READ AND WRITE: the rotation
// operates on values this workgroup already holds in registers, so a
// `head_dim`-wide store followed by a `head_dim`-wide load by another
// dispatch becomes nothing at all, and the barrier between them goes with it.
//
// # The rotary extent is not always the head
//
// gemma-4 rotates a quarter of each full-attention head and all of each
// sliding one, so `rotary` is a push constant here where `neox.slang` reads
// it off `gl_NumWorkGroups.x`. That is a consequence of the shape change and
// not a choice: this kernel's grid is one workgroup per head, so the x extent
// no longer carries the pair count and something has to.
//
// Channels at or above `rotary` are NORMED AND STORED UNROTATED. Dropping
// them would be wrong and storing them unnormed would be worse; a partial
// rotation still norms the whole head.

#include "common/bf16.slang"
#include "common/reduce.slang"

#ifndef N_READS
#define N_READS 4
#endif

#define PIE_GROUP_X 256

PIE_BUFFER_RW(0, x);
PIE_BUFFER_RO(1, w);
struct Push {
    float eps;
    uint axis_size;
    uint w_stride;
    uint plus_one;
    float gain;
    uint row_pitch;
    uint rotary;
    float scale;
    float base_or_mscale;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;
[[vk::binding(2)]] StructuredBuffer<int> position;
#if defined(PIE_FREQS)
[[vk::binding(3)]] StructuredBuffer<float> inv_freq;
#endif

/// The gain at element `i`, in float, and the same convention `rms.slang`
/// states: gemma stores every RMSNorm weight as `w` and applies it as
/// `1 + w`, folded in FLOAT before the bf16 round because MLX materialises
/// `add(weight, 1.0f)` in float and a parity walk has to make the same
/// choice.
float gain_at(uint i) {
    float wv = PIE_LOAD(w[pc.w_stride * i]);
    return pc.gain * (pc.plus_one != 0u ? (1.0 + wv) : wv);
}

/// The angle for rotary pair `i`, in whichever of the three schedules this
/// instantiation was built for. `half_rot` is the pair count, which is the
/// divisor the geometric ladder uses and which `neox.slang` reads off its
/// grid.
float theta_at(uint i, uint half_rot, int at) {
#if defined(PIE_PROP)
    // The proportional form divides by the WHOLE head extent rather than by
    // the pair count -- `2*i/head_dim` against `i/half_rot`. Those agree only
    // when the rotation covers the whole head, which is exactly the case a
    // partial rotary breaks, so the two spellings are kept apart here as they
    // are in `neox.slang`.
    float d = 2.0 * float(i) / float(pc.axis_size);
    return pc.scale * float(at) * exp2(-d * pc.base_or_mscale);
#elif defined(PIE_FREQS)
    return pc.scale * float(at) * inv_freq[i];
#else
    float d = float(i) / float(half_rot);
    return pc.scale * float(at) * exp2(-d * pc.base_or_mscale);
#endif
}

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    // y is the head and z is the row, which is `norm.rs`'s `per_head_row`
    // and Metal's launch and NOT this kernel's own natural shape -- x would
    // do for the head. The reason it does not is that a Vulkan lane count is
    // in THREADS on the x axis and the driver div_ceils it by `[numthreads]`,
    // so putting the head on x would spell it `heads * 256` and make the head
    // count a thing recovered by division rather than stated. The whole
    // family already answers this the same way.
    uint head = group.y;
    uint row = group.z;
    // Two-level, exactly as `rms.slang`'s `PIE_HEAD_ROWS` base is: a token
    // holds `n_head` per-head rows packed `axis_size` apart, and the next
    // token is a uniform `row_pitch` away. One grid axis cannot carry both
    // terms, which is why there are two.
    uint base = row * pc.row_pitch + head * pc.axis_size;
    uint axis_size = pc.axis_size;
    uint span = uint(PIE_GROUP_X) * uint(N_READS);
    uint lid = local.x;

#if defined(PIE_DECODE)
    int at = position[0];
#else
    int at = position[row];
#endif

    float acc = 0.0;
    for (uint start = lid * uint(N_READS); start < axis_size; start += span) {
        [ForceUnroll] for (uint i = 0u; i < uint(N_READS); i++) {
            if (start + i < axis_size) {
                float xi = PIE_LOAD(x[base + start + i]);
                acc += xi * xi;
            }
        }
    }
    float inv = pie_inv_rms(acc, axis_size, pc.eps);

    uint rot = pc.rotary;
    uint half_rot = rot / 2u;

    // The rotation, one thread per PAIR. A pair's two channels are `half_rot`
    // apart, so a thread that owns `i` also owns `i + half_rot` and no two
    // threads touch the same element -- which is what makes this loop safe
    // without a barrier after the reduction's own.
    for (uint i = lid; i < half_rot; i += uint(PIE_GROUP_X)) {
        uint i1 = base + i;
        uint i2 = i1 + half_rot;
        float x1 = gain_at(i) * (PIE_LOAD(x[i1]) * inv);
        float x2 = gain_at(i + half_rot) * (PIE_LOAD(x[i2]) * inv);
        float th = theta_at(i, half_rot, at);
        float c = cos(th);
        float s = sin(th);
#if defined(PIE_FREQS)
        float out_scale = pc.base_or_mscale;
#else
        float out_scale = 1.0;
#endif
        x[i1] = PIE_STORE(out_scale * (x1 * c - x2 * s));
        x[i2] = PIE_STORE(out_scale * (x1 * s + x2 * c));
    }

    // The tail the rotation does not reach, normed and stored where it lies.
    // Empty whenever `rotary == head_dim`, which is every model in this tree
    // except gemma-4's full-attention layers.
    for (uint i = rot + lid; i < axis_size; i += uint(PIE_GROUP_X)) {
        x[base + i] = PIE_STORE(gain_at(i) * (PIE_LOAD(x[base + i]) * inv));
    }
}

// The reduction reads the row BEFORE the rotation writes it, and both halves
// of every pair are read by the thread that writes them, so the only ordering
// this kernel needs is the one `pie_inv_rms` already contains. There is no
// second `GroupMemoryBarrierWithGroupSync` here and its absence is load-
// bearing rather than an oversight: adding one would be harmless, but the
// claim that none is needed is what says the merge cost no arithmetic.

// pie:instantiate rms_rope_bfloat16 N_READS=4
// pie:instantiate rms_rope_decode_bfloat16 N_READS=4 PIE_DECODE=1
// pie:instantiate rms_rope_freqs_bfloat16 N_READS=4 PIE_FREQS=1
// pie:instantiate rms_rope_freqs_decode_bfloat16 N_READS=4 PIE_FREQS=1 PIE_DECODE=1
// pie:instantiate rms_rope_prop_bfloat16 N_READS=4 PIE_PROP=1
// pie:instantiate rms_rope_prop_decode_bfloat16 N_READS=4 PIE_PROP=1 PIE_DECODE=1
