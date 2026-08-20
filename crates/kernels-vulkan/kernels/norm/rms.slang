// RMSNorm: `out = w * (x / rms(x))`, one workgroup per row.
//
// The Vulkan port of `kernels-metal/kernels/norm/rms.metal`, which is itself
// MLX's `rms_single_row` scoped to this driver. Five entrypoints off one body:
// the plain row, the strided row a prefill wants, the two-level strided head
// row the per-head q/k norms want, and the two residual-folding forms gemma4
// wants.
//
// ## What moved, and what did not
//
// The arithmetic did not. The accumulate is fp32, the row is strided in chunks
// of `workgroup * N_READS` so a 5120-wide hidden does not ask for a
// 1280-thread workgroup that no implementation allows, and the gain is applied
// in float before the single bf16 round on the way out.
//
// What moved is where the scalars ride. Every one of them is in the push block
// below, which is this backend's rule stated whole: a scalar rides the push
// range, and there is nothing else left here for one to ride.
//
// IT USED TO BE BOTH AT ONCE, and this was the shader the rule was written
// against. `eps`, `axis_size`, `w_stride`, `plus_one` and `gain` were
// `RmsParams` on a storage binding -- Metal's buffer-3 layout, carried over
// whole through `norm/rms_params.slang` -- while `row_pitch` was a push
// constant, so the module declared a descriptor for twenty live bytes AND a
// push range beside it. That is the one arrangement `binding::params` cannot
// serve: it asks push first, and a module answering with the right push size
// "is not also hiding a parameter buffer", so the two roads could only ever
// work because the driver's table path knew this kernel by name.
//
// `norm::rms_single_row` and its four siblings state all five as `Const` marks
// now -- words 0 through 4 of the same statement run the struct was staged
// from, in the struct's own order because that order is the statement's -- and
// `Encoder::words` builds the block from the arguments the BODY passes, in the
// order it passes them. So `row_pitch` is LAST: a stride folded in ahead of the
// five would renumber the fields all five entrypoints share.
//
// The struct's departure also RENUMBERED THE BINDINGS, which no other kernel in
// this sweep needed. `PIE_PARAMS` held binding 3 with the residual arms' `r`
// and `s` declared after it, and a descriptor set is written from a DENSE list
// -- `encode` pushes one buffer per bound operand and `declared.bindings`
// counts them -- so a hole at 3 is an arity refusal, not a gap. `r` is binding
// 3 now and `s` is 4.
//
// The one real divergence from Metal is the reduction: Metal's `simd_sum` over
// a 32-wide simdgroup becomes `WaveActiveSum` over a subgroup whose width the
// implementation chooses, and the partials array is sized for the worst case
// rather than for 32. `common/reduce.slang` is where that argument lives.

#include "common/bf16.slang"
#include "common/reduce.slang"

// Four elements per lane, which is Metal's `N_READS` and the same number for
// the same reason: it is the widest load that stays a single vec4's worth of
// registers per lane at every width the tree sees.
#ifndef N_READS
#define N_READS 4
#endif

#define PIE_GROUP_X 256

PIE_BUFFER_RO(0, x);
PIE_BUFFER_RO(1, w);
PIE_BUFFER_RW(2, out_);

// BINDING 3, WHICH WAS 4, AND 4, WHICH WAS 5 -- see the header. The
// conditional bindings still come after the unconditional ones, so folding the
// residual in does not renumber the three every form shares.
#if defined(PIE_RESIDUAL)
PIE_BUFFER_RO(3, r);
#if defined(PIE_SCALED)
PIE_BUFFER_RO(4, s);
#endif
#endif

// Twenty bytes, or twenty-four where a stride rides along, against the 128
// `Device::max_push` guarantees. `PIE_HEAD_ROWS` is only ever built with
// `PIE_STRIDED` beside it -- see the instantiation lines at the foot of this
// file -- so the two-level form takes the six-word arm as well.
#if defined(PIE_STRIDED)
struct Push {
    float eps;
    uint axis_size;
    uint w_stride;
    uint plus_one;
    float gain;
    int row_pitch;
};
#else
struct Push {
    float eps;
    uint axis_size;
    uint w_stride;
    uint plus_one;
    float gain;
};
#endif
[[vk::push_constant]] ConstantBuffer<Push> pc;

/// The row this workgroup owns, as an element offset.
///
/// Three shapes, and the third is why the head form cannot be the strided one
/// with a different pitch: a token holds `n_rows` per-head norms packed
/// `axis_size` apart, and the next token is a uniform `row_pitch` away, so the
/// base is two-level and one grid axis cannot carry both terms. The launch
/// gives it two -- the group id's `y` is the head, `z` is the token -- exactly
/// as Metal's does.
uint row_base(uint3 group) {
#if defined(PIE_HEAD_ROWS)
    return group.z * uint(pc.row_pitch)
         + group.y * pc.axis_size;
#elif defined(PIE_STRIDED)
    return group.x * uint(pc.row_pitch);
#else
    return group.x * pc.axis_size;
#endif
}

/// The gain at element `i`, in float.
///
/// `plus_one` is the gemma convention -- every RMSNorm's weight is stored as
/// `w` and applied as `1 + w` -- and it is folded in FLOAT, before the bf16
/// round, because MLX materialises `add(weight, 1.0f)` in float and a parity
/// walk against it has to make the same choice.
float gain_at(uint i) {
    float wv = PIE_LOAD(w[pc.w_stride * i]);
    return pc.gain * (pc.plus_one != 0u ? (1.0 + wv) : wv);
}

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint base = row_base(group);
    uint axis_size = pc.axis_size;
    uint span = uint(PIE_GROUP_X) * uint(N_READS);
    uint lid = local.x;

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

#if defined(PIE_RESIDUAL) && defined(PIE_SCALED)
    // One number for the whole row, and every invocation reads it: a broadcast
    // load, which every implementation coalesces, rather than a shared slot
    // that would need a barrier this loop does not otherwise want.
    float post = PIE_LOAD(s[0]);
#elif defined(PIE_RESIDUAL)
    float post = 1.0;
#endif

    for (uint start = lid * uint(N_READS); start < axis_size; start += span) {
        [ForceUnroll] for (uint i = 0u; i < uint(N_READS); i++) {
            uint at = start + i;
            if (at < axis_size) {
                float normed = gain_at(at) * (PIE_LOAD(x[base + at]) * inv);
#if defined(PIE_RESIDUAL)
                out_[base + at] = PIE_STORE((normed + PIE_LOAD(r[base + at])) * post);
#else
                out_[base + at] = PIE_STORE(normed);
#endif
            }
        }
    }
}

// pie:instantiate rms_single_row_bfloat16 N_READS=4
// pie:instantiate rms_strided_row_bfloat16 N_READS=4 PIE_STRIDED=1
// pie:instantiate rms_strided_head_row_bfloat16 N_READS=4 PIE_STRIDED=1 PIE_HEAD_ROWS=1
// pie:instantiate rms_residual_bfloat16 N_READS=4 PIE_RESIDUAL=1
// pie:instantiate rms_residual_scaled_bfloat16 N_READS=4 PIE_RESIDUAL=1 PIE_SCALED=1
