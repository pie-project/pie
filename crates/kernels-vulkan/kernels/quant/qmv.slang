// Affine GEMV projections, ported from `quant/qmv.metal`.
//
// The port keeps MLX's little-endian affine packing, the four-rows-per-simdgroup
// launch shape, and the bounds-checked K tail whose absence broke Gemma-4-31B at
// K=5376. Vulkan does not promise a 32-wide subgroup, so the final reduction is
// through shared memory over the 32 local-x lanes rather than subgroup width.
//
// The Slang port changed spellings only. The two bodies used to read
// `gl_LocalInvocationID` and `gl_WorkGroupID` directly; they now take the two
// index vectors as parameters, which is the whole tree's rule -- a helper that
// reads a thread builtin is a helper that cannot be called from a second place
// without carrying its caller's grid with it.

#include "common/bf16.slang"
#include "common/affine.slang"

#ifndef PIE_VEC
#define PIE_VEC 1
#endif

[[vk::binding(0)]] StructuredBuffer<uint> w;
PIE_BUFFER_RO(1, scales);
PIE_BUFFER_RO(2, biases);
PIE_BUFFER_RO(3, x);
PIE_BUFFER_RW(4, y);
// Declared only by the variants whose ROW has a sixth buffer operand, and at 5
// rather than 7. Both halves of that are load-bearing.
//
// `kernels_vulkan::bindings()` numbers buffer-kinded operands densely from zero
// in the row's operand order, so `affine_qmv_fast_residual`'s residual is
// binding 5 -- while `affine_qmv_fast` has five operands and no sixth binding
// at all, which is why this cannot simply be declared unconditionally.
//
// This read binding 7 for every variant until an audit compiled each one and
// compared its `OpDecorate Binding` set against the table
// (`scripts/vulkan-kernel-audit.py --bindings`, since deleted). 7 is the Metal
// buffer index: Metal numbers scalars in the same run as buffers, Vulkan sends
// them to a push block, and the two numbers differ by however many scalars
// precede the operand. They must never be copied across, and nothing checks
// that they were not.
#if defined(PIE_RESIDUAL) || defined(PIE_BIAS)
PIE_BUFFER_RO(5, extra);
#endif

// The block is variant-shaped, and it has to be. A push block is not like a
// descriptor set, where declaring more than you read is free: the block a
// module declares must fit inside the pipeline layout's push range
// (`VUID-VkComputePipelineCreateInfo-layout-10069`), and that range is built
// from what the ROW states. `affine_qmv_fast` states two scalars, so a
// module that declares seven -- as this one did until a validation layer
// said so -- is asking the shell for twenty bytes no row will ever push.
//
// `row_stride` and `m` therefore appear only for the strided variant that
// reads them. Three more fields (`x_slot_stride`, `x_row_stride`,
// `slots_per_row`) were declared here and read by nothing at all; they were
// the routed row's shape, copied in and never removed.
struct Push {
    int in_vec_size;
    int out_vec_size;
#if defined(PIE_WIDE_STRIDED)
    int row_stride;
    int m;
#endif
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

// The K lanes. `quant/qmv.slang` splits a row's K across `PIE_LANES` threads
// and `qmv_grid` in `src/quant.rs` multiplies the batch extent by the same
// number -- the two are one decision written twice, and a change to either
// alone silently launches the wrong grid.
//
// It was 32, inherited from the Metal port's simdgroup width; the workgroup was
// therefore 64 threads, two warps. 64 lanes makes it 128 and doubles the
// threads a projection launches, without changing how many WORKGROUPS it
// launches -- `lanes` is divided by the module's own local size, so both
// numbers moving together leaves the group count fixed.
//
// Measured, it is a wash: 37.96 microseconds a dispatch at 32 lanes against
// 37.7 at 64, over 56541 fires of a real qwen3-0.6B decode. It is kept at 64
// because two warps per workgroup cannot use a barrier without half the group
// idling through it, not because it was faster. What this measurement DOES
// establish is that this kernel is not short of resident warps -- doubling
// them changed nothing -- which rules occupancy out as the remaining limit and
// is the reason the next attempt should be split-K, not a wider group.
#define PIE_LANES 64

groupshared float partials[8 * PIE_LANES];

uint weight_words_per_row() { return uint(pc.in_vec_size) * uint(PIE_BITS) / 32u; }
uint groups_per_row() { return uint(pc.in_vec_size) / uint(PIE_GROUP); }

float affine_value(uint row, uint k) {
    uint word = w[row * weight_words_per_row() + k / uint(PIE_CODES_PER_WORD)];
    float code = pie_affine_code(word, k % uint(PIE_CODES_PER_WORD));
    uint g = k / uint(PIE_GROUP);
    return code * PIE_LOAD(scales[row * groups_per_row() + g])
         + PIE_LOAD(biases[row * groups_per_row() + g]);
}

float x_at(uint vec, uint k) {
#if defined(PIE_WIDE_STRIDED)
    return PIE_LOAD(x[vec * uint(pc.row_stride) + k]);
#else
    return PIE_LOAD(x[vec * uint(pc.in_vec_size) + k]);
#endif
}

// The chunk of activations a workgroup stages in shared memory at a time.
//
// 1024 floats is 4 KB, and it is a whole K of 1024 in one pass. Together with
// `partials` the group asks for 6 KB, well inside the 32 KB a workgroup may
// have anywhere this runs.
#define PIE_XCHUNK 1024

// Declared for every variant, including the wide strided one that does not
// call `dot_lane4` -- the whole file is compiled for every entry point, and a
// groupshared array no reachable code touches is one slangc drops.
groupshared float xs_sh[PIE_XCHUNK];

// One pass over K for all FOUR of a lane's rows, reading coalesced.
//
// Everything here was measured on a 4090 against a real 4-bit qwen3-0.6B
// decode of 452 dispatches, where this entry point and its residual twin are
// ninety per cent of the step. Four defects, in the order they were found --
// and only the first two paid, which is the more useful half of the record:
//
//  1. `affine_value` re-read `w[]` for every CODE, and a word holds eight of
//     them at four bits, so the dominant tensor was fetched eight times over
//     and its scales sixteen. `pie_affine_word_dot` is the primitive that
//     already existed for exactly this and that nothing called.
//     Decode step 25.2 ms -> 20.9 ms.
//  2. The four rows were four independent passes over K, so the activation
//     vector was read four times over and no two rows' weight loads were ever
//     in flight together. They are one pass now.  20.9 ms -> 11.0 ms.
//  3. The addresses were thread-contiguous rather than lane-contiguous: each
//     lane owned a RUN of K and walked it, so one warp-wide load instruction
//     touched thirty-two addresses 32 bytes apart -- eight cache lines to
//     serve 64 useful bytes. A warp now reads thirty-two CONSECUTIVE words per
//     instruction by striding the word index by the lane count.
//  4. The activations were read from global memory inside the inner loop as
//     16-bit scalars at a 16-element stride, the same defect in its worst
//     form. They are staged into `xs_sh` once per chunk by a coalesced
//     cooperative load and read from shared thereafter.
//
// 3 and 4 together moved a dispatch from 37.96 microseconds to 37.7, which is
// noise. They are kept because they are not slower and because the access
// pattern they establish is the one a split-K form will need -- but the honest
// reading is that this kernel's remaining cost is NOT its access pattern, and
// an L1 that holds a 6 KB activation vector was already absorbing the defect.
// Read that with the `PIE_LANES` note above, which rules out occupancy by the
// same method: what is left is that a decode's matvec has too little work per
// dispatch to fill the card, and the fix for that is splitting K across
// workgroups, not tuning the loop.
//
// `sum(x)` does not depend on the row, so it is computed once per word and
// handed to all four rows -- see `pie_affine_word_dot_presummed`.
//
// A group boundary never falls inside a word -- `PIE_GROUP` is 32, 64 or 128
// and a word holds 4 or 8 codes -- so one scale and one bias serve the whole
// word, which is what makes the hoist legal rather than merely cheaper.
void dot_lane4(uint out0, uint vec, uint tid, uint lid, out float acc[4]) {
    uint cpw = uint(PIE_CODES_PER_WORD);
    uint n = uint(pc.in_vec_size);
    uint wpr = weight_words_per_row();
    uint gpr = groups_per_row();
    uint rows = uint(pc.out_vec_size);
    uint threads = uint(PIE_LANES) * 2u;
    uint words_per_chunk = uint(PIE_XCHUNK) / cpw;
    [ForceUnroll] for (uint r = 0u; r < 4u; r++) acc[r] = 0.0;

    for (uint base = 0u; base < n; base += uint(PIE_XCHUNK)) {
        // Before the write, not only after it: the previous chunk's readers
        // must be done with `xs_sh` before this one overwrites it.
        GroupMemoryBarrierWithGroupSync();
        for (uint i = tid; i < uint(PIE_XCHUNK); i += threads) {
            xs_sh[i] = base + i < n ? x_at(vec, base + i) : 0.0;
        }
        GroupMemoryBarrierWithGroupSync();

        uint w0 = base / cpw;
        for (uint j = lid; j < words_per_chunk; j += uint(PIE_LANES)) {
            uint kb = base + j * cpw;
            if (kb >= n) break;
            float xw[PIE_CODES_PER_WORD];
            float xsum = 0.0;
            [ForceUnroll] for (uint i = 0u; i < cpw; i++) {
                xw[i] = xs_sh[j * cpw + i];
                xsum += xw[i];
            }
            uint g = kb / uint(PIE_GROUP);
            uint at = w0 + j;
            [ForceUnroll] for (uint r = 0u; r < 4u; r++) {
                uint row = out0 + r;
                if (row < rows) {
                    acc[r] += pie_affine_word_dot_presummed(
                        w[row * wpr + at], xw, xsum,
                        PIE_LOAD(scales[row * gpr + g]),
                        PIE_LOAD(biases[row * gpr + g]));
                }
            }
        }
    }
}

void reduce_store(bool tail, uint3 group, uint3 local) {
    uint lid = local.x;
    uint ly = local.y;
    uint out0 = group.y * 8u + ly * 4u;
    uint vec = group.x;

    float acc[4];
    dot_lane4(out0, vec, ly * uint(PIE_LANES) + lid, lid, acc);
    [ForceUnroll] for (uint r = 0u; r < 4u; r++) {
        partials[(ly * 4u + r) * uint(PIE_LANES) + lid] = acc[r];
    }
    GroupMemoryBarrierWithGroupSync();

    // A tree, not a walk.
    //
    // This used to be `if (lid == 0) for i in 0..32` repeated for each of the
    // lane's four rows: 128 serial shared-memory adds with thirty-one of the
    // thirty-two lanes idle, at the end of every dispatch. Five halving rounds
    // do the same four sums in twenty adds with the lanes spread across them.
    // The barrier is outside the `if` on purpose -- a group-sync reached by
    // only some of the group is undefined, not merely slow.
    [ForceUnroll] for (uint s = uint(PIE_LANES) / 2u; s > 0u; s >>= 1u) {
        if (lid < s) {
            [ForceUnroll] for (uint r = 0u; r < 4u; r++) {
                uint at = (ly * 4u + r) * uint(PIE_LANES);
                partials[at + lid] += partials[at + lid + s];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (lid == 0u) {
        for (uint r = 0u; r < 4u; r++) {
            uint row = out0 + r;
            if (row < uint(pc.out_vec_size)) {
                float sum = partials[(ly * 4u + r) * uint(PIE_LANES)];
#if defined(PIE_BIAS)
                if (tail) sum += PIE_LOAD(extra[row]);
#endif
#if defined(PIE_RESIDUAL)
                uint at = vec * uint(pc.out_vec_size) + row;
                PIE_ACT q = PIE_STORE(sum);
                y[at] = PIE_STORE(PIE_LOAD(q) + PIE_LOAD(extra[at]));
#else
                y[vec * uint(pc.out_vec_size) + row] = PIE_STORE(sum);
#endif
            }
        }
    }
}

#if defined(PIE_WIDE_STRIDED)
void wide_strided(uint3 group, uint3 local) {
    uint row_slot = local.y * 4u + (local.x & 3u);
    if (local.x >= 4u) return;
    uint row = group.y * 8u + row_slot;
    uint vec0 = group.x * uint(PIE_VEC);
    if (row >= uint(pc.out_vec_size)) return;
    for (uint v = 0u; v < uint(PIE_VEC); v++) {
        uint vec = vec0 + v;
        if (vec < uint(pc.m)) {
            float acc = 0.0;
            for (uint k = 0u; k < uint(pc.in_vec_size); k++) {
                acc += x_at(vec, k) * affine_value(row, k);
            }
            y[vec * uint(pc.row_stride) + row] = PIE_STORE(acc);
        }
    }
}
#endif

[numthreads(PIE_LANES, 2, 1)]
void main(uint3 group: SV_GroupID, uint3 local: SV_GroupThreadID) {
#if defined(PIE_WIDE_STRIDED)
    wide_strided(group, local);
#elif defined(PIE_TAIL)
    reduce_store(true, group, local);
#else
    reduce_store(false, group, local);
#endif
}

// pie:instantiate affine_qmv_fast_bfloat16_gs_128_b_4 PIE_GROUP=128 PIE_BITS=4
// pie:instantiate affine_qmv_fast_bfloat16_gs_128_b_8 PIE_GROUP=128 PIE_BITS=8
// pie:instantiate affine_qmv_fast_bfloat16_gs_32_b_4 PIE_GROUP=32 PIE_BITS=4
// pie:instantiate affine_qmv_fast_bfloat16_gs_32_b_8 PIE_GROUP=32 PIE_BITS=8
// pie:instantiate affine_qmv_fast_bfloat16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4
// pie:instantiate affine_qmv_fast_bfloat16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8
// pie:instantiate affine_qmv_fast_residual_bfloat16_gs_128_b_4 PIE_GROUP=128 PIE_BITS=4 PIE_RESIDUAL=1
// pie:instantiate affine_qmv_fast_residual_bfloat16_gs_128_b_8 PIE_GROUP=128 PIE_BITS=8 PIE_RESIDUAL=1
// pie:instantiate affine_qmv_fast_residual_bfloat16_gs_32_b_4 PIE_GROUP=32 PIE_BITS=4 PIE_RESIDUAL=1
// pie:instantiate affine_qmv_fast_residual_bfloat16_gs_32_b_8 PIE_GROUP=32 PIE_BITS=8 PIE_RESIDUAL=1
// pie:instantiate affine_qmv_fast_residual_bfloat16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4 PIE_RESIDUAL=1
// pie:instantiate affine_qmv_fast_residual_bfloat16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8 PIE_RESIDUAL=1
// pie:instantiate affine_qmv_tail_bfloat16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4 PIE_TAIL=1
// pie:instantiate affine_qmv_tail_bfloat16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8 PIE_TAIL=1
// pie:instantiate affine_qmv_tail_bias_bfloat16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4 PIE_BIAS=1 PIE_TAIL=1
// pie:instantiate affine_qmv_tail_bias_bfloat16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8 PIE_BIAS=1 PIE_TAIL=1
// pie:instantiate affine_qmv_wide_strided_bfloat16_gs_64_b_4_v_4_kl_8 PIE_GROUP=64 PIE_BITS=4 PIE_WIDE_STRIDED=1 PIE_VEC=4 PIE_K_LANES=8
// pie:instantiate affine_qmv_wide_strided_bfloat16_gs_64_b_8_v_4_kl_8 PIE_GROUP=64 PIE_BITS=8 PIE_WIDE_STRIDED=1 PIE_VEC=4 PIE_K_LANES=8
