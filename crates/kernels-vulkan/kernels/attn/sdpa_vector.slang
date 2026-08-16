#include "common/bf16.slang"
#include "attn/sdpa_online.slang"

// Dense decode SDPA. The Metal path splits keys across simdgroups and merges
// with subgroup reductions; this Vulkan body keeps the same online softmax and
// owns one output channel per invocation, with the score reduced across the
// workgroup in shared memory rather than in a subgroup. That is deliberately
// subgroup-width agnostic and preserves the row/binding contract.

#ifndef PIE_HEAD_DIM
#error "sdpa_vector requires -DPIE_HEAD_DIM=<64|128|256>"
#endif

PIE_BUFFER_RO(0, queries);
PIE_BUFFER_RO(1, keys);
PIE_BUFFER_RO(2, values);
PIE_BUFFER_RW(3, out_);

struct Push {
    int gqa_factor;
    int n;
    PIE_STRIDE k_head_stride;
    PIE_STRIDE k_seq_stride;
    PIE_STRIDE v_head_stride;
    PIE_STRIDE v_seq_stride;
    float scale;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

// The score is computed ONCE per key, not once per thread.
//
// This kernel used to give every thread its own copy of `dot_qk`, so all
// `PIE_HEAD_DIM` invocations walked the entire query and key vectors to reach
// the same scalar. The dot product is the whole cost of decode attention, so
// the whole cost was being paid `PIE_HEAD_DIM` times over -- 128 times at the
// usual head width.
//
// That is not a new diagnosis. It is the same defect, in the same shape, that
// `sdpa_paged.slang` carried until it was measured: 306 ms at 1536 tokens of
// context became 110 ms when the workgroup added its terms in a tree instead.
// This body is that fix transcribed, and the reason it needed transcribing
// separately is that these kernels share an online softmax but not a dot
// product, so fixing one could not fix the others and nothing pointed from
// any of them to its siblings. It was found by dumping the grids a decode
// launches, looking for something else entirely. `sdpa_sliding.slang` is the
// third of the family and had it too.
//
// Each thread contributes its own term and the workgroup sums them pairwise,
// which is both less arithmetic and a slightly better-conditioned sum than
// the sequential one it replaces. The output accumulation is untouched, one
// dimension per thread, because that part was never redundant.
//
// Measured rather than assumed, at `d_128`, 16 query heads, one row, timing
// n=4096 against n=256 so that the difference is the key loop alone and not
// the buffer setup around it. Best of five, this card:
//
//   before   67.7 ms
//   after    42.5 ms, 46.6 ms on a second run
//
// About 1.5x, and NOT the hundred-and-twenty-eight-fold the arithmetic count
// would suggest -- which is worth stating, because the arithmetic count is
// what the fix was reasoned from. Two things bound it. The old body read each
// key's 128 values 128 times but read them from cache, so the DRAM traffic
// was always the same and it is the traffic this kernel is really bound by.
// And the tree costs seven barriers per key, which the sequential version
// did not pay at all. A subgroup reduction would pay neither, and is left
// alone deliberately: this body is subgroup-width agnostic by design, and
// that is a promise worth more here than the remaining factor.
//
// Every thread of the workgroup shares one row and one key position -- the
// loop bound `pc.n` is uniform and there is no mask here to make it otherwise
// -- so the barriers below are reached by all threads or by none.
groupshared float pie_sdpa_part[PIE_HEAD_DIM];

[numthreads(PIE_HEAD_DIM, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint q_head = group.x;
    uint row = group.y;
    uint d_out = local.x;
    uint n_rows = gl_NumWorkGroups.y;
    uint kv_head = q_head / uint(pc.gqa_factor);
    uint q_offset = (q_head * n_rows + row) * uint(PIE_HEAD_DIM);
    uint o_offset = q_offset;

    float q_d = PIE_LOAD(queries[q_offset + d_out]);
    float max_score = PIE_SDPA_NEG_INF;
    float sum_exp = 0.0;
    float out_acc = 0.0;
    for (int i = 0; i < pc.n; i++) {
        uint k_base = kv_head * PIE_LOW(pc.k_head_stride) + uint(i) * PIE_LOW(pc.k_seq_stride);
        pie_sdpa_part[d_out] = pc.scale * q_d * PIE_LOAD(keys[k_base + d_out]);
        GroupMemoryBarrierWithGroupSync();
        for (uint half_ = uint(PIE_HEAD_DIM) >> 1; half_ > 0u; half_ >>= 1) {
            if (d_out < half_) pie_sdpa_part[d_out] += pie_sdpa_part[d_out + half_];
            GroupMemoryBarrierWithGroupSync();
        }
        float score = pie_sdpa_part[0];
        float hist, prob;
        sdpa_online_update(score, max_score, sum_exp, hist, prob);
        uint v_at = kv_head * PIE_LOW(pc.v_head_stride) + uint(i) * PIE_LOW(pc.v_seq_stride) + d_out;
        out_acc = out_acc * hist + prob * PIE_LOAD(values[v_at]);
        // The next key overwrites the scratch, so no thread may run ahead of
        // the slowest reader of `pie_sdpa_part[0]`.
        GroupMemoryBarrierWithGroupSync();
    }
    out_[o_offset + d_out] = PIE_STORE(sum_exp == 0.0 ? out_acc : out_acc / sum_exp);
}

// pie:instantiate sdpa_vector_decode_bfloat16_d_64 PIE_HEAD_DIM=64
// pie:instantiate sdpa_vector_decode_bfloat16_d_128 PIE_HEAD_DIM=128
// pie:instantiate sdpa_vector_decode_bfloat16_d_256 PIE_HEAD_DIM=256
