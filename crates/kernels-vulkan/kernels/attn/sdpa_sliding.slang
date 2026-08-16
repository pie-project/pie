#include "common/bf16.slang"
#include "attn/sdpa_online.slang"

// Sliding-window dense SDPA. It is the vector decode body with the row-specific
// causal end moved to `N - (M-1-row)` and optional learned sinks folded into the
// denominator without a value term.

#ifndef PIE_HEAD_DIM
#error "sdpa_sliding requires -DPIE_HEAD_DIM=<64|256|512>"
#endif

PIE_BUFFER_RO(0, queries);
PIE_BUFFER_RO(1, keys);
PIE_BUFFER_RO(2, values);
PIE_BUFFER_RW(3, out_);
#if defined(PIE_WITH_SINK)
PIE_BUFFER_RO(4, sinks);
#endif

struct Push {
    int gqa_factor;
    int n;
    PIE_STRIDE k_head_stride;
    PIE_STRIDE k_seq_stride;
    PIE_STRIDE v_head_stride;
    PIE_STRIDE v_seq_stride;
    float scale;
    int window;
    int q_row_stride;
    int o_row_stride;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

// The score is computed ONCE per key, not once per thread. See
// `sdpa_vector.slang`, which carried the same defect in the same words: every
// invocation used to walk the whole query and key vector to reach one scalar,
// so the dot product -- the entire cost of decode attention -- was paid
// `PIE_HEAD_DIM` times over.
//
// This is the third kernel in this directory to need the fix, after
// `sdpa_paged.slang` and `sdpa_vector.slang`. Three bodies descended from one
// Metal template, sharing an online softmax but no dot product, so each had
// to be found and corrected separately.
//
// The barriers are safe here for a reason worth stating, because this kernel
// is the one where it is not obvious: the loop bounds `kv_start` and `n_row`
// are computed from `row`, which is the group id's `y` and therefore the same
// for every thread of the workgroup. The window does not vary within a
// workgroup, so the loop is entered the same number of times by all threads
// or by none.
groupshared float pie_sdpa_part[PIE_HEAD_DIM];

[numthreads(PIE_HEAD_DIM, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint q_head = group.x;
    uint row = group.y;
    uint d_out = local.x;
    int n_rows = int(gl_NumWorkGroups.y);
    int n_row = pc.n - (n_rows - 1 - int(row));
    int kv_start = (pc.window > 0 && n_row > pc.window) ? (n_row - pc.window) : 0;
    uint kv_head = q_head / uint(pc.gqa_factor);
    uint q_base = (pc.q_row_stride > 0 ? row * uint(pc.q_row_stride) : row * gl_NumWorkGroups.x * uint(PIE_HEAD_DIM)) + q_head * uint(PIE_HEAD_DIM);
    uint o_base = (pc.o_row_stride > 0 ? row * uint(pc.o_row_stride) : row * gl_NumWorkGroups.x * uint(PIE_HEAD_DIM)) + q_head * uint(PIE_HEAD_DIM);

    float q_d = PIE_LOAD(queries[q_base + d_out]);
    float max_score = PIE_SDPA_NEG_INF;
    float sum_exp = 0.0;
    float out_acc = 0.0;
    for (int i = kv_start; i < n_row; i++) {
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
#if defined(PIE_WITH_SINK)
    float orescale = sdpa_merge_sink(PIE_LOAD(sinks[q_head]), max_score, sum_exp);
    out_acc *= orescale;
#endif
    out_[o_base + d_out] = PIE_STORE(sum_exp == 0.0 ? out_acc : out_acc / sum_exp);
}

// pie:instantiate sdpa_vector_decode_swa_bfloat16_d_256 PIE_HEAD_DIM=256
// pie:instantiate sdpa_vector_decode_swa_bfloat16_d_512 PIE_HEAD_DIM=512
// pie:instantiate sdpa_vector_decode_sink_bfloat16_d_64 PIE_HEAD_DIM=64 PIE_WITH_SINK=1
