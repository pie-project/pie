// Fused MoE routing kernel: softmax + top-K + normalize + fused_scales
//
// Replaces ~10 PyTorch dispatches with a single Metal dispatch.
// For decode (num_tokens=1), operates on tiny [E] vectors (E=32 typical).
//
// Input:  [E] bf16 routing logits
// Output: [K] int32 expert_ids (with local_expert_offset subtracted)
//         [K] float32 fused_scales (routing_weight * output2_scale)
//
// Params: [E_float, K_float, output2_scale, local_expert_offset]
//
// Grid: (1, 1, 1), Group: (1, 1, 1)  — single thread, E is tiny

#include <metal_stdlib>
using namespace metal;

kernel void moe_route_topk_bf16(
    device const bfloat*  logits         [[buffer(0)]],  // [E]
    device int*           expert_ids     [[buffer(1)]],  // [K]
    device float*         fused_scales   [[buffer(2)]],  // [K]
    device const float*   params         [[buffer(3)]],  // [E, K, output2_scale, local_offset]
    uint tid [[thread_position_in_grid]]
) {
    const int E = (int)params[0];
    const int K = (int)params[1];
    const float output2_scale = params[2];
    const int local_offset = (int)params[3];

    // 1. Load logits into registers and find max (for numerically stable softmax)
    //    E is small (32 typical), so this fits in registers easily.
    float vals[64];  // max E supported
    float max_val = -INFINITY;
    for (int i = 0; i < E; ++i) {
        vals[i] = float(logits[i]);
        max_val = max(max_val, vals[i]);
    }

    // 2. Compute softmax: exp(x - max) / sum(exp(x - max))
    float sum_exp = 0.0f;
    for (int i = 0; i < E; ++i) {
        vals[i] = exp(vals[i] - max_val);
        sum_exp += vals[i];
    }
    float inv_sum = 1.0f / sum_exp;
    for (int i = 0; i < E; ++i) {
        vals[i] *= inv_sum;
    }

    // 3. Top-K selection via partial sort (insertion sort for K elements)
    //    K is small (4 typical), so O(E*K) is fast.
    int   topk_idx[8];    // max K supported
    float topk_val[8];
    for (int k = 0; k < K; ++k) {
        topk_val[k] = -1.0f;
        topk_idx[k] = -1;
    }

    for (int i = 0; i < E; ++i) {
        float v = vals[i];
        // Check if v belongs in top-K
        if (v > topk_val[K - 1]) {
            // Insert in sorted position (descending)
            int pos = K - 1;
            while (pos > 0 && v > topk_val[pos - 1]) {
                topk_val[pos] = topk_val[pos - 1];
                topk_idx[pos] = topk_idx[pos - 1];
                pos--;
            }
            topk_val[pos] = v;
            topk_idx[pos] = i;
        }
    }

    // 4. Normalize top-K weights (sum to 1)
    float topk_sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        topk_sum += topk_val[k];
    }
    float inv_topk_sum = 1.0f / topk_sum;

    // 5. Write outputs
    for (int k = 0; k < K; ++k) {
        expert_ids[k] = topk_idx[k] - local_offset;
        fused_scales[k] = topk_val[k] * inv_topk_sum * output2_scale;
    }
}

// Half-precision (float16) variant
kernel void moe_route_topk_f16(
    device const half*    logits         [[buffer(0)]],
    device int*           expert_ids     [[buffer(1)]],
    device float*         fused_scales   [[buffer(2)]],
    device const float*   params         [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    const int E = (int)params[0];
    const int K = (int)params[1];
    const float output2_scale = params[2];
    const int local_offset = (int)params[3];

    float vals[64];
    float max_val = -INFINITY;
    for (int i = 0; i < E; ++i) {
        vals[i] = float(logits[i]);
        max_val = max(max_val, vals[i]);
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < E; ++i) {
        vals[i] = exp(vals[i] - max_val);
        sum_exp += vals[i];
    }
    float inv_sum = 1.0f / sum_exp;
    for (int i = 0; i < E; ++i) {
        vals[i] *= inv_sum;
    }

    int   topk_idx[8];
    float topk_val[8];
    for (int k = 0; k < K; ++k) {
        topk_val[k] = -1.0f;
        topk_idx[k] = -1;
    }

    for (int i = 0; i < E; ++i) {
        float v = vals[i];
        if (v > topk_val[K - 1]) {
            int pos = K - 1;
            while (pos > 0 && v > topk_val[pos - 1]) {
                topk_val[pos] = topk_val[pos - 1];
                topk_idx[pos] = topk_idx[pos - 1];
                pos--;
            }
            topk_val[pos] = v;
            topk_idx[pos] = i;
        }
    }

    float topk_sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        topk_sum += topk_val[k];
    }
    float inv_topk_sum = 1.0f / topk_sum;

    for (int k = 0; k < K; ++k) {
        expert_ids[k] = topk_idx[k] - local_offset;
        fused_scales[k] = topk_val[k] * inv_topk_sum * output2_scale;
    }
}
