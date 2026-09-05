#ifndef PIE_VULKAN_SDPA_ONLINE_SLANG
#define PIE_VULKAN_SDPA_ONLINE_SLANG

static const float PIE_SDPA_NEG_INF = -3.0e38;

void sdpa_online_update(float score, inout float max_score,
                        inout float sum_exp_score, out float history_scale,
                        out float score_scale) {
    float new_max = max(max_score, score);
    history_scale = exp(max_score - new_max);
    score_scale = exp(score - new_max);
    max_score = new_max;
    sum_exp_score = sum_exp_score * history_scale + score_scale;
}

float sdpa_merge_sink(float sink, float reference_max,
                      inout float sum_exp_score) {
    float merged_max = max(reference_max, sink);
    float output_scale = exp(reference_max - merged_max);
    sum_exp_score = sum_exp_score * output_scale + exp(sink - merged_max);
    return output_scale;
}

float sdpa_lse_base2(float max_score, float sum_exp_score) {
    static const float kLog2E = 1.44269504088896340736;
    if (sum_exp_score > 0.0) {
        return max_score * kLog2E + log2(sum_exp_score);
    }
    return asfloat(0xff800000u);
}

#endif
