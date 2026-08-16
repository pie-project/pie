// The online-softmax rescaling every SDPA body in this directory shares. The
// Slang half of `attn/sdpa_online.glsl`, and the only thing the change of
// language touched is `const` becoming `static const`: Slang's `inout` and
// `out` parameters are GLSL's, and the arithmetic is untouched because it is
// the arithmetic the four bodies are compared against.

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

#endif
