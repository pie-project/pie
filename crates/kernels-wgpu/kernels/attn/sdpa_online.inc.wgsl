const PIE_SDPA_NEG_INF: f32 = -3.0e38;
const PIE_SDPA_LOG2E: f32 = 1.44269504088896340736;

fn sdpa_online_scales(score: f32, max_score: f32) -> vec2<f32> {
    let new_max = max(max_score, score);
    return vec2<f32>(exp(max_score - new_max), exp(score - new_max));
}

fn sdpa_lse_base2(max_score: f32, sum_exp_score: f32) -> f32 {
    if (sum_exp_score > 0.0) {
        return max_score * PIE_SDPA_LOG2E + log2(sum_exp_score);
    }
    return bitcast<f32>(0xff800000u);
}
