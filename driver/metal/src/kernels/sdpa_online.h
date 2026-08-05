#ifndef PIE_METAL_SDPA_ONLINE_H
#define PIE_METAL_SDPA_ONLINE_H

METAL_FUNC void sdpa_online_update(
    float score, thread float& max_score, thread float& sum_exp_score,
    thread float& history_scale, thread float& score_scale) {
  const float new_max = max(max_score, score);
  history_scale = fast::exp(max_score - new_max);
  score_scale = fast::exp(score - new_max);
  max_score = new_max;
  sum_exp_score = sum_exp_score * history_scale + score_scale;
}

METAL_FUNC float sdpa_merge_sink(
    float sink, float reference_max, thread float& sum_exp_score) {
  const float merged_max = max(reference_max, sink);
  const float output_scale = fast::exp(reference_max - merged_max);
  sum_exp_score =
      sum_exp_score * output_scale + fast::exp(sink - merged_max);
  return output_scale;
}

#endif
