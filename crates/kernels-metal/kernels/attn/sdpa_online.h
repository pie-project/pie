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

/// The cross-simdgroup reduction and writeback that ends every vector SDPA.
///
/// A macro and not a function, deliberately. It closes over eleven names from
/// the kernel's scope -- `o`, `max_score`, `max_scores`, `outputs`, `BD`,
/// `simd_gid` ... -- so a template would take eleven parameters, three of them
/// threadgroup array references with the tile widths in their types, to say
/// what the text already says. And a macro is CHECKABLE in a way a rewrite is
/// not: the preprocessed token stream is identical to what was inlined before,
/// which is the only proof available on a machine with no Metal compiler.
///
/// It was copied verbatim into `sdpa_vector.metal` and `sdpa_sliding.metal` --
/// twenty-three byte-identical lines, and the largest of six shared runs
/// between those two files.
#define SDPA_ONLINE_FINISH()                                              \
  if (simd_lid == 0) {                                             \
    max_scores[simd_gid] = max_score;                              \
    sum_exp_scores[simd_gid] = sum_exp_score;                      \
  }                                                                \
  threadgroup_barrier(mem_flags::mem_threadgroup);                 \
  max_score = max_scores[simd_lid];                                \
  U new_max = simd_max(max_score);                                 \
  U factor = fast::exp(max_score - new_max);                       \
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);     \
                                                                   \
  for (int i = 0; i < v_per_thread; i++) {                         \
    outputs[simd_lid * BD + simd_gid] = o[i];                      \
    threadgroup_barrier(mem_flags::mem_threadgroup);               \
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);   \
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);     \
    threadgroup_barrier(mem_flags::mem_threadgroup);               \
  }                                                                \
                                                                   \
  if (simd_lid == 0) {                                             \
    for (int i = 0; i < v_per_thread; i++) {                       \
      out[i] = static_cast<T>(o[i]);                               \
    }                                                              \
  }                                                               
