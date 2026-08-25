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

// THE BASE AN LSE LEAVES THIS PLANE IN, and it is the floor's base and not
// this file's.
//
// `sdpa_online_update` above accumulates in NATURAL log: the score is
// `scale * q.k` and nothing here folds `log2(e)` into `scale`, so `max_score`
// and `sum_exp_score` are an `exp` pair. flashinfer's are an `exp2` pair
// because its host multiplies `sm_scale` by `log2(e)` before the launch, and
// `attention.decode_lse` states THAT base -- base two -- for every plane,
// because it is the one the plane whose kernel we do not own has for free.
//
// So the rebase happens once, here, at the single point where the number
// stops being an accumulator and becomes an operand of the next statement.
// One multiply against a launch per reading, and `attention.sink` on the far
// side multiplies by `ln(2)` to meet the checkpoint's natural-log sink logit.
// A plane that published `ln` instead would answer every value check in this
// tree and disagree with cuda by a factor of 0.693 at the one place two bases
// meet, which is the defect `attn/attn_sink.cuh`'s header was written for.
//
// `sum_exp_score == 0` is a row that kept no key -- causally masked out, or a
// window with nothing in it. flashinfer publishes `-inf` there and
// `attn_sink_rescale` tests `isfinite` for exactly that row, so `-inf` is the
// stated value rather than whatever `log2(0)` happens to be.
METAL_FUNC float sdpa_lse_base2(float max_score, float sum_exp_score) {
  constexpr float kLog2E = 1.44269504088896340736f;
  return sum_exp_score > 0.0f ? (max_score * kLog2E + log2(sum_exp_score))
                              : -INFINITY;
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
