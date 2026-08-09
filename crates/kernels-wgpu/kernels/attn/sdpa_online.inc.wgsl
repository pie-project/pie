// The online softmax, shared by every SDPA body in this family.
//
// One file, for the reason `attn/sdpa_online.glsl` is one file next door: the
// vector, sliding, paged and tiled bodies all run this recurrence, and two
// copies of it would be two answers to "what does this backend compute" that
// nothing compares. A parity walk against `kernels-metal` or `kernels-vulkan`
// is only meaningful while the recurrence has one definition.
//
// ## Why these return a struct where the GLSL takes `inout`
//
// WGSL has no `inout` and no `out` parameter. It has pointers, but a
// `ptr<function, f32>` parameter buys nothing here except four more places a
// caller can pass the wrong variable, so the running state crosses as a VALUE
// the caller reassigns. The arithmetic is unchanged, term for term.

// The initial running maximum. Not `-inf`: `exp(-inf - -inf)` is NaN, and the
// first key of a row would poison the whole accumulation. A finite floor below
// every representable score gives `history_scale` a clean 0.0 on that first
// step instead.
const PIE_SDPA_NEG_INF = -3.0e38;

// One key's worth of the recurrence: the new running state, plus the two
// scales the caller applies to its own accumulator.
struct PieSdpaStep {
    // The running maximum, after this key.
    max_score: f32,
    // The running denominator, after this key.
    sum_exp: f32,
    // What everything accumulated BEFORE this key must be multiplied by.
    history_scale: f32,
    // This key's own softmax weight, before the final division.
    score_scale: f32,
}

fn pie_sdpa_online_update(score: f32, max_score: f32, sum_exp: f32) -> PieSdpaStep {
    let new_max = max(max_score, score);
    let history_scale = exp(max_score - new_max);
    let score_scale = exp(score - new_max);
    return PieSdpaStep(
        new_max,
        sum_exp * history_scale + score_scale,
        history_scale,
        score_scale,
    );
}

// A learned sink folded into the denominator, after the last key.
//
// gpt-oss's: a per-head logit that joins the softmax with no VALUE behind it,
// so it moves the running maximum and the sum but contributes nothing to the
// numerator. The caller rescales its accumulator by `output_scale` and divides
// by the returned `sum_exp`.
struct PieSdpaSink {
    output_scale: f32,
    sum_exp: f32,
}

fn pie_sdpa_merge_sink(sink: f32, reference_max: f32, sum_exp: f32) -> PieSdpaSink {
    let merged_max = max(reference_max, sink);
    let output_scale = exp(reference_max - merged_max);
    return PieSdpaSink(output_scale, sum_exp * output_scale + exp(sink - merged_max));
}
