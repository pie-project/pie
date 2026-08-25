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

// THE BASE AN LSE LEAVES THIS PLANE IN, and it is the floor's base and not
// this file's.
//
// `pie_sdpa_online_update` above accumulates in NATURAL log. WGSL's `exp` is
// e^x — `exp2` is the base-two one and nothing in this family calls it — and
// nothing folds `log2(e)` into `params.scale` on the way in, so `max_score`
// and `sum_exp` are an `exp` pair. flashinfer's are an `exp2` pair, because
// its HOST multiplies `sm_scale` by `log2(e)` before the launch, and
// `attention.decode_lse` states THAT base — base two — for every plane,
// because it is the one the plane whose kernel this tree does not own has for
// free.
//
// So the rebase happens ONCE, here, at the single point where the number stops
// being an accumulator and becomes an operand of the next statement. One
// multiply per `(row, query head)` against a launch per reading, never a launch
// of its own; and `attention.sink` on the far side multiplies by `ln(2)` to
// meet the checkpoint's natural-log sink logit. A plane that published `ln`
// instead would answer every value check in this tree and disagree with cuda by
// a factor of 0.693 at the one place two bases meet, which is the defect
// `kernels-cuda/kernels/attn/attn_sink.cuh`'s header was written for.
//
// `sum_exp == 0` is a row that kept no key — causally masked out, a window with
// nothing in it, or a mask that took the row's own key away. flashinfer
// publishes `-inf` there and `attn/attn_sink.wgsl` tests the exponent bits for
// exactly that row, so `-inf` is the STATED value rather than whatever `log2(0)`
// happens to be. It is spelled as a bit pattern because WGSL has no infinity
// literal and a const expression that divides by zero is a shader-creation
// error — the same reason `sample/argmax.wgsl` spells its own floor that way.
//
// Not a `select`: both arms of a `select` are evaluated, and `log2(0.0)` on the
// dead arm is precisely the value this function exists not to publish.
fn pie_sdpa_lse_base2(max_score: f32, sum_exp: f32) -> f32 {
    const kLog2E = 1.44269504088896340736;
    if (sum_exp > 0.0) {
        return max_score * kLog2E + log2(sum_exp);
    }
    return bitcast<f32>(0xff800000u);
}
