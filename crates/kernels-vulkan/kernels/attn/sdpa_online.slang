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

// THE BASE AN LSE LEAVES THIS PLANE IN, and it is the floor's base and not
// this file's.
//
// MEASURED, not assumed. `sdpa_online_update` above accumulates on Slang's
// `exp`, and what that lowers to on this target was read off the compiled
// module rather than off the language reference: at `-target spirv` it is
// `OpExtInst %1 Exp`, GLSL.std.450's NATURAL exponent, with no second operand
// and no `log2(e)` anywhere near it. Slang has `exp2` and this family calls it
// nowhere; nothing folds `log2(e)` into `pc.scale` on the way in either. So
// `max_score` and `sum_exp_score` are an `exp` pair.
//
// flashinfer's are an `exp2` pair, because its HOST multiplies `sm_scale` by
// `log2(e)` before the launch, and `attention.decode_lse` states THAT base --
// base two -- for every plane, because it is the one the plane whose kernel
// this tree does not own has for free.
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
// `sum_exp_score == 0` is a row that kept no key -- causally masked out, a
// window with nothing in it, or a mask that took the row's own key away.
// flashinfer publishes `-inf` there and `attn/attn_sink.slang` tests the
// exponent bits for exactly that row, so `-inf` is the STATED value rather than
// whatever `log2(0)` happens to be.
//
// Spelled as a bit pattern, which is a CHOICE here where it was forced next
// door. WGSL has no infinity literal and rejects a const expression that
// divides by zero, so `attn/sdpa_online.inc.wgsl` had to bitcast; Slang folds
// `-1.0 / 0.0` to the same `-0x1p+128` and would accept it. The bit pattern is
// written anyway because it is the operand `attn/attn_sink.slang` masks and
// compares against, and one spelling for both ends of the point is worth more
// than the shorter one at this end. `asfloat` is `common/bf16.slang`'s own
// spelling for a bit cast.
//
// Not a ternary over both arms: `log2(0.0)` is precisely the value this
// function exists not to publish, and an `if` says so where `?:` invites a
// reader to think both sides are computed.
float sdpa_lse_base2(float max_score, float sum_exp_score) {
    static const float kLog2E = 1.44269504088896340736;
    if (sum_exp_score > 0.0) {
        return max_score * kLog2E + log2(sum_exp_score);
    }
    return asfloat(0xff800000u);
}

#endif
