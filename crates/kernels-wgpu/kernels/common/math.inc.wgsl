// Transcendentals, for the arguments a shader is actually handed.

// `tanh`, without the overflow hole in the middle of its own saturation.
//
// WGSL's `tanh` is `(exp(2x) - 1) / (exp(2x) + 1)` on this backend. `exp(2x)`
// overflows f32 at `2x > ln(f32::MAX) = 88.7`, and past there the expression
// is `inf / inf`, which is **NaN** -- for arguments whose correct answer is
// the most obvious one the function has, exactly 1.
//
// The negative side is fine and for a different reason: `exp(2x)` UNDERFLOWS
// to zero, `(0 - 1) / (0 + 1)` is -1, and that is right. So this is a
// one-sided defect that a symmetric test misses.
//
// Measured on this machine, twice, at the boundary the overflow predicts
// (`44.36 = 88.7 / 2`):
//
//   * `logit_softcap` at `cap = 12.5` returned the cap for a logit of 512
//     (`x = 40.96`) and NaN for 552 (`x = 44.16`).
//   * `gelu_tanh`'s inner term is `0.798 * (g + 0.0447 g^3)`, which crosses
//     44.36 at a GATE OF 10.5 -- an ordinary FFN activation. Measured: 10.0
//     answered 10.0 and 10.5 answered NaN.
//
// The second one is the reason this file exists. A logit of 552 wants a
// miscalibrated head; a gate of 10.5 wants nothing at all, and every gemma
// GeGLU on this backend was one such activation away from a NaN that
// propagates through the rest of the layer.
//
// Clamping to 16 changes NOTHING that was right. `tanh(x)` is exactly 1.0 in
// f32 for `x >= 9.02` -- `1 - tanh(x) ~ 2*exp(-2x)` falls below the half-ulp
// below 1.0 there -- so every argument the clamp touches already had 1.0 as
// its correctly rounded answer. 16 rather than 9.02 because the margin costs
// nothing: `exp(32)` is 7.9e13, nowhere near the overflow.
fn pie_tanh(x: f32) -> f32 {
    return tanh(clamp(x, -16.0, 16.0));
}

// `log(1 + t)`, for a `t` that `1 + t` would round away.
//
// WGSL HAS NO `log1p`. Neither does MSL, which is why
// `kernels-metal/kernels/moe/route.metal` spells its `sqrt_softplus` as
// `log(1 + exp(x))` and records the divergence from cuda's `log1pf(expf(x))`
// as a fact it accepted. The divergence is real and it is not small where it
// bites: `1 + t` rounds to exactly 1.0 for every `t` below 2^-24 (5.96e-8), so
// the naive spelling returns exactly ZERO there while `log1p` returns `t`.
// Through the `sqrt` above it, a routing logit of -22 comes out as a weight of
// 0 where cuda's is 1.67e-5.
//
// # The compensated form is DEAD ON THIS PLANE, and that is a measurement
//
// Kahan's is the three-line answer and it was written here first: `u = 1 + t`,
// `d = u - 1` -- which Sterbenz makes an EXACT recovery of the part of `t` that
// survived the rounding -- and then `log(u) * (t / d)`, falling back to `t`
// when `d` is zero.
//
// It returns ZERO on an L40S through naga and the NVIDIA Vulkan driver, and
// the reason is that `d` is never observed to be zero. Probed directly, by
// replacing the whole body with `select(999.0, 111.0, d == 0.0)` and routing a
// row whose every logit is around -22: every weight came back `sqrt(999)`.
// The compiler reassociates `(1.0 + t) - 1.0` to `t`, which makes `t / d` one
// and `log(u)` `log(1.0)`, and the product is the zero the trick exists to
// avoid. WGSL permits that -- it does not require IEEE evaluation order and
// there is no `NoContraction` to ask for -- so a compensation that depends on
// a rounding SURVIVING cannot be written in this language.
//
// # So the small argument is served by its own series
//
// A polynomial has no identity to reassociate. Through `t^5/5` the truncation
// is `t^6/6`, which at the crossover `t = 1/16` is 1.6e-7 of `t` -- inside f32
// -- and it falls as `t^5` below that. Above the crossover the naive `1 + t`
// keeps all but 2^-24 of an argument no smaller than 1/16, so `log(1 + t)` is
// good to 9.5e-7 relative and better everywhere further up.
//
// The two halves meet inside f32, so this plane does NOT inherit metal's
// divergence: `sqrt_softplus` here agrees with `kernels-cuda`'s at every logit,
// including the tail that made the note worth writing. Measured against
// `pie::moe::topk_sqrtsoftplus` on the same card, over a routing whose logits
// run from -30 to -19: worst relative difference 7.9e-7.
fn pie_log1p(t: f32) -> f32 {
    if (abs(t) < 0.0625) {
        return t * (1.0 - t * (0.5 - t * (0.33333333333 - t * (0.25 - t * 0.2))));
    }
    return log(1.0 + t);
}
