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
