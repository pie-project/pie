// `eta.diffusion` — a block-diffusion text's reference sampler pieces (which
// canvas positions to accept, at what temperature), a port of
// `inferlet::eta::diffusion`. The pass itself is an ordinary `ForwardPass`
// set to the model's canvas path by `switch`.

import { Dtype } from './ir.js';
import { Tensor, and, cast, cumsum, eq, iota, le, lt, neg, reduceSum, scatterSet, sortDesc } from './value.js';

/** The reference schedule `tMin + (tMax - tMin) * remaining / max`, with
 * `remaining` counting DOWN from `maxSteps` to 1. Host arithmetic; it reaches
 * the program through a control channel the host `set`s. */
export function linearTemperature(remaining: number, maxSteps: number, tMax: number, tMin: number): number {
  return Math.fround(tMin + (tMax - tMin) * Math.fround(remaining / Math.max(maxSteps, 1)));
}

/** The entropy-bound acceptance rule over one canvas: accept the
 * lowest-entropy positions while `sum(H) - max(H) <= bound` over the accepted
 * set (Ben-Hamu et al., 2505.24857). `entropy` is `[n]` f32; the answer is
 * `[n]` bool in canvas order. */
export function entropyBoundAccept(entropy: Tensor, bound: number): Tensor {
  const n = entropy.shape[0];
  const [negSorted, order] = sortDesc(neg(entropy));
  const sorted = neg(negSorted);
  const below = le(cumsum(sorted).sub(sorted), bound);
  const none = lt(iota(n), 0);
  return scatterSet(none, order, below);
}

/** The reference stopping rule for one canvas: the argmax canvas did not move
 * since the previous step AND the mean per-position entropy is under
 * `threshold`. `argmax`/`previous` are `[n]` i32, `entropy` `[n]` f32; the
 * answer is a bool scalar. */
export function stableAndConfident(argmax: Tensor, previous: Tensor, entropy: Tensor, threshold: number): Tensor {
  const n = argmax.shape[0];
  const unchanged = reduceSum(cast(eq(argmax, previous), Dtype.I32));
  const stable = eq(unchanged, n);
  const mean = reduceSum(entropy).div(n);
  return and(stable, lt(mean, threshold));
}
