// Zeroth-order optimization — wraps pie:zo/zo WIT interface.

import * as _zo from 'pie:zo/zo';
import type { ForwardPass } from 'pie:core/inference';
import type { Adapter } from 'pie:core/adapter';

/** Sets the adapter seed on a forward pass for ZO exploration. */
export function adapterSeed(pass: ForwardPass, seed: bigint): void {
    _zo.adapterSeed(pass, seed);
}

/** Initializes the ZO adapter with CMA-ES parameters. */
export function initialize(
    adapter: Adapter,
    rank: number,
    alpha: number,
    populationSize: number,
    muFraction: number,
    initialSigma: number,
): void {
    _zo.initialize(adapter, rank, alpha, populationSize, muFraction, initialSigma);
}

/** Updates the ZO adapter with fitness scores and seeds. */
export function update(
    adapter: Adapter,
    scores: Float32Array,
    seeds: BigInt64Array,
    maxSigma: number,
): void {
    _zo.update(adapter, scores, seeds, maxSigma);
}
