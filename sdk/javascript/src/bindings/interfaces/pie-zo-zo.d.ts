/** @module Interface pie:zo/zo **/
export function adapterSeed(self: ForwardPass, seed: bigint): void;
export function initialize(self: Adapter, rank: number, alpha: number, populationSize: number, muFraction: number, initialSigma: number): void;
export function update(self: Adapter, scores: Float32Array, seeds: BigInt64Array, maxSigma: number): void;
export type Error = import('./pie-core-types.js').Error;
export type ForwardPass = import('./pie-core-inference.js').ForwardPass;
export type Adapter = import('./pie-core-adapter.js').Adapter;
