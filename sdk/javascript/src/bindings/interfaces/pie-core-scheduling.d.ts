/** @module Interface pie:core/scheduling **/
/**
 * ── Market ─────────────────────────────────────────────────────
 * Cost to produce one new KV page (constant = 1 credit).
 */
export function price(): number;
/**
 * Rent: clearing price from the last knapsack auction on this context's device.
 */
export function rent(ctx: Context): number;
/**
 * Dividend received last step (endowment-proportional share of revenue).
 */
export function dividend(model: Model): number;
/**
 * ── Device ─────────────────────────────────────────────────────
 * Per-tick latency of this context's device (seconds), updated each tick.
 */
export function latency(ctx: Context): number;
/**
 * ── Account ────────────────────────────────────────────────────
 * Current credit balance (global, usable on any device).
 */
export function balance(model: Model): number;
export type Model = import('./pie-core-model.js').Model;
export type Context = import('./pie-core-context.js').Context;
