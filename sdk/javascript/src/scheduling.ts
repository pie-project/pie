// Market / scheduling accessors.
//
// Thin wrappers over `pie:core/scheduling` for power users writing custom
// bid strategies. Most callers should leave bidding alone — the
// `Generator` auto-bids using a budget-exhausting strategy that spreads
// the wallet over the horizon.

import * as _scheduling from 'pie:core/scheduling';

import type { Context } from './context.js';
import type { Model } from './model.js';

/** Current credit balance (global, usable on any device). */
export function balance(model: Model): number {
  return _scheduling.balance(model._handle);
}

/** Rent: clearing price from the last knapsack auction on this context's
 *  device. */
export function rent(ctx: Context): number {
  return _scheduling.rent(ctx._handle);
}

/** Dividend received last step (endowment-proportional share of revenue). */
export function dividend(model: Model): number {
  return _scheduling.dividend(model._handle);
}

/** Per-tick latency of this context's device (seconds). */
export function latency(ctx: Context): number {
  return _scheduling.latency(ctx._handle);
}

/** Cost to produce one new KV page (constant = 1 credit). */
export function price(): number {
  return _scheduling.price();
}
