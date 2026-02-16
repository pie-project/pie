/** @module Interface pie:instruct/chat **/
/**
 * Fill roles for history replay
 */
export function system(ctx: Context, message: string): void;
export function user(ctx: Context, message: string): void;
export function assistant(ctx: Context, message: string): void;
/**
 * Cue the model to generate (fills generation header)
 */
export function cue(ctx: Context): void;
/**
 * Seal the current turn (insert stop token)
 */
export function seal(ctx: Context): void;
/**
 * Returns the stop token IDs for the model
 */
export function stopTokens(model: Model): Uint32Array;
/**
 * Create a decoder to classify generated tokens
 */
export function createDecoder(model: Model): Decoder;
export type Context = import('./pie-core-context.js').Context;
export type Model = import('./pie-core-model.js').Model;
export type Error = import('./pie-core-types.js').Error;
export type Event = EventDelta | EventInterrupt | EventDone;
/**
 * Generated text chunk
 */
export interface EventDelta {
  tag: 'delta',
  val: string,
}
/**
 * Special token encountered (token ID)
 */
export interface EventInterrupt {
  tag: 'interrupt',
  val: number,
}
/**
 * Generation complete (full accumulated text)
 */
export interface EventDone {
  tag: 'done',
  val: string,
}

export class Decoder {
  /**
   * This type does not have a public constructor.
   */
  private constructor();
  feed(tokens: Uint32Array): Event;
  reset(): void;
}
