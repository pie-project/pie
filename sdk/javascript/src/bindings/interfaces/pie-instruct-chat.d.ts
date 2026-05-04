/** @module Interface pie:instruct/chat **/
/**
 * Fill roles for history replay
 */
export function system(model: Model, message: string): Uint32Array;
export function user(model: Model, message: string): Uint32Array;
export function assistant(model: Model, message: string): Uint32Array;
/**
 * Cue the model to generate (fills generation header)
 */
export function cue(model: Model): Uint32Array;
/**
 * Seal the current turn (insert stop token)
 */
export function seal(model: Model): Uint32Array;
/**
 * Returns the stop token IDs for the model
 */
export function stopTokens(model: Model): Uint32Array;
/**
 * Create a decoder to classify generated tokens
 */
export function createDecoder(model: Model): Decoder;
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
