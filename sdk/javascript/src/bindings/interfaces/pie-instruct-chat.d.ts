/** @module Interface pie:instruct/chat **/
/**
 * Fill roles for history replay
 */
export function system(message: string): Uint32Array;
export function firstUser(message: string): Uint32Array;
export function user(message: string): Uint32Array;
export function systemUser(system: string, user: string): Uint32Array;
export function assistant(message: string): Uint32Array;
/**
 * Cue the model to generate (fills generation header)
 */
export function cue(): Uint32Array;
/**
 * Seal the current turn (insert stop token)
 */
export function seal(): Uint32Array;
/**
 * Returns the stop token IDs for the model
 */
export function stopTokens(): Uint32Array;
/**
 * Create a decoder to classify generated tokens
 */
export function createDecoder(): Decoder;
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
