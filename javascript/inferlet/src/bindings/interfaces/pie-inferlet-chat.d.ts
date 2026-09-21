/** @module Interface pie:inferlet/chat@0.3.0 **/
/**
 * The tokens a conversation or a raw completion prompt starts with —
 * `<bos>` where the model needs one, empty otherwise. The role fillers
 * below include it; a raw prompt is `prefix() ++ encode(text)`.
 */
export function prefix(): Uint32Array;
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
export type Error = import('./pie-inferlet-types.js').Error;
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
  constructor()
  feed(tokens: Uint32Array): Event;
  reset(): void;
}
