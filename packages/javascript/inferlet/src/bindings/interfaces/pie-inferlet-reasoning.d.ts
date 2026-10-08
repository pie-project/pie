/** @module Interface pie:inferlet/reasoning@0.3.0 **/
export type Error = import('./pie-inferlet-types.js').Error;
export type Event = EventStart | EventDelta | EventComplete;
/**
 * Reasoning block started
 */
export interface EventStart {
  tag: 'start',
}
/**
 * Reasoning text chunk
 */
export interface EventDelta {
  tag: 'delta',
  val: string,
}
/**
 * Reasoning complete (full reasoning text)
 */
export interface EventComplete {
  tag: 'complete',
  val: string,
}

export class Decoder {
  constructor()
  feed(tokens: Uint32Array): Event;
  reset(): void;
}
