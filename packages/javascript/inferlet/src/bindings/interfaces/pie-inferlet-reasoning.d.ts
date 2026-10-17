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
  /**
   * A decoder for the turn `cue` began: inside a thinking block from
   * the first token when the cue left one open (a template that writes
   * `<think>` into its generation cue), so no `start` comes.
   */
  static forCue(cue: Uint32Array): Decoder;
  feed(tokens: Uint32Array): Event;
  reset(): void;
  /**
   * Whether the first token is already inside a thinking block.
   */
  startsInside(): boolean;
}
