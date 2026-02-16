/** @module Interface pie:instruct/reasoning **/
/**
 * Create a decoder to detect reasoning blocks in generated tokens
 */
export function createDecoder(model: Model): Decoder;
export type Model = import('./pie-core-model.js').Model;
export type Error = import('./pie-core-types.js').Error;
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
  /**
   * This type does not have a public constructor.
   */
  private constructor();
  feed(tokens: Uint32Array): Event;
  reset(): void;
}
