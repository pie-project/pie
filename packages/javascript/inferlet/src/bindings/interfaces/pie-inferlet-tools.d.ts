/** @module Interface pie:inferlet/tools@0.3.0 **/
/**
 * Tool results (name, value) that follow one another, as the template
 * writes them together. One result is a list of one; tools are declared
 * by `chat.open`.
 */
export function answers(results: Array<[string, string]>): Uint32Array;
/**
 * An earlier assistant turn: `message`, then the calls it made in the
 * template's call syntax. `current` marks a turn after the
 * conversation's last user query, which some templates replay with its
 * reasoning.
 */
export function assistantCalls(message: string, calls: Array<ToolCall>, current: boolean): Uint32Array;
/**
 * Returns the grammar that constrains well-formed tool-call output for
 * this model and toolset, or none if the model has no enforceable format.
 */
export function format(tools: Array<string>): Grammar | undefined;
export type Grammar = import('./pie-inferlet-grammar.js').Grammar;
export type Error = import('./pie-inferlet-types.js').Error;
/**
 * A complete tool call parsed out of the model's output.
 */
export interface ToolCall {
  name: string,
  argumentsJson: string,
}
export type Event = EventStart | EventCall;
/**
 * Tool call detected
 */
export interface EventStart {
  tag: 'start',
}
/**
 * Complete tool call
 */
export interface EventCall {
  tag: 'call',
  val: ToolCall,
}

export class Decoder {
  constructor()
  feed(tokens: Uint32Array): Event;
  reset(): void;
}
