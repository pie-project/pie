/** @module Interface pie:inferlet/tools@0.3.0 **/
/**
 * Register available tools (list of JSON schema strings)
 */
export function equip(tools: Array<string>): Uint32Array;
/**
 * Provide a tool result (after a tool-call reply)
 */
export function answer(name: string, value: string): Uint32Array;
/**
 * Returns the grammar that constrains well-formed tool-call output for
 * this model and toolset, or none if the model has no enforceable format.
 */
export function format(tools: Array<string>): Grammar | undefined;
/**
 * Create a grammar matcher to force-generate tool calls
 */
export function createMatcher(tools: Array<string>): Matcher;
export type Grammar = import('./pie-inferlet-grammar.js').Grammar;
export type Matcher = import('./pie-inferlet-grammar.js').Matcher;
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
