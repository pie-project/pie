/** @module Interface pie:inferlet/tokenizer@0.3.0 **/
/**
 * The tokenizer half of the former `model` interface: global functions over
 * the single bound model's tokenizer (no resource handle).
 * Converts input text into a list of token IDs
 */
export function encode(text: string): Uint32Array;
/**
 * Converts token IDs back into a decoded string
 */
export function decode(tokens: Uint32Array): string;
/**
 * Returns the model's vocabulary as a list of byte sequences (tokens)
 */
export function vocabs(): [Uint32Array, Array<Uint8Array>];
/**
 * Returns the split regular expression used by the tokenizer
 */
export function splitRegex(): string;
/**
 * Returns the special tokens recognized by the model
 */
export function specialTokens(): [Uint32Array, Array<Uint8Array>];
export type Error = import('./pie-inferlet-types.js').Error;
