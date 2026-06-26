/** @module Interface pie:core/model **/
/**
 * The engine serves exactly one model; these are global functions over
 * that single bound model (no `model`/`tokenizer` resource handles).
 * Returns the name of the bound model
 */
export function name(): string;
/**
 * Returns the model architecture identifier (e.g. "gemma4", "qwen3_6")
 */
export function architecture(): string;
/**
 * Whether the bound model enables system speculation by default
 */
export function defaultSystemSpeculation(): boolean;
/**
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
export type Error = import('./pie-core-types.js').Error;
