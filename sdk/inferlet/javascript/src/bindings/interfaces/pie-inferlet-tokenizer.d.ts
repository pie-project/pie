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
 * Returns the model's vocabulary
 * 
 * IT IS THE EXPENSIVE DOOR, AND MOST CALLERS WANT ONE OF THE TWO BELOW.
 * A vocabulary is a quarter of a million records; lowering that list
 * across this boundary and looping over it guest-side cost token healing
 * ~115 ms on every launch (palo build log 23), to answer a byte-prefix
 * question the host can answer by scanning a table it already holds.
 */
export function vocabs(): Array<Token>;
/**
 * The raw bytes each of `tokens` stands for, in order. An id the
 * vocabulary does not hold answers an empty list, which is what indexing
 * a table built from `vocabs` gave for the same id.
 */
export function tokenBytes(tokens: Uint32Array): Array<Uint8Array>;
/**
 * Every token id whose bytes begin with `prefix`, ascending.
 * 
 * The host-side spelling of the loop a caller would otherwise write over
 * `vocabs`, and the answer is the same set. An empty prefix matches every
 * token in the vocabulary, because every byte string starts with nothing.
 */
export function tokensWithPrefix(prefix: Uint8Array): Uint32Array;
/**
 * Returns the split regular expression used by the tokenizer
 */
export function splitRegex(): string;
/**
 * Returns the special tokens recognized by the model
 */
export function specialTokens(): Array<Token>;
export type Error = import('./pie-inferlet-types.js').Error;
/**
 * One entry of a token table: the id the model uses and the raw bytes it
 * stands for. A record rather than two parallel lists, so a caller cannot
 * zip them wrong and no comment is needed to say which is which.
 */
export interface Token {
  id: number,
  bytes: Uint8Array,
}
