// Tokenizer accessors for the single bound model.
//
// These used to be part of `pie:core/model`. The WIT split moved them to
// their own `pie:inferlet/tokenizer` interface: `model` now carries identity
// and memory-shaping capabilities only. Same functions, new module.

import * as _tokenizer from 'pie:inferlet/tokenizer@0.3.0';

/** Encodes text into token IDs. */
export function encode(text: string): Uint32Array {
  return _tokenizer.encode(text);
}

/** Decodes token IDs back into text. Takes any array of ids (a plain
 *  `number[]` of sampled tokens, or the `Uint32Array` `encode` returns). */
export function decode(tokens: ArrayLike<number>): string {
  return _tokenizer.decode(Uint32Array.from(tokens));
}

function split(tokens: _tokenizer.Token[]): [Uint32Array, Uint8Array[]] {
  return [Uint32Array.from(tokens, (t) => t.id), tokens.map((t) => t.bytes)];
}

/** Returns the full vocabulary: [tokenIds, byteSequences]. */
export function vocabs(): [Uint32Array, Uint8Array[]] {
  return split(_tokenizer.vocabs());
}

/** The byte sequence of each token id. */
export function tokenBytes(tokens: ArrayLike<number>): Uint8Array[] {
  return _tokenizer.tokenBytes(Uint32Array.from(tokens));
}

/** Every token id whose bytes start with `prefix`. */
export function tokensWithPrefix(prefix: Uint8Array): Uint32Array {
  return _tokenizer.tokensWithPrefix(prefix);
}

/** Returns the split regex used by the tokenizer. */
export function splitRegex(): string {
  return _tokenizer.splitRegex();
}

/** Returns special tokens: [tokenIds, byteSequences]. */
export function specialTokens(): [Uint32Array, Uint8Array[]] {
  return split(_tokenizer.specialTokens());
}
