// Tokenizer accessors for the single bound model.
//
// These used to be part of `pie:core/model`. The WIT split moved them to
// their own `pie:inferlet/tokenizer` interface: `model` now carries identity
// and memory-shaping capabilities only. Same functions, new module.

import * as _tokenizer from 'pie:inferlet/tokenizer';

/** Encodes text into token IDs. */
export function encode(text: string): Uint32Array {
  return _tokenizer.encode(text);
}

/** Decodes token IDs back into text. */
export function decode(tokens: Uint32Array): string {
  return _tokenizer.decode(tokens);
}

/** Returns the full vocabulary: [tokenIds, byteSequences]. */
export function vocabs(): [Uint32Array, Uint8Array[]] {
  return _tokenizer.vocabs();
}

/** Returns the split regex used by the tokenizer. */
export function splitRegex(): string {
  return _tokenizer.splitRegex();
}

/** Returns special tokens: [tokenIds, byteSequences]. */
export function specialTokens(): [Uint32Array, Uint8Array[]] {
  return _tokenizer.specialTokens();
}
