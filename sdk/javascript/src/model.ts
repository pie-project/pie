// Accessors for the single bound model.
//
// The engine serves exactly one model, so model and tokenizer operations
// are module-level functions over `pie:core/model` — there is no model or
// tokenizer handle to load or pass around.

import * as _model from 'pie:core/model';

/** Name of the bound model. */
export function name(): string {
  return _model.name();
}

/** Model architecture identifier (e.g. "gemma4", "qwen3_6"). */
export function architecture(): string {
  return _model.architecture();
}

/** Whether greedy generation should use the system drafter by default. */
export function defaultSystemSpeculation(): boolean {
  return _model.defaultSystemSpeculation();
}

/** Encodes text into token IDs. */
export function encode(text: string): Uint32Array {
  return _model.encode(text);
}

/** Decodes token IDs back into text. */
export function decode(tokens: Uint32Array): string {
  return _model.decode(tokens);
}

/** Returns the full vocabulary: [tokenIds, byteSequences]. */
export function vocabs(): [Uint32Array, Uint8Array[]] {
  return _model.vocabs();
}

/** Returns the split regex used by the tokenizer. */
export function splitRegex(): string {
  return _model.splitRegex();
}

/** Returns special tokens: [tokenIds, byteSequences]. */
export function specialTokens(): [Uint32Array, Uint8Array[]] {
  return _model.specialTokens();
}
