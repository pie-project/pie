// Accessors for the single bound model.
//
// The engine serves exactly one model, so these are module-level functions
// over `pie:inferlet/model` — there is no model handle to load or pass
// around. The tokenizer surface (encode/decode/vocabs/special-tokens/
// split-regex) moved to the sibling `tokenizer` module when the WIT split
// separated the two interfaces.

import * as _model from 'pie:inferlet/model';

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

/** Whether the bound model carries irreversibly-folded recurrent state. */
export function isLinear(): boolean {
  return _model.isLinear();
}

/**
 * Which forward-pass interface the bound model requires. Selects the binding
 * surface; do not derive it by parsing `architecture()`.
 */
export function passKind(): _model.ForwardKind {
  return _model.passKind();
}

/** Logits/output dimension. May exceed the tokenizer's vocabulary size. */
export function outputVocabSize(): number {
  return _model.outputVocabSize();
}

/** Tokens per KV page for the bound model/driver. */
export function kvPageSize(): number {
  return _model.kvPageSize();
}

/** Waves per frame (k). A `submit` takes exactly this many ordered slots. */
export function frameSize(): number {
  return _model.frameSize();
}

/**
 * Host-reader channel capacity, in cells, that sustains the engine's
 * run-ahead. Read it per run — unlike `frameSize` it is not promised static.
 */
export function channelCapacity(): number {
  return _model.channelCapacity();
}

/** Max embed tokens in a single pass — the prefill chunk budget. */
export function maxEmbedLength(): number {
  return _model.maxEmbedLength();
}

/** Bytes in one folded recurrent-state object. 0 for pure attention. */
export function rsStateSize(): bigint {
  return _model.rsStateSize();
}

/** Tokens per buffered RS page. 0 if the model has no recurrent state. */
export function rsBufferPageSize(): number {
  return _model.rsBufferPageSize();
}

/** Fold granularity in tokens. 1 (or 0) means unconstrained. */
export function rsFoldGranularity(): number {
  return _model.rsFoldGranularity();
}

/** Bytes in one unified-arena accounting block. */
export function arenaBlockSize(): bigint {
  return _model.arenaBlockSize();
}

export type { ForwardKind } from 'pie:inferlet/model';
