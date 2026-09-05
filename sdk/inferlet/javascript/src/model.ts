// Accessors for the single bound model.
//
// The engine serves exactly one model, so these are module-level functions
// over `pie:inferlet/model` — there is no model handle to load or pass
// around. The tokenizer surface (encode/decode/vocabs/special-tokens/
// split-regex) moved to the sibling `tokenizer` module when the WIT split
// separated the two interfaces.

import * as _model from 'pie:inferlet/model@0.3.0';
import type { BlockDrafter, CanvasShape } from 'pie:inferlet/model@0.3.0';

// The tokenizer surface lives in the sibling `tokenizer` module and is
// re-exported here, so `model.encode`/`model.decode` read off `model` the way
// they do in the Rust SDK.
export { decode, encode, specialTokens, splitRegex, tokenBytes, tokensWithPrefix, vocabs } from './tokenizer.js';

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

/** The draft head's chain depth; 0 for a model with no draft head. */
export function mtpDepth(): number {
  return _model.mtpDepth();
}

/** How long a pipeline may hold a frame's wait-set, in microseconds. */
export function submitDeadlineUs(): number {
  return Number(_model.submitDeadlineUs());
}

/** Whether the bound model carries irreversibly-folded recurrent state (a
 *  `recurrent` or `hybrid` pass kind). */
export function isLinear(): boolean {
  const k = _model.passKind();
  return k === 'recurrent' || k === 'hybrid';
}

/** The bound model's block drafter, if it carries one. */
export function draftBlock(): BlockDrafter | undefined {
  return _model.draftBlock();
}

/** The diffusion canvas; `undefined` for every other pass kind. */
export function canvas(): CanvasShape | undefined {
  return _model.canvas();
}

/** Fires one lane may have submitted and not yet taken. */
export function runAheadWindow(): number {
  return _model.runAheadWindow();
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

/** Tokens per KV page for the bound model/engine. */
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
export function rsStateSize(): number {
  return Number(_model.rsStateSize());
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
export function arenaBlockSize(): number {
  return Number(_model.arenaBlockSize());
}

export type { BlockDrafter, CanvasShape, ForwardKind } from 'pie:inferlet/model@0.3.0';
