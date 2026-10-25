// Accessors for the single bound model.
//
// The engine serves exactly one model, so these are module-level functions
// over `pie:inferlet/model` — there is no model handle to load or pass
// around.

import * as _model from 'pie:inferlet/model@0.3.0';
import type { BlockDrafter, LaneDecl, LatentSpace, ScheduleFact, SwitchDecl } from 'pie:inferlet/model@0.3.0';

// The tokenizer surface lives in the sibling `tokenizer` module and is
// re-exported here, so `model.encode`/`model.decode` read off `model` the way
// they do in the Rust `inferlet` crate.
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

/** The bound model's block drafter, if it carries one. */
export function draftBlock(): BlockDrafter | undefined {
  return _model.draftBlock();
}

/** Fires one lane may have submitted and not yet taken. */
export function runAheadWindow(): number {
  return _model.runAheadWindow();
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

/** The most rows one pass may read out through `kind`; the pass itself may be any length. */
export function maxReadoutRows(kind: _model.ReadoutKind): number {
  return _model.maxReadoutRows(kind);
}

/** Bytes in one folded recurrent-state object. 0 for pure attention. */
export function rsStateSize(): number {
  return Number(_model.rsStateSize());
}

// -- generative facts ---------------------------------------------------------

/** Every switch the bound model declares (`ForwardPass.switch`), each
 *  value list's first its default; empty for a model with one path (every
 *  text model). */
export function switches(): SwitchDecl[] {
  return _model.switches();
}

/** The switch `name` the model declares, if any. */
export function switchDecl(name: string): SwitchDecl | undefined {
  return _model.switches().find((d) => d.name === name);
}

/** The lane a pass of these switches is: the inputs it binds (exactly
 *  these, by `ForwardPass.input`), whether it binds `state` and `embed`, and
 *  what it reads out. Switches are `[name, value]` pairs of `switches()`;
 *  one left unset takes its default. */
export function lane(switches: Array<[string, string]>): LaneDecl {
  return _model.lane(switches);
}

/** The latent space a denoiser works in; `undefined` without one. */
export function latent(): LatentSpace | undefined {
  return _model.latent();
}

/** The schedule the denoiser was trained under; `undefined` when nothing
 *  denoises. */
export function schedule(): ScheduleFact | undefined {
  return _model.schedule();
}

/** The most rows one lane may carry: the guest sizes its row inputs under this. */
export function maxLatentRows(): number {
  return _model.maxLatentRows();
}

export type {
  BlockDrafter,
  InputAxis,
  InputDecl,
  LaneDecl,
  LatentSpace,
  Readout,
  ReadoutKind,
  RowCount,
  ScheduleFact,
  ScheduleKind,
  SwitchDecl,
} from 'pie:inferlet/model@0.3.0';
