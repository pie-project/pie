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
/**
 * ── Working-set / arena capabilities (global, over the bound model) ──
 * Memory-shaping parameters of the bound model's driver, so an inferlet
 * can size working sets and validate fold lengths before allocating.
 * Size in bytes of one folded recurrent-state object. 0 if the model has
 * no recurrent state (pure attention).
 */
export function rsStateSize(): bigint;
/**
 * Tokens per buffered RS page. 0 if the model has no recurrent state.
 */
export function rsBufferPageSize(): number;
/**
 * Fold granularity in tokens: `forward-pass.fold-buffered(n)` requires `n`
 * to be a positive multiple of this value. 1 (or 0) means unconstrained;
 * 0 also implies the model has no recurrent state. (Token-causal RS models
 * — Qwen3.5 GDN, Nemotron-H Mamba2 — report 1.)
 */
export function rsFoldGranularity(): number;
/**
 * Size in bytes of one unified-arena accounting block. In v1 the KV page is
 * exactly one block, so this is the byte size of one KV page; an RS slab
 * occupies an integer number of these blocks.
 */
export function arenaBlockSize(): bigint;
export type Error = import('./pie-core-types.js').Error;
