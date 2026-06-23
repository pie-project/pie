/** @module Interface pie:core/inference **/
export type Pollable = import('./wasi-io-poll.js').Pollable;
export type Error = import('./pie-core-types.js').Error;
export type Context = import('./pie-core-context.js').Context;
export type Model = import('./pie-core-model.js').Model;
export type Tokenizer = import('./pie-core-model.js').Tokenizer;
export type Adapter = import('./pie-core-adapter.js').Adapter;
export type PageId = import('./pie-core-context.js').PageId;
export type Image = import('./pie-core-media.js').Image;
export type Audio = import('./pie-core-media.js').Audio;
/**
 * binary run-length encoding
 */
export type Brle = Uint32Array;
export type Sampler = SamplerMultinomial | SamplerTopK | SamplerTopP | SamplerMinP | SamplerTopKTopP | SamplerEmbedding | SamplerDist | SamplerRawLogits | SamplerLogprob | SamplerLogprobs | SamplerEntropy;
export interface SamplerMultinomial {
  tag: 'multinomial',
  val: [number, number],
}
export interface SamplerTopK {
  tag: 'top-k',
  val: [number, number],
}
export interface SamplerTopP {
  tag: 'top-p',
  val: [number, number],
}
export interface SamplerMinP {
  tag: 'min-p',
  val: [number, number],
}
export interface SamplerTopKTopP {
  tag: 'top-k-top-p',
  val: [number, number, number],
}
export interface SamplerEmbedding {
  tag: 'embedding',
}
export interface SamplerDist {
  tag: 'dist',
  val: [number, number],
}
/**
 * Returns the model's raw, unscaled logits (pre-softmax) for each
 * requested position. No sampling is performed. The host returns the
 * logits via output.logits as a packed little-endian f32 byte buffer
 * of length `vocab-size * 4` per requested position.
 */
export interface SamplerRawLogits {
  tag: 'raw-logits',
}
/**
 * Returns log p(token | context) — the natural log of the model's
 * probability for the given token at this position. Computed via
 * log_softmax(logits) with NO temperature scaling. Result is a single
 * f32 per slot, surfaced via output.logprobs as a length-1 inner list.
 */
export interface SamplerLogprob {
  tag: 'logprob',
  val: number,
}
/**
 * Returns log p(t | context) for each `t` in the supplied list at the
 * same position — useful for multi-candidate scoring (yes/no, A/B/C/D,
 * reranking). One inner list per slot, one f32 per candidate, in the
 * order the labels were provided.
 */
export interface SamplerLogprobs {
  tag: 'logprobs',
  val: Uint32Array,
}
/**
 * Returns the Shannon entropy H(p) = -sum(p * log p) of the
 * (unscaled) next-token distribution at this position. One f32 per
 * slot via output.entropies. Useful for uncertainty / confidence /
 * adaptive-sampling decisions.
 */
export interface SamplerEntropy {
  tag: 'entropy',
}
/**
 * One typed result per `forward-pass.sampler(...)` slot, in the order
 * the sampler calls were attached. Lets a single forward pass mix
 * arbitrary sampler kinds (e.g. multinomial AND entropy on the same
 * position) without losing any output to a single-variant pick.
 */
export type SlotOutput = SlotOutputToken | SlotOutputDistribution | SlotOutputLogits | SlotOutputLogprobs | SlotOutputEntropy | SlotOutputEmbedding;
/**
 * Sampled token id. Produced by multinomial / top-k / top-p /
 * min-p / top-k-top-p samplers, and by spec-mode verification.
 */
export interface SlotOutputToken {
  tag: 'token',
  val: number,
}
/**
 * Top-k token ids paired with their (post-softmax, temperature-
 * scaled) probabilities. Produced by `sampler.dist`.
 */
export interface SlotOutputDistribution {
  tag: 'distribution',
  val: [Uint32Array, Float32Array],
}
/**
 * Native-endian f32 bytes, length = vocab-size * 4. Produced by
 * `sampler.raw-logits`.
 */
export interface SlotOutputLogits {
  tag: 'logits',
  val: Uint8Array,
}
/**
 * Length 1 for `sampler.logprob`; length K for `sampler.logprobs`
 * (parallel to the requested label list). Computed via log-softmax
 * with no temperature scaling.
 */
export interface SlotOutputLogprobs {
  tag: 'logprobs',
  val: Float32Array,
}
/**
 * Shannon entropy `H(p)` of the unscaled distribution at this
 * position. Produced by `sampler.entropy`.
 */
export interface SlotOutputEntropy {
  tag: 'entropy',
  val: number,
}
/**
 * Hidden-state embedding bytes for this position. Produced by
 * `sampler.embedding` (placeholder; not yet wired end-to-end).
 */
export interface SlotOutputEmbedding {
  tag: 'embedding',
  val: Uint8Array,
}
/**
 * Result of one `forward-pass.execute`. `slots` mirrors the order of
 * `forward-pass.sampler` calls. `spec-tokens` / `spec-positions` are a
 * per-request side channel for the next iteration's draft tokens
 * (empty in non-speculative flows).
 */
export interface Output {
  slots: Array<SlotOutput>,
  specTokens: Uint32Array,
  specPositions: Uint32Array,
}

export class ForwardPass {
  constructor(model: Model)
  context(context: Context): void;
  inputTokens(tokens: Uint32Array, positions: Uint32Array): void;
  /**
  * Splice an encoded visual span (image or video clip) at sequence
  * position `anchor`. The driver runs the vision encoder and scatters
  * the projected rows into the hidden state for this span. See
  * MULTIMODAL.md.
  */
  inputImage(image: Image, anchor: number): void;
  /**
  * Splice an encoded audio clip at sequence position `anchor`. The
  * driver runs the gemma4_audio encoder and scatters the projected
  * soft-token rows into the hidden state. See audio_frontend.md.
  */
  inputAudio(audio: Audio, anchor: number): void;
  inputSpeculativeTokens(tokens: Uint32Array, positions: Uint32Array): void;
  /**
  * enabled by default
  */
  outputSpeculativeTokens(flag: boolean): void;
  /**
  * Controls runtime pass-level speculation for this execute only.
  */
  passSpeculation(flag: boolean): void;
  /**
  * if not provided, fallback to causal mask
  */
  attentionMask(mask: Array<Brle>): void;
  /**
  * if not provided, fallback to all ones (no masking)
  */
  logitMask(mask: Brle): void;
  sampler(indices: Uint32Array, sampler: Sampler): void;
  adapter(adapter: Adapter): void;
  execute(): FutureOutput;
}

export class FutureOutput {
  /**
   * This type does not have a public constructor.
   */
  private constructor();
  /**
  * Returns a pollable object to check when the result is ready
  */
  pollable(): Pollable;
  get(): Output | undefined;
}

export class Grammar {
  /**
   * This type does not have a public constructor.
   */
  private constructor();
  /**
  * Construct from a JSON Schema string.
  */
  static fromJsonSchema(schema: string): Grammar;
  /**
  * Construct a built-in free-form JSON grammar (any valid JSON).
  */
  static json(): Grammar;
  /**
  * Construct from a regular expression pattern.
  */
  static fromRegex(pattern: string): Grammar;
  /**
  * Construct from an EBNF grammar string.
  */
  static fromEbnf(ebnf: string): Grammar;
  /**
  * Debug representation of the grammar. Format is unspecified and
  * may differ across grammar kinds; do not parse.
  */
  toString(): string;
}

export class Matcher {
  /**
  * Create a new matcher from a grammar and tokenizer.
  */
  constructor(grammar: Grammar, tokenizer: Tokenizer)
  /**
  * Accept one or more decoded tokens, advancing the matcher state.
  * Returns an error if any token violates the grammar.
  */
  acceptTokens(tokenIds: Uint32Array): void;
  /**
  * Fill the next-token bitmask.  The returned BRLE encodes which
  * token ids in the vocabulary are allowed at the current position.
  */
  nextTokenLogitMask(): Brle;
  /**
  * Check whether the matcher has reached a terminal state.
  */
  isTerminated(): boolean;
  /**
  * Reset the matcher to its initial state so it can be reused.
  */
  reset(): void;
}
