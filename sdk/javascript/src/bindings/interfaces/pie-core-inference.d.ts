/** @module Interface pie:core/inference **/
export type Pollable = import('./wasi-io-poll.js').Pollable;
export type Error = import('./pie-core-types.js').Error;
export type Context = import('./pie-core-context.js').Context;
export type Model = import('./pie-core-model.js').Model;
export type Tokenizer = import('./pie-core-model.js').Tokenizer;
export type Adapter = import('./pie-core-adapter.js').Adapter;
export type PageId = import('./pie-core-context.js').PageId;
/**
 * binary run-length encoding
 */
export type Brle = Uint32Array;
export type Sampler = SamplerMultinomial | SamplerTopK | SamplerTopP | SamplerMinP | SamplerTopKTopP | SamplerEmbedding | SamplerDist;
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
export type Output = OutputNone | OutputTokens | OutputTokensWithSpeculation | OutputEmbeddings | OutputDistributions;
export interface OutputNone {
  tag: 'none',
}
export interface OutputTokens {
  tag: 'tokens',
  val: Uint32Array,
}
export interface OutputTokensWithSpeculation {
  tag: 'tokens-with-speculation',
  val: [Uint32Array, Uint32Array, Uint32Array],
}
/**
 * accepted tokens, next spec tokens, next spec positions
 */
export interface OutputEmbeddings {
  tag: 'embeddings',
  val: Array<Uint8Array>,
}
export interface OutputDistributions {
  tag: 'distributions',
  val: Array<[Uint32Array, Float32Array]>,
}

export class ForwardPass {
  constructor(model: Model)
  context(context: Context): void;
  inputTokens(tokens: Uint32Array, positions: Uint32Array): void;
  inputSpeculativeTokens(tokens: Uint32Array, positions: Uint32Array): void;
  /**
  * enabled by default
  */
  outputSpeculativeTokens(flag: boolean): void;
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
