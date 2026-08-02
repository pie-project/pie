/** @module Interface pie:inferlet/grammar@0.3.0 **/
export type Error = import('./pie-inferlet-types.js').Error;

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
  * Create a new matcher from a grammar.
  */
  constructor(grammar: Grammar)
  /**
  * Accept one or more decoded tokens, advancing the matcher state.
  * Returns an error if any token violates the grammar.
  */
  acceptTokens(tokenIds: Uint32Array): void;
  /**
  * The packed allowed-token bitmask at the current position: a
  * `[ceil(vocab/32)]` u32 array where bit `j` (word `j/32`, bit `j%32`)
  * is 1 iff token `j` is allowed, 0 if disallowed. Applied in-program
  * by the sampler's `mask-apply` op (the de-hardwired grammar mask
  * channel); multiple constraints compose by word-wise bitwise-AND
  * host-side. Replaces the removed BRLE `next-token-logit-mask`.
  */
  mask(): Uint32Array;
  /**
  * Check whether the matcher has reached a terminal state.
  */
  isTerminated(): boolean;
  /**
  * Reset the matcher to its initial state so it can be reused.
  */
  reset(): void;
  /**
  * Split off an independent matcher positioned exactly where this one
  * is. The two matchers do not share state afterwards. Search
  * algorithms that branch -- beam search, tree drafting, speculative
  * decoding -- need one grammar state per live branch; without this,
  * a branch has to be replayed token by token from a fresh matcher.
  */
  fork(): Matcher;
  /**
  * Undo the last `num-tokens` accepted tokens, returning the matcher
  * to the state it had before them. Values above
  * `rollback-capacity` undo as much history as is retained. This is
  * the reject path of speculative decoding: the drafter proposes k
  * tokens, the verifier accepts j < k, and the matcher must give back
  * the other k - j.
  */
  rollback(numTokens: number): void;
  /**
  * How many accepted tokens `rollback` can still undo. History is
  * bounded, so a caller that drafts deeply should check this rather
  * than assume an unlimited undo log.
  */
  rollbackCapacity(): number;
}
