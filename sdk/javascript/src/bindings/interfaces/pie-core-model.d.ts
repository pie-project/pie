/** @module Interface pie:core/model **/
export type Error = import('./pie-core-types.js').Error;

export class Model {
  /**
   * This type does not have a public constructor.
   */
  private constructor();
  static load(name: string): Model;
  tokenizer(): Tokenizer;
}

export class Tokenizer {
  /**
   * This type does not have a public constructor.
   */
  private constructor();
  /**
  * Converts input text into a list of token IDs
  */
  encode(text: string): Uint32Array;
  /**
  * Converts token IDs back into a decoded string
  */
  decode(tokens: Uint32Array): string;
  /**
  * Returns the tokenizer's vocabulary as a list of byte sequences (tokens)
  */
  vocabs(): [Uint32Array, Array<Uint8Array>];
  /**
  * Returns the split regular expression used by the tokenizer
  */
  splitRegex(): string;
  /**
  * Returns the special tokens recognized by the tokenizer
  */
  specialTokens(): [Uint32Array, Array<Uint8Array>];
}
