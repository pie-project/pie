// Model and Tokenizer wrappers over pie:core/model WIT resources.

import {
  Model as _Model,
  Tokenizer as _Tokenizer,
} from 'pie:core/model';

/**
 * A loaded model instance.
 *
 * Wraps the `pie:core/model.Model` WIT resource.
 */
export class Model {
  /** @internal */
  readonly _handle: _Model;

  private constructor(handle: _Model) {
    this._handle = handle;
  }

  /** Load a model by name. */
  static load(name: string): Model {
    return new Model(_Model.load(name));
  }

  /** Get the tokenizer for this model. */
  tokenizer(): Tokenizer {
    return new Tokenizer(this._handle.tokenizer());
  }
}

/**
 * Tokenizer for encoding and decoding text ↔ token IDs.
 *
 * Wraps the `pie:core/model.Tokenizer` WIT resource.
 */
export class Tokenizer {
  /** @internal */
  readonly _handle: _Tokenizer;

  /** @internal */
  constructor(handle: _Tokenizer) {
    this._handle = handle;
  }

  /** Encodes text into token IDs. */
  encode(text: string): Uint32Array {
    return this._handle.encode(text);
  }

  /** Decodes token IDs back into text. */
  decode(tokens: Uint32Array): string {
    return this._handle.decode(tokens);
  }

  /** Returns the full vocabulary: [tokenIds, byteSequences]. */
  vocabs(): [Uint32Array, Uint8Array[]] {
    return this._handle.vocabs();
  }

  /** Returns the split regex used by the tokenizer. */
  splitRegex(): string {
    return this._handle.splitRegex();
  }

  /** Returns special tokens: [tokenIds, byteSequences]. */
  specialTokens(): [Uint32Array, Uint8Array[]] {
    return this._handle.specialTokens();
  }
}
