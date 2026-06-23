/** @module Interface pie:core/media **/
export type Error = import('./pie-core-types.js').Error;
export type Model = import('./pie-core-model.js').Model;

export class Image {
  /**
   * This type does not have a public constructor.
   */
  private constructor();
  /**
  * Encoded still image (PNG / JPEG / ...).
  */
  static fromBytes(model: Model, bytes: Uint8Array): Image;
  /**
  * Hidden-state rows / KV slots this visual span occupies.
  */
  tokenCount(): number;
  /**
  * How far the 1-D sequence cursor advances past this span.
  */
  positionSpan(): number;
  /**
  * (t, h, w) in merged-token units.
  */
  grid(): [number, number, number];
  /**
  * Model-specific delimiter tokens placed before this span.
  */
  prefixTokens(): Uint32Array;
  /**
  * Model-specific delimiter tokens placed after this span.
  */
  suffixTokens(): Uint32Array;
}

export class Video {
  /**
   * This type does not have a public constructor.
   */
  private constructor();
  /**
  * Encoded animated container (for example GIF).
  */
  static fromBytes(model: Model, bytes: Uint8Array, maxFrames: number): Video;
  /**
  * Number of sampled frames.
  */
  frameCount(): number;
  /**
  * The `index`-th sampled frame as an owned image span.
  */
  frame(index: number): Image;
  /**
  * Timestamp in seconds of the `index`-th sampled frame.
  */
  timestamp(index: number): number;
}

export class Audio {
  /**
   * This type does not have a public constructor.
   */
  private constructor();
  /**
  * Encoded audio (WAV / RIFF).
  */
  static fromBytes(model: Model, bytes: Uint8Array): Audio;
  /**
  * Hidden-state rows / KV slots this clip occupies.
  */
  tokenCount(): number;
  /**
  * How far the 1-D sequence cursor advances past this clip.
  */
  positionSpan(): number;
  /**
  * Model-specific delimiter tokens placed before this span.
  */
  prefixTokens(): Uint32Array;
  /**
  * Model-specific delimiter tokens placed after this span.
  */
  suffixTokens(): Uint32Array;
}
