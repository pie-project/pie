/** @module Interface pie:core/media **/
export type Error = import('./pie-core-types.js').Error;
export type Model = import('./pie-core-model.js').Model;

export class Audio {
  /**
   * This type does not have a public constructor.
   */
  private constructor();
  /**
  * Encoded audio (WAV / RIFF). The host decodes to mono PCM, resamples to
  * the model's rate, and computes its log-mel features
  * (Gemma4AudioFeatureExtractor). Errors if the model has no audio
  * front-end or the bytes don't decode.
  */
  static fromBytes(model: Model, bytes: Uint8Array): Audio;
  /**
  * Hidden-state rows / KV slots this clip occupies == audio soft tokens.
  */
  tokenCount(): number;
  /**
  * How far the 1-D sequence cursor advances past this clip. Equals
  * token-count for Gemma (1-D RoPE).
  */
  positionSpan(): number;
  /**
  * Model-specific delimiter tokens placed before / after the span (e.g.
  * Gemma `<|audio>` / `<audio|>`); empty for models that need none.
  */
  prefixTokens(): Uint32Array;
  suffixTokens(): Uint32Array;
}

export class Image {
  /**
   * This type does not have a public constructor.
   */
  private constructor();
  /**
  * Encoded still image (PNG / JPEG / …). The host decodes it, then
  * resizes + patchifies + normalizes exactly as the bound model's image
  * processor requires (Gemma SigLIP2 vs Qwen smart-resize, etc.). Errors
  * if the model has no vision front-end or the bytes don't decode.
  */
  static fromBytes(model: Model, bytes: Uint8Array): Image;
  /**
  * Hidden-state rows / KV slots this visual span occupies.
  */
  tokenCount(): number;
  /**
  * How far the 1-D sequence cursor advances past this span. Equals
  * token-count for Gemma (1-D RoPE); equals max(t, h, w) for Qwen
  * (M-RoPE), where the next text token's three components all begin.
  */
  positionSpan(): number;
  /**
  * (t, h, w) in merged-token units.
  */
  grid(): [number, number, number];
  /**
  * Model-specific delimiter tokens the context must place immediately
  * before / after this span (e.g. Qwen `<|vision_start|>` /
  * `<|vision_end|>`); empty for models that need none. The SDK's
  * `append-image` applies these so the inferlet stays model-agnostic.
  */
  prefixTokens(): Uint32Array;
  suffixTokens(): Uint32Array;
}

export class Video {
  /**
   * This type does not have a public constructor.
   */
  private constructor();
  /**
  * Encoded animated container (e.g. GIF). The host decodes every frame,
  * uniformly samples `<= max-frames`, and preprocesses each per the bound
  * model. Errors if the model has no vision front-end or the bytes don't
  * decode as an animation.
  */
  static fromBytes(model: Model, bytes: Uint8Array, maxFrames: number): Video;
  /**
  * Number of sampled frames.
  */
  frameCount(): number;
  /**
  * The `index`-th sampled frame as an owned `image` span.
  */
  frame(index: number): Image;
  /**
  * Timestamp (seconds) of the `index`-th sampled frame.
  */
  timestamp(index: number): number;
}
