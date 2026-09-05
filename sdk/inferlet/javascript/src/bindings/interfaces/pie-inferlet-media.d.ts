/** @module Interface pie:inferlet/media@0.3.0 **/
export type Error = import('./pie-inferlet-types.js').Error;
/**
 * A visual span's extent in merged-token units.
 */
export interface MergedGrid {
  t: number,
  h: number,
  w: number,
}

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
  static fromBytes(bytes: Uint8Array): Audio;
  /**
  * THE SPAN, AS THE SEQUENCE CARRIES IT — `image.tokens()`'s contract,
  * with this model's audio delimiters and pad. One ledger for both
  * modalities is why `forward.media` takes a variant rather than two
  * verbs: audio joined the door without changing it.
  */
  tokens(): Uint32Array;
  /**
  * A stable content hash of the PREPROCESSED clip — log-mel features,
  * not source bytes. The cache statute in this interface's header binds
  * audio runs exactly as it binds visual ones. 32 bytes.
  */
  digest(): Uint8Array;
  /**
  * Hidden-state rows / KV slots this clip occupies == audio soft tokens.
  * 
  * Introspection: `tokens()` already places exactly this many.
  */
  tokenCount(): number;
  /**
  * How far the 1-D sequence cursor advances past this clip. Equals
  * token-count for Gemma (1-D RoPE).
  */
  positionSpan(): number;
  /**
  * The delimiter tokens `tokens()` already placed around the run (e.g.
  * Gemma `<|audio>` / `<audio|>`); empty for models that need none.
  * Introspection, like `image`'s — see the note there.
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
  static fromBytes(bytes: Uint8Array): Image;
  /**
  * THE SPAN, AS THE SEQUENCE CARRIES IT: prefix + placeholder run
  * (exactly `token-count` long) + suffix, in the bound model's own ids.
  * 
  * This is the one ledger (media-door.md §0). A span enters the context
  * as `toks.extend(img.tokens())` and as nothing else — no anchor list,
  * no guest-side splicing of `prefix-tokens` around a run the guest
  * built itself, no second bookkeeping structure beside the tokens. The
  * handle then crosses again beside them, through
  * `forward.forward-pass.media`, carrying only the payload.
  * 
  * The ids are the host's answer, never the guest's spelling: the guest
  * hardcodes nothing and stays model-agnostic. See the cache statute in
  * this interface's header before keying anything on the result.
  */
  tokens(): Uint32Array;
  /**
  * A stable content hash of the PREPROCESSED span — patches, not source
  * bytes, so two encodings of one image collide correctly and two
  * images do not. 32 bytes.
  * 
  * Exists for the cache statute above: `tokens()` cannot tell two
  * images apart and this can, cheaply, so a guest that caches across a
  * media run has no excuse for keying on the run alone.
  */
  digest(): Uint8Array;
  /**
  * Hidden-state rows / KV slots this visual span occupies.
  * 
  * Introspection: `tokens()` already places exactly this many
  * placeholders, so nothing on the happy path needs to ask.
  */
  tokenCount(): number;
  /**
  * How far the 1-D sequence cursor advances past this span. Equals
  * token-count for Gemma (1-D RoPE); equals max(t, h, w) for Qwen
  * (M-RoPE), where the next text token's three components all begin.
  */
  positionSpan(): number;
  /**
  * Extent in merged-token units.
  */
  grid(): MergedGrid;
  /**
  * The delimiter tokens `tokens()` already placed around the run (e.g.
  * Qwen `<|vision_start|>` / `<|vision_end|>`); empty for models that
  * need none.
  * 
  * INTROSPECTION, NOT A STEP (media-door.md §1). These were the guest's
  * splicing instructions back when the guest assembled the run; they
  * stay because a caller inspecting what a span will cost is a real
  * question, and because a WIT verb is cheap to keep and expensive to
  * re-add. Nothing on the happy path calls them.
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
  static fromBytes(bytes: Uint8Array, maxFrames: number): Video;
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
