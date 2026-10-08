/** @module Interface pie:inferlet/frames@0.3.0 **/
export type Error = import('./pie-inferlet-types.js').Error;
export type Blob = import('./pie-inferlet-types.js').Blob;
export type Channel = import('./pie-inferlet-channel.js').Channel;
/**
 * How a `frames` handle is serialized.
 * 
 * `png`, `jpeg` and `webp` are STILLS: they are valid only when
 * `count() == 1`, and a multi-frame handle refuses them by name rather
 * than silently encoding frame 0. `webp` is the lossless VP8L encoder
 * (pure Rust); it is not the animated form.
 * 
 * `raw-rgb8` is the parity format — exactly the bytes the handle holds,
 * `count * height * width * 3`, no header — so a test can compare against
 * what it put in. `y4m` is the parity format for a clip: an uncompressed
 * sequence any decoder reads, which is what makes an mp4 regression
 * provable rather than merely visible.
 * 
 * `mp4-h264` is the real video format: H.264 High profile from NVENC,
 * muxed into ISO base media (avc1 + avcC). It needs an NVIDIA encoder on
 * the machine and a runtime built with the CUDA shell; without either it
 * refuses by name, saying which half is missing.
 * # Variants
 * 
 * ## `"png"`
 * 
 * ## `"jpeg"`
 * 
 * ## `"webp"`
 * 
 * ## `"raw-rgb8"`
 * 
 * ## `"y4m"`
 * 
 * ## `"mp4-h264"`
 */
export type ImageFormat = 'png' | 'jpeg' | 'webp' | 'raw-rgb8' | 'y4m' | 'mp4-h264';
/**
 * How a `pcm` handle is serialized. `wav` is 16-bit PCM RIFF (the form
 * every player takes); `raw-f32` is the parity format — the samples as
 * little-endian f32, interleaved, no header.
 * # Variants
 * 
 * ## `"wav"`
 * 
 * ## `"raw-f32"`
 */
export type AudioFormat = 'wav' | 'raw-f32';

export class Frames {
  /**
   * This type does not have a public constructor.
   */
  private constructor();
  /**
  * Build a handle from raw interleaved RGB8. `bytes` must be exactly
  * `count * height * width * 3` long; every other length is refused by
  * name with both figures.
  * 
  * This is the constructor that lets a test, or a guest holding pixels
  * it made itself, reach the encoders before a VAE decode reading
  * exists to produce a handle the other way.
  */
  static fromRgb8(bytes: Blob, width: number, height: number, count: number, fps: number): Frames;
  /**
  * Build a handle from a channel's committed cell, WITHOUT the pixels
  * entering linear memory (design D8/D11). This is the VAE road: a
  * `vae.decode` pass's epilogue reads `intrinsics::pixels()` and puts
  * the plane on `ch`; this takes that cell host-side, the way
  * `channel.take` would, and turns it into frames.
  * 
  * The cell must be `count * height * width * 3` f32 in the model's own
  * range, [-1, 1] — what the `pixels` seam lands for an RGB decoder —
  * one row per output voxel in `(t, h, w)` order, `w` fastest, which is
  * exactly presentation order. Values are mapped to RGB8 by
  * `(x + 1) / 2` and CLAMPED, so a decoder that overshoots produces a
  * saturated pixel rather than an error. Any other length, or a channel
  * that is not f32, is refused by name with both figures.
  * 
  * The cell is CONSUMED, like `take`: the same channel feeds the next
  * fire. It blocks on an empty cell under the same discipline as
  * `channel.take-blocking` — the guest's task waits for the fire that
  * fills it, and a poisoned channel surfaces that fire's failure.
  */
  static fromChannel(ch: Channel, width: number, height: number, count: number, fps: number): Frames;
  /**
  * **THE WAY IN**: decode an encoded still (PNG / JPEG / GIF / WebP)
  * into a handle. The format is SNIFFED from the bytes — a caller
  * holding a file holds its magic too, and a stated format that
  * disagreed with the bytes would be one more thing to get wrong.
  * 
  * This is `encode`'s pair, and it is what lets a guest hand a
  * PICTURE to a `vae.encode` reading: img2img, inpainting, a
  * reference lane. `media.image` is the other door and the wrong one
  * here — it patchifies for a vision tower and answers TOKENS, and a
  * diffusion model has no vision front-end for it to ask.
  * 
  * Errors, by name and with the byte count, when the bytes decode to
  * nothing. A picture that did not decode is not a black picture.
  */
  static decode(bytes: Blob): Frames;
  /**
  * **THE VAE ROAD, RUN BACKWARDS**: put these pixels into `ch`'s
  * cell, the inverse of `from-channel` and to the same statute — the
  * bytes do not enter linear memory.
  * 
  * The cell is written as `count * height * width * 3` f32 in the
  * model's own range, [-1, 1], one row per voxel in `(t, h, w)`
  * order with `w` fastest, which is what a `vae.encode` reading's
  * pixel port reads. RGB8 maps by `2 * (x / 255) - 1`, exactly
  * undoing `from-channel`'s `(x + 1) / 2`.
  * 
  * The put is `channel.put`'s — the SEED a channel takes before its
  * first fire, not `set`'s rewrite of a cell already in the ring
  * (a channel has no ring until a fire has run). So this is a
  * channel a pass binds as an INPUT, seeded exactly once, and not
  * one an epilogue writes. A channel whose declared shape is not
  * this many f32 is refused by name with both figures.
  */
  toChannel(ch: Channel): void;
  /**
  * Frame width in pixels.
  */
  width(): number;
  /**
  * Frame height in pixels.
  */
  height(): number;
  /**
  * Number of frames. 1 for a still.
  */
  count(): number;
  /**
  * Frames per second. 0 for a still, and for a clip whose producer
  * stated no rate.
  */
  fps(): number;
  /**
  * Encode into `format` and return the bytes AS A GUEST VALUE — a copy
  * into linear memory. Prefer `session.send-frames` when the bytes are
  * headed for the client and nothing in the guest reads them.
  * 
  * Errors: a still format against a multi-frame handle, `mp4-h264`
  * without an NVIDIA encoder, or an encoder failure — each by name.
  */
  encode(format: ImageFormat): Blob;
}

export class Pcm {
  /**
   * This type does not have a public constructor.
   */
  private constructor();
  /**
  * Build a handle from interleaved f32 samples. `channels` must divide
  * the sample count.
  */
  static fromF32(samples: Float32Array, rate: number, channels: number): Pcm;
  /**
  * Sample rate in Hz.
  */
  rate(): number;
  /**
  * Channel count (interleaved).
  */
  channels(): number;
  /**
  * Encode into `format` and return the bytes as a guest value. Total,
  * because both formats are a header plus a transcode of samples the
  * handle already holds: there is nothing here that can fail.
  */
  encode(format: AudioFormat): Blob;
}
