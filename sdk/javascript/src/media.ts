// Multimodal media wrappers over pie:core/media.

import {
  Image as _Image,
  Video as _Video,
  Audio as _Audio,
} from 'pie:core/media';

import type { Model } from './model.js';

function asBytes(bytes: Uint8Array | ArrayBuffer | Iterable<number>): Uint8Array {
  if (bytes instanceof Uint8Array) return bytes;
  if (bytes instanceof ArrayBuffer) return new Uint8Array(bytes);
  return new Uint8Array(Array.from(bytes));
}

/** Host-side preprocessed still image. */
export class Image {
  /** @internal */
  readonly _handle: _Image;

  /** @internal */
  constructor(handle: _Image) {
    this._handle = handle;
  }

  /** Decode image bytes for `model`. */
  static fromBytes(model: Model, bytes: Uint8Array | ArrayBuffer | Iterable<number>): Image {
    return new Image(_Image.fromBytes(model._handle, asBytes(bytes)));
  }

  /** Soft-token rows / KV slots occupied by this span. */
  tokenCount(): number { return this._handle.tokenCount(); }

  /** Sequence-position span consumed by this image. */
  positionSpan(): number { return this._handle.positionSpan(); }

  /** `(t, h, w)` merged-token grid. */
  grid(): [number, number, number] { return this._handle.grid(); }

  /** Model-specific delimiter tokens before the image span. */
  prefixTokens(): Uint32Array { return this._handle.prefixTokens(); }

  /** Model-specific delimiter tokens after the image span. */
  suffixTokens(): Uint32Array { return this._handle.suffixTokens(); }
}

/** Host-side decoded and sampled video clip. */
export class Video {
  /** @internal */
  readonly _handle: _Video;

  /** @internal */
  constructor(handle: _Video) {
    this._handle = handle;
  }

  /** Decode and uniformly sample up to `maxFrames` frames for `model`. */
  static fromBytes(
    model: Model,
    bytes: Uint8Array | ArrayBuffer | Iterable<number>,
    maxFrames: number,
  ): Video {
    return new Video(_Video.fromBytes(model._handle, asBytes(bytes), maxFrames));
  }

  /** Number of sampled frames. */
  frameCount(): number { return this._handle.frameCount(); }

  /** The `index`-th sampled frame as an image span. */
  frame(index: number): Image { return new Image(this._handle.frame(index)); }

  /** Timestamp in seconds for the `index`-th sampled frame. */
  timestamp(index: number): number { return this._handle.timestamp(index); }
}

/** Host-side preprocessed audio clip. */
export class Audio {
  /** @internal */
  readonly _handle: _Audio;

  /** @internal */
  constructor(handle: _Audio) {
    this._handle = handle;
  }

  /** Decode audio bytes for `model`. */
  static fromBytes(model: Model, bytes: Uint8Array | ArrayBuffer | Iterable<number>): Audio {
    return new Audio(_Audio.fromBytes(model._handle, asBytes(bytes)));
  }

  /** Soft-token rows / KV slots occupied by this clip. */
  tokenCount(): number { return this._handle.tokenCount(); }

  /** Sequence-position span consumed by this audio clip. */
  positionSpan(): number { return this._handle.positionSpan(); }

  /** Model-specific delimiter tokens before the audio span. */
  prefixTokens(): Uint32Array { return this._handle.prefixTokens(); }

  /** Model-specific delimiter tokens after the audio span. */
  suffixTokens(): Uint32Array { return this._handle.suffixTokens(); }
}
