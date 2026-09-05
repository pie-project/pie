// Multimodal input — `pie:inferlet/media`.
//
// The inferlet hands the host raw encoded bytes (`Image.fromBytes` for
// PNG/JPEG, `Video.fromBytes` for animated GIF, `Audio.fromBytes` for WAV)
// and the host decodes + preprocesses per the bound model. A span enters the
// sequence as the token run its handle answers, and the handle crosses again
// beside the tokens (`fwd.media([{ tag: 'image', val: img.handle }])`).

import * as _media from 'pie:inferlet/media@0.3.0';

export class MediaError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'MediaError';
  }
}

function wit<T>(what: string, f: () => T): T {
  try {
    return f();
  } catch (e: unknown) {
    const payload = (e as { payload?: unknown })?.payload;
    throw new MediaError(`${what}: ${typeof payload === 'string' ? payload : String(e)}`);
  }
}

export interface MergedGrid {
  t: number;
  h: number;
  w: number;
}

export class Image {
  /** @internal */
  constructor(readonly handle: _media.Image) {}

  static fromBytes(data: Uint8Array): Image {
    return new Image(wit('image', () => _media.Image.fromBytes(data)));
  }

  /** The placeholder run this span enters the sequence as. */
  tokens(): Uint32Array {
    return this.handle.tokens();
  }
  digest(): Uint8Array {
    return this.handle.digest();
  }
  tokenCount(): number {
    return this.handle.tokenCount();
  }
  positionSpan(): number {
    return this.handle.positionSpan();
  }
  grid(): MergedGrid {
    return this.handle.grid();
  }
  prefixTokens(): Uint32Array {
    return this.handle.prefixTokens();
  }
  suffixTokens(): Uint32Array {
    return this.handle.suffixTokens();
  }
}

export class Audio {
  /** @internal */
  constructor(readonly handle: _media.Audio) {}

  static fromBytes(data: Uint8Array): Audio {
    return new Audio(wit('audio', () => _media.Audio.fromBytes(data)));
  }

  tokens(): Uint32Array {
    return this.handle.tokens();
  }
  digest(): Uint8Array {
    return this.handle.digest();
  }
  tokenCount(): number {
    return this.handle.tokenCount();
  }
  positionSpan(): number {
    return this.handle.positionSpan();
  }
  prefixTokens(): Uint32Array {
    return this.handle.prefixTokens();
  }
  suffixTokens(): Uint32Array {
    return this.handle.suffixTokens();
  }
}

export class Video {
  /** @internal */
  constructor(readonly handle: _media.Video) {}

  static fromBytes(data: Uint8Array, maxFrames: number): Video {
    return new Video(wit('video', () => _media.Video.fromBytes(data, maxFrames)));
  }

  frameCount(): number {
    return this.handle.frameCount();
  }
  frame(index: number): Image {
    return new Image(wit(`video frame ${index}`, () => this.handle.frame(index)));
  }
  timestamp(index: number): number {
    return this.handle.timestamp(index);
  }
}
