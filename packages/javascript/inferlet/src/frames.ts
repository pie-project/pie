// Pictures and sound the host holds — `pie:inferlet/frames`.
//
// They stay out of linear memory: a guest makes them from a channel or bytes,
// and hands them to the client, or takes them as a client moved them in. A
// still is frames of one.

import * as _frames from 'pie:inferlet/frames@0.3.0';
import * as _session from 'pie:inferlet/session@0.3.0';

export class FramesError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'FramesError';
  }
}

function wit<T>(what: string, f: () => T): T {
  try {
    return f();
  } catch (e: unknown) {
    const payload = (e as { payload?: unknown })?.payload;
    throw new FramesError(`${what}: ${typeof payload === 'string' ? payload : String(e)}`);
  }
}

/** `count` frames of `width` x `height` RGB8, held by the host. */
export class Frames {
  /** @internal */
  constructor(readonly handle: _frames.Frames) {}

  static fromRgb8(data: Uint8Array, width: number, height: number, count = 1, fps = 0): Frames {
    return new Frames(wit('frames', () => _frames.Frames.fromRgb8(data, width, height, count, fps)));
  }

  /** The handle the launch moved in under `name`, once. */
  static take(name: string): Frames {
    const inner = _session.takeFrames(name);
    if (inner === undefined) throw new FramesError(`the launch moved in no frames \`${name}\``);
    return new Frames(inner);
  }

  /** Hands these pictures to the client under the launch's `name`. Unusable after. */
  give(name: string): void {
    wit('give', () => _session.giveFrames(name, this.handle));
  }

  get width(): number {
    return this.handle.width();
  }
  get height(): number {
    return this.handle.height();
  }
  get count(): number {
    return this.handle.count();
  }
  get fps(): number {
    return this.handle.fps();
  }
}

/** Sound held by the host: interleaved samples at `rate`. */
export class Pcm {
  /** @internal */
  constructor(readonly handle: _frames.Pcm) {}

  static fromF32(samples: Float32Array, rate: number, channels = 1): Pcm {
    return new Pcm(wit('pcm', () => _frames.Pcm.fromF32(samples, rate, channels)));
  }

  /** The handle the launch moved in under `name`, once. */
  static take(name: string): Pcm {
    const inner = _session.takePcm(name);
    if (inner === undefined) throw new FramesError(`the launch moved in no pcm \`${name}\``);
    return new Pcm(inner);
  }

  /** Hands this sound to the client under the launch's `name`. Unusable after. */
  give(name: string): void {
    wit('give', () => _session.givePcm(name, this.handle));
  }

  get rate(): number {
    return this.handle.rate();
  }
  get channels(): number {
    return this.handle.channels();
  }
}
