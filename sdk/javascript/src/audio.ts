// Audio-output wrappers over pie:core/audio-out.

import {
  Speech as _Speech,
  type Voice,
} from 'pie:core/audio-out';

import type { Model } from './model.js';

export type { Voice };

/** Generated audio clip. */
export class Speech {
  /** @internal */
  readonly _handle: _Speech;

  /** @internal */
  constructor(handle: _Speech) {
    this._handle = handle;
  }

  /** Output sample rate in Hz. */
  sampleRate(): number { return this._handle.sampleRate(); }

  /** Channel count. */
  channels(): number { return this._handle.channels(); }

  /** Duration in milliseconds. */
  durationMs(): number { return this._handle.durationMs(); }

  /** Decoded PCM samples in [-1, 1]. */
  pcm(): Float32Array { return this._handle.pcm(); }

  /** Encode mono f32 PCM as canonical 16-bit PCM WAV. */
  toWav(): Uint8Array {
    return writeWav(this.pcm(), this.sampleRate());
  }
}

/** Builder for model-agnostic speech synthesis. */
export class SpeechBuilder {
  readonly #model: Model;
  readonly #text: string;
  #voice: Voice = { tag: 'speaker', val: 0 };
  #maxDurationMs: number | undefined;

  /** @internal */
  constructor(model: Model, text: string) {
    this.#model = model;
    this.#text = text;
  }

  /** Set the voice directly. Defaults to speaker 0. */
  voice(voice: Voice): this {
    this.#voice = voice;
    return this;
  }

  /** Convenience for `voice({ tag: 'speaker', val: id })`. */
  speaker(id: number): this {
    this.#voice = { tag: 'speaker', val: id };
    return this;
  }

  /** Convenience for named voices on models that support them. */
  named(name: string): this {
    this.#voice = { tag: 'named', val: name };
    return this;
  }

  /** Cap generated audio length in milliseconds. */
  maxDurationMs(ms: number): this {
    this.#maxDurationMs = Math.max(0, Math.floor(ms));
    return this;
  }

  /** Cap generated audio length in seconds. */
  maxDurationSeconds(seconds: number): this {
    return this.maxDurationMs(seconds * 1000);
  }

  /** Synthesize speech on the bound model. */
  async generate(): Promise<Speech> {
    const speech = _Speech.generate(this.#model._handle, {
      text: this.#text,
      voice: this.#voice,
      maxDurationMs: this.#maxDurationMs,
    });
    return new Speech(speech);
  }
}

/** Write mono f32 PCM (`[-1, 1]`) as canonical 16-bit PCM WAV. */
export function writeWav(pcm: Float32Array | Iterable<number>, sampleRate: number): Uint8Array {
  const samples = pcm instanceof Float32Array ? pcm : new Float32Array(Array.from(pcm));
  const dataBytes = samples.length * 2;
  const out = new Uint8Array(44 + dataBytes);
  const view = new DataView(out.buffer);

  writeAscii(out, 0, 'RIFF');
  view.setUint32(4, 36 + dataBytes, true);
  writeAscii(out, 8, 'WAVE');
  writeAscii(out, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(out, 36, 'data');
  view.setUint32(40, dataBytes, true);

  for (let i = 0; i < samples.length; i++) {
    const clamped = Math.max(-1, Math.min(1, samples[i]!));
    view.setInt16(44 + i * 2, Math.round(clamped * 32767), true);
  }
  return out;
}

function writeAscii(out: Uint8Array, offset: number, s: string): void {
  for (let i = 0; i < s.length; i++) out[offset + i] = s.charCodeAt(i);
}
