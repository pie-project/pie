/** @module Interface pie:core/audio-out **/
export type Error = import('./pie-core-types.js').Error;
export type Model = import('./pie-core-model.js').Model;

export type Voice = VoiceSpeaker | VoiceNamed;
export interface VoiceSpeaker {
  tag: 'speaker',
  val: number,
}
export interface VoiceNamed {
  tag: 'named',
  val: string,
}

export interface SpeechRequest {
  text: string,
  voice: Voice,
  maxDurationMs?: number,
}

export class Speech {
  /**
   * This type does not have a public constructor.
   */
  private constructor();
  /**
  * Synthesize speech on `model`.
  */
  static generate(model: Model, req: SpeechRequest): Speech;
  /**
  * Output sample rate in Hz.
  */
  sampleRate(): number;
  /**
  * Channel count.
  */
  channels(): number;
  /**
  * Duration in milliseconds.
  */
  durationMs(): number;
  /**
  * Decoded PCM samples in [-1, 1].
  */
  pcm(): Float32Array;
}
