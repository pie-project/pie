/** @module Interface pie:core/audio-out **/
export type Error = import('./pie-core-types.js').Error;
/**
 * Which voice to speak in. A neutral selector; the host maps it onto the
 * bound model's own conditioning (CSM bakes an integer speaker id into its
 * "[id]text" frame). `named` is for models that pick voices by name — CSM
 * has none and returns a clear error for it.
 */
export type Voice = VoiceSpeaker | VoiceNamed;
export interface VoiceSpeaker {
  tag: 'speaker',
  val: number,
}
export interface VoiceNamed {
  tag: 'named',
  val: string,
}
/**
 * What to synthesize. Plain intent only: the host owns the model-specific
 * prompt framing, so the inferlet never writes a "[speaker]text" string or a
 * BOS/EOS token id of its own.
 */
export interface SpeechRequest {
  /**
   * The text to speak.
   */
  text: string,
  /**
   * The voice to speak it in.
   */
  voice: Voice,
  /**
   * Stop after roughly this much audio. A neutral unit (milliseconds); the
   * host converts it to the model's internal frame count (CSM: 12.5 Hz Mimi
   * frames, 80 ms each). `none` => the model's default cap. Generation also
   * stops early on the model's end-of-audio signal regardless.
   */
  maxDurationMs?: number,
}

export class Speech {
  /**
   * This type does not have a public constructor.
   */
  private constructor();
  /**
  * Synthesize `req` on the bound model. Errors if the model has no
  * audio-output front-end (i.e. is not a CSM checkpoint).
  */
  static generate(req: SpeechRequest): Speech;
  /**
  * Output sample rate in Hz (CSM: 24000).
  */
  sampleRate(): number;
  /**
  * Channel count (CSM: 1, mono).
  */
  channels(): number;
  /**
  * Duration of the generated clip in milliseconds.
  */
  durationMs(): number;
  /**
  * Decoded PCM samples in [-1, 1] (mono => one sample per frame).
  */
  pcm(): Float32Array;
}
