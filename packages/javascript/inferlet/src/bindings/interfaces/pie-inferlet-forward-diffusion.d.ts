/** @module Interface pie:inferlet/forward-diffusion@0.3.0 **/
/**
 * Submit ONE FRAME on `on`: exactly `model.frame-size()` ordered slots.
 * `forward.submit`'s contract, unchanged; an encode pass and a denoise
 * pass may share a frame.
 */
export function submit(on: Pipeline, slots: Array<ForwardPass | undefined>): void;
export type Error = import('./pie-inferlet-types.js').Error;
export type Data = import('./pie-inferlet-types.js').Data;
export type Channel = import('./pie-inferlet-channel.js').Channel;
export type WorkingSet = import('./pie-inferlet-working-set.js').WorkingSet;
export type Pipeline = import('./pie-inferlet-pipeline.js').Pipeline;
export type MediaSpan = import('./pie-inferlet-forward.js').MediaSpan;
export type StateGeometry = import('./pie-inferlet-forward.js').StateGeometry;
export type LaneStream = import('./pie-inferlet-model.js').LaneStream;
/**
 * Which of the model's two readings a pass runs. See the file note.
 * # Variants
 * 
 * ## `"encode"`
 * 
 * ## `"denoise"`
 */
export type Mode = 'encode' | 'denoise';

export class ForwardPass {
  constructor()
  /**
  * `forward.state`. On a `denoise` pass the geometry describes the
  * canvas, the same on every step, so one pass serves the whole
  * denoising loop.
  */
  state(lanes: Array<WorkingSet>, geom: StateGeometry): void;
  /**
  * `forward.reading`, unchanged: which declared reading this pass
  * runs. Optional when the model declares at most one.
  */
  reading(name: string): void;
  /**
  * `forward.input`, unchanged: bind a channel to the model's input
  * `name`, read (committed cell) at every submit. A denoiser's
  * self-conditioning taps are two such inputs.
  */
  input(name: string, ch: Channel): void;
  /**
  * `forward.stream`, unchanged: which lane stream this pass is.
  */
  stream(s: LaneStream): void;
  /**
  * `forward.group`, unchanged: the attention group this pass's
  * lanes join within a frame.
  */
  group(id: number): void;
  /**
  * `forward.peer`, unchanged.
  */
  peer(group: number): void;
  /**
  * `forward.attention-classes`, unchanged.
  */
  attentionClasses(classes: Int32Array, table: Uint8Array, count: number): void;
  /**
  * The reading. REQUIRED: a diffusion pass with no mode is not
  * submittable, because neither reading is a default the host may
  * pick for the guest. Set before `program`; a pass keeps one mode
  * for its life, so a loop holds one encode pass and one denoise
  * pass rather than flipping either.
  * 
  * A `denoise` pass on an engine whose attention cannot lift its
  * causal bound is refused by name at submit, never read causally.
  */
  canvas(mode: Mode): void;
  /**
  * Bind embedding token ids and CSR row indptr. Both are channels.
  * On a `denoise` pass the tokens are the canvas — random ids on the
  * first step, whatever the guest's sampler wrote after. Required
  * on a reading that embeds tokens, refused on one that does not
  * (`forward.embed`'s rule).
  */
  embed(tokens: Channel, indptr: Channel): void;
  /**
  * Bind an optional readout-index channel separately from embedding.
  * A denoiser reads every canvas row.
  */
  readout(indices: Channel): void;
  /**
  * `forward.media`, unchanged: an `encode` pass's concern, since a
  * prompt is what carries images.
  */
  media(spans: Array<MediaSpan>): void;
  /**
  * Attach canonical ETA bytes and channel handles in dense declaration
  * order. Validation uses the engine-owned ModelProfile and rejects
  * any stage this interface does not admit.
  */
  program(containerBytes: Data, channels: Array<Channel>): void;
}
