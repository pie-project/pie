/** @module Interface pie:inferlet/forward-diffusion@0.3.0 **/
/**
 * Submit ONE FRAME on `on`: exactly `model.frame-size()` ordered slots.
 * `forward.submit`'s contract, unchanged; an encode pass and a denoise
 * pass may share a frame.
 */
export function submit(on: Pipeline, slots: Array<ForwardPass | undefined>): void;
/**
 * Leave the frame's wait-set on `on` until this pipeline submits again.
 * `forward.park`'s contract, unchanged.
 */
export function park(on: Pipeline): void;
export type Error = import('./pie-inferlet-types.js').Error;
export type Data = import('./pie-inferlet-types.js').Data;
export type Channel = import('./pie-inferlet-channel.js').Channel;
export type KvWorkingSet = import('./pie-inferlet-working-set.js').KvWorkingSet;
export type PageSpan = import('./pie-inferlet-working-set.js').PageSpan;
export type Pipeline = import('./pie-inferlet-pipeline.js').Pipeline;
export type MediaSpan = import('./pie-inferlet-forward.js').MediaSpan;
/**
 * Attention geometry, field for field `forward`'s. Declared separately
 * per interface ON PURPOSE (D8/D3).
 * 
 * On a `denoise` pass the geometry describes the canvas: `positions`
 * are the canvas's (the prefix length onward), `w-slot`/`w-off` land
 * its rows in the canvas pages, and `kv-len` is prefix plus canvas — the
 * same numbers on every step, which is what lets one pass be
 * resubmitted for the whole denoising loop. `mask`, when bound, narrows
 * the bidirectional reading; it never has to widen anything.
 */
export interface KvGeometry {
  readablePages: PageSpan,
  writablePages: PageSpan,
  kvLen: Channel,
  pages: Channel,
  pageIndptr: Channel,
  wSlot: Channel,
  wOff: Channel,
  positions: Channel,
  mask?: Channel,
}
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
  * State binding. REQUIRED — `forward.attention`, unchanged.
  */
  attention(kv: KvWorkingSet, geom: KvGeometry): void;
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
  * **THE SELF-CONDITIONING SIGNAL, AS THE GUEST'S TAPS.** The
  * reference denoiser feeds each step the previous step's
  * distribution, as `softmax(logits / T) · E`. Whose distribution
  * that is — which temperature, how much of the tail — is the
  * sampler's business, so the guest hands the model the
  * distribution's TAPS: per canvas row, `self-cond-taps` token ids
  * and their probabilities (`model.canvas().self-cond-taps`, row
  * major, `rows.len() == weights.len() == length * taps`). The
  * model gathers those rows of its embedding table with those
  * weights and runs its self-conditioning block over the sum. Zero
  * weights are "no signal" — the reference's first step.
  * 
  * A payload beside the ledger, like `media`: staged for the NEXT
  * submit of this pass and consumed by it, one at a time. A denoise
  * pass submitted with nothing staged runs with no signal. Staging
  * twice without a submit between is refused (the first payload
  * would be silently lost); so is staging on an `encode` pass.
  */
  selfConditioning(rows: Uint32Array, weights: Float32Array): void;
  /**
  * Bind embedding token ids and CSR row indptr. Both are channels.
  * On a `denoise` pass the tokens are the canvas — random ids on the
  * first step, whatever the guest's sampler wrote after.
  */
  embed(tokens: Channel, indptr: Channel): void;
  /**
  * Bind an optional readout-index channel separately from embedding.
  * A denoiser reads every canvas row.
  */
  readout(indices: Channel): void;
  /**
  * The media payload beside the ledger, exactly as `forward.media`
  * states it — an `encode` pass's concern, since a prompt is what
  * carries images.
  */
  media(spans: Array<MediaSpan>): void;
  /**
  * tart: run only the first `max-layers` layers and take the head
  * there. Zero rejected; unset = full model. Per pass, so an early
  * denoising step may run shallow and the last ones deep.
  */
  setMaxLayers(maxLayers: number): void;
  /**
  * The attention interface's verb, same host half: these rows are a
  * block drafter's proposal, not the sequence's own. A denoiser has no
  * use for it; it is here so the four pass interfaces stay one shape.
  */
  setDraftingBlock(on: boolean): void;
  /**
  * Attach canonical ETA bytes and channel handles in dense declaration
  * order. Validation uses the engine-owned ModelProfile and rejects
  * any stage this interface does not admit.
  */
  program(containerBytes: Data, channels: Array<Channel>): void;
}
