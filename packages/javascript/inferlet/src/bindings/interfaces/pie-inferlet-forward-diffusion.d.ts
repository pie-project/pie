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
export type LaneStream = import('./pie-inferlet-model.js').LaneStream;
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
  * State binding. REQUIRED — `forward.attention`, unchanged — on
  * every reading that declares a KV space (a diffusion text's one
  * implicit reading does); refused by name on one that does not.
  */
  attention(kv: KvWorkingSet, geom: KvGeometry): void;
  /**
  * `forward.reading`, unchanged: which declared reading this pass
  * runs. Optional when the model declares at most one.
  */
  reading(name: string): void;
  /**
  * `forward.input`, unchanged: bind a channel to one of the
  * reading's float ports, read (committed cell) at every submit.
  */
  input(port: string, ch: Channel): void;
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
  * Name the attention group whose same-stream lane is this lane's
  * PEER: the lane whose velocity this one's epilogue reads with
  * `intrinsics::peer_velocity(width)`.
  * 
  * This is the classifier-free guidance verb, and it names a group
  * rather than a lane of one's own because guidance's two branches
  * must NOT attend each other — they are two independent denoisings
  * of the same canvas, one holding the prompt and one the negative
  * one, and putting them in one attention group would make each see
  * the other's rows. So they are two groups of ONE fire, and this
  * says which of them is the other.
  * 
  * The peer is the lane of that group carrying THIS pass's stream;
  * a group seating no such lane, or more than one, is refused by
  * name at submit rather than guessed at. Naming one's own group is
  * refused too: a lane is not its own peer, and `u + s(u - u)` is a
  * picture that looks fine and is not guided.
  * 
  * It costs a row offset and nothing else. The velocity plane is
  * fire-wide and written by the forward walk before any epilogue
  * block runs, so both branches' predictions are already sitting in
  * one rectangle by the time either epilogue reads. A pass that
  * reads `peer_velocity` without naming a peer is refused, so
  * guidance cannot silently degrade into a lane reading its own
  * rows twice.
  * 
  * Set before `program`, and after `group`. A model that predicts
  * no velocity refuses the whole program at bind.
  */
  peer(group: number): void;
  /**
  * A structured attention mask over this pass's rows, applied to the
  * reading's group-packed attention: `classes` names one class in
  * `0..count` per row (or -1: the row attends, and is attended by,
  * every row); `table[q * count + kv]` non-zero lets a q-class row
  * attend a kv-class row. `count` is at most 64. Every lane of one
  * attention group states the same table; rows of lanes that state
  * nothing are class -1. Regional prompting, attention coupling and
  * cross-frame restriction are all one table. Set before `program`.
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
  * The same taps, read off two of this pass's own channels at every
  * submit instead of staged from the host: `rows` a `[length, taps]`
  * u32 channel of ids, `weights` a `[length, taps]` f32 channel. A
  * persistent binding, set once before the loop; each submit reads
  * the channels' COMMITTED cells, so an epilogue that writes the
  * next step's taps keeps them loop-carried (`take` then `put`) and
  * the guest seeds them with zeros for the first step. The whole
  * denoising loop then runs without a host copy a step.
  */
  selfConditioningFrom(rows: Channel, weights: Channel): void;
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
