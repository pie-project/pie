/** @module Interface pie:inferlet/forward@0.3.0 **/
/**
 * Submit ONE FRAME on `on`: exactly `model.frame-size()` ordered slots.
 * Slot i executes in wave i of the frame; the composer never moves a
 * slot to another wave. `none` is a no-op for that wave. The slot list
 * may repeat the same handle (a plain decode frame is the same pass in
 * every slot) and may be heterogeneous (prefill chunks in the earliest
 * slots, decode in the rest). A frame is one call, not a resource.
 * 
 * Submission validation (deterministic, structural — never timing):
 * - `slots` must hold exactly `frame-size()` entries with at least one
 *   non-no-op slot.
 * - Every host-writer channel bound by the frame's non-no-op slots must
 *   be *staged* (it holds at least as many staged cells as the frame's
 *   fires will consume), *device-advanced* (the program contains an
 *   advance rule for it), or *latest-value* (the program only reads it —
 *   a control word; one committed cell suffices and host `set` may
 *   replace it at any time).
 * - Per host-reader channel, occupancy plus cells reserved by accepted
 *   unsettled frames plus the cells this frame will write must fit the
 *   capacity — overflow is prevented here, never by back-pressure.
 * 
 * Each non-no-op slot prepares and enqueues RUN-AHEAD in slot order,
 * including passes that bind recurrent state: an RS mapping publishes at
 * prepare, so slot i+1 sees slot i's decision without waiting for it to
 * execute, and the state's contents are ordered by the stream. The first
 * submit of a pass binds seeds; steady-state resubmits carry the
 * identity hash + instance only. Staged host-writer puts coalesce into
 * the slot fires in order.
 * 
 * The result covers validation/preparation/enqueue only. A fire that
 * fails AFTER submit poisons the pass's host-reader channels (surfaced
 * at `take`/`read` with the fire's error) and fails the pass — further
 * submits of it error with the root cause. End-of-stream is a PIPELINE
 * event, not a fire attribute: call `pipeline.close` after the last
 * submit.
 */
export function submit(on: Pipeline, slots: Array<ForwardPass | undefined>): void;
export type Error = import('./pie-inferlet-types.js').Error;
export type Data = import('./pie-inferlet-types.js').Data;
export type Channel = import('./pie-inferlet-channel.js').Channel;
export type KvWorkingSet = import('./pie-inferlet-working-set.js').KvWorkingSet;
export type PageSpan = import('./pie-inferlet-working-set.js').PageSpan;
export type Pipeline = import('./pie-inferlet-pipeline.js').Pipeline;
/**
 * Attention geometry. Every input is an individual channel; the record
 * exists so the whole group can be handled as a unit (and, in
 * `forward-hybrid`, made optional as a unit). `none` on `mask` omits the
 * PTIR AttnMask port; `some(mask)` binds that channel to it.
 * 
 * Declared separately per interface ON PURPOSE (D8/D3).
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

export class ForwardPass {
  constructor()
  /**
  * State binding. REQUIRED -- a pass with no attention binding is not
  * submittable.
  */
  attention(kv: KvWorkingSet, geom: KvGeometry): void;
  /**
  * Bind embedding token ids and CSR row indptr. Both are channels.
  */
  embed(tokens: Channel, indptr: Channel): void;
  /**
  * Bind an optional readout-index channel separately from embedding.
  */
  readout(indices: Channel): void;
  /**
  * Attach canonical PTIR bytes and channel handles in dense declaration
  * order. Validation uses the engine-owned ModelProfile, and now also
  * rejects any stage this interface does not admit -- see the per-file
  * note on which stages are legal.
  */
  program(containerBytes: Data, channels: Array<Channel>): void;
}
