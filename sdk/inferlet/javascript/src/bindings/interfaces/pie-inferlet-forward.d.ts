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
/**
 * Leave the frame's wait-set on `on` until this pipeline submits again.
 * 
 * A frame batches one slot per participating pipeline and does not
 * dispatch until every member has submitted, so membership is a promise
 * to keep submitting. A pipeline that stops — it is blocked on a user
 * turn, waiting on a peer, or simply idle — would otherwise hold the
 * frame. `model.submit-deadline-us` bounds that promise: hold the
 * wait-set that long with nothing owed to you and the engine stops
 * waiting for you — the slot is dropped from the frame, work you already
 * submitted still runs, and your next submit rejoins. That costs you a
 * boundary, not your instance. Staying silent much longer than that
 * WITHOUT parking is a different matter: it reads as an abandoned
 * pipeline and the instance is terminated. `park` is how a pipeline
 * stops without breaking the promise — it states the intent the engine
 * cannot infer, leaves the wait-set, and stops that clock entirely, for
 * as long as you like.
 * 
 * Ordered against this pipeline's own submits, not against the call: it
 * takes effect once every frame submitted before it has sealed, so it is
 * legal — and expected — to park with fires still outstanding. Their
 * results are delivered as usual; the exit simply follows them. There is
 * no rejoin call, because a member that had joined but not yet submitted
 * would reopen exactly the gap this closes: the next `submit` rejoins
 * the wait-set atomically with the slot it contributes.
 * 
 * Parking twice with no submit in between is a no-op, as is parking a
 * pipeline that never fired. Outside frame mode there is no wait-set and
 * this does nothing.
 */
export function park(on: Pipeline): void;
export type Error = import('./pie-inferlet-types.js').Error;
export type Data = import('./pie-inferlet-types.js').Data;
export type Channel = import('./pie-inferlet-channel.js').Channel;
export type KvWorkingSet = import('./pie-inferlet-working-set.js').KvWorkingSet;
export type PageSpan = import('./pie-inferlet-working-set.js').PageSpan;
export type Pipeline = import('./pie-inferlet-pipeline.js').Pipeline;
export type Image = import('./pie-inferlet-media.js').Image;
export type Audio = import('./pie-inferlet-media.js').Audio;
/**
 * One attached media span, by the resource the guest is holding.
 * 
 * **A VARIANT FROM DAY ONE** (media-door.md §2). Audio input's tower is
 * its own wave and its front-end is not written yet, but the shape it
 * will arrive in is decided here: a second modality joins by taking a
 * case, not by growing a second verb beside `media`. The host reads both
 * cases into the same span record — the two differ in how they were
 * computed and in nothing the sequence can see.
 * 
 * Borrowed, not owned: the pass reads the payload out and the guest keeps
 * its handle, which is what lets one decoded image be submitted to two
 * passes without decoding twice.
 */
export type MediaSpan = MediaSpanImage | MediaSpanAudio;
export interface MediaSpanImage {
  tag: 'image',
  val: Image,
}
export interface MediaSpanAudio {
  tag: 'audio',
  val: Audio,
}
/**
 * Attention geometry. Every input is an individual channel; the record
 * exists so the whole group can be handled as a unit (and, in
 * `forward-hybrid`, made optional as a unit). `none` on `mask` omits the
 * ETA AttnMask port; `some(mask)` binds that channel to it.
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
  * tart STRUCTURAL v0 (re-ported to 0.3): run only the first
  * `max-layers` transformer layers for this pass's fires and take
  * the head there (the layerskip-draft / logit-lens class).
  * Values at or above the model's depth are the identity; zero is
  * rejected. Unset = the full model. Mixed-depth co-fires run as
  * banded depth when the deployment supports it.
  */
  setMaxLayers(maxLayers: number): void;
  /**
  * **THESE ROWS ARE A BLOCK DRAFTER'S PROPOSAL, NOT THE SEQUENCE'S.**
  * A block drafter proposes many tokens in one pass over a block
  * whose first row is the correction the target just made and whose
  * rest is the model's mask token; the trunk must not run over them,
  * and a plan that carries such a drafter guards itself on this.
  * 
  * It cannot be inferred from what the pass reads, the way drafting
  * is: what makes a fire a draft is the anchor chosen from the
  * accepted prefix, which only the guest knows. Unset = an ordinary
  * pass.
  */
  setDraftingBlock(on: boolean): void;
  /**
  * **THE PAYLOAD BESIDE THE LEDGER** (media-door.md §0/§3): the spans
  * this pass's tokens carry, in the order their placeholder runs
  * appear.
  * 
  * **THE TOKENS ARE THE LEDGER AND THIS IS NOT A SECOND ONE.** A span
  * entered the sequence through `embed`, as the run `image.tokens()`
  * answered and as nothing else; there is no anchor here, no offset,
  * no length. This call carries only what a token id cannot: the
  * preprocessed patches the tower consumes. The correspondence between
  * the two is not asserted by the guest — it is SCANNED by the host,
  * which finds the model's reserved placeholder runs in the submitted
  * tokens (a tokenizer never emits that id from text) and matches them
  * to these spans in order.
  * 
  * **EVERY DISAGREEMENT IS REFUSED BY NAME, BEFORE ANYTHING LAUNCHES.**
  * More runs than spans or more spans than runs; a run whose length is
  * not that span's `token-count` (which is what catches a guest that
  * sliced a run in half, or spliced two spans' runs together); media
  * attached to a model with no tower. None of these can reach the
  * device, and none of them are diagnosed by a wrong answer.
  * 
  * Ordered with `embed`, not against it: call either first. The scan
  * runs at `submit`, when both the tokens and the spans are final.
  */
  media(spans: Array<MediaSpan>): void;
  /**
  * Attach canonical ETA bytes and channel handles in dense declaration
  * order. Validation uses the engine-owned ModelProfile, and now also
  * rejects any stage this interface does not admit -- see the per-file
  * note on which stages are legal.
  */
  program(containerBytes: Data, channels: Array<Channel>): void;
}
