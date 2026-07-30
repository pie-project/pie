/** @module Interface pie:inferlet/forward-hybrid@0.3.0 **/
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
export type RsWorkingSet = import('./pie-inferlet-working-set.js').RsWorkingSet;
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
/**
 * The recurrent counterpart of `kv-geometry`: where this fire's recurrent
 * state lives, and where its folded boundary lands.
 * 
 * A linear model's context is TWO adjacent spans -- a compressed, frozen
 * folded prefix `[0, F)` and an uncompressed, mutable buffer `[F, F+B)`
 * holding each token's pre-recurrence activations. Reading the state means
 * starting from the folded prefix and scanning the buffer; `fold` just
 * moves `F` right. Folding is therefore semantically a NO-OP -- it trades
 * optionality for memory and compute, and is legal exactly when the tokens
 * it absorbs will never be modified again. Only the guest knows that,
 * which is why this is an API rather than a runtime heuristic.
 * 
 * The guest states POSITIONS; the runtime derives ADDRESSES. Where each
 * token's activations land, which pages a row occupies, how many tokens
 * are live, and how far back a replay reaches are all functions of the
 * store's own occupancy, which the runtime is authoritative for -- so
 * they are not on this record. They used to be, and the runtime computed
 * them anyway and refused any fire whose copy disagreed: six channels of
 * page arithmetic with exactly one satisfying assignment. The FOLDED
 * state slot is likewise not named here -- it comes from the working-set
 * handle.
 * 
 * This rides on the STATE BINDING, not on a fold-mode method: once the
 * buffer is understood as half the state, its addressing is an input to
 * the recurrence itself, needed by every fire rather than by two of them.
 * A plain prefill names an empty buffer and a full `fold-len`, which is
 * degenerate but meaningful -- not the dummy geometry this refactor
 * exists to delete.
 */
export interface RsGeometry {
  /**
   * How far the folded boundary advances, per request -- the twin of
   * `kv-geometry.kv-len`. Counted over the concatenation
   * `[buffer | this fire's tokens]`, so `0` buffers everything and
   * `buffer-len + fire tokens` folds everything. CLAMPED to that tail,
   * which is what lets "fold everything" be a fire-invariant constant
   * (`u32::MAX`) even though the token count varies per fire.
   * 
   * **A fire that replays buffered tokens produces NO USABLE LOGITS.**
   * The replay reloads each linear layer's cached in-projection
   * activations and stops at the recurrence -- it never runs the output
   * projection, so those layers contribute nothing to the residual
   * stream and whatever reaches `lm-head` is missing them. Read the
   * logits from the fire that BUFFERED the tokens, which ran the full
   * backbone; the fire that later folds them is a state advance, and
   * its readout is good for nothing but a completion signal.
   * 
   * This is about REPLAY, not about folding. A fire that folds its own
   * new tokens -- a prefill, or any ordinary fire that never buffers --
   * runs the full stack and its logits are perfectly good. Folding is
   * the common case on a linear model, so a rule phrased as "a fold
   * produces no logits" would condemn nearly every fire there is.
   * 
   * A CHANNEL rather than a scalar because a speculative decode
   * computes its accepted count ON DEVICE: as a channel the verify and
   * the commit fuse into one fire, instead of the guest round-tripping
   * the count through the host first. `kv-len` is already permitted to
   * be device-resident in exactly this way.
   */
  foldLen: Channel,
  /**
   * How many buffer pages this fire may occupy, per request.
   * 
   * A CAPACITY GRANT, not an address. Allocation is the one buffer
   * decision the guest still owns -- a fire that needs a page it was
   * not granted must fail rather than have the runtime quietly find
   * one -- so it stays on the API. WHERE within the grant each token
   * lands is not a decision at all: new tokens append at the buffer
   * tail, and the runtime is the only party that knows where the tail
   * is.
   * 
   * There is deliberately no `readable` / `writable` split. Reading and
   * writing the buffer really are different intents -- a write may
   * allocate, a read must not, and a read of a merely-reserved page
   * would gather uninitialized activations straight into the recurrence
   * -- but that distinction is enforced by the store's own write
   * targets, not by anything the guest can say. Two spans only let the
   * guest state it twice, inconsistently.
   */
  buffer: PageSpan,
}
/**
 * Groups the KV half so it can be made optional AS A UNIT. Without this
 * wrapper a commit-only fire -- one that folds buffered tokens and runs
 * no attention -- could not omit KV without inventing the dummy geometry
 * this refactor exists to delete.
 */
export interface KvBinding {
  workingSet: KvWorkingSet,
  geometry: KvGeometry,
}

export class ForwardPass {
  constructor()
  /**
  * State binding -- a hybrid binds KV and RS in ONE call (D7). They are
  * one set of geometry describing different layers of the same forward,
  * not two independent decisions.
  * 
  * `kv` is `none` in exactly one case: a fire that runs recurrent
  * layers only. The RS working sets stay REQUIRED even there -- the
  * driver still needs the folded slot and the buffered CSR -- which is
  * why `rs` is a plain list and only `kv` is optional.
  * 
  * This optional is also what preserves FRAME MONOMORPHISM: `submit`'s
  * `slots` list is monomorphic and WIT has no existential types, so a
  * frame mixing a recurrent-only fire with a normal fire is expressible
  * only if both are the SAME resource type.
  */
  attention(kv: KvBinding | undefined, rs: Array<RsWorkingSet>, rsGeom: RsGeometry): void;
  /**
  * There is deliberately NO `fold` / `buffer` / `fold-buffered` triple.
  * Those were three values of one scalar -- where the folded boundary
  * lands -- so they collapsed into `rs-geometry.fold-len`. An open
  * enumeration of KINDS became a closed number over POSITIONS, which is
  * also why folding it into a record is safe: a variant has cases to
  * add, a number does not.
  * 
  * `start-token` disappeared with them. New tokens always append at the
  * buffer tail, which the runtime already tracks; letting the guest
  * state a runtime-owned value is what made multi-chunk buffering
  * expressible and wrong.
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
