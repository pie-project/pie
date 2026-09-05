/** @module Interface pie:inferlet/forward-recurrent@0.3.0 **/
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
export type RsWorkingSet = import('./pie-inferlet-working-set.js').RsWorkingSet;
export type PageSpan = import('./pie-inferlet-working-set.js').PageSpan;
export type Pipeline = import('./pie-inferlet-pipeline.js').Pipeline;
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

export class ForwardPass {
  constructor()
  /**
  * State binding. REQUIRED. The mechanism is a recurrence, but the
  * SLOT NAME is `attention` in all three interfaces (D6) -- the role
  * inside a pass is the same regardless of mechanism; only the
  * signature varies.
  * 
  * `rs` is one working set per request, in resolved request order --
  * the state's IDENTITY. `geom` is where that state lives for this
  * fire and where its boundary lands. The same division as
  * `attention(kv, kv-geometry)` in `forward`.
  */
  attention(rs: Array<RsWorkingSet>, geom: RsGeometry): void;
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
  * tart (0.3 re-port): run only the first `max-layers` transformer
  * layers and take the head there. Zero rejected; unset = full
  * model. On recurrent/hybrid passes the truncation covers the
  * whole backbone prefix [0, max-layers).
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
  * Attach canonical ETA bytes and channel handles in dense declaration
  * order. Validation uses the engine-owned ModelProfile, and now also
  * rejects any stage this interface does not admit -- see the per-file
  * note on which stages are legal.
  */
  program(containerBytes: Data, channels: Array<Channel>): void;
}
