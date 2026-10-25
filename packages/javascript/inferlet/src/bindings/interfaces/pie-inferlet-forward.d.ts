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
 * without waiting for the slot before it to execute, including passes
 * that bind recurrent state, whose contents the stream orders. The
 * first submit of a pass binds seeds; steady-state resubmits carry the
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
export type WorkingSet = import('./pie-inferlet-working-set.js').WorkingSet;
export type Pipeline = import('./pie-inferlet-pipeline.js').Pipeline;
export type Image = import('./pie-inferlet-media.js').Image;
export type Audio = import('./pie-inferlet-media.js').Audio;
/**
 * One attached media span, borrowed: the pass reads the payload out and
 * the guest keeps its handle, so one decoded image may feed two passes.
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
 * Where a pass's rows go and what they read. Every field is optional;
 * what is left out takes the default, so a plain prefill or decode binds
 * nothing at all. Each `some` binds that channel to the program's port.
 * 
 * - `positions`: per row, the RoPE position. Default: the row's slot.
 * - `slots`: per row, the slot it writes, at or above `folded`; a lane's
 *   rows write consecutive slots. Default: after the last slot written.
 *   Writing a slot again replaces it.
 * - `kv-len`: per lane, the lane reads slots `[0, kv-len)`. Default:
 *   causal.
 * - `fold`: per lane, how many more slots freeze after this fire;
 *   `u32::MAX` folds everything. Default: none. May be computed on the
 *   device; lanes sharing a working set fold by the first one's count.
 */
export interface StateGeometry {
  positions?: Channel,
  slots?: Channel,
  kvLen?: Channel,
  fold?: Channel,
}

export class ForwardPass {
  constructor()
  /**
  * One working set per lane, in lane order; lanes sharing one state
  * their slots. Required on a lane `model.lane` says binds `state`,
  * refused on any other.
  */
  state(lanes: Array<WorkingSet>, geom: StateGeometry): void;
  /**
  * Slots at which a recurrent state is kept, so a later pass over the
  * same tokens resumes there. Nothing on a model without one.
  */
  checkpoints(slots: Uint32Array): void;
  /**
  * Bind embedding token ids and CSR row indptr. Both are channels.
  * Required on a lane that embeds tokens (`model.lane`'s `embeds`),
  * refused on one that does not.
  */
  embed(tokens: Channel, indptr: Channel): void;
  /**
  * Set lane `lane`'s model switch `name` (`model.switches()`) to
  * `value`; an unset switch takes its first declared value. Set
  * before `program`.
  */
  'switch'(lane: number, name: string, value: string): void;
  /**
  * Bind lane `lane`'s model input `name` (`model.lane`) to `ch`, whose
  * committed cell is read at every submit: `[width]`, `[rows, width]`
  * or one `[t?, h, w, width]` clip. A lane binds exactly what it reads.
  */
  input(lane: number, name: string, ch: Channel): void;
  /**
  * Put lane `lane` in this pass's attention group `id`: lanes naming
  * one group attend each other's rows; a lane naming none attends its
  * own. Set before `program`.
  */
  group(lane: number, id: number): void;
  /**
  * `[rows, n]` bools: per row, the slots `[0, kv-len)` it attends when
  * lanes read kv, else its attention group's rows in lane order. Set
  * before `program`.
  */
  mask(rows: Channel): void;
  /**
  * Bind an optional readout-index channel separately from embedding.
  */
  readout(indices: Channel): void;
  /**
  * The payloads of the media spans lane `lane`'s tokens carry, in
  * placeholder order; a count or run length that disagrees is refused
  * at `submit`.
  */
  media(lane: number, spans: Array<MediaSpan>): void;
  /**
  * Attach canonical ETA bytes and channel handles in dense declaration
  * order. Validation checks the program against what each lane reads
  * out and rejects any stage this interface does not admit.
  */
  program(containerBytes: Data, channels: Array<Channel>): void;
}
