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
 * to keep submitting. While its guest computes, or waits on its own
 * fires and channels, a pipeline is awaited: the frame holds for it.
 * While its guest waits on input from outside (a session receive, a
 * timer, an HTTP response), the host parks it, and its next submit
 * rejoins. `park` is how a pipeline that stops for any other reason
 * says so: it leaves the wait-set for as long as you like. A pipeline
 * whose guest computes without submitting is held for only while
 * another frame covers the device: once holding for it would leave the
 * device idle, the frame seals past it and its next submit rejoins. No
 * pipeline is terminated for being silent.
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
 * - `parents`: `[lanes]` u32, a beam search's: lane `b` continues lane
 *   `parents[b]` of the fire before. Every lane names the same working
 *   set, whose reserve past its frozen full pages is a pool each lane
 *   holds its own pages of; the engine keeps the lanes' page lists and
 *   reorders them by the parents before each fire reads or appends.
 *   Computed on the device by
 *   the epilogue before; the first fire ignores it. A parent past the
 *   lanes faults its lane. Default: none.
 *   A search owns its set's state: the rows every live beam shares
 *   (below the group's common ancestor) fold into the set as the search
 *   runs, so a search left without adopting
 *   a beam leaves the set folded past its prompt. A guest that wants the
 *   prompt state back forks the set before the search.
 */
export interface StateGeometry {
  positions?: Channel,
  slots?: Channel,
  kvLen?: Channel,
  fold?: Channel,
  parents?: Channel,
  /**
   * With `parents`: the slot past the search's last row (its folded
   * slots and its fires), which its page lists are laid for;
   * required there, absent otherwise.
   */
  parentsEnd?: number,
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
  * same tokens resumes there: one in the scope its working set shares
  * in (`working-set.share`). Nothing on a model without one. Refused
  * on a lane whose set shares nothing, which no pass could resume.
  */
  checkpoints(slots: Uint32Array): void;
  /**
  * States that this pass's sampled rows are final: once a fire of it
  * samples a row, the program never retracts, truncates, rolls back
  * or rewinds the working sets it writes below that row. Its lanes
  * then fold every row at fire and carry none into the next.
  * Unstated a pass is not final: on a set with a
  * recurrent state or windows its run-ahead fires fold only what the
  * program has accepted, so a stop can take the rest back. The
  * runtime never infers this; only the statement counts. The SDKs
  * state it on every pass the program does not declare retractable
  * (`ForwardPass::retractable`), so final is the inferlet's default.
  * `pipeline.retract` over a final fire is refused by name. Set
  * before `program`; refused on a pass binding `parents`.
  */
  finalRows(): void;
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
  "switch"(lane: number, name: string, value: string): void;
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
  * A beam search's finish: once the
  * pass's last fire has landed, its working set holds beam `beam`'s
  * hypothesis at its slots `[folded, end)` in its own order, as its
  * page list names them (its recurrent rows too, on a hybrid); the
  * other beams' pages are let go. Later passes bind the set as any
  * (no `parents`). The pass must bind `parents`.
  */
  adoptBeam(beam: number, end: number): void;
  /**
  * `[rows, n]` bools. Where lanes read kv: per row, which slots of its
  * working set's buffer and of the pass's own rows it attends. Column
  * `c` is slot `folded + c`, `folded` the set's frozen length when the
  * pass runs; `n` covers the buffer and the lane's rows. The frozen
  * part is not the mask's: every row attends it under the model's own
  * bounds. A row attends no row of its lane after it, whatever its
  * bits say. Where lanes read no kv: per row, its attention group's
  * rows in lane order. Set before `program`.
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
