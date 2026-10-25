/** @module Interface pie:inferlet/model@0.3.0 **/
/**
 * The engine serves exactly one model; these are global functions over
 * that single bound model (no `model` resource handle). Identity + memory-
 * shaping capabilities only — the tokenizer surface (encode/decode/vocabs/
 * special-tokens/split-regex) lives in the sibling `tokenizer` interface.
 * Returns the name of the bound model
 */
export function name(): string;
/**
 * Returns the model architecture identifier (e.g. "gemma4", "qwen3_6")
 */
export function architecture(): string;
/**
 * Whether the bound model enables system speculation by default
 */
export function defaultSystemSpeculation(): boolean;
/**
 * The draft head's chain depth: how many tokens past each readout row
 * the `mtp_drafts` intrinsic carries (`[n_out × depth]`, row-major,
 * each row's chain conditioned on the trunk's argmax at that row). Zero
 * for a model with no draft head, in which case `mtp_drafts` and
 * `mtp_logits` are unavailable. Static, like `frame-size`.
 */
export function mtpDepth(): number;
export function draftBlock(): BlockDrafter | undefined;
export function passKind(): ForwardKind;
export function canvas(): CanvasShape | undefined;
/**
 * Every reading the bound model declares, in index order. Empty for a
 * model with one implicit reading (every text row), on which
 * `forward-pass.reading` is refused. Static.
 */
export function readings(): Array<ReadingFact>;
/**
 * The lane a pass of these switches is. Switches are `(name, value)`
 * pairs: `("reading", name)`, `("stream", name)` and, on a diffusion
 * model, `("canvas", "encode" | "denoise")`; one left unset takes its
 * default (the model's one reading, the `text` stream, `encode`).
 */
export function lane(switches: Array<[string, string]>): LaneDecl;
export function latent(): LatentSpace | undefined;
export function schedule(): ScheduleFact | undefined;
/**
 * The most rows one lane may carry: the guest sizes its row inputs
 * (`[rows, width]`) under this, the rows the engine's arena holds a
 * fire. Static.
 */
export function maxLatentRows(): number;
/**
 * Logits/output dimension (= hf_config.vocab_size). May EXCEED the
 * tokenizer's vocabs() token count due to padding. Use THIS for
 * sampler-program lowering and any logits-shaped op; use vocabs() only for
 * token-space work. Sourced from the model (not hardcoded) so the
 * inferlet's lowering vocab == the engine's logits / recognizer-table vocab.
 */
export function outputVocabSize(): number;
/**
 * Tokens per KV page for the bound model/engine.
 */
export function kvPageSize(): number;
/**
 * Waves per frame (k) for this deployment — a static constant fixed at
 * engine start, exactly like kv-page-size. `forward.submit` takes exactly
 * this many ordered slots; slot i executes in wave i. Guests must be
 * output-correct for any k: it decides submission granularity and
 * resource sizing, never token-level behavior. Never renegotiated per
 * frame and never adapted from runtime timing.
 */
export function frameSize(): number;
/**
 * How long (microseconds) a pipeline may hold a frame's wait-set without
 * submitting before the engine stops waiting for it.
 * 
 * A frame does not dispatch until every participating pipeline has
 * submitted its slot, so membership is a promise to keep submitting, and
 * this is that promise's bound. The clock is per pipeline, measures
 * CONSECUTIVE silence while actually blocking a seal, and is suspended
 * whenever the engine is what the pipeline is waiting on — so it is not a
 * latency budget and a pipeline that keeps submitting can never trip it
 * however long it runs.
 * 
 * A pipeline that intends to stop — blocked on a user turn, waiting on a
 * peer, or simply done for now — calls `forward.park` instead, which
 * leaves the wait-set and stops the clock. What this bound catches is
 * therefore only the case the engine cannot otherwise distinguish: a
 * member that will never submit again.
 * 
 * Small (50ms by default) because it measures a much narrower interval
 * than its size suggests: run-ahead, an unsettled dispatch, a bind in
 * flight and `forward.park` all stop the clock, so what is left is a
 * pipeline hard-blocking the seal with nothing owed to it. It is not
 * smaller because the one-off cost of BUILDING the first pass of a loop
 * — tracing and compiling it after the prefill result lands — is on the
 * clock and was measured between 5ms and 20ms; a steady decode turnaround
 * is far below that.
 * 
 * The corollary is the one thing a pipeline has to be deliberate about:
 * work done between receiving a result and submitting the next fire is
 * ON the clock, because during it the fleet is stalled on this pipeline
 * and nothing is owed back. Anything that can outlast this bound —
 * tokenizing a large document, compiling a grammar, an RPC, a decision
 * that waits on another pipeline — should be preceded by `forward.park`
 * and is rejoined by simply submitting again. Parking is cheap and
 * unparking is implicit, so when in doubt, park.
 * 
 * Overrunning it is not fatal. The engine simply stops waiting: the slot
 * leaves the frame, already-submitted work still runs, and the next
 * submit rejoins. What it costs is a boundary — so treat this as the
 * budget that keeps you IN each frame, not as a deadline you die to.
 * (Termination is reserved for a pipeline that goes silent for orders of
 * magnitude longer without ever parking, i.e. an abandoned one.)
 * 
 * Static, like frame-size and unlike channel-capacity: guests read it
 * once and size their work against it, so it must not move.
 */
export function submitDeadlineUs(): bigint;
/**
 * Host-reader channel capacity, in cells, that lets one lane sustain the
 * engine's run-ahead without the ring becoming the bottleneck. Size every
 * host-reader channel to at least this; the staging margin is already
 * included, so guests need no arithmetic.
 * 
 * Under-sizing does not fail loudly. At k = 1 there is no frame to
 * validate, so an undersized ring simply serialises the lane with no
 * diagnostic — and because the scheduler waits for ALL lanes, a serialised
 * lane holds up every co-resident tenant, not just itself.
 * 
 * UNLIKE frame-size this is not promised to be static: it derives from the
 * host resubmit turnaround, which the runtime may later adapt. Read it,
 * don't cache it across a run.
 */
export function channelCapacity(): number;
/**
 * Fires one lane may have submitted and not yet taken -- the run-ahead
 * window, in fires. A host-reader ring of `channel-capacity` cells holds
 * exactly this many plus the visibility margin, so a guest that keeps
 * this many fires in flight never serialises and never overruns. The
 * runtime derives both numbers from one source (`engine::runahead`);
 * guests read this rather than recovering it from `channel-capacity`.
 * Not static, for the same reason `channel-capacity` is not.
 */
export function runAheadWindow(): number;
/**
 * The most rows one pass may read out through `kind`: what a pass returns,
 * not how long it is. A pass whose rows cannot run in pieces is refused
 * past one fire.
 */
export function maxReadoutRows(kind: ReadoutKind): number;
/**
 * Size in bytes of one folded recurrent state. 0 if the model keeps
 * none (attention only, windowed or not).
 */
export function rsStateSize(): bigint;
export type Dtype = import('./pie-inferlet-types.js').Dtype;
export type Error = import('./pie-inferlet-types.js').Error;
/**
 * What a guest needs to seed a BLOCK drafter's draft pass: the rows one
 * pass carries (the anchor and `rows - 1` mask slots) and the id every
 * row but the first carries in. Facts the head was trained
 * at, not policy: which rows to verify, and whether to draft at all, stay
 * the guest's. `none` for a model with no block drafter (a chained head
 * is `mtp-depth`). Static, like `mtp-depth`.
 */
export interface BlockDrafter {
  rows: number,
  maskToken: number,
  /**
   * The first row whose readout is a proposal: 1 when the anchor row
   * proposes nothing, 0 when every row does (the anchor's row then
   * predicts the token after it).
   */
  proposalsFrom: number,
}
/**
 * Which forward-pass interface the bound model takes: `forward` for every
 * text model, whatever state it keeps; `forward-diffusion` for a
 * block-diffusion text.
 * 
 * Do NOT derive this by parsing `architecture()` -- that is an open set.
 * # Variants
 * 
 * ## `"text"`
 * 
 * Tokens one by one -> `forward`
 * ## `"diffusion"`
 * 
 * A canvas the model denoises in place, one trunk read causally or
 * bidirectionally per pass -> `forward-diffusion`
 */
export type ForwardKind = 'text' | 'diffusion';
/**
 * The canvas a diffusion model denoises: how many tokens one block is,
 * and the trunk's hidden width. `none` for every other kind — the same
 * fact as `pass-kind() == diffusion`, stated once with its numbers.
 */
export interface CanvasShape {
  length: number,
  hidden: number,
}
/**
 * ── Generative facts ────────────────────────────────────────────────
 * What a guest sizes an image/video job from, so it never parses
 * `architecture()`: the family's readings (the arms a pass can run),
 * its latent space, the schedule it was trained under, and the widest
 * float lane one pass carries. A text model answers an empty list,
 * `none`, `none` and 0: it has one implicit reading and no latent.
 * A lane's stream — which of the model's rectangles its rows belong to
 * (a lane is one request's rows of ONE stream; a request with text and
 * image rows submits one pass per stream, both in one attention group).
 * Codes agree with `poem_ir::Stream`.
 * # Variants
 * 
 * ## `"text"`
 * 
 * ## `"image"`
 * 
 * ## `"video"`
 * 
 * ## `"audio"`
 * 
 * ## `"context"`
 * 
 * ## `"reference"`
 */
export type LaneStream = 'text' | 'image' | 'video' | 'audio' | 'context' | 'reference';
/**
 * What one axis of a positions input means. The rotary space a
 * DiT was trained in is 1..=4 axes wide and every family orders them
 * differently; this names them so a guest can fill the grid without
 * knowing the family.
 * # Variants
 * 
 * ## `"time"`
 * 
 * Frame / reference / sequence index (FLUX's `T`, Z-Image's `t`).
 * ## `"height"`
 * 
 * The latent grid's row.
 * ## `"width"`
 * 
 * The latent grid's column.
 * ## `"index"`
 * 
 * A plain row-ordinal axis, carried by one lane's rows alone
 * (FLUX's fourth `L` axis, which numbers the text rows).
 */
export type AxisRole = 'time' | 'height' | 'width' | 'index';
/**
 * WHERE A READING'S LANES SIT IN THE ROTARY SPACE.
 * 
 * A denoise reading's positions input takes one coordinate vector
 * per row, and which coordinate goes where is a family contract the
 * guest must not spell (`flux_2`'s `(0, h, w, j)`, `z_image`'s
 * `(L + 1, a, b)`). This fact states it, so `latent::positions_for`
 * builds both from the same code:
 * 
 *   * a TEXT or CONTEXT lane's row `j` sits at `text-origin + j` on
 *     axis `text-axis`, and at 0 on every other axis;
 *   * an IMAGE lane's patch `(a, b)` sits at `a` on the `height` axis
 *     and `b` on the `width` axis; on `text-axis` it sits at
 *     `text-origin + text-rows` when `image-follows-text`, else 0; on
 *     every remaining axis, 0.
 *   * a REFERENCE lane's patch `(a, b)` of reference `i` sits where an
 *     image lane's would, except on the `time` axis, where it sits at
 *     `reference-stride * (i + 1)` -- the offset that keeps each
 *     reference picture in a rotary neighbourhood of its own, clear of
 *     the target grid at 0.
 * 
 * `axes` has exactly the positions input's `width` entries. A family whose
 * positions do not fit this shape states `none` and its guests build
 * their own.
 */
export interface PositionConvention {
  axes: Array<AxisRole>,
  textAxis: number,
  textOrigin: number,
  imageFollowsText: boolean,
  /**
   * WHERE REFERENCE `i` SITS ON THE `time` AXIS: at
   * `reference-stride * (i + 1)`, so the first reference clears the
   * target grid's `T = 0` and each further one clears the last
   * (FLUX.2's `10*(i + 1)`). `none` -- the answer for every family
   * that declares no `reference` stream -- means this row states no
   * reference convention, and a guest that would bind a reference
   * lane must refuse rather than invent an offset.
   */
  referenceStride?: number,
}
/**
 * Which export seam a lane's epilogue reads: `logits()` for a text
 * head, `velocity()` for a denoiser, `hidden()` for an encoder,
 * `pixels` for a VAE reading on the voxel axis (one row per output
 * voxel of the pass's clip, `readout-width` wide).
 * # Variants
 * 
 * ## `"logits"`
 * 
 * ## `"velocity"`
 * 
 * ## `"hidden"`
 * 
 * ## `"pixels"`
 */
export type ReadoutKind = 'logits' | 'velocity' | 'hidden' | 'pixels';
/**
 * One reading of the bound model: the arm a pass runs when it names
 * this reading (`forward-pass.reading`). `index` is the fact value a
 * lane is stamped with; `has-kv` says whether `state` is required
 * (else refused) on its pass, `takes-tokens` the same for `embed`;
 * `streams` are the lane streams a request submits (one pass per
 * stream, `forward-pass.stream`; empty means text only); `positions`
 * says where its lanes' rows sit in the rotary space;
 * `readout`/`readout-width` say what the epilogue may read and how wide
 * one row is (the `velocity(width)` / `hidden(width)` declaration to
 * make).
 */
export interface ReadingFact {
  name: string,
  index: number,
  hasKv: boolean,
  takesTokens: boolean,
  streams: Array<LaneStream>,
  /**
   * How this reading's lanes place their rows in the rotary space,
   * when it states a convention. `none` otherwise.
   */
  positions?: PositionConvention,
  readout: ReadoutKind,
  readoutWidth: number,
}
/**
 * What one entry of an input belongs to: its lane, each row of its
 * lane, or each voxel of its lane's clip.
 * # Variants
 * 
 * ## `"lane"`
 * 
 * ## `"row"`
 * 
 * ## `"voxel"`
 */
export type InputAxis = 'lane' | 'row' | 'voxel';
/**
 * One input the model reads from its guest, by name: `width` values a
 * lane, a row or a voxel, as `dtype` (`f32` for a float input, `i32`
 * for an i32 one; the host marshals to the model's own element type).
 * One name is one input wherever the model reads it. `rows` is the row
 * count a lane binding it carries when the family fixes it (Wan 2.2's
 * 512 context rows: its reference zero-pads umT5's answer to 512 and
 * attends every pad row), `none` nearly always.
 */
export interface InputDecl {
  name: string,
  axis: InputAxis,
  width: number,
  dtype: Dtype,
  rows?: number,
}
/**
 * What a lane's epilogue reads out, and how wide one row is.
 */
export interface Readout {
  kind: ReadoutKind,
  width: number,
}
/**
 * Everything about a lane of the given switches: the inputs it must
 * bind (exactly these, by `forward-pass.input`), whether it binds
 * `state` and `embed`, and what it reads out.
 */
export interface LaneDecl {
  inputs: Array<InputDecl>,
  readsKv: boolean,
  embeds: boolean,
  readout: Readout,
}
/**
 * The latent space a denoiser works in: one latent row is
 * `patch-t × patch-h × patch-w` cells of `channels` values, one cell
 * is `spatial-compression²` pixels by `temporal-compression` frames.
 * `none` for a model with no latent.
 */
export interface LatentSpace {
  channels: number,
  patchT: number,
  patchH: number,
  patchW: number,
  spatialCompression: number,
  temporalCompression: number,
}
/**
 * The prediction target a schedule's velocity is.
 * # Variants
 * 
 * ## `"flow"`
 * 
 * Rectified flow: `x <- x + (sigma' - sigma) · v`.
 * ## `"epsilon"`
 * 
 * Epsilon prediction.
 * ## `"v"`
 * 
 * v-prediction.
 */
export type ScheduleKind = 'flow' | 'epsilon' | 'v';
/**
 * The noise schedule the denoiser was trained under: `shift` is the
 * trained timestep shift (1.0 = none; a flow model's base `mu` when it
 * shifts by token count), `train-steps` scales the timestep axis,
 * `boundary` is the sigma a two-backbone family hands over at (`none`
 * for one backbone), `pinned-sigmas` a distilled model's fixed sigma
 * list, descending, without the final 0 (empty when the guest builds
 * its own), and `stream-shifts` names the per-modality shifts of a
 * family that runs several schedules at once. The library's
 * `latent::FlowMatchEuler` is built from this.
 * `none` when no reading denoises.
 */
export interface StreamShift {
  /**
   * `stream` is a WIT keyword, so the lane's stream is spelled `lane`.
   */
  lane: LaneStream,
  shift: number,
}
export interface ScheduleFact {
  kind: ScheduleKind,
  shift: number,
  trainSteps: number,
  boundary?: number,
  pinnedSigmas: Float32Array,
  /**
   * The shift each STREAM's own sigma grid is built at, for a family
   * whose modalities advance on different schedules inside ONE
   * evaluation (MiniMax H3 runs video at 12 and audio at 3). Empty
   * when `shift` serves every lane, which is every other family; a
   * stream absent from the list takes `shift`.
   */
  streamShifts: Array<StreamShift>,
}
