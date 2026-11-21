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
export function draftBlock(): BlockDrafter | undefined;
/**
 * How many tokens the model's own draft head proposes in one pass, each
 * fed the one before it: the leading entries of the `mtp.drafts` output
 * that are a chain, whatever width that output has. A guest drafting from
 * the head drafts at most this many a round. 0 for a model whose text
 * states no draft head. Static.
 */
export function draftDepth(): number;
/**
 * Every switch the bound model declares; empty for a one-path model.
 */
export function switches(): Array<SwitchDecl>;
/**
 * The lane of these `(name, value)` switches; unset ones take defaults.
 */
export function lane(switches: Array<[string, string]>): LaneDecl;
export function latent(): LatentSpace | undefined;
export function schedule(): ScheduleFact | undefined;
/**
 * The most rows one pass may bind, across all its lanes. Static.
 */
export function maxPassRows(): number;
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
 * reads its drafts out as an output). Static.
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
 * A switch the model declares and the values `forward-pass.switch` may
 * set it to, the first its default.
 */
export interface SwitchDecl {
  name: string,
  values: Array<string>,
}
/**
 * Whether an input or output is one per lane, per row, or per voxel of the clip.
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
 * The rows a lane binding an input carries: exactly `n`, or the fewest
 * multiple of `n` holding the guest's own; the guest zero-pads to them.
 */
export type RowCount = RowCountExact | RowCountMultiple;
export interface RowCountExact {
  tag: 'exact',
  val: number,
}
export interface RowCountMultiple {
  tag: 'multiple',
  val: number,
}
/**
 * One input the model reads from its guest: `width` values of `dtype`
 * per `axis` entry; `rows` is set when the family fixes the row count
 * (Wan 2.2 attends its reference's 512 zero-padded context rows).
 */
export interface InputDecl {
  name: string,
  axis: InputAxis,
  width: number,
  dtype: Dtype,
  rows?: RowCount,
}
/**
 * One output a lane's epilogue may read, by the name the model gives it:
 * `width` values of `dtype` per `axis` entry, stacked `layers` deep when
 * the model writes it per layer (0 when it does not); at most `max-rows`
 * rows a pass. `readout`: its rows are the lane's readout rows, which
 * `forward-pass.readout` chooses. `planes`: how one entry's `width`
 * values factor, outermost first, each run contiguous: `[q-heads,
 * keys]` for `attn.scores`, so a guest folds over the heads the model
 * states. Empty when the entry is one flat run.
 */
export interface OutputDecl {
  name: string,
  axis: InputAxis,
  width: number,
  dtype: Dtype,
  layers: number,
  maxRows: number,
  readout: boolean,
  planes: Uint32Array,
}
/**
 * A lane of given switches: the inputs it binds (exactly these), whether
 * it binds `state` and `embed`, and the outputs it may read.
 */
export interface LaneDecl {
  inputs: Array<InputDecl>,
  state: boolean,
  embeds: boolean,
  outputs: Array<OutputDecl>,
}
/**
 * ── Generative facts ────────────────────────────────────────────────
 * A text model answers `none` to each.
 * The VAE's `[t, h, w, channels]` latent; latent frame 0 covers
 * `first-frames` pixel frames, each later one `temporal-compression`.
 */
export interface LatentSpace {
  channels: number,
  spatialCompression: number,
  temporalCompression: number,
  firstFrames: number,
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
 * `none` when the model does not denoise.
 */
export interface StreamShift {
  /**
   * A `stream` switch value (`stream` is a WIT keyword).
   */
  lane: string,
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
