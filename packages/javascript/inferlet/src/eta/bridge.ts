// The author-facing ETA bridge over the WIT forward surface — port of
// `crates/inferlet/src/eta.rs`.
//
// `ForwardPass` wraps the `pie:inferlet/forward` resource and drives the
// neutral `Builder`, lowering author stage closures to the ETA container. A
// `Channel` owns both the trace declaration and the WIT resource.
//
// Host readback is SYNCHRONOUS here: a JS guest cannot lower the world's
// `async func` imports (componentize-js has no component-model async), so it
// reads through the host's blocking twins (`take-blocking` / `read-blocking`)
// and the guest's task simply blocks until the cell fills.

import * as witChannel from 'pie:inferlet/channel@0.3.0';
import * as witForward from 'pie:inferlet/forward@0.3.0';
import * as witModel from 'pie:inferlet/model@0.3.0';
import * as witPipeline from 'pie:inferlet/pipeline@0.3.0';
import * as witWs from 'pie:inferlet/working-set@0.3.0';

import { Audio, Image } from '../media.js';
import { Builder, DslChannel } from './builder.js';
import { kernel } from './intrinsics.js';
import { Dtype, Port, Shape, Stage, numel, portName, shapeOf } from './ir.js';
import { ChannelState, TraceError } from './trace.js';
import { ConstData, ConstLike, Tensor, constData, reshape, unpackElems } from './value.js';

/** A host-side refusal (`pie:inferlet` declares `type error = string`). */
export class InferletError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'InferletError';
  }
}

function wit<T>(what: string, f: () => T): T {
  try {
    return f();
  } catch (e: unknown) {
    const payload = (e as { payload?: unknown })?.payload;
    const msg = typeof payload === 'string' ? payload : e instanceof Error ? e.message : String(e);
    throw new InferletError(what ? `${what}: ${msg}` : msg);
  }
}

// ---------------------------------------------------------------------------
// Host handles
// ---------------------------------------------------------------------------

const WIT_DTYPE: Record<Dtype, witChannel.Dtype> = {
  [Dtype.F32]: 'f32',
  [Dtype.I32]: 'i32',
  [Dtype.U32]: 'u32',
  [Dtype.BOOL]: 'bool',
};

/** The WIT `channel` resource behind a trace channel, created from its
 * declaration on first host use (which is what lets `capacity()` still widen
 * a channel nobody has touched). */
function hostChannel(state: ChannelState): witChannel.Channel {
  if (state.host === null) {
    state.host = new witChannel.Channel(new Uint32Array(state.shape), WIT_DTYPE[state.dtype], state.capacity);
  }
  return state.host as witChannel.Channel;
}

// ---------------------------------------------------------------------------
// Channel
// ---------------------------------------------------------------------------

export const TOKEN_PAD = -1;

export function padTokens(tokens: ArrayLike<number>, envelope: number): number[] {
  if (tokens.length > envelope) throw new Error(`window of ${tokens.length} tokens exceeds its envelope of ${envelope}`);
  const out = Array.from(tokens, (t) => Math.trunc(t));
  while (out.length < envelope) out.push(TOKEN_PAD);
  return out;
}

export function unpadTokens(window: ArrayLike<number>): number[] {
  return Array.from(window).filter((t) => t !== TOKEN_PAD);
}

/**
 * A GPU-resident bounded queue, backing both the trace and the WIT `channel`
 * resource. `new Channel([1], dtype.i32)` declares an empty capacity-1
 * channel; `Channel.from([0, 1], dtype.u32)` a channel seeded full. Inside a
 * stage body, `take()`/`read()`/`put(tensor)` record device ops; on the host,
 * `put(data)` stages a cell and `takeHost()` reads one back (blocking).
 */
export class Channel {
  private dsl_: DslChannel | null;
  private pending: { capacity?: number; name?: string } | null = null;

  constructor(shape: readonly number[], dtype: Dtype);
  /** @internal */
  constructor(dsl: DslChannel);
  constructor(shapeOrDsl: readonly number[] | DslChannel, dtype?: Dtype) {
    if (shapeOrDsl instanceof DslChannel) {
      this.dsl_ = shapeOrDsl;
      return;
    }
    if (dtype === undefined) throw new TypeError('new Channel(shape, dtype)');
    this.dsl_ = DslChannel.new(shapeOf(shapeOrDsl), dtype);
  }

  /** @internal An output channel declared by a stage body's return value:
   * its shape and dtype are the returned tensor's, known once the pass
   * traces (at first submit). `capacity`/`named` before that are kept. */
  static lazy(): Channel {
    const ch = new Channel(DslChannel.new([1], Dtype.I32));
    ch.dsl_ = null;
    ch.pending = {};
    return ch;
  }

  /** @internal Inside a stage body: declare this lazy channel as the
   * returned tensor's cell and record the put. */
  materialize(value: Tensor): void {
    const one = numel(value.shape) === 1 && value.shape.length !== 1;
    const shape = one ? [1] : value.shape;
    if (one) value = reshape(value, shape);
    this.dsl_ = DslChannel.new(shape, value.dtype);
    const pending = this.pending ?? {};
    this.pending = null;
    if (pending.capacity !== undefined) this.dsl_.capacity(pending.capacity);
    if (pending.name !== undefined) this.dsl_.named(pending.name);
    this.dsl_.putTensor(value);
  }

  /** @internal The trace-side half. */
  get dsl(): DslChannel {
    if (this.dsl_ === null) throw new InferletError('this output channel is declared by its stage body; submit the pass first');
    return this.dsl_;
  }

  /** An initially empty channel whose producer is the host. */
  static writer(shape: readonly number[], dtype: Dtype): Channel {
    const ch = new Channel(shape, dtype);
    ch.dsl.noteHostPut();
    return ch;
  }

  /** A channel seeded full with the per-instance value `v`. */
  static from(v: ConstLike, dtype?: Dtype): Channel {
    return Channel.seededWith(constData(v, dtype));
  }

  /** Like `from`, but reinterprets the flat seed under `shape`. */
  static fromShaped(shape: readonly number[], v: ConstLike, dtype?: Dtype): Channel {
    const d0 = constData(v, dtype);
    const s = shapeOf(shape);
    if (numel(s) !== numel(d0.shape)) throw new Error('fromShaped: element count mismatch');
    return Channel.seededWith(new ConstData(s, d0.dtype, d0.data));
  }

  private static seededWith(data: ConstData): Channel {
    const ch = new Channel(DslChannel.fromConst(data));
    wit('stage seed on a fresh channel', () => ch.wit().put(data.data));
    return ch;
  }

  /** A seeded channel whose seed value is supplied at instantiation. */
  static seeded(shape: readonly number[], dtype: Dtype): Channel {
    return new Channel(DslChannel.seeded(shapeOf(shape), dtype));
  }

  /** @internal */
  wit(): witChannel.Channel {
    return hostChannel(this.dsl.state);
  }

  /** Widen the ring to `n` cells (deeper run-ahead). Must precede first use. */
  capacity(n: number): this {
    if (this.dsl_ === null) {
      this.pending!.capacity = n;
      return this;
    }
    if (this.dsl.state.host !== null) throw new TraceError('capacity must be set before the channel is used');
    this.dsl.capacity(n);
    return this;
  }

  named(name: string): this {
    if (this.dsl_ === null) {
      this.pending!.name = name;
      return this;
    }
    this.dsl.named(name);
    return this;
  }

  /** Declaration order — the container keys channels by it. */
  get gid(): number {
    return this.dsl.gid;
  }
  get name(): string {
    return this.dsl.name;
  }
  get dtype(): Dtype {
    return this.dsl.dtype;
  }
  get shape(): Shape {
    return this.dsl.shape;
  }

  /** Consume a cell inside a stage body — records a `ChanTake`. */
  take(): Tensor {
    return this.dsl.take();
  }

  /** Peek a cell inside a stage body — records a `ChanRead`. */
  read(): Tensor {
    return this.dsl.read();
  }

  /** In a stage body with a `Tensor`: a device `ChanPut`. On the host with
   * data: stage the next cell for the following submit. */
  put(v: Tensor | ConstLike, dtype?: Dtype): void {
    if (v instanceof Tensor) {
      this.dsl.putTensor(v);
      return;
    }
    const data = constData(v, dtype ?? this.dtype);
    if (data.dtype !== this.dtype) throw new TypeError(`channel ${this.name} holds dtype ${this.dtype}, put ${data.dtype}`);
    this.dsl.noteHostPut();
    try {
      this.wit().put(data.data);
    } catch {
      // Fire-and-forget: failures surface at take.
    }
  }

  /** Atomically replace the committed front cell (a host operation). */
  set(v: ConstLike, dtype?: Dtype): void {
    const data = constData(v, dtype ?? this.dtype);
    wit(`${this.name} set`, () => this.wit().set(data.data));
  }

  /** Consume a cell on the host, decoded to numbers (booleans for a bool
   * channel). Blocks until an in-flight fire fills it; a poisoned channel
   * throws `InferletError`. */
  takeHost(): number[] | boolean[] {
    this.dsl.noteHostTake();
    const raw = wit(`${this.name} take`, () => this.wit().takeBlocking());
    return unpackElems(raw, this.dtype);
  }

  /** Peek a cell on the host (leaves it full). */
  readHost(): number[] | boolean[] {
    this.dsl.noteHostRead();
    const raw = wit(`${this.name} read`, () => this.wit().readBlocking());
    return unpackElems(raw, this.dtype);
  }

  /** `takeHost()` for a one-element cell. */
  takeScalar(): number {
    const v = this.takeHost();
    if (!v.length) throw new InferletError(`${this.name} take: channel cell is empty`);
    return Number(v[0]);
  }

  readScalar(): number {
    const v = this.readHost();
    if (!v.length) throw new InferletError(`${this.name} read: channel cell is empty`);
    return Number(v[0]);
  }
}

// ---------------------------------------------------------------------------
// Working sets
// ---------------------------------------------------------------------------

/** One sequence's model state; slots below `folded()` are frozen, the rest is a buffer. */
export class WorkingSet {
  /** @internal */
  ws: witWs.WorkingSet;

  constructor() {
    this.ws = new witWs.WorkingSet();
  }

  private static wrap(ws: witWs.WorkingSet): WorkingSet {
    const out: WorkingSet = Object.create(WorkingSet.prototype);
    out.ws = ws;
    return out;
  }

  /** Lets passes write slots below `end`; needed only when the device computes the slots. */
  reserve(end: number): void {
    wit('reserve', () => this.ws.reserve(end));
  }

  /** The frozen length; while a device-counted fold is in flight, the length before it. */
  folded(): number {
    return this.ws.folded();
  }

  /** Slots the runtime mapped in from state it already held rather than computed. */
  cachedTokens(): number {
    return this.ws.cachedTokens();
  }

  /** A child sharing the frozen part, with a copy of the buffer. */
  fork(on: Pipeline): WorkingSet {
    return WorkingSet.wrap(wit('fork', () => this.ws.fork(on.wit)));
  }
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/** A run-ahead ordering domain. Concurrent streams need separate pipelines. */
export class Pipeline {
  readonly wit: witPipeline.Pipeline = new witPipeline.Pipeline();

  /** End the stream; already-submitted fires still drain. */
  close(): void {
    this.wit.close();
  }

  /** Leave the frame wait-set until this pipeline submits again. */
  park(): void {
    witForward.park(this.wit);
  }

  /** Enqueue one pass as a single-slot frame. */
  submit(fwd: ForwardPass): void {
    this.submitFrame([fwd]);
  }

  /** Submit ONE FRAME: up to `frameSize()` slots, slot i executing in wave
   * i; trailing slots pad with no-ops. */
  submitFrame(slots: (ForwardPass | undefined)[]): void {
    const k = frameSize();
    if (slots.length > k) throw new InferletError(`frame holds ${slots.length} slot(s); model.frame-size() is ${k}`);
    const live = slots.filter((p): p is ForwardPass => p !== undefined);
    for (const p of live) p.attachProgram();
    if (!live.length) return;
    const borrows: (witForward.ForwardPass | undefined)[] = slots.map((p) => p?.wit);
    while (borrows.length < k) borrows.push(undefined);
    wit('submit', () => witForward.submit(this.wit, borrows));
  }

  /**
   * Keep the runtime's run-ahead window full while `onToken` consumes
   * results, until `budget` fires submit or `onToken` returns `false`.
   * Returns the run count. `onToken` is synchronous here (host reads block).
   */
  runAhead(fwd: ForwardPass, budget: number, onToken: () => boolean | void): number {
    if (budget === 0) return 0;
    // Live slots per frame. A pass that binds a dense device mask takes one;
    // so does a pass on a model with recurrent state, whose frame's waves
    // carry the same sequence's state one into the next (a frame of one live
    // slot measured faster than a full one on Qwen3.5-0.8B), and a pass off
    // its model's default path (a canvas redrawn whole), which redraws what
    // the last wrote. Any other pass fills the frame.
    const r = fwd.bindsDeviceMask() || rsStateSize() > 0 || fwd.switched() ? 1 : frameSize();
    const windowFrames = Math.max(Math.floor(witModel.runAheadWindow() / Math.max(r, 1)), 1);
    let submitted = 0;
    let consumed = 0;
    const submitOneFrame = () => {
      const live = Math.min(r, budget - submitted);
      if (live === 0) return;
      this.submitFrame(new Array(live).fill(fwd));
      submitted += live;
    };
    for (let i = 0; i < windowFrames; i++) {
      if (submitted >= budget) break;
      submitOneFrame();
    }
    let ended = false;
    if (submitted >= budget && !ended) {
      this.close();
      ended = true;
    }
    while (consumed < submitted) {
      if (onToken() === false) {
        if (!ended) this.close();
        return consumed + 1;
      }
      consumed += 1;
      if (submitted < budget && submitted - consumed <= (windowFrames - 1) * r) submitOneFrame();
      if (submitted >= budget && !ended) {
        this.close();
        ended = true;
      }
    }
    if (!ended) this.close();
    return consumed;
  }
}

// ---------------------------------------------------------------------------
// State geometry
// ---------------------------------------------------------------------------

/** How many more slots each lane freezes after a fire: none, `'all'`, or a u32 count channel. */
export type Fold = undefined | 'all' | Channel;

/** Where a pass's rows land and what they read; omitted fields mean a causal append. */
export interface StateGeometry {
  /** Per row, the RoPE position. */
  positions?: Channel;
  /** Per row, the slot it writes; a lane's rows write consecutive slots. */
  slots?: Channel;
  /** Per lane, how many slots the lane reads. */
  kvLen?: Channel;
  fold?: Fold;
}

// ---------------------------------------------------------------------------
// Model constants
// ---------------------------------------------------------------------------

const cache = new Map<string, number>();
function cached(key: string, f: () => number): number {
  let v = cache.get(key);
  if (v === undefined) {
    v = f();
    cache.set(key, v);
  }
  return v;
}

/** Waves per frame (k) for this deployment (cached). */
export const frameSize = () => cached('frameSize', () => Math.max(witModel.frameSize(), 1));
export const submitDeadlineUs = () => cached('submitDeadlineUs', () => Number(witModel.submitDeadlineUs()));
/** Host-reader channel capacity that sustains run-ahead (not cached). */
export const channelCapacity = () => Math.max(witModel.channelCapacity(), 2);
export const kvPageSize = () => cached('kvPageSize', () => witModel.kvPageSize());
const rsStateSize = () => cached('rsStateSize', () => Number(witModel.rsStateSize()));
/** The most rows one pass may read out through `kind`; the pass itself may be any length. */
export const maxReadoutRows = (kind: witModel.ReadoutKind) =>
  cached(`maxReadoutRows:${kind}`, () => Math.max(witModel.maxReadoutRows(kind), 1));

function evenSpans(n: number, cap: number): [number, number][] {
  if (n === 0) return [];
  cap = Math.max(Math.min(cap, n), 1);
  const k = Math.max(Math.ceil(n / cap), 1);
  const q = Math.floor(n / k);
  const r = n % k;
  const out: [number, number][] = [];
  let base = 0;
  for (let i = 0; i < k; i++) {
    const end = base + q + (i < r ? 1 : 0);
    out.push([base, end]);
    base = end;
  }
  return out;
}

/** Even `[start, end)` spans of `n` rows, each at most `maxReadoutRows(kind)`. */
export function readoutSpans(n: number, kind: witModel.ReadoutKind): [number, number][] {
  return evenSpans(n, maxReadoutRows(kind));
}

// ---------------------------------------------------------------------------
// ForwardPass
// ---------------------------------------------------------------------------

const SITE_BITS: Record<string, number> = { q: 1, k: 2, v: 4, o: 8, gate_up: 16, down: 32 };

/**
 * The forward-pass builder over `pie:inferlet/forward`: what a model's lanes
 * bind, the switches they run under, and the program the fires run.
 */
export class ForwardPass {
  readonly wit: witForward.ForwardPass;
  private ports: [Port, Channel][] = [];
  /** Each lane's bound inputs, by lane. */
  private inputs: [number, string, Channel][] = [];
  private stages: [Stage, () => void][] = [];
  private readonly vocab: number;
  private readonly pageSize: number;
  private programAttached = false;
  private adapterLowrankSites = 0;
  private adapterScaleSites = 0;
  /** Keeps the bound working sets alive while the pass is bound. */
  private lanes: WorkingSet[] = [];
  private foldAll: Channel | undefined;
  /** Each lane's set switches, by lane. */
  private switches: [number, string, string][] = [];

  constructor() {
    this.wit = new witForward.ForwardPass();
    this.vocab = witModel.outputVocabSize();
    this.pageSize = kvPageSize();
  }

  private ensurePortsAvailable(ports: Port[]) {
    if (this.programAttached) throw new InferletError('forward pass program is already attached');
    const bound = new Set(this.ports.map(([p]) => p));
    for (const p of ports) if (bound.has(p)) throw new InferletError(`forward pass port ${portName(p)} is already bound`);
  }

  bindsDeviceMask(): boolean {
    return this.ports.some(([p]) => p === Port.ATTN_MASK);
  }

  /** Bind token ids and the CSR row indptr. A token array becomes an i32
   * channel; `indptr` omitted is the one row `[0, n]`, an array is a u32
   * channel. */
  embed(tokens: Channel | ArrayLike<number>, indptr?: Channel | ArrayLike<number>): void {
    if (!(tokens instanceof Channel)) tokens = Channel.from(Array.from(tokens), Dtype.I32);
    if (indptr === undefined) indptr = Channel.from([0, numel(tokens.shape)], Dtype.U32);
    else if (!(indptr instanceof Channel)) indptr = Channel.from(Array.from(indptr), Dtype.U32);
    this.ensurePortsAvailable([Port.EMBED_TOKENS, Port.EMBED_INDPTR]);
    wit('embed', () => this.wit.embed(tokens.wit(), indptr.wit()));
    this.ports.push([Port.EMBED_TOKENS, tokens], [Port.EMBED_INDPTR, indptr]);
  }

  readout(indices: Channel): void {
    this.ensurePortsAvailable([Port.READOUT]);
    wit('readout', () => this.wit.readout(indices.wit()));
    this.ports.push([Port.READOUT, indices]);
  }

  /** Carry the payloads of the `media.Image` / `media.Audio` spans lane
   * `lane`'s tokens carry, order-matched to their placeholder token runs. */
  media(lane: number, spans: (Image | Audio)[]): void {
    const wrapped: witForward.MediaSpan[] = spans.map((s) => {
      if (s instanceof Audio) return { tag: 'audio', val: s.handle };
      if (s instanceof Image) return { tag: 'image', val: s.handle };
      throw new TypeError('media span must be a media.Image or media.Audio');
    });
    wit('media', () => this.wit.media(lane, wrapped));
  }

  /** Bind one working set per lane, in lane order; lanes sharing a set must state `slots`. */
  state(lanes: WorkingSet[], geom: StateGeometry = {}): void {
    const rebind = this.programAttached;
    const fold = geom.fold === 'all' ? (this.foldAll ??= Channel.from([0xffff_ffff], Dtype.U32)) : geom.fold;
    const stated: [Port, Channel | undefined][] = [
      [Port.POSITIONS, geom.positions],
      [Port.SLOTS, geom.slots],
      [Port.KV_LEN, geom.kvLen],
      [Port.FOLD, fold],
    ];
    if (!rebind) this.ensurePortsAvailable(stated.filter(([, ch]) => ch !== undefined).map(([port]) => port));
    const g: witForward.StateGeometry = {};
    if (geom.positions) g.positions = geom.positions.wit();
    if (geom.slots) g.slots = geom.slots.wit();
    if (geom.kvLen) g.kvLen = geom.kvLen.wit();
    if (fold) g.fold = fold.wit();
    wit('state', () => this.wit.state(lanes.map((l) => l.ws), g));
    this.lanes = [...lanes];
    if (!rebind) for (const [port, ch] of stated) if (ch !== undefined) this.ports.push([port, ch]);
  }

  /** Slots to keep a recurrent state at, so a later pass over these tokens resumes there. */
  checkpoints(slots: ArrayLike<number>): void {
    wit('checkpoints', () => this.wit.checkpoints(Uint32Array.from(slots)));
  }

  /** Set lane `lane`'s model switch `name` to `value` (`model.switches()`):
   * which path that lane runs. Before the program. */
  switch(lane: number, name: string, value: string): void {
    if (this.programAttached) throw new InferletError('forward pass program is already attached');
    wit('switch', () => this.wit.switch(lane, name, value));
    this.switches.push([lane, name, value]);
  }

  /** Whether a switch of this pass is off its model's default: a path a
   * model runs apart from its default one (a canvas a block-diffusion text
   * redraws whole) posts one fire at a time. */
  switched(): boolean {
    if (!this.switches.length) return false;
    const declared = witModel.switches();
    return this.switches.some(([, name, value]) => {
      const decl = declared.find((d) => d.name === name);
      return decl !== undefined && decl.values[0] !== value;
    });
  }

  /** Bind `ch` to lane `lane`'s model input `name` (`model.lane(...).inputs`),
   * read at every submit from its committed cell. The host validates the shape. */
  input(lane: number, name: string, ch: Channel): void {
    if (this.programAttached) throw new InferletError('forward pass program is already attached');
    if (this.inputs.some(([at, bound]) => at === lane && bound === name)) {
      throw new InferletError(`input \`${name}\` is already bound on lane ${lane}`);
    }
    wit('input', () => this.wit.input(lane, name, ch.wit()));
    this.inputs.push([lane, name, ch]);
  }

  /** Put lane `lane` in this pass's attention group `id`. */
  group(lane: number, id: number): void {
    if (this.programAttached) throw new InferletError('forward pass program is already attached');
    wit('group', () => this.wit.group(lane, id));
  }

  /** Per row of the pass, the rows it attends: kv slots on a pass whose lanes
   * read kv, its attention group's rows on one whose lanes do not. */
  mask(rows: Channel): void {
    this.ensurePortsAvailable([Port.ATTN_MASK]);
    wit('mask', () => this.wit.mask(rows.wit()));
    this.ports.push([Port.ATTN_MASK, rows]);
  }

  /** Attach a PEFT adapter at `site` ("q"|"k"|"v"|"o"|"gate_up"|"down"). */
  adapter(site: string, f: (x: AdapterExpr, y: AdapterExpr) => AdapterExpr): void {
    const bit = SITE_BITS[site];
    if (bit === undefined) throw new InferletError(`adapter: unknown site ${site}`);
    const expr = f(new AdapterExpr('x'), new AdapterExpr('y'));
    const isLowrank = (e: AdapterExpr): [Channel, Channel] | null => {
      if (e.kind !== 'add') return null;
      const [lhs, rhs] = e.args as [AdapterExpr, AdapterExpr];
      const delta = lhs.kind === 'y' ? rhs : rhs.kind === 'y' ? lhs : null;
      if (!delta || delta.kind !== 'mm') return null;
      const [b, mid] = delta.args as [Channel, AdapterExpr];
      if (mid.kind !== 'mm') return null;
      const [a, x] = mid.args as [Channel, AdapterExpr];
      if (x.kind !== 'x') return null;
      return [a, b];
    };
    if (expr.kind === 'scale') {
      const [l, inner] = expr.args as [Channel, AdapterExpr];
      const lr = isLowrank(inner);
      if (lr) {
        const [a, b] = lr;
        if ((this.adapterLowrankSites | this.adapterScaleSites) & bit) throw new InferletError(`adapter: site ${site} already carries an adapter on this pass`);
        this.adapterLowrankSites |= bit;
        this.adapterScaleSites |= bit;
        this.prologue(() => {
          kernel.lora(a.read(), b.read(), Tensor.constant(bit, Dtype.U32));
          kernel.adapterScale(l.read(), Tensor.constant(bit, Dtype.U32));
        });
        return;
      }
      if (inner.kind === 'y') {
        if (this.adapterScaleSites & bit) throw new InferletError(`adapter: site ${site} already carries a scale on this pass`);
        this.adapterScaleSites |= bit;
        this.prologue(() => kernel.adapterScale(l.read(), Tensor.constant(bit, Dtype.U32)));
        return;
      }
    }
    const lr = isLowrank(expr);
    if (!lr) throw new InferletError('adapter: form not lowerable (v0 lowers `y + mm(b, mm(a, x))`, `scale(y, l)`)');
    const [a, b] = lr;
    if (this.adapterLowrankSites & bit) throw new InferletError(`adapter: site ${site} already carries an adapter on this pass`);
    this.adapterLowrankSites |= bit;
    this.prologue(() => kernel.lora(a.read(), b.read(), Tensor.constant(bit, Dtype.U32)));
  }

  private setStage(stage: Stage, body: () => void): void {
    if (this.programAttached) throw new InferletError('stage attachment is construction-only');
    const i = this.stages.findIndex(([s]) => s === stage);
    if (i >= 0) this.stages[i] = [stage, body];
    else this.stages.push([stage, body]);
  }

  prologue(body: () => void): void {
    this.setStage(Stage.PROLOGUE, body);
  }
  /** Attach the `epilogue` stage (sampling programs; after the forward). A
   * body that returns a tensor declares an output channel of that cell, put
   * once per fire; the channel is returned and read on the host
   * (`out.takeScalar()`). `out.capacity(n)` widens it for run-ahead. */
  epilogue(body: () => Tensor | void): Channel {
    const out = Channel.lazy();
    this.setStage(Stage.EPILOGUE, () => {
      const value = body();
      if (value === undefined) return;
      if (!(value instanceof Tensor)) throw new TypeError('an epilogue returns a Tensor to publish, or nothing');
      out.materialize(value);
    });
    return out;
  }
  onAttnProj(body: () => void): void {
    this.setStage(Stage.ON_ATTN_PROJ, body);
  }
  onAttn(body: () => void): void {
    this.setStage(Stage.ON_ATTN, body);
  }

  attachProgram(): void {
    if (this.programAttached) return;
    const builder = new Builder(this.vocab, this.pageSize);
    for (const [port, ch] of this.ports) builder.bindPort(port, ch.dsl);
    // A channel several lanes read (a CFG step's one timestep) is one input
    // of the program.
    const listed: Channel[] = [];
    for (const [, , ch] of this.inputs) {
      if (!listed.includes(ch)) {
        listed.push(ch);
        builder.bindInput(ch.dsl);
      }
    }
    for (const [stage, body] of this.stages) builder.stage(stage, body);
    const traced = builder.build();
    const handles = traced.channels.map(hostChannel);
    wit('program', () => this.wit.program(traced.encode(), handles));
    this.programAttached = true;
  }
}

export class AdapterExpr {
  readonly args: unknown[];
  constructor(
    readonly kind: string,
    ...args: unknown[]
  ) {
    this.args = args;
  }
  add(other: AdapterExpr): AdapterExpr {
    return new AdapterExpr('add', this, other);
  }
}

/** `mm(w, e)` — multiply by the channel-borne weight. */
export function mm(w: Channel, e: AdapterExpr): AdapterExpr {
  return new AdapterExpr('mm', w, e);
}

/** `scale(e, l)` — elementwise multiply by the channel-borne vector. */
export function scale(e: AdapterExpr, l: Channel): AdapterExpr {
  return new AdapterExpr('scale', l, e);
}

/** Submit ONE FRAME on `on`: up to `frameSize()` slots. */
