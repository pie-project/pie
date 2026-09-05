// The author-facing ETA bridge over the WIT forward surface — port of
// `crates/inferlet/src/eta.rs`.
//
// `ForwardPass` wraps the `pie:inferlet/forward*` resources and drives the
// neutral `Builder`, lowering author stage closures to the ETA container. A
// `Channel` owns both the trace declaration and the WIT resource.
//
// Host readback is SYNCHRONOUS here: a JS guest cannot lower the world's
// `async func` imports (componentize-js has no component-model async), so it
// reads through the host's blocking twins (`take-blocking` / `read-blocking`)
// and the guest's task simply blocks until the cell fills.

import * as witChannel from 'pie:inferlet/channel@0.3.0';
import * as witAttention from 'pie:inferlet/forward@0.3.0';
import * as witHybrid from 'pie:inferlet/forward-hybrid@0.3.0';
import * as witRecurrent from 'pie:inferlet/forward-recurrent@0.3.0';
import * as witModel from 'pie:inferlet/model@0.3.0';
import * as witPipeline from 'pie:inferlet/pipeline@0.3.0';
import * as witWs from 'pie:inferlet/working-set@0.3.0';

import { Audio, Image } from '../media.js';
import { Builder, DslChannel } from './builder.js';
import { kernel } from './intrinsics.js';
import { Dtype, Port, Shape, Stage, numel, portName, shapeOf } from './ir.js';
import { ChannelState, TraceError } from './trace.js';
import { ConstData, ConstLike, Tensor, constData, unpackElems } from './value.js';

export type ForwardKind = witModel.ForwardKind;

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
  /** @internal The trace-side half. */
  readonly dsl: DslChannel;

  constructor(shape: readonly number[], dtype: Dtype);
  /** @internal */
  constructor(dsl: DslChannel);
  constructor(shapeOrDsl: readonly number[] | DslChannel, dtype?: Dtype) {
    if (shapeOrDsl instanceof DslChannel) {
      this.dsl = shapeOrDsl;
      return;
    }
    if (dtype === undefined) throw new TypeError('new Channel(shape, dtype)');
    this.dsl = DslChannel.new(shapeOf(shapeOrDsl), dtype);
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
    if (this.dsl.state.host !== null) throw new TraceError('capacity must be set before the channel is used');
    this.dsl.capacity(n);
    return this;
  }

  named(name: string): this {
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
      // Fire-and-forget, like the Rust SDK: failures surface at take.
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

export interface PageRange {
  start: number;
  len: number;
}

/** A grant of fresh logical page indexes from `WorkingSet.reserve`. */
export class PageGrant {
  readonly ids: number[];
  constructor(
    readonly start: number,
    length: number,
  ) {
    this.ids = Array.from({ length }, (_, i) => start + i);
  }
  range(): PageRange {
    return { start: this.start, len: this.ids.length };
  }
}

/** The attention working set — a logical page address space over the KV
 * mapping trie. Every page reference is working-set-relative. */
export class WorkingSet {
  readonly kv: witWs.KvWorkingSet;

  constructor(kv?: witWs.KvWorkingSet) {
    this.kv = kv ?? new witWs.KvWorkingSet();
  }

  pageLen(): number {
    return this.kv.pageLen();
  }

  reserve(pages: number): PageGrant {
    const r = wit('reserve KV', () => this.kv.reserve(pages));
    return new PageGrant(r.start, r.len);
  }

  updateIndex(key: Uint8Array): void {
    wit('updateIndex', () => this.kv.updateIndex(key));
  }

  static fromIndex(key: Uint8Array): WorkingSet | undefined {
    const kv = wit('fromIndex', () => witWs.KvWorkingSet.fromIndex(key));
    return kv ? new WorkingSet(kv) : undefined;
  }

  static removeIndex(key: Uint8Array): boolean {
    return wit('removeIndex', () => witWs.KvWorkingSet.removeIndex(key));
  }

  discard(on: Pipeline, ranges: PageRange[]): void {
    wit('discard', () => this.kv.discard(on.wit, ranges));
  }

  fork(on: Pipeline): WorkingSet {
    return new WorkingSet(wit('fork', () => this.kv.fork(on.wit)));
  }

  slice(on: Pipeline, start: number, len: number): WorkingSet {
    return new WorkingSet(wit('slice', () => this.kv.slice(on.wit, { start, len })));
  }

  copyInto(on: Pipeline, dstPageIds: ArrayLike<number>, dstTokIdx: ArrayLike<number>, srcPageIds: ArrayLike<number>, srcTokIdx: ArrayLike<number>): void {
    wit('copyInto', () =>
      this.kv.copyInto(on.wit, Uint32Array.from(dstPageIds), Uint32Array.from(dstTokIdx), Uint32Array.from(srcPageIds), Uint32Array.from(srcTokIdx)),
    );
  }
}

/** Runtime recurrent-state slots for hybrid / linear-attention models. */
export class RsWorkingSet {
  readonly rs: witWs.RsWorkingSet;

  constructor(rs?: witWs.RsWorkingSet) {
    this.rs = rs ?? new witWs.RsWorkingSet();
  }

  stateSize(): number {
    return Number(witModel.rsStateSize());
  }
  bufferSize(): number {
    return this.rs.bufferSize();
  }
  bufferPageSize(): number {
    return witModel.rsBufferPageSize();
  }
  allocBuffer(n: number): PageRange {
    return wit('allocBuffer', () => this.rs.allocBuffer(n));
  }
  freeBuffer(indices: ArrayLike<number>): void {
    wit('freeBuffer', () => this.rs.freeBuffer(Uint32Array.from(indices)));
  }
  discardBuffered(count: number): void {
    wit('discardBuffered', () => this.rs.discardBuffered(count));
  }
  reorderBuffer(perm: ArrayLike<number>): void {
    wit('reorderBuffer', () => this.rs.reorderBuffer(Uint32Array.from(perm)));
  }
  fork(on: Pipeline): RsWorkingSet {
    return new RsWorkingSet(wit('fork', () => this.rs.fork(on.wit)));
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
    witAttention.park(this.wit);
  }
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

/** A page span: `undefined` = everything; a number = `start..`; `[start, end]`
 * (end may be undefined) = `start..end`. */
export type PageDecl = undefined | number | [number, number | undefined];

function pageDecl(r: PageDecl): [number, number | undefined] {
  if (r === undefined) return [0, undefined];
  if (typeof r === 'number') return [r, undefined];
  const [start, end] = r;
  if (end !== undefined && start > end) throw new Error(`attention page-span start ${start} exceeds end ${end}`);
  return [start ?? 0, end];
}

function pageSpan(d: [number, number | undefined]): witWs.PageSpan {
  return d[1] === undefined ? { start: d[0] } : { start: d[0], end: d[1] };
}

/** The attention geometry of one fire — mirrors WIT `kv-geometry`. */
export interface KvGeometry {
  kvLen: Channel;
  pages: Channel;
  pageIndptr: Channel;
  wSlot: Channel;
  wOff: Channel;
  positions: Channel;
  mask?: Channel;
  readablePages?: PageDecl;
  writablePages?: PageDecl;
}

/** Where the bound recurrent state's folded boundary lands. `foldLen`
 * absent folds everything. */
export interface RsGeometry {
  foldLen?: Channel;
  buffer?: PageDecl;
}

export interface KvBinding {
  workingSet: WorkingSet;
  geometry: KvGeometry;
}

// ---------------------------------------------------------------------------
// Model constants (cached where the Rust SDK caches)
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
/** Live slots per frame: k for dense, 1 for recurrent (linear/hybrid). */
export const liveSlots = () => cached('liveSlots', () => (witModel.passKind() !== 'attention' ? 1 : frameSize()));
export const kvPageSize = () => cached('kvPageSize', () => witModel.kvPageSize());
export const maxEmbedLength = () => cached('maxEmbedLength', () => Math.max(witModel.maxEmbedLength(), 1));

export function evenSpans(n: number, cap: number): [number, number][] {
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

/** The `[start, end)` spans a prompt of `n` tokens must be prefilled in. */
export function prefillChunks(n: number, cap?: number): [number, number][] {
  const c = Math.min(cap ?? 0xffff_ffff, Math.max(maxEmbedLength(), 1));
  return evenSpans(n, c);
}

// ---------------------------------------------------------------------------
// ForwardPass
// ---------------------------------------------------------------------------

type AttentionMod = typeof witAttention;
type HybridMod = typeof witHybrid;
type RecurrentMod = typeof witRecurrent;
type AnyPass = witAttention.ForwardPass | witHybrid.ForwardPass | witRecurrent.ForwardPass;

interface StagedKv {
  ws: witWs.KvWorkingSet;
  readable: [number, number | undefined];
  writable: [number, number | undefined];
  kvLen: witChannel.Channel;
  pages: witChannel.Channel;
  pageIndptr: witChannel.Channel;
  wSlot: witChannel.Channel;
  wOff: witChannel.Channel;
  positions: witChannel.Channel;
  mask: witChannel.Channel | undefined;
}

const SITE_BITS: Record<string, number> = { q: 1, k: 2, v: 4, o: 8, gate_up: 16, down: 32 };

/**
 * The forward-pass builder over one `pie:inferlet/forward*` interface,
 * selected by `kind` (default: `model.passKind()`). Attach stage bodies with
 * `epilogue(fn)`, bind state with `bindState(...)` (kind-independent; or the
 * per-kind `attention` / `bindHybrid` / `bindRecurrent`), and `submit(pipe)`.
 * The first submit traces the stage bodies once.
 */
export class ForwardPass {
  readonly kind: ForwardKind;
  readonly wit: AnyPass;
  private readonly mod: AttentionMod | HybridMod | RecurrentMod;
  private ports: [Port, Channel][] = [];
  private stages: [Stage, () => void][] = [];
  private readonly vocab: number;
  private readonly pageSize: number;
  private programAttached = false;
  private adapterLowrankSites = 0;
  private adapterScaleSites = 0;
  private foldAll: Channel | undefined;

  constructor(kind?: ForwardKind) {
    this.kind = kind ?? witModel.passKind();
    if (this.kind === 'attention') this.mod = witAttention;
    else if (this.kind === 'hybrid') this.mod = witHybrid;
    else if (this.kind === 'recurrent') this.mod = witRecurrent;
    else throw new Error(`unknown forward kind ${String(this.kind)}`);
    this.wit = new this.mod.ForwardPass();
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

  /** Bind token ids and CSR row indptr (both channels). */
  embed(tokens: Channel, indptr: Channel): void {
    this.ensurePortsAvailable([Port.EMBED_TOKENS, Port.EMBED_INDPTR]);
    wit('embed', () => this.wit.embed(tokens.wit(), indptr.wit()));
    this.ports.push([Port.EMBED_TOKENS, tokens], [Port.EMBED_INDPTR, indptr]);
  }

  readout(indices: Channel): void {
    this.ensurePortsAvailable([Port.READOUT]);
    wit('readout', () => this.wit.readout(indices.wit()));
    this.ports.push([Port.READOUT, indices]);
  }

  setMaxLayers(maxLayers: number): void {
    wit('setMaxLayers', () => this.wit.setMaxLayers(maxLayers));
  }

  setDraftingBlock(on: boolean): void {
    wit('setDraftingBlock', () => this.wit.setDraftingBlock(on));
  }

  /** Carry the payloads of `media.Image` / `media.Audio` spans, order-matched
   * to their placeholder token runs in the embed. */
  media(spans: (Image | Audio)[]): void {
    if (this.kind === 'recurrent') throw new InferletError('a recurrent-only pass carries no media');
    const wrapped: witAttention.MediaSpan[] = spans.map((s) => {
      if (s instanceof Audio) return { tag: 'audio', val: s.handle };
      if (s instanceof Image) return { tag: 'image', val: s.handle };
      throw new TypeError('media span must be a media.Image or media.Audio');
    });
    wit('media', () => (this.wit as witAttention.ForwardPass).media(wrapped));
  }

  private stageKv(ws: WorkingSet, geom: KvGeometry): StagedKv {
    const rebind = this.programAttached;
    if (!rebind) {
      const ports = [Port.KV_LEN, Port.PAGES, Port.PAGE_INDPTR, Port.W_SLOT, Port.W_OFF, Port.POSITIONS];
      if (geom.mask) ports.push(Port.ATTN_MASK);
      this.ensurePortsAvailable(ports);
    }
    const staged: StagedKv = {
      ws: ws.kv,
      readable: pageDecl(geom.readablePages),
      writable: pageDecl(geom.writablePages),
      kvLen: geom.kvLen.wit(),
      pages: geom.pages.wit(),
      pageIndptr: geom.pageIndptr.wit(),
      wSlot: geom.wSlot.wit(),
      wOff: geom.wOff.wit(),
      positions: geom.positions.wit(),
      mask: geom.mask?.wit(),
    };
    if (!rebind) {
      this.ports.push(
        [Port.KV_LEN, geom.kvLen],
        [Port.PAGES, geom.pages],
        [Port.PAGE_INDPTR, geom.pageIndptr],
        [Port.W_SLOT, geom.wSlot],
        [Port.W_OFF, geom.wOff],
        [Port.POSITIONS, geom.positions],
      );
      if (geom.mask) this.ports.push([Port.ATTN_MASK, geom.mask]);
    }
    return staged;
  }

  private kvGeometryWit(s: StagedKv): witAttention.KvGeometry {
    const g: witAttention.KvGeometry = {
      readablePages: pageSpan(s.readable),
      writablePages: pageSpan(s.writable),
      kvLen: s.kvLen,
      pages: s.pages,
      pageIndptr: s.pageIndptr,
      wSlot: s.wSlot,
      wOff: s.wOff,
      positions: s.positions,
    };
    if (s.mask) g.mask = s.mask;
    return g;
  }

  private stageRs(workingSets: RsWorkingSet[], geom: RsGeometry) {
    if (!workingSets.length) throw new InferletError('forward pass needs one recurrent-state working set per request');
    const buffer = pageDecl(geom.buffer ?? [0, 0]);
    let foldLen: witChannel.Channel;
    if (geom.foldLen) {
      if (!this.programAttached) {
        this.ensurePortsAvailable([Port.RS_FOLD_LEN]);
        this.ports.push([Port.RS_FOLD_LEN, geom.foldLen]);
      }
      foldLen = geom.foldLen.wit();
    } else {
      this.foldAll ??= Channel.from([0xffff_ffff], Dtype.U32);
      foldLen = this.foldAll.wit();
    }
    return { workingSets: workingSets.map((r) => r.rs), foldLen, buffer };
  }

  private requireKind(kind: ForwardKind, what: string): void {
    if (this.kind !== kind) throw new InferletError(`${what} binds a ${kind} pass; this pass is ${this.kind}`);
  }

  /** Bind the KV working set + geometry of an attention-only pass. */
  attention(ws: WorkingSet, geom: KvGeometry): void {
    this.requireKind('attention', 'attention');
    const kv = this.stageKv(ws, geom);
    wit('attention', () => (this.wit as witAttention.ForwardPass).attention(kv.ws, this.kvGeometryWit(kv)));
  }

  /** Bind a hybrid pass: the KV binding (`undefined` for a fire that touches
   * no attention layer) plus the recurrent working set(s) and where their
   * folded boundary lands. */
  bindHybrid(kv: KvBinding | undefined, rs: RsWorkingSet[], rsGeom: RsGeometry): void {
    this.requireKind('hybrid', 'bindHybrid');
    const stagedKv = kv ? this.stageKv(kv.workingSet, kv.geometry) : undefined;
    const stagedRs = this.stageRs(rs, rsGeom);
    const binding = stagedKv ? { workingSet: stagedKv.ws, geometry: this.kvGeometryWit(stagedKv) } : undefined;
    wit('bindHybrid', () =>
      (this.wit as witHybrid.ForwardPass).attention(binding, stagedRs.workingSets, { foldLen: stagedRs.foldLen, buffer: pageSpan(stagedRs.buffer) }),
    );
  }

  /** Bind a recurrent-only pass: the recurrent working set(s) and where
   * their folded boundary lands. */
  bindRecurrent(rs: RsWorkingSet[], geom: RsGeometry): void {
    this.requireKind('recurrent', 'bindRecurrent');
    const stagedRs = this.stageRs(rs, geom);
    wit('bindRecurrent', () =>
      (this.wit as witRecurrent.ForwardPass).attention(stagedRs.workingSets, { foldLen: stagedRs.foldLen, buffer: pageSpan(stagedRs.buffer) }),
    );
  }

  /** Kind-independent binding for the common text program: the KV geometry,
   * plus — on a hybrid model — the recurrent working set(s), folding every
   * token straight into the recurrence. */
  bindState(ws: WorkingSet, geom: KvGeometry, rs: RsWorkingSet[] = []): void {
    if (this.kind === 'attention') this.attention(ws, geom);
    else if (this.kind === 'hybrid') this.bindHybrid({ workingSet: ws, geometry: geom }, rs, { buffer: [0, 0] });
    else throw new InferletError('bindState: a recurrent-only model has no KV geometry to bind');
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
  /** Attach the `epilogue` stage (sampling programs; after the forward). */
  epilogue(body: () => void): void {
    this.setStage(Stage.EPILOGUE, body);
  }
  onAttnProj(body: () => void): void {
    if (this.kind === 'recurrent') throw new InferletError('a recurrent-only pass has no attention layer to tap');
    this.setStage(Stage.ON_ATTN_PROJ, body);
  }
  onAttn(body: () => void): void {
    if (this.kind === 'recurrent') throw new InferletError('a recurrent-only pass has no attention layer to tap');
    this.setStage(Stage.ON_ATTN, body);
  }

  attachProgram(): void {
    if (this.programAttached) return;
    const builder = new Builder(this.vocab, this.pageSize);
    for (const [port, ch] of this.ports) builder.bindPort(port, ch.dsl);
    for (const [stage, body] of this.stages) builder.stage(stage, body);
    const traced = builder.build();
    const handles = traced.channels.map(hostChannel);
    wit('program', () => this.wit.program(traced.encode(), handles));
    this.programAttached = true;
  }

  /** Enqueue this pass as a single-slot frame on `on`. */
  submit(on: Pipeline): void {
    submitFrame(on, [this]);
  }

  /** @internal */
  submitWith(on: Pipeline, borrows: (AnyPass | undefined)[]): void {
    wit('submit', () => (this.mod.submit as (on: witPipeline.Pipeline, slots: (AnyPass | undefined)[]) => void)(on.wit, borrows));
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
export function submitFrame(on: Pipeline, slots: (ForwardPass | undefined)[]): void {
  const k = frameSize();
  if (slots.length > k) throw new InferletError(`frame holds ${slots.length} slot(s); model.frame-size() is ${k}`);
  const live = slots.filter((p): p is ForwardPass => p !== undefined);
  for (const p of live) p.attachProgram();
  if (!live.length) return;
  const borrows: (AnyPass | undefined)[] = slots.map((p) => p?.wit);
  while (borrows.length < k) borrows.push(undefined);
  live[0].submitWith(on, borrows);
}

/**
 * Keep the runtime's run-ahead window full while `onToken` consumes results,
 * until `budget` fires submit or `onToken` returns `false`. Returns the run
 * count. `onToken` is synchronous here (host reads block).
 */
export function runAhead(on: Pipeline, fwd: ForwardPass, budget: number, onToken: () => boolean | void): number {
  if (budget === 0) return 0;
  const r = fwd.bindsDeviceMask() ? 1 : liveSlots();
  const windowFrames = Math.max(Math.floor((channelCapacity() - 1) / Math.max(r, 1)), 1);
  let submitted = 0;
  let consumed = 0;
  const submitOneFrame = () => {
    const live = Math.min(r, budget - submitted);
    if (live === 0) return;
    submitFrame(on, new Array(live).fill(fwd));
    submitted += live;
  };
  for (let i = 0; i < windowFrames; i++) {
    if (submitted >= budget) break;
    submitOneFrame();
  }
  let ended = false;
  if (submitted >= budget && !ended) {
    on.close();
    ended = true;
  }
  while (consumed < submitted) {
    if (onToken() === false) {
      if (!ended) on.close();
      return consumed + 1;
    }
    consumed += 1;
    if (submitted < budget && submitted - consumed <= (windowFrames - 1) * r) submitOneFrame();
    if (submitted >= budget && !ended) {
      on.close();
      ended = true;
    }
  }
  if (!ended) on.close();
  return consumed;
}
