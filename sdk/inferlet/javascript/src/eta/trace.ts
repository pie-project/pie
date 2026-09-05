// The trace-recording context — port of `eta-dsl/src/context.rs`.
//
// A module-level session holds the stage currently being traced. Channels
// are plain objects the author holds; a trace interns the ones it touches.
// Single-threaded by construction (wasm inferlets).

import { Dtype, Op, Shape, SinkScope, Stage } from './ir.js';

/** An authoring mistake found while tracing. */
export class TraceError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'TraceError';
  }
}

export interface ValueType {
  readonly shape: Shape;
  readonly dtype: Dtype;
}

export function vt(shape: Shape, dtype: Dtype): ValueType {
  return { shape, dtype };
}

export interface ChannelState {
  gid: number;
  name: string;
  shape: Shape;
  dtype: Dtype;
  capacity: number;
  seeded: boolean;
  hasSeed: boolean;
  progPuts: Stage[];
  progTakes: Stage[];
  progReads: Stage[];
  hostPuts: number;
  hostTakes: number;
  hostReads: number;
  descTakes: number;
  descReads: number;
  /** The bridge's host-side handle (the WIT resource), created on first
   * host use. Opaque to the trace. */
  host: unknown;
}

export function elemTy(st: ChannelState): ValueType {
  return vt(st.shape, st.dtype);
}

export interface SinkCall {
  name: string;
  scope: SinkScope;
}

class Recorder {
  ops: Op[] = [];
  types: ValueType[] = [];
  sinks: SinkCall[] = [];
  constructor(
    public stage: Stage,
    public rows: number,
  ) {}

  push(op: Op, resultTys: readonly ValueType[]): number {
    const base = this.types.length;
    if (op.resultCount !== resultTys.length) {
      throw new Error(
        `result arity mismatch for ${op.name}: recording ${resultTys.length} types against ${op.resultCount} results would shift every later value id`,
      );
    }
    this.types.push(...resultTys);
    this.ops.push(op);
    return base;
  }
}

export interface StageResult {
  stage: Stage;
  ops: Op[];
  sinks: SinkCall[];
}

class Session {
  chanByGid = new Map<number, number>();
  channels: ChannelState[] = [];
  current: Recorder | null = null;
  names: string[] = [];

  intern(ch: ChannelState): number {
    const idx = this.chanByGid.get(ch.gid);
    if (idx !== undefined) return idx;
    const next = this.channels.length;
    this.chanByGid.set(ch.gid, next);
    this.channels.push(ch);
    return next;
  }
}

let session: Session | null = null;
let nextGidCounter = 1;

let modelVocab = 32_000;
let modelPageSize = 16;

export function nextGid(): number {
  return nextGidCounter++;
}

export function isTracing(): boolean {
  return session !== null && session.current !== null;
}

function sess(): Session {
  if (session === null) throw new TraceError('no trace session is active');
  return session;
}

function rec(): Recorder {
  const s = sess();
  if (s.current === null) throw new TraceError('op emitted outside a traced stage');
  return s.current;
}

export function internChannel(ch: ChannelState): number {
  return sess().intern(ch);
}

export function withSession<R>(f: () => R): { result: R; channels: ChannelState[]; names: string[] } {
  if (session !== null) throw new TraceError('nested trace session');
  session = new Session();
  try {
    const result = f();
    return { result, channels: session.channels, names: session.names };
  } finally {
    session = null;
  }
}

export function traceStage(stage: Stage, rows: number, body: () => void): StageResult {
  const s = sess();
  if (s.current !== null) throw new TraceError('nested stage');
  s.current = new Recorder(stage, rows);
  let r: Recorder;
  try {
    body();
  } finally {
    r = s.current;
    s.current = null;
  }
  return { stage: r.stage, ops: r.ops, sinks: r.sinks };
}

export function currentRows(): number {
  if (session === null || session.current === null) return 1;
  return session.current.rows;
}

export function emit(op: Op, resultTys: readonly ValueType[]): number {
  return rec().push(op, resultTys);
}

export function recordChannelRead(ch: ChannelState, consume: boolean): [number, ValueType] {
  const s = sess();
  const dense = s.intern(ch);
  const elem = elemTy(ch);
  const r = rec();
  let op: Op;
  if (consume) {
    ch.progTakes.push(r.stage);
    op = Op.chanTake(dense);
  } else {
    ch.progReads.push(r.stage);
    op = Op.chanRead(dense);
  }
  const vid = r.push(op, [elem]);
  return [vid, elem];
}

export function recordChannelPut(ch: ChannelState, value: number): void {
  const s = sess();
  const dense = s.intern(ch);
  const r = rec();
  const peekedPort = ch.descReads > 0 && ch.descTakes === 0;
  const drain = peekedPort && ch.progTakes.length === 0;
  if (drain) {
    ch.progTakes.push(r.stage);
    r.push(Op.chanTake(dense), [elemTy(ch)]);
  }
  ch.progPuts.push(r.stage);
  r.push(Op.chanPut(dense, value), []);
}

export function internName(name: string): number {
  const s = sess();
  const i = s.names.indexOf(name);
  if (i >= 0) return i;
  s.names.push(name);
  return s.names.length - 1;
}

export function recordSink(name: string, scope: SinkScope): void {
  rec().sinks.push({ name, scope });
}

export function withConstants<R>(vocab: number, pageSize: number, f: () => R): R {
  const prev = [modelVocab, modelPageSize];
  modelVocab = vocab;
  modelPageSize = pageSize;
  try {
    return f();
  } finally {
    [modelVocab, modelPageSize] = prev;
  }
}

export function vocab(): number {
  return modelVocab;
}

export function pageSize(): number {
  return modelPageSize;
}
