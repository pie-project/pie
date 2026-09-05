// ETA IR — the representation layer, ported from `crates/eta-ir`.
//
// Stage-tagged op programs, channel declarations, descriptor-port bindings,
// and the versioned trace container whose canonical bytes are the pass's
// identity (`containerHash`). The bytes agree with the Rust encoder byte for
// byte: the same program traced from JavaScript, Python and Rust hashes to
// the same FNV-1a value and shares the host's program cache.

export enum Dtype {
  F32 = 0,
  I32 = 1,
  U32 = 2,
  BOOL = 3,
}

/** `dtype.f32` / `dtype.i32` / `dtype.u32` / `dtype.bool`. */
export const dtype = {
  f32: Dtype.F32,
  i32: Dtype.I32,
  u32: Dtype.U32,
  bool: Dtype.BOOL,
} as const;

export function dtypeName(d: Dtype): string {
  return ['f32', 'i32', 'u32', 'bool'][d];
}

export function elemSize(d: Dtype): number {
  return d === Dtype.BOOL ? 1 : 4;
}

export const MAX_RANK = 4;

/** A shape: extents outermost first; `[]` is the scalar. Treated as immutable. */
export type Shape = readonly number[];

export const SCALAR: Shape = Object.freeze([]);

export function shapeOf(dims: readonly number[] | number): Shape {
  const shape = typeof dims === 'number' ? [dims] : dims.map((d) => Math.trunc(d));
  if (shape.length > MAX_RANK) {
    throw new Error(`shape [${shape}] has rank ${shape.length}, above MAX_RANK=${MAX_RANK}`);
  }
  for (const d of shape) {
    if (!(d >= 1)) throw new Error(`shape [${shape}] has a non-positive extent; every extent must be >= 1`);
    if (d > 0xffff_ffff) throw new Error(`shape [${shape}] has an extent that does not fit u32`);
  }
  return Object.freeze(shape);
}

export function shapeEq(a: Shape, b: Shape): boolean {
  return a.length === b.length && a.every((d, i) => d === b[i]);
}

export function numel(shape: Shape): number {
  let n = 1;
  for (const d of shape) n *= d;
  return n;
}

export function rows(shape: Shape): number {
  if (shape.length <= 1) return 1;
  return numel(shape.slice(0, -1));
}

export function dropLast(shape: Shape): Shape {
  if (shape.length === 0) return SCALAR;
  return Object.freeze(shape.slice(0, -1));
}

export enum RngKind {
  UNIFORM = 0,
  GUMBEL = 1,
}

// ---------------------------------------------------------------------------
// Registry: stages, ports, intrinsics, sinks
// ---------------------------------------------------------------------------

export enum Stage {
  PROLOGUE = 0,
  ON_ATTN_PROJ = 1,
  ON_ATTN = 2,
  EPILOGUE = 3,
}

export const STAGE_NAMES = ['prologue', 'on_attn_proj', 'on_attn', 'epilogue'] as const;

export enum Port {
  EMBED_TOKENS = 0,
  EMBED_INDPTR = 1,
  POSITIONS = 2,
  PAGES = 3,
  PAGE_INDPTR = 4,
  KV_LEN = 5,
  W_SLOT = 6,
  W_OFF = 7,
  READOUT = 8,
  ATTN_MASK = 9,
  RS_BUFFER_PAGES = 10,
  RS_BUFFER_INDPTR = 11,
  RS_BUFFER_LEN = 12,
  RS_W_SLOT = 13,
  RS_W_OFF = 14,
  RS_FOLD_LEN = 15,
}

/** True iff a channel bound to this port is consumed (take) by the pass. */
export function portConsumes(p: Port): boolean {
  switch (p) {
    case Port.EMBED_TOKENS:
    case Port.POSITIONS:
    case Port.W_SLOT:
    case Port.W_OFF:
    case Port.RS_W_SLOT:
    case Port.RS_W_OFF:
    case Port.RS_FOLD_LEN:
      return true;
    default:
      return false;
  }
}

export function portName(p: Port): string {
  return Port[p].toLowerCase();
}

export enum Intrinsic {
  LOGITS = 0,
  MTP_LOGITS = 1,
  HIDDEN = 2,
  QUERY = 3,
  VALUE_HEAD = 4,
  LAYER = 5,
  MTP_DRAFTS = 6,
  ATTN_SCORE = 7,
}

export enum SinkScope {
  PASS_WIDE = 0,
  ATTENTION = 1,
}

export const ATTN_SCORE_KV_MAX = 2048;

export enum HostRole {
  NONE = 0,
  WRITER = 1,
  READER = 2,
}

export const PRED_RANK_LE = 0;
export const PRED_CUMMASS_LE = 1;
export const PRED_PROB_GE = 2;

// ---------------------------------------------------------------------------
// The op table
// ---------------------------------------------------------------------------

export const enum Field {
  VALUE,
  CHAN,
  IMM,
  DTYPE,
  SHAPE,
  RNG_KIND,
  PREDICATE,
  LITERAL,
  NAME,
  INTRINSIC,
  ARGS,
}

export const tags = {
  EXP: 0x01,
  LOG: 0x02,
  NEG: 0x03,
  RECIP: 0x04,
  ABS: 0x05,
  SIGN: 0x06,
  CAST: 0x07,
  ADD: 0x10,
  SUB: 0x11,
  MUL: 0x12,
  DIV: 0x13,
  MAX_ELEM: 0x14,
  MIN_ELEM: 0x15,
  GT: 0x16,
  GE: 0x17,
  EQ: 0x18,
  NE: 0x19,
  LT: 0x1a,
  LE: 0x1b,
  AND: 0x1c,
  OR: 0x1d,
  NOT: 0x1e,
  REM: 0x1f,
  SELECT: 0x20,
  REDUCE_SUM: 0x30,
  REDUCE_MAX: 0x31,
  REDUCE_MIN: 0x32,
  REDUCE_ARGMAX: 0x33,
  BROADCAST: 0x38,
  RESHAPE: 0x39,
  TRANSPOSE: 0x3a,
  CUMSUM: 0x40,
  CUMPROD: 0x41,
  SORT_DESC: 0x50,
  TOP_K: 0x51,
  MATMUL: 0x55,
  PIVOT_THRESHOLD: 0x58,
  GATHER: 0x60,
  GATHER_ROW: 0x61,
  SCATTER_ADD: 0x62,
  SCATTER_SET: 0x63,
  IOTA: 0x64,
  MASK_APPLY_PACKED: 0x65,
  CAUSAL_MASK: 0x66,
  SLIDING_WINDOW_MASK: 0x67,
  SINK_WINDOW_MASK: 0x68,
  RNG: 0x70,
  RNG_KEYED: 0x71,
  CONST: 0x81,
  CHAN_TAKE: 0x90,
  CHAN_READ: 0x91,
  CHAN_PUT: 0x92,
  INTRINSIC_VAL: 0xa0,
  KERNEL_CALL: 0xa1,
  SINK_CALL: 0xa2,
} as const;

type Row = readonly [name: string, results: number, layout: readonly Field[]];

const V = Field.VALUE;
export const OP_TABLE: ReadonlyMap<number, Row> = new Map<number, Row>([
  [tags.EXP, ['exp', 1, [V]]],
  [tags.LOG, ['log', 1, [V]]],
  [tags.NEG, ['neg', 1, [V]]],
  [tags.RECIP, ['recip', 1, [V]]],
  [tags.ABS, ['abs', 1, [V]]],
  [tags.SIGN, ['sign', 1, [V]]],
  [tags.CAST, ['cast', 1, [V, Field.DTYPE]]],
  [tags.ADD, ['add', 1, [V, V]]],
  [tags.SUB, ['sub', 1, [V, V]]],
  [tags.MUL, ['mul', 1, [V, V]]],
  [tags.DIV, ['div', 1, [V, V]]],
  [tags.MAX_ELEM, ['max_elem', 1, [V, V]]],
  [tags.MIN_ELEM, ['min_elem', 1, [V, V]]],
  [tags.GT, ['gt', 1, [V, V]]],
  [tags.GE, ['ge', 1, [V, V]]],
  [tags.EQ, ['eq', 1, [V, V]]],
  [tags.NE, ['ne', 1, [V, V]]],
  [tags.LT, ['lt', 1, [V, V]]],
  [tags.LE, ['le', 1, [V, V]]],
  [tags.AND, ['and', 1, [V, V]]],
  [tags.OR, ['or', 1, [V, V]]],
  [tags.NOT, ['not', 1, [V]]],
  [tags.REM, ['rem', 1, [V, V]]],
  [tags.SELECT, ['select', 1, [V, V, V]]],
  [tags.REDUCE_SUM, ['reduce_sum', 1, [V]]],
  [tags.REDUCE_MAX, ['reduce_max', 1, [V]]],
  [tags.REDUCE_MIN, ['reduce_min', 1, [V]]],
  [tags.REDUCE_ARGMAX, ['reduce_argmax', 1, [V]]],
  [tags.BROADCAST, ['broadcast', 1, [V, Field.SHAPE]]],
  [tags.RESHAPE, ['reshape', 1, [V, Field.SHAPE]]],
  [tags.TRANSPOSE, ['transpose', 1, [V]]],
  [tags.CUMSUM, ['cumsum', 1, [V]]],
  [tags.CUMPROD, ['cumprod', 1, [V]]],
  [tags.SORT_DESC, ['sort_desc', 2, [V]]],
  [tags.TOP_K, ['top_k', 2, [V, Field.IMM]]],
  [tags.MATMUL, ['matmul', 1, [V, V]]],
  [tags.PIVOT_THRESHOLD, ['pivot_threshold', 1, [V, Field.PREDICATE]]],
  [tags.GATHER, ['gather', 1, [V, V]]],
  [tags.GATHER_ROW, ['gather_row', 1, [V, V]]],
  [tags.SCATTER_ADD, ['scatter_add', 1, [V, V, V]]],
  [tags.SCATTER_SET, ['scatter_set', 1, [V, V, V]]],
  [tags.IOTA, ['iota', 1, [Field.IMM]]],
  [tags.MASK_APPLY_PACKED, ['mask_apply_packed', 1, [V, V]]],
  [tags.CAUSAL_MASK, ['causal_mask', 1, [V, Field.IMM]]],
  [tags.SLIDING_WINDOW_MASK, ['sliding_window_mask', 1, [V, Field.IMM, Field.IMM]]],
  [tags.SINK_WINDOW_MASK, ['sink_window_mask', 1, [V, Field.IMM, Field.IMM, Field.IMM]]],
  [tags.RNG, ['rng', 1, [Field.IMM, Field.SHAPE, Field.RNG_KIND]]],
  [tags.RNG_KEYED, ['rng_keyed', 1, [V, Field.SHAPE, Field.RNG_KIND]]],
  [tags.CONST, ['const', 1, [Field.LITERAL]]],
  [tags.CHAN_TAKE, ['chan_take', 1, [Field.CHAN]]],
  [tags.CHAN_READ, ['chan_read', 1, [Field.CHAN]]],
  [tags.CHAN_PUT, ['chan_put', 0, [Field.CHAN, V]]],
  [tags.INTRINSIC_VAL, ['intrinsic_val', 1, [Field.INTRINSIC, Field.DTYPE, Field.SHAPE]]],
  [tags.KERNEL_CALL, ['kernel_call', 1, [Field.NAME, Field.DTYPE, Field.SHAPE, Field.ARGS]]],
  [tags.SINK_CALL, ['sink_call', 0, [Field.NAME, Field.ARGS]]],
]);

/** One op as its flat wire record — `eta_ir::wire::OpWire`. */
export class Op {
  tag: number;
  args: number[];
  chan = -1;
  imms: number[] = [];
  dtype = 0;
  shape: Shape = SCALAR;
  kind = 0;
  predTag = 0;
  predPayload = 0;
  litDtype = 0;
  litBits = 0;
  nameIdx = 0;
  intr = 0;

  constructor(tag: number, args: number[] = []) {
    this.tag = tag;
    this.args = args;
  }

  get name(): string {
    return OP_TABLE.get(this.tag)![0];
  }

  get resultCount(): number {
    return OP_TABLE.get(this.tag)![1];
  }

  static unary(tag: number, a: number): Op {
    return new Op(tag, [a]);
  }
  static binary(tag: number, a: number, b: number): Op {
    return new Op(tag, [a, b]);
  }
  static ternary(tag: number, a: number, b: number, c: number): Op {
    return new Op(tag, [a, b, c]);
  }
  static const(dt: Dtype, value: number | boolean): Op {
    const op = new Op(tags.CONST);
    op.litDtype = dt;
    op.litBits = literalBits(dt, value);
    return op;
  }
  static cast(value: number, dt: Dtype): Op {
    const op = new Op(tags.CAST, [value]);
    op.dtype = dt;
    return op;
  }
  static reshape(value: number, shape: Shape): Op {
    const op = new Op(tags.RESHAPE, [value]);
    op.shape = shape;
    return op;
  }
  static broadcast(value: number, shape: Shape): Op {
    const op = new Op(tags.BROADCAST, [value]);
    op.shape = shape;
    return op;
  }
  static iota(length: number): Op {
    const op = new Op(tags.IOTA);
    op.imms = [length];
    return op;
  }
  static topK(value: number, k: number): Op {
    const op = new Op(tags.TOP_K, [value]);
    op.imms = [k];
    return op;
  }
  static pivotThreshold(value: number, predTag: number, predValue: number): Op {
    const op = new Op(tags.PIVOT_THRESHOLD, [value]);
    op.predTag = predTag;
    op.predPayload = predValue;
    return op;
  }
  static causalMask(positions: number, length: number): Op {
    const op = new Op(tags.CAUSAL_MASK, [positions]);
    op.imms = [length];
    return op;
  }
  static slidingWindowMask(positions: number, length: number, window: number): Op {
    const op = new Op(tags.SLIDING_WINDOW_MASK, [positions]);
    op.imms = [length, window];
    return op;
  }
  static sinkWindowMask(positions: number, length: number, sink: number, window: number): Op {
    const op = new Op(tags.SINK_WINDOW_MASK, [positions]);
    op.imms = [length, sink, window];
    return op;
  }
  static rng(stream: number, shape: Shape, kind: RngKind): Op {
    const op = new Op(tags.RNG);
    op.imms = [stream];
    op.shape = shape;
    op.kind = kind;
    return op;
  }
  static rngKeyed(state: number, shape: Shape, kind: RngKind): Op {
    const op = new Op(tags.RNG_KEYED, [state]);
    op.shape = shape;
    op.kind = kind;
    return op;
  }
  static chanTake(chan: number): Op {
    const op = new Op(tags.CHAN_TAKE);
    op.chan = chan;
    return op;
  }
  static chanRead(chan: number): Op {
    const op = new Op(tags.CHAN_READ);
    op.chan = chan;
    return op;
  }
  static chanPut(chan: number, value: number): Op {
    const op = new Op(tags.CHAN_PUT, [value]);
    op.chan = chan;
    return op;
  }
  static intrinsicVal(intr: Intrinsic, shape: Shape, dt: Dtype): Op {
    const op = new Op(tags.INTRINSIC_VAL);
    op.intr = intr;
    op.shape = shape;
    op.dtype = dt;
    return op;
  }
  static kernelCall(name: number, args: number[], shape: Shape, dt: Dtype): Op {
    const op = new Op(tags.KERNEL_CALL, [...args]);
    op.nameIdx = name;
    op.shape = shape;
    op.dtype = dt;
    return op;
  }
  static sinkCall(name: number, args: number[]): Op {
    const op = new Op(tags.SINK_CALL, [...args]);
    op.nameIdx = name;
    return op;
  }
}

const scratch = new DataView(new ArrayBuffer(8));

/** The 4 raw payload bytes of a `const` literal, as a u32. */
export function literalBits(dt: Dtype, value: number | boolean): number {
  switch (dt) {
    case Dtype.F32:
      scratch.setFloat32(0, Number(value), true);
      return scratch.getUint32(0, true);
    case Dtype.I32:
      scratch.setInt32(0, Number(value) | 0, true);
      return scratch.getUint32(0, true);
    case Dtype.U32: {
      const v = Number(value);
      if (!(v >= 0 && v <= 0xffff_ffff && Number.isInteger(v))) throw new Error(`${v} does not fit u32`);
      return v;
    }
    case Dtype.BOOL:
      return value ? 1 : 0;
  }
}

// ---------------------------------------------------------------------------
// Container
// ---------------------------------------------------------------------------

export const ETA_MAGIC = new Uint8Array([0x45, 0x54, 0x41, 0x00]); // "ETA\0"
export const ETA_VERSION = 1;
export const ETA_VERSION_EXTERN = 2;

export interface ChannelDecl {
  shape: Shape;
  dtype: Dtype;
  capacity: number;
  hostRole: HostRole;
  seeded: boolean;
}

export interface PortBinding {
  port: Port;
  channel: number;
}

export interface StageProgram {
  stage: Stage;
  ops: Op[];
}

export interface ExternDecl {
  name: number;
  direction: number;
  chan: number;
}

export interface TraceContainer {
  names: string[];
  channels: ChannelDecl[];
  ports: PortBinding[];
  stages: StageProgram[];
  externs: ExternDecl[];
}

class Writer {
  private buf = new Uint8Array(1024);
  private len = 0;

  private grow(n: number) {
    if (this.len + n <= this.buf.length) return;
    let cap = this.buf.length * 2;
    while (cap < this.len + n) cap *= 2;
    const next = new Uint8Array(cap);
    next.set(this.buf.subarray(0, this.len));
    this.buf = next;
  }
  u8(v: number) {
    if (!(v >= 0 && v <= 0xff)) throw new Error(`${v} exceeds its u8 wire width`);
    this.grow(1);
    this.buf[this.len++] = v;
  }
  u16(v: number) {
    if (!(v >= 0 && v <= 0xffff)) throw new Error(`${v} exceeds its u16 wire width`);
    this.grow(2);
    this.buf[this.len++] = v & 0xff;
    this.buf[this.len++] = (v >>> 8) & 0xff;
  }
  u32(v: number) {
    if (!(v >= 0 && v <= 0xffff_ffff)) throw new Error(`${v} exceeds its u32 wire width`);
    this.grow(4);
    this.buf[this.len++] = v & 0xff;
    this.buf[this.len++] = (v >>> 8) & 0xff;
    this.buf[this.len++] = (v >>> 16) & 0xff;
    this.buf[this.len++] = (v >>> 24) & 0xff;
  }
  bytes(b: Uint8Array) {
    this.grow(b.length);
    this.buf.set(b, this.len);
    this.len += b.length;
  }
  finish(): Uint8Array {
    return this.buf.slice(0, this.len);
  }
}

function encodeShape(w: Writer, shape: Shape) {
  w.u8(shape.length);
  for (const d of shape) w.u32(d);
}

export function encodeOp(w: Writer, op: Op): void {
  w.u8(op.tag);
  const layout = OP_TABLE.get(op.tag)![2];
  let value = 0;
  let imm = 0;
  for (const f of layout) {
    switch (f) {
      case Field.VALUE:
        w.u32(op.args[value++]);
        break;
      case Field.CHAN:
        if (op.chan < 0) throw new Error(`${op.name} carries no channel index`);
        w.u32(op.chan);
        break;
      case Field.IMM:
        w.u32(op.imms[imm++]);
        break;
      case Field.DTYPE:
        w.u8(op.dtype);
        break;
      case Field.SHAPE:
        encodeShape(w, op.shape);
        break;
      case Field.RNG_KIND:
        w.u8(op.kind);
        break;
      case Field.PREDICATE:
        w.u8(op.predTag);
        w.u32(op.predPayload);
        break;
      case Field.LITERAL:
        w.u8(op.litDtype);
        w.u32(op.litBits);
        break;
      case Field.NAME:
        w.u16(op.nameIdx);
        break;
      case Field.INTRINSIC:
        w.u16(op.intr);
        break;
      case Field.ARGS: {
        const rest = op.args.slice(value);
        w.u8(rest.length);
        for (const a of rest) w.u32(a);
        break;
      }
    }
  }
}

const utf8 = new TextEncoder();

export function encode(c: TraceContainer): Uint8Array {
  const w = new Writer();
  w.bytes(ETA_MAGIC);
  const v2 = c.externs.length > 0;
  w.u16(v2 ? ETA_VERSION_EXTERN : ETA_VERSION);
  w.u16(0);
  w.u32(c.names.length);
  w.u32(c.channels.length);
  w.u32(c.ports.length);
  w.u32(c.stages.length);
  if (v2) w.u32(c.externs.length);
  for (const n of c.names) {
    const b = utf8.encode(n);
    w.u16(b.length);
    w.bytes(b);
  }
  for (const ch of c.channels) {
    w.u8(ch.dtype);
    encodeShape(w, ch.shape);
    w.u32(ch.capacity);
    w.u8(ch.hostRole);
    w.u8(ch.seeded ? 1 : 0);
  }
  for (const p of c.ports) {
    w.u8(p.port);
    w.u8(0);
    w.u32(p.channel);
  }
  for (const s of c.stages) {
    w.u8(s.stage);
    w.u32(s.ops.length);
    for (const op of s.ops) encodeOp(w, op);
  }
  for (const e of c.externs) {
    w.u16(e.name);
    w.u8(e.direction);
    w.u32(e.chan);
  }
  return w.finish();
}

const FNV_OFFSET = 0xcbf29ce484222325n;
const FNV_PRIME = 0x100000001b3n;
const MASK64 = 0xffff_ffff_ffff_ffffn;

export function fnv1a64(data: Uint8Array): bigint {
  let h = FNV_OFFSET;
  for (const b of data) {
    h ^= BigInt(b);
    h = (h * FNV_PRIME) & MASK64;
  }
  return h;
}

/** FNV-1a 64 over the canonical container bytes — the pass's identity. */
export function containerHash(containerBytes: Uint8Array): bigint {
  return fnv1a64(containerBytes);
}

// ---------------------------------------------------------------------------
// Expansions (`eta_ir::expand`)
// ---------------------------------------------------------------------------

export const enum Step {
  ROW,
  REDUCED,
  SCALAR,
  ROW_MASK,
  REDUCED_INDEX,
}

export type Push = (op: Op, step: Step) => number;

export function expandGumbel(push: Push, state: number, shape: Shape): number {
  return push(Op.rngKeyed(state, shape, RngKind.GUMBEL), Step.ROW);
}

export function expandMaskApply(push: Push, logits: number, mask: number): number {
  const ninf = push(Op.const(Dtype.F32, -Infinity), Step.SCALAR);
  return push(Op.ternary(tags.SELECT, mask, logits, ninf), Step.ROW);
}

export function expandSoftmax(push: Push, x: number, shape: Shape): number {
  const m = push(Op.unary(tags.REDUCE_MAX, x), Step.REDUCED);
  const mb = push(Op.broadcast(m, shape), Step.ROW);
  const c = push(Op.binary(tags.SUB, x, mb), Step.ROW);
  const e = push(Op.unary(tags.EXP, c), Step.ROW);
  const s = push(Op.unary(tags.REDUCE_SUM, e), Step.REDUCED);
  const sb = push(Op.broadcast(s, shape), Step.ROW);
  return push(Op.binary(tags.DIV, e, sb), Step.ROW);
}

export function expandLogSoftmax(push: Push, x: number, shape: Shape): number {
  const m = push(Op.unary(tags.REDUCE_MAX, x), Step.REDUCED);
  const mb = push(Op.broadcast(m, shape), Step.ROW);
  const c = push(Op.binary(tags.SUB, x, mb), Step.ROW);
  const e = push(Op.unary(tags.EXP, c), Step.ROW);
  const s = push(Op.unary(tags.REDUCE_SUM, e), Step.REDUCED);
  const lg = push(Op.unary(tags.LOG, s), Step.REDUCED);
  const lb = push(Op.broadcast(lg, shape), Step.ROW);
  return push(Op.binary(tags.SUB, c, lb), Step.ROW);
}

export function expandL2norm(push: Push, x: number, shape: Shape): number {
  const sq = push(Op.binary(tags.MUL, x, x), Step.ROW);
  const s = push(Op.unary(tags.REDUCE_SUM, sq), Step.REDUCED);
  const lg = push(Op.unary(tags.LOG, s), Step.REDUCED);
  const half = push(Op.const(Dtype.F32, 0.5), Step.SCALAR);
  const h = push(Op.binary(tags.MUL, lg, half), Step.REDUCED);
  const rt = push(Op.unary(tags.EXP, h), Step.REDUCED);
  const rb = push(Op.broadcast(rt, shape), Step.ROW);
  return push(Op.binary(tags.DIV, x, rb), Step.ROW);
}

export function expandNucleusSample(push: Push, logits: number, topP: number, state: number, shape: Shape): number {
  const probabilities = expandSoftmax(push, logits, shape);
  const keep = push(Op.pivotThreshold(probabilities, PRED_CUMMASS_LE, topP), Step.ROW_MASK);
  const masked = expandMaskApply(push, logits, keep);
  const noise = expandGumbel(push, state, shape);
  const perturbed = push(Op.binary(tags.ADD, masked, noise), Step.ROW);
  return push(Op.unary(tags.REDUCE_ARGMAX, perturbed), Step.REDUCED_INDEX);
}
