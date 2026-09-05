// `Tensor` — an SSA value — plus the free-function op surface. Port of
// `eta-dsl/src/value.rs`: every op emits the IR's canonical `Op`, and the
// composed ops inline the IR's expansions so the emitted op stream is
// identical to the Rust SDK's.
//
// JavaScript has no operator overloading, so arithmetic is spelled as
// methods: `a.add(b)`, `a.sub(1)`, `a.div(pageSize)`, `a.rem(pageSize)`,
// `a.neg()`, `a.divCeil(pageSize)`. A JS `number` operand takes the dtype of
// the tensor it is combined with (an integer-valued number defaults to i32,
// a fractional one to f32 when there is no partner).

import {
  Dtype,
  Intrinsic,
  MAX_RANK,
  Op,
  PRED_CUMMASS_LE,
  PRED_PROB_GE,
  PRED_RANK_LE,
  RngKind,
  SCALAR,
  Shape,
  Step,
  dropLast,
  dtypeName,
  elemSize,
  expandL2norm,
  expandLogSoftmax,
  expandMaskApply,
  expandNucleusSample,
  expandSoftmax,
  numel,
  shapeEq,
  shapeOf,
  tags,
} from './ir.js';
import { TraceError, ValueType, emit, vt } from './trace.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** A trace-known constant value: raw little-endian element bytes. */
export class ConstData {
  constructor(
    public shape: Shape,
    public dtype: Dtype,
    public data: Uint8Array,
  ) {}

  elem(i: number): number {
    return elemAt(this.dtype, this.data, i);
  }
}

function packScalar(v: number | boolean, dt: Dtype): Uint8Array {
  const out = new Uint8Array(elemSize(dt));
  const dv = new DataView(out.buffer);
  switch (dt) {
    case Dtype.F32:
      dv.setFloat32(0, Number(v), true);
      break;
    case Dtype.I32:
      dv.setInt32(0, Number(v) | 0, true);
      break;
    case Dtype.U32:
      dv.setUint32(0, Number(v) >>> 0, true);
      break;
    case Dtype.BOOL:
      out[0] = v ? 1 : 0;
      break;
  }
  return out;
}

function elemAt(dt: Dtype, data: Uint8Array, i: number): number {
  if (dt === Dtype.BOOL) return data[i] ? 1 : 0;
  const dv = new DataView(data.buffer, data.byteOffset, data.byteLength);
  switch (dt) {
    case Dtype.F32:
      return dv.getFloat32(i * 4, true);
    case Dtype.I32:
      return dv.getInt32(i * 4, true);
    default:
      return dv.getUint32(i * 4, true);
  }
}

/** `values` as a little-endian payload of `dt` elements. */
export function packElems(values: ArrayLike<number | boolean>, dt: Dtype): Uint8Array {
  const n = values.length;
  const out = new Uint8Array(n * elemSize(dt));
  if (dt === Dtype.BOOL) {
    for (let i = 0; i < n; i++) out[i] = values[i] ? 1 : 0;
    return out;
  }
  const dv = new DataView(out.buffer);
  for (let i = 0; i < n; i++) {
    const v = Number(values[i]);
    if (dt === Dtype.F32) dv.setFloat32(i * 4, v, true);
    else if (dt === Dtype.I32) dv.setInt32(i * 4, v | 0, true);
    else dv.setUint32(i * 4, v >>> 0, true);
  }
  return out;
}

/** The inverse of `packElems`: a payload as an array of numbers/booleans. */
export function unpackElems(data: Uint8Array, dt: Dtype): number[] | boolean[] {
  if (dt === Dtype.BOOL) return Array.from(data, (b) => b !== 0);
  const n = Math.floor(data.byteLength / 4);
  const dv = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    out[i] = dt === Dtype.F32 ? dv.getFloat32(i * 4, true) : dt === Dtype.I32 ? dv.getInt32(i * 4, true) : dv.getUint32(i * 4, true);
  }
  return out;
}

export type ConstLike = number | boolean | ConstData | ArrayLike<number | boolean> | Iterable<number | boolean>;

/**
 * Coerce a JS value into a `ConstData`. A `boolean` → scalar bool; an
 * integer-valued `number` → i32 (or `dt`); a fractional one → f32. A
 * sequence of booleans → `[n] bool`; a sequence of numbers **needs an
 * explicit `dt`** (`dtype.i32` for tokens, `dtype.u32` for geometry), the
 * way the Rust SDK needs a literal suffix.
 */
export function constData(v: ConstLike, dt?: Dtype): ConstData {
  if (v instanceof ConstData) {
    if (dt !== undefined && dt !== v.dtype) throw new TypeError(`constant is ${dtypeName(v.dtype)}, asked for ${dtypeName(dt)}`);
    return v;
  }
  if (v instanceof Tensor) throw new TypeError('a Tensor is not a constant');
  if (typeof v === 'boolean') {
    const d = dt ?? Dtype.BOOL;
    return new ConstData(SCALAR, d, packScalar(v, d));
  }
  if (typeof v === 'number') {
    const d = dt ?? (Number.isInteger(v) ? Dtype.I32 : Dtype.F32);
    return new ConstData(SCALAR, d, packScalar(v, d));
  }
  if (v instanceof Uint8Array && dt !== undefined) {
    const n = Math.floor(v.length / elemSize(dt));
    if (n * elemSize(dt) !== v.length) throw new Error(`${v.length} bytes is not a whole number of ${dtypeName(dt)} elements`);
    return new ConstData(n ? shapeOf([n]) : SCALAR, dt, new Uint8Array(v));
  }
  const items: (number | boolean)[] = Array.from(v as Iterable<number | boolean>);
  let d = dt;
  if (d === undefined) {
    if (items.length && items.every((x) => typeof x === 'boolean')) d = Dtype.BOOL;
    else if (items.length && items.every((x) => typeof x === 'number'))
      throw new TypeError('a number sequence needs an explicit dtype (dtype.i32 for tokens, dtype.u32 for geometry)');
    else throw new TypeError('cannot infer a dtype for this sequence; pass a dtype');
  }
  if (items.length === 0) throw new Error('a constant needs at least one element (every extent is >= 1)');
  return new ConstData(shapeOf([items.length]), d, packElems(items, d));
}

// ---------------------------------------------------------------------------
// Tensor
// ---------------------------------------------------------------------------

export type Operand = Tensor | number | boolean | ConstData;

export class Tensor {
  private constructor(
    private readonly id: number | null,
    private readonly ty_: ValueType | null,
    readonly konst: ConstData | null,
  ) {}

  static node(vid: number, ty: ValueType): Tensor {
    return new Tensor(vid, ty, null);
  }

  /** A trace-known constant. Only a uniform tensor (broadcast) or a `u32`
   * affine ramp (iota) can lower; bulk data belongs in a seeded channel. */
  static constant(v: ConstLike, dt?: Dtype): Tensor {
    return new Tensor(null, null, constData(v, dt));
  }

  static fromConst(c: ConstData): Tensor {
    return new Tensor(null, null, c);
  }

  get isConst(): boolean {
    return this.konst !== null;
  }

  get ty(): ValueType {
    if (this.konst !== null) return vt(this.konst.shape, this.konst.dtype);
    return this.ty_!;
  }

  get dtype(): Dtype {
    return this.ty.dtype;
  }

  get shape(): Shape {
    return this.ty.shape;
  }

  /** @internal */
  nodeId(): number {
    return this.id!;
  }

  toString(): string {
    const t = this.ty;
    return this.konst ? `Tensor(const ${dtypeName(t.dtype)}[${t.shape}])` : `Tensor(%${this.id}: ${dtypeName(t.dtype)}[${t.shape}])`;
  }

  add(o: Operand): Tensor {
    return add(this, o);
  }
  sub(o: Operand): Tensor {
    return sub(this, o);
  }
  mul(o: Operand): Tensor {
    return mul(this, o);
  }
  div(o: Operand): Tensor {
    return div(this, o);
  }
  rem(o: Operand): Tensor {
    return rem(this, o);
  }
  neg(): Tensor {
    return neg(this);
  }

  /** Ceiling division, spelled like `u32::div_ceil`. */
  divCeil(rhs: Operand): Tensor {
    const d = toArg(rhs);
    const v = constScalar(d);
    if (v !== null) {
      const oneLess = Tensor.fromConst(new ConstData(SCALAR, d.dtype, packScalar(v - 1, d.dtype)));
      return this.add(oneLess).div(d);
    }
    return this.add(d).sub(constant(1, Dtype.U32)).div(d);
  }
}

export function constant(v: ConstLike, dt?: Dtype): Tensor {
  return Tensor.constant(v, dt);
}

// ---------------------------------------------------------------------------
// Operand plumbing
// ---------------------------------------------------------------------------

function toArg(x: Operand): Tensor {
  if (x instanceof Tensor) return x;
  if (typeof x === 'number' || typeof x === 'boolean' || x instanceof ConstData) return Tensor.fromConst(constData(x));
  throw new TypeError(`${typeof x} is not a tensor operand`);
}

function constScalar(a: Tensor): number | null {
  const c = a.konst;
  if (c === null || c.shape.length !== 0) return null;
  return c.elem(0);
}

function scalarLiteralOp(dt: Dtype, data: Uint8Array): Op {
  return Op.const(dt, dt === Dtype.BOOL ? data[0] !== 0 : elemAt(dt, data, 0));
}

function materializeConst(c: ConstData): [number, ValueType] {
  const ty = vt(c.shape, c.dtype);
  if (c.shape.length === 0) {
    const vid = emit(scalarLiteralOp(c.dtype, c.data), [vt(SCALAR, c.dtype)]);
    return [vid, ty];
  }
  const n = numel(c.shape);
  const vals: number[] = [];
  for (let i = 0; i < n; i++) vals.push(c.elem(i));
  if (vals.length && vals.every((v) => v === vals[0])) {
    const s = emit(scalarLiteralOp(c.dtype, c.data.subarray(0, elemSize(c.dtype))), [vt(SCALAR, c.dtype)]);
    const vid = emit(Op.broadcast(s, c.shape), [ty]);
    return [vid, ty];
  }
  if (c.dtype === Dtype.U32 && n >= 2) {
    const a = vals[0];
    const b = vals[1] - vals[0];
    if (b >= 0 && vals.every((v, i) => v === a + b * i)) {
      let cur = emit(Op.iota(n), [ty]);
      if (b !== 1) {
        const bc = emit(Op.const(Dtype.U32, b), [vt(SCALAR, Dtype.U32)]);
        cur = emit(Op.binary(tags.MUL, cur, bc), [ty]);
      }
      if (a !== 0) {
        const ac = emit(Op.const(Dtype.U32, a), [vt(SCALAR, Dtype.U32)]);
        cur = emit(Op.binary(tags.ADD, cur, ac), [ty]);
      }
      return [cur, ty];
    }
  }
  throw new TraceError(
    `a ${dtypeName(c.dtype)} constant of shape [${c.shape}] is bulk data, and the op set carries constants as scalars: ` +
      '`const` holds one literal, so only a uniform tensor (broadcast) and a u32 affine ramp a+b*i (iota) are ' +
      'reachable from it. Seed a channel with the values and read it in the body — `Channel.from(values)` — or ' +
      'build the tensor from an arithmetic expression',
  );
}

/** Resolve an operand to an SSA id + type inside a traced stage. */
export function materialize(x: Operand): [number, ValueType] {
  const a = toArg(x);
  if (a.konst !== null) return materializeConst(a.konst);
  return [a.nodeId(), a.ty];
}

function coerce(c: ConstData, to: Dtype): ConstData | null {
  if (c.dtype === to || c.shape.length !== 0) return null;
  return new ConstData(SCALAR, to, packScalar(c.elem(0), to));
}

function reconcile(a: Tensor, b: Tensor): [Tensor, Tensor] {
  if (a.konst !== null && b.konst === null) {
    const c = coerce(a.konst, b.dtype);
    if (c !== null) return [Tensor.fromConst(c), b];
  } else if (a.konst === null && b.konst !== null) {
    const c = coerce(b.konst, a.dtype);
    if (c !== null) return [a, Tensor.fromConst(c)];
  }
  return [a, b];
}

function nonScalarShape(a: Shape, b: Shape): Shape {
  return a.length === 0 ? b : a;
}

function emitUnary(x: Operand, tag: number, out: (t: ValueType) => ValueType): Tensor {
  const [vid, ty] = materialize(x);
  const rty = out(ty);
  return Tensor.node(emit(Op.unary(tag, vid), [rty]), rty);
}

function emitBinary(a: Operand, b: Operand, tag: number, resultDtype: (d: Dtype) => Dtype): Tensor {
  const [aa, bb] = reconcile(toArg(a), toArg(b));
  const shape = nonScalarShape(aa.shape, bb.shape);
  const [ia, tya] = materialize(aa);
  const [ib] = materialize(bb);
  const rty = vt(shape, resultDtype(tya.dtype));
  return Tensor.node(emit(Op.binary(tag, ia, ib), [rty]), rty);
}

const same = (t: ValueType) => t;
const identityDtype = (d: Dtype) => d;
const boolDtype = () => Dtype.BOOL;
const reduced = (t: ValueType) => vt(dropLast(t.shape), t.dtype);

// ---------------------------------------------------------------------------
// The free-function op surface
// ---------------------------------------------------------------------------

export const neg = (x: Operand) => emitUnary(x, tags.NEG, same);
export const abs = (x: Operand) => emitUnary(x, tags.ABS, same);
export const sign = (x: Operand) => emitUnary(x, tags.SIGN, same);
export const recip = (x: Operand) => emitUnary(x, tags.RECIP, same);
export const exp = (x: Operand) => emitUnary(x, tags.EXP, same);
export const log = (x: Operand) => emitUnary(x, tags.LOG, same);

/** `x` converted elementwise to `to`; the identity cast emits nothing. */
export function cast(x: Operand, to: Dtype): Tensor {
  const t = toArg(x);
  if (t.dtype === to) return t;
  const [vid, ty] = materialize(t);
  const rty = vt(ty.shape, to);
  return Tensor.node(emit(Op.cast(vid, to), [rty]), rty);
}

export const add = (a: Operand, b: Operand) => emitBinary(a, b, tags.ADD, identityDtype);
export const sub = (a: Operand, b: Operand) => emitBinary(a, b, tags.SUB, identityDtype);
export const mul = (a: Operand, b: Operand) => emitBinary(a, b, tags.MUL, identityDtype);
export const div = (a: Operand, b: Operand) => emitBinary(a, b, tags.DIV, identityDtype);
export const rem = (a: Operand, b: Operand) => emitBinary(a, b, tags.REM, identityDtype);
export const maxElem = (a: Operand, b: Operand) => emitBinary(a, b, tags.MAX_ELEM, identityDtype);
export const minElem = (a: Operand, b: Operand) => emitBinary(a, b, tags.MIN_ELEM, identityDtype);
export const eq = (a: Operand, b: Operand) => emitBinary(a, b, tags.EQ, boolDtype);
export const ne = (a: Operand, b: Operand) => emitBinary(a, b, tags.NE, boolDtype);
export const lt = (a: Operand, b: Operand) => emitBinary(a, b, tags.LT, boolDtype);
export const le = (a: Operand, b: Operand) => emitBinary(a, b, tags.LE, boolDtype);
export const gt = (a: Operand, b: Operand) => emitBinary(a, b, tags.GT, boolDtype);
export const ge = (a: Operand, b: Operand) => emitBinary(a, b, tags.GE, boolDtype);
export const and = (a: Operand, b: Operand) => emitBinary(a, b, tags.AND, boolDtype);
export const or = (a: Operand, b: Operand) => emitBinary(a, b, tags.OR, boolDtype);
export const not = (x: Operand) => emitUnary(x, tags.NOT, (t) => vt(t.shape, Dtype.BOOL));

export function select(cond: Operand, a: Operand, b: Operand): Tensor {
  const [ca] = materialize(cond);
  const [aa, bb] = reconcile(toArg(a), toArg(b));
  const shape = nonScalarShape(aa.shape, bb.shape);
  const [ia, tya] = materialize(aa);
  const [ib] = materialize(bb);
  const rty = vt(shape, tya.dtype);
  return Tensor.node(emit(Op.ternary(tags.SELECT, ca, ia, ib), [rty]), rty);
}

export function reshape(x: Operand, shape: readonly number[]): Tensor {
  const s = shapeOf(shape);
  const [vid, ty] = materialize(x);
  const rty = vt(s, ty.dtype);
  return Tensor.node(emit(Op.reshape(vid, s), [rty]), rty);
}

export function broadcast(x: Operand, shape: readonly number[]): Tensor {
  const s = shapeOf(shape);
  const [vid, ty] = materialize(x);
  const rty = vt(s, ty.dtype);
  return Tensor.node(emit(Op.broadcast(vid, s), [rty]), rty);
}

export function transpose(x: Operand): Tensor {
  return emitUnary(x, tags.TRANSPOSE, (t) => {
    const d = t.shape;
    const s = d.length === 2 ? shapeOf([d[1], d[0]]) : d;
    return vt(s, t.dtype);
  });
}

export function iota(length: number): Tensor {
  const ty = vt(shapeOf([length]), Dtype.U32);
  return Tensor.node(emit(Op.iota(length), [ty]), ty);
}

/** The CSR row-offset vector for `rows` runs of equal length `runLen`. */
export function indptr(rows: number, runLen: Operand): Tensor {
  const n = rows + 1;
  return mul(iota(n), broadcast(runLen, [n]));
}

export function gather(src: Operand, idx: Operand): Tensor {
  const [is, tys] = materialize(src);
  const [ii, tyi] = materialize(idx);
  const dims = [...tyi.shape, ...tys.shape.slice(Math.min(tys.shape.length, 1))];
  let rshape: Shape;
  try {
    rshape = shapeOf(dims);
  } catch {
    throw new TraceError(`gather of [${tys.shape}] by [${tyi.shape}] has result shape [${dims}], whose rank exceeds ${MAX_RANK}`);
  }
  const rty = vt(rshape, tys.dtype);
  return Tensor.node(emit(Op.binary(tags.GATHER, is, ii), [rty]), rty);
}

export function gatherRow(src: Operand, idx: Operand): Tensor {
  const [is, tys] = materialize(src);
  const [ii] = materialize(idx);
  const m = tys.shape.length ? tys.shape[0] : 0;
  const rty = vt(shapeOf([m]), tys.dtype);
  return Tensor.node(emit(Op.binary(tags.GATHER_ROW, is, ii), [rty]), rty);
}

// A scalar-literal `vals` takes the base's dtype (a JS `1` is `1.0` into an
// f32 base), the way the Rust author's literal suffix would say.
function scatter(tag: number, base: Operand, idx: Operand, vals: Operand): Tensor {
  const [bb, vv] = reconcile(toArg(base), toArg(vals));
  const [ib, tyb] = materialize(bb);
  const [ii] = materialize(idx);
  const [iv] = materialize(vv);
  return Tensor.node(emit(Op.ternary(tag, ib, ii, iv), [tyb]), tyb);
}

export const scatterSet = (base: Operand, idx: Operand, vals: Operand) => scatter(tags.SCATTER_SET, base, idx, vals);
export const scatterAdd = (base: Operand, idx: Operand, vals: Operand) => scatter(tags.SCATTER_ADD, base, idx, vals);

export const reduceSum = (x: Operand) => emitUnary(x, tags.REDUCE_SUM, reduced);
export const reduceMax = (x: Operand) => emitUnary(x, tags.REDUCE_MAX, reduced);
export const reduceMin = (x: Operand) => emitUnary(x, tags.REDUCE_MIN, reduced);
export const reduceArgmax = (x: Operand) => emitUnary(x, tags.REDUCE_ARGMAX, (t) => vt(dropLast(t.shape), Dtype.I32));
export const cumsum = (x: Operand) => emitUnary(x, tags.CUMSUM, same);
export const cumprod = (x: Operand) => emitUnary(x, tags.CUMPROD, same);

// -- expansions ---------------------------------------------------------------

function tracedPush(row: ValueType) {
  const red = vt(dropLast(row.shape), row.dtype);
  return (op: Op, step: Step): number => {
    let ty: ValueType;
    switch (step) {
      case Step.ROW:
        ty = row;
        break;
      case Step.REDUCED:
        ty = red;
        break;
      case Step.SCALAR:
        ty = vt(SCALAR, Dtype.F32);
        break;
      case Step.ROW_MASK:
        ty = vt(row.shape, Dtype.BOOL);
        break;
      default:
        ty = vt(red.shape, Dtype.I32);
    }
    return emit(op, [ty]);
  };
}

function expanded(x: Operand, seq: (push: ReturnType<typeof tracedPush>, x: number, shape: Shape) => number): Tensor {
  const [xid, ty] = materialize(x);
  const row = vt(ty.shape, Dtype.F32);
  return Tensor.node(seq(tracedPush(row), xid, ty.shape), row);
}

export const softmax = (x: Operand) => expanded(x, expandSoftmax);
export const logSoftmax = (x: Operand) => expanded(x, expandLogSoftmax);
export const l2norm = (x: Operand) => expanded(x, expandL2norm);

// -- order --------------------------------------------------------------------

export function topK(x: Operand, k: number): [Tensor, Tensor] {
  const [ix, tyx] = materialize(x);
  const dims = [...tyx.shape];
  if (dims.length) dims[dims.length - 1] = k;
  let outShape: Shape;
  try {
    outShape = shapeOf(dims);
  } catch {
    outShape = shapeOf([k]);
  }
  const valTy = vt(outShape, tyx.dtype);
  const idxTy = vt(outShape, Dtype.U32);
  const base = emit(Op.topK(ix, k), [valTy, idxTy]);
  return [Tensor.node(base, valTy), Tensor.node(base + 1, idxTy)];
}

export function sortDesc(x: Operand): [Tensor, Tensor] {
  const [ix, tyx] = materialize(x);
  const n = tyx.shape.length ? tyx.shape[tyx.shape.length - 1] : 0;
  const valTy = vt(shapeOf([n]), Dtype.F32);
  const idxTy = vt(shapeOf([n]), Dtype.U32);
  const base = emit(Op.unary(tags.SORT_DESC, ix), [valTy, idxTy]);
  return [Tensor.node(base, valTy), Tensor.node(base + 1, idxTy)];
}

export class Predicate {
  constructor(
    readonly tag: number,
    readonly arg: Tensor,
  ) {}
}

export const rankLe = (k: Operand) => new Predicate(PRED_RANK_LE, toArg(k));
export const cummassLe = (p: Operand) => new Predicate(PRED_CUMMASS_LE, toArg(p));
export const probGe = (thr: Operand) => new Predicate(PRED_PROB_GE, toArg(thr));

export function pivotThreshold(x: Operand, predicate: Predicate): Tensor {
  const [ii, tyi] = materialize(x);
  const [pv] = materialize(predicate.arg);
  const rty = vt(tyi.shape, Dtype.BOOL);
  return Tensor.node(emit(Op.pivotThreshold(ii, predicate.tag, pv), [rty]), rty);
}

export function matmul(a: Operand, b: Operand): Tensor {
  const [ia, tya] = materialize(a);
  const [ib, tyb] = materialize(b);
  const m = tya.shape.length ? tya.shape[0] : 0;
  const n = tyb.shape.length ? tyb.shape[tyb.shape.length - 1] : 0;
  const rty = vt(shapeOf([m, n]), Dtype.F32);
  return Tensor.node(emit(Op.binary(tags.MATMUL, ia, ib), [rty]), rty);
}

// -- sampling -----------------------------------------------------------------

function rngNoise(state: Operand, shape: readonly number[], kind: RngKind): Tensor {
  const s = shapeOf(shape);
  const [istate] = materialize(state);
  const rty = vt(s, Dtype.F32);
  return Tensor.node(emit(Op.rngKeyed(istate, s, kind), [rty]), rty);
}

export const gumbel = (state: Operand, shape: readonly number[]) => rngNoise(state, shape, RngKind.GUMBEL);
export const rng = (state: Operand, shape: readonly number[]) => rngNoise(state, shape, RngKind.UNIFORM);

export function maskApply(logits: Operand, mask: Operand): Tensor {
  const [il, tyl] = materialize(logits);
  const [im] = materialize(mask);
  return Tensor.node(expandMaskApply(tracedPush(tyl), il, im), tyl);
}

function appendMaskAxis(shape: Shape, length: number): Shape {
  const dims = [...shape, length];
  try {
    return shapeOf(dims);
  } catch {
    throw new TraceError(`a structured mask over [${shape}] with length ${length} has shape [${dims}], whose rank exceeds ${MAX_RANK}`);
  }
}

export function causalMask(positions: Operand, length: number): Tensor {
  const [p, ty] = materialize(positions);
  const rty = vt(appendMaskAxis(ty.shape, length), Dtype.BOOL);
  return Tensor.node(emit(Op.causalMask(p, length), [rty]), rty);
}

export function slidingWindowMask(positions: Operand, length: number, window: number): Tensor {
  const [p, ty] = materialize(positions);
  const rty = vt(appendMaskAxis(ty.shape, length), Dtype.BOOL);
  return Tensor.node(emit(Op.slidingWindowMask(p, length, window), [rty]), rty);
}

export function sinkWindowMask(positions: Operand, length: number, sink: number, window: number): Tensor {
  const [p, ty] = materialize(positions);
  const rty = vt(appendMaskAxis(ty.shape, length), Dtype.BOOL);
  return Tensor.node(emit(Op.sinkWindowMask(p, length, sink, window), [rty]), rty);
}

/** For every row and key, whether the key occurs anywhere in the row. */
export function rowMembership(rows: Operand, keys: Operand): Tensor {
  const rowsT = toArg(rows);
  const keysT = toArg(keys);
  const rowType = rowsT.ty;
  const keyType = keysT.ty;
  if (rowType.shape.length !== 2) throw new TraceError(`row_membership rows must have shape [R, D], got [${rowType.shape}]`);
  const [rowCount, depth] = rowType.shape;
  if (keyType.shape.length !== 1) throw new TraceError(`row_membership keys must have shape [K], got [${keyType.shape}]`);
  const [keyCount] = keyType.shape;
  if (rowType.dtype !== keyType.dtype) {
    throw new TraceError(`row_membership rows and keys must have the same dtype, got ${dtypeName(rowType.dtype)} and ${dtypeName(keyType.dtype)}`);
  }
  const rowStride = keyCount * depth;
  const rowFlatLen = rowCount * depth;
  const flatLen = rowCount * keyCount * depth;
  if (Math.max(rowStride, rowFlatLen, flatLen) > 0xffff_ffff) {
    throw new TraceError(`row_membership over ${rowCount} rows x ${keyCount} keys x depth ${depth} needs a ${flatLen}-element intermediate, which overflows the wire's u32 extents`);
  }
  const [rid] = materialize(rowsT);
  const [kid] = materialize(keysT);
  const rowsN = Tensor.node(rid, rowType);
  const keysN = Tensor.node(kid, keyType);
  const linear = iota(flatLen);
  const rowIndex = div(linear, constant(rowStride, Dtype.U32));
  const depthIndex = rem(linear, constant(depth, Dtype.U32));
  const rowValueIndex = add(mul(rowIndex, constant(depth, Dtype.U32)), depthIndex);
  const rowValues = gather(reshape(rowsN, [rowFlatLen]), rowValueIndex);
  const keyIndex = rem(div(linear, constant(depth, Dtype.U32)), constant(keyCount, Dtype.U32));
  const keyValues = gather(keysN, keyIndex);
  const matches = eq(reshape(rowValues, [rowCount, keyCount, depth]), reshape(keyValues, [rowCount, keyCount, depth]));
  return cast(reduceMax(cast(matches, Dtype.U32)), Dtype.BOOL);
}

export function maskedArgmax(logits: Operand, mask: Operand): Tensor {
  const [lid, lty] = materialize(logits);
  const [mid] = materialize(mask);
  const resultType = vt(dropLast(lty.shape), Dtype.I32);
  const ninf = emit(Op.const(Dtype.F32, -Infinity), [vt(SCALAR, Dtype.F32)]);
  const masked = emit(Op.ternary(tags.SELECT, mid, lid, ninf), [lty]);
  const result = emit(Op.unary(tags.REDUCE_ARGMAX, masked), [resultType]);
  return Tensor.node(result, resultType);
}

/** Semantic Gumbel-max sampler over the input's complete shape. */
export function gumbelMax(logits: Operand, state: Operand): Tensor {
  const [lid, lty] = materialize(logits);
  const [sid] = materialize(state);
  const resultType = vt(dropLast(lty.shape), Dtype.I32);
  const noise = emit(Op.rngKeyed(sid, lty.shape, RngKind.GUMBEL), [vt(lty.shape, Dtype.F32)]);
  const perturbed = emit(Op.binary(tags.ADD, lid, noise), [lty]);
  const result = emit(Op.unary(tags.REDUCE_ARGMAX, perturbed), [resultType]);
  return Tensor.node(result, resultType);
}

export function entropy(probabilities: Operand): Tensor {
  const [pid, pty] = materialize(probabilities);
  const resultType = vt(dropLast(pty.shape), Dtype.F32);
  const lp = emit(Op.unary(tags.LOG, pid), [pty]);
  const terms = emit(Op.binary(tags.MUL, pid, lp), [pty]);
  const s = emit(Op.unary(tags.REDUCE_SUM, terms), [resultType]);
  const result = emit(Op.unary(tags.NEG, s), [resultType]);
  return Tensor.node(result, resultType);
}

export function entropyFromLogprobs(probabilities: Operand, logProbabilities: Operand): Tensor {
  const [pid, pty] = materialize(probabilities);
  const [lid] = materialize(logProbabilities);
  const resultType = vt(dropLast(pty.shape), Dtype.F32);
  const terms = emit(Op.binary(tags.MUL, pid, lid), [pty]);
  const s = emit(Op.unary(tags.REDUCE_SUM, terms), [resultType]);
  const result = emit(Op.unary(tags.NEG, s), [resultType]);
  return Tensor.node(result, resultType);
}

export function scalarGather(src: Operand, index: Operand): Tensor {
  const [sid, sty] = materialize(src);
  const [iid, ity] = materialize(index);
  let op: Op;
  let resultShape: Shape;
  if (sty.shape.length === 2) {
    const r = sty.shape[0];
    if (!(ity.shape.length === 1 && ity.shape[0] === r)) {
      throw new TraceError(`scalar_gather over a [${sty.shape}] matrix requires one index per row ([${r}]), got [${ity.shape}]`);
    }
    op = Op.binary(tags.GATHER_ROW, sid, iid);
    resultShape = shapeOf([r]);
  } else {
    const dims = [...ity.shape, ...sty.shape.slice(Math.min(sty.shape.length, 1))];
    try {
      resultShape = shapeOf(dims);
    } catch {
      throw new TraceError(`scalar_gather of [${sty.shape}] by [${ity.shape}] has result shape [${dims}], whose rank exceeds ${MAX_RANK}`);
    }
    op = Op.binary(tags.GATHER, sid, iid);
  }
  const resultType = vt(resultShape, sty.dtype);
  return Tensor.node(emit(op, [resultType]), resultType);
}

/** Exact nucleus sampler as ordinary composable SSA. */
export function nucleusSample(logits: Operand, topP: Operand, state: Operand): Tensor {
  const [lid, lty] = materialize(logits);
  const [pid] = materialize(topP);
  const [sid] = materialize(state);
  const tokenType = vt(dropLast(lty.shape), Dtype.I32);
  const result = expandNucleusSample(tracedPush(lty), lid, pid, sid, lty.shape);
  return Tensor.node(result, tokenType);
}

// -- intrinsic leaf / internal helpers ------------------------------------------

export function intrinsicVal(intr: Intrinsic, shape: Shape, dt: Dtype): Tensor {
  const ty = vt(shape, dt);
  return Tensor.node(emit(Op.intrinsicVal(intr, shape, dt), [ty]), ty);
}

/** Reshape a value id to `target` if it differs (a `put` fitting a scalar into a `[1]` cell). */
export function reshapeIdTo(vid: number, frm: ValueType, target: Shape): number {
  if (shapeEq(frm.shape, target)) return vid;
  return emit(Op.reshape(vid, target), [vt(target, frm.dtype)]);
}
