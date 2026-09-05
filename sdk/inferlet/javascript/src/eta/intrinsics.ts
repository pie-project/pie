// `intrinsics.*`: first-party stage-scoped values and model constants, and
// the `intrinsics.kernel.*` second-party surface. Port of
// `eta-dsl/src/intrinsics.rs`.

import { ATTN_SCORE_KV_MAX, Dtype, Intrinsic, Op, SCALAR, SinkScope, shapeOf } from './ir.js';
import { currentRows, emit, internName, recordSink, pageSize as tracePageSize, vocab as traceVocab, vt } from './trace.js';
import { Operand, Tensor, constant, intrinsicVal, materialize, reshape } from './value.js';

/** The interpreter-visible activation dtype. Always F32. */
export const activationType = Dtype.F32;

export function vocab(): number {
  return traceVocab();
}

export function pageSize(): number {
  return tracePageSize();
}

function singleRowReshape(t: Tensor): Tensor {
  const s = t.shape;
  if (s.length === 2 && s[0] === 1) return reshape(t, [s[1]]);
  return t;
}

/** The LM-head logits, `[n_out, vocab]` f32 — `[vocab]` for a single read-out row. */
export function logits(): Tensor {
  const rows = Math.max(currentRows(), 1);
  return singleRowReshape(intrinsicVal(Intrinsic.LOGITS, shapeOf([rows, vocab()]), Dtype.F32));
}

export function mtpLogits(k: number): Tensor {
  return intrinsicVal(Intrinsic.MTP_LOGITS, shapeOf([k, vocab()]), Dtype.F32);
}

export function mtpDrafts(n: number): Tensor {
  return intrinsicVal(Intrinsic.MTP_DRAFTS, shapeOf([Math.max(n, 1)]), Dtype.I32);
}

export function hidden(width: number): Tensor {
  const rows = Math.max(currentRows(), 1);
  return intrinsicVal(Intrinsic.HIDDEN, shapeOf([rows, Math.max(width, 1)]), activationType);
}

export function query(width: number): Tensor {
  return intrinsicVal(Intrinsic.QUERY, shapeOf([Math.max(width, 1)]), activationType);
}

export function valueHead(): Tensor {
  return intrinsicVal(Intrinsic.VALUE_HEAD, shapeOf([Math.max(currentRows(), 1)]), Dtype.F32);
}

export function layer(): Tensor {
  return intrinsicVal(Intrinsic.LAYER, SCALAR, Dtype.U32);
}

export function attnScore(planes: number): Tensor {
  return intrinsicVal(Intrinsic.ATTN_SCORE, shapeOf([Math.max(planes, 1), ATTN_SCORE_KV_MAX]), Dtype.F32);
}

export function attnScoreKvMax(): number {
  return ATTN_SCORE_KV_MAX;
}

/** Second-party kernels and configuration sinks. */
export const kernel = {
  envelopeDot(pMax: number): Tensor {
    const queryTy = vt(shapeOf([1]), activationType);
    const q = emit(Op.intrinsicVal(Intrinsic.QUERY, queryTy.shape, queryTy.dtype), [queryTy]);
    const scoreTy = vt(shapeOf([pMax]), Dtype.F32);
    const name = internName('envelope_dot');
    return Tensor.node(emit(Op.kernelCall(name, [q], scoreTy.shape, scoreTy.dtype), [scoreTy]), scoreTy);
  },

  attnPageMask(mask: Operand): void {
    const [mid] = materialize(mask);
    const name = internName('attn_page_mask');
    emit(Op.sinkCall(name, [mid]), []);
    recordSink('attn_page_mask', SinkScope.ATTENTION);
  },

  lora(a: Operand, b: Operand, sites: Operand): void {
    const [aid] = materialize(a);
    const [bid] = materialize(b);
    const [sid] = materialize(sites);
    const name = internName('lora');
    emit(Op.sinkCall(name, [aid, bid, sid]), []);
    recordSink('lora', SinkScope.PASS_WIDE);
  },

  adapterScale(l: Operand, sites: Operand): void {
    const [lid] = materialize(l);
    const [sid] = materialize(sites);
    const name = internName('lora');
    emit(Op.sinkCall(name, [lid, sid]), []);
    recordSink('lora', SinkScope.PASS_WIDE);
  },
};

export { constant };
