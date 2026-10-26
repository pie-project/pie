// `intrinsics.*`: first-party stage-scoped values and model constants, and
// the `intrinsics.kernel.*` second-party surface. Port of
// `eta-dsl/src/intrinsics.rs`.

import { Axis, Dtype, Op, OutputDecl, shapeOf } from './ir.js';
import { currentRows, declaredOutput, emit, internName, recordSink, pageSize as tracePageSize, vocab as traceVocab } from './trace.js';
import { Operand, Tensor, constant, materialize, outputVal, reshape } from './value.js';

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

function declared(lane: number, name: string): OutputDecl {
  const decl = declaredOutput(lane, name);
  if (!decl) throw new Error(`lane ${lane} of this pass reads out no output \`${name}\``);
  return decl;
}

/**
 * Lane `lane`'s output `name` as the model declares it: its readout rows (a
 * lone row read as its width alone), or its layers stacked.
 */
export function output(lane: number, name: string): Tensor {
  const decl = declared(lane, name);
  let rows: number;
  if (decl.axis === Axis.LANE) rows = Math.max(decl.layers, 1);
  else if (decl.readout) rows = Math.max(currentRows(), 1);
  else throw new Error(`output \`${name}\` is one row a token or voxel of lane ${lane}: \`outputRows\` reads the rows it names`);
  const t = outputVal(internName(name), lane, shapeOf([rows, decl.width]), decl.dtype);
  return decl.readout ? singleRowReshape(t) : t;
}

/** `rows` rows of lane `lane`'s output `name`, one a token or voxel. */
export function outputRows(lane: number, name: string, rows: number): Tensor {
  const decl = declared(lane, name);
  return outputVal(internName(name), lane, shapeOf([Math.max(rows, 1), decl.width]), decl.dtype);
}

/** Second-party kernels and configuration sinks. */
export const kernel = {
  attnPageMask(mask: Operand): void {
    const [mid] = materialize(mask);
    const name = internName('attn_page_mask');
    emit(Op.sinkCall(name, [mid]), []);
    recordSink('attn_page_mask');
  },

  lora(a: Operand, b: Operand, sites: Operand): void {
    const [aid] = materialize(a);
    const [bid] = materialize(b);
    const [sid] = materialize(sites);
    const name = internName('lora');
    emit(Op.sinkCall(name, [aid, bid, sid]), []);
    recordSink('lora');
  },

  adapterScale(l: Operand, sites: Operand): void {
    const [lid] = materialize(l);
    const [sid] = materialize(sites);
    const name = internName('lora');
    emit(Op.sinkCall(name, [lid, sid]), []);
    recordSink('lora');
  },
};

export { constant };
