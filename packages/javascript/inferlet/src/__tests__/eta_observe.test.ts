// A program that reads attention scores states what it observes on its
// builder: the container carries the optional observe section after every
// table, byte for byte as the Rust `eta-ir` encoder writes it (tag 0x4f, a
// kind byte, and for `rows` the u32 count); a program that states nothing
// carries no section, its bytes as they always were.

import { describe, expect, it } from 'vitest';

import { Builder, DslChannel } from '../eta/builder.js';
import * as intrinsics from '../eta/intrinsics.js';
import { Axis, Dtype, Observe, OBSERVE_SECTION, OutputDecl, Stage, dtype } from '../eta/ir.js';
import { reduceArgmax } from '../eta/value.js';

const VOCAB = 8;

function traced(observe?: Observe): Uint8Array {
  const out = DslChannel.new([1], dtype.i32);
  const b = new Builder(VOCAB, 2);
  const logits: OutputDecl = { name: 'logits', axis: Axis.ROW, width: VOCAB, dtype: Dtype.F32, layers: 0, readout: true };
  b.setOutputs([[logits]]);
  if (observe !== undefined) b.observe(observe);
  b.stage(Stage.EPILOGUE, () => {
    out.putTensor(reduceArgmax(intrinsics.output(0, 'logits')));
  });
  return b.build().encode();
}

describe('the observe section', () => {
  it('round-trips each observation and leaves a plain container as it was', () => {
    const plain = traced();
    expect(plain[plain.length - 2]).not.toBe(OBSERVE_SECTION);
    const rows = traced({ rows: 32 });
    expect(Array.from(rows.slice(0, plain.length))).toEqual(Array.from(plain));
    expect(Array.from(rows.slice(plain.length))).toEqual([OBSERVE_SECTION, 0, 32, 0, 0, 0]);
    const fire = traced('fire-rows');
    expect(Array.from(fire.slice(0, plain.length))).toEqual(Array.from(plain));
    expect(Array.from(fire.slice(plain.length))).toEqual([OBSERVE_SECTION, 1]);
  });
});
