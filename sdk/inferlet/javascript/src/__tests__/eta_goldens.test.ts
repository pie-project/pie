// Byte-identity of the JavaScript ETA port against the Rust SDK.
//
// `sdk/inferlet/python/tests/goldens/eta_containers.txt` holds, per program,
// the FNV-1a identity hash and the canonical container bytes as the Rust
// `eta-dsl` `Builder` emits them. Each test rebuilds the same program through
// the JS `Builder` and compares bytes — a JS inferlet, a Python inferlet and a
// Rust inferlet tracing the same pass share one container hash.

import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

import { Builder, DslChannel, Traced } from '../eta/builder.js';
import * as intrinsics from '../eta/intrinsics.js';
import { Dtype, Port, Stage, dtype, fnv1a64 } from '../eta/ir.js';
import { TraceError } from '../eta/trace.js';
import {
  Tensor,
  abs,
  and,
  cast,
  causalMask,
  constData,
  cummassLe,
  cumprod,
  cumsum,
  entropy,
  entropyFromLogprobs,
  eq,
  exp,
  gather,
  ge,
  gt,
  gumbel,
  gumbelMax,
  indptr,
  iota,
  l2norm,
  le,
  log,
  logSoftmax,
  lt,
  maskApply,
  maskedArgmax,
  matmul,
  maxElem,
  minElem,
  ne,
  neg,
  not,
  nucleusSample,
  or,
  pivotThreshold,
  probGe,
  rankLe,
  recip,
  reduceArgmax,
  reduceMax,
  reduceMin,
  reduceSum,
  rem,
  reshape,
  rng,
  rowMembership,
  scalarGather,
  scatterAdd,
  scatterSet,
  select,
  sign,
  sinkWindowMask,
  slidingWindowMask,
  softmax,
  sortDesc,
  topK,
  transpose,
} from '../eta/value.js';

const VOCAB = 151_936;
const PAGE = 32;

const here = dirname(fileURLToPath(import.meta.url));
const goldensPath = join(here, '..', '..', '..', 'python', 'tests', 'goldens', 'eta_containers.txt');
const GOLDENS = new Map<string, { hash: bigint; bytes: Uint8Array }>();
for (const line of readFileSync(goldensPath, 'utf8').split('\n')) {
  if (!line.trim()) continue;
  const [name, h, hex] = line.split(' ');
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < bytes.length; i++) bytes[i] = parseInt(hex.slice(2 * i, 2 * i + 2), 16);
  GOLDENS.set(name, { hash: BigInt(h), bytes });
}

const chFrom = (v: Iterable<number | boolean> | number[], dt: Dtype, name: string) => DslChannel.fromConst(constData(v, dt)).named(name);
const chNew = (shape: number[], dt: Dtype, name: string) => DslChannel.new(shape, dt).named(name);
const hostPut = (ch: DslChannel, v: Iterable<number | boolean>, dt?: Dtype) => {
  constData(v, dt);
  ch.noteHostPut();
};
const range = (a: number, b?: number) => {
  const [lo, hi] = b === undefined ? [0, a] : [a, b];
  return Array.from({ length: hi - lo }, (_, i) => lo + i);
};
const divCeil = (a: number, b: number) => Math.floor((a + b - 1) / b);

function check(name: string, traced: Traced) {
  const want = GOLDENS.get(name)!;
  const got = traced.encode();
  expect(Buffer.from(got).toString('hex')).toBe(Buffer.from(want.bytes).toString('hex'));
  expect(traced.identityHash()).toBe(want.hash);
  expect(fnv1a64(got)).toBe(want.hash);
}

function bindGeometry(b: Builder, ports: [Port, DslChannel][]) {
  for (const [p, c] of ports) b.bindPort(p, c);
}

describe('eta goldens', () => {
  it('s3 matches lowering.rs', () => {
    const vocab = 32_000;
    const ctr1 = Tensor.constant([0, 1], Dtype.U32);
    const tok = chNew([1], dtype.i32, 'tok');
    const ind = chFrom([0, 1], dtype.u32, 'indptr');
    const out = chNew([1], dtype.i32, 'out');
    const mask = chNew([vocab], dtype.bool, 'mask');
    const len = chFrom([1], dtype.u32, 'len');
    const rngCh = chFrom([7, 0], dtype.u32, 'rng');
    hostPut(tok, [1], dtype.i32);
    const b = new Builder(vocab, 16);
    b.bindPort(Port.EMBED_TOKENS, tok);
    b.bindPort(Port.EMBED_INDPTR, ind);
    b.bindPort(Port.KV_LEN, len);
    b.stage(Stage.EPILOGUE, () => {
      const logits = intrinsics.logits();
      const r = rngCh.take();
      const g = gumbel(r, [intrinsics.vocab()]);
      const t = reduceArgmax(maskApply(logits, mask.take()).add(g));
      rngCh.putTensor(r.add(ctr1));
      tok.putTensor(t);
      len.putTensor(len.take().add(1));
      out.putTensor(t);
    });
    hostPut(mask, new Array(vocab).fill(true));
    const traced = b.build();
    expect(traced.identityHash()).toBe(4213522552817221928n);
    check('s3', traced);
  });

  it('text_completion_decode', () => {
    const n = 5;
    const tokIn = chFrom([42], dtype.i32, 'tok_in');
    const embedIndptr = chFrom([0, 1], dtype.u32, 'embed_indptr');
    const positions = chFrom([n], dtype.u32, 'positions');
    const pages = chFrom(range(3), dtype.u32, 'pages');
    const pageIndptr = chFrom([0, divCeil(n + 1, PAGE)], dtype.u32, 'page_indptr');
    const wSlot = chFrom([Math.floor(n / PAGE)], dtype.u32, 'w_slot');
    const wOff = chFrom([n % PAGE], dtype.u32, 'w_off');
    const kvLen = chFrom([n + 1], dtype.u32, 'kv_len');
    const tokOut = chNew([1], dtype.i32, 'tok_out');
    const b = new Builder(VOCAB, PAGE);
    bindGeometry(b, [
      [Port.EMBED_TOKENS, tokIn],
      [Port.EMBED_INDPTR, embedIndptr],
      [Port.KV_LEN, kvLen],
      [Port.PAGES, pages],
      [Port.PAGE_INDPTR, pageIndptr],
      [Port.W_SLOT, wSlot],
      [Port.W_OFF, wOff],
      [Port.POSITIONS, positions],
    ]);
    b.stage(Stage.EPILOGUE, () => {
      const length = kvLen.take();
      const nextLength = length.add(1);
      const pageCount = nextLength.divCeil(PAGE);
      kvLen.putTensor(nextLength);
      positions.putTensor(length);
      wSlot.putTensor(length.div(PAGE));
      wOff.putTensor(length.rem(PAGE));
      pageIndptr.putTensor(indptr(1, pageCount));
      tokOut.putTensor(reshape(reduceArgmax(intrinsics.logits()), [1]));
    });
    tokOut.noteHostTake();
    check('text_completion_decode', b.build());
  });

  it('naive_decode', () => {
    const n = 7;
    const temperature = 0.7;
    const cap = 8;
    const tokIn = chFrom([3], dtype.i32, 'tok_in');
    const rngCh = chFrom([0x7ce1 ^ 0x5bd1, 0], dtype.u32, 'rng');
    const tokOut = chNew([1], dtype.i32, 'tok_out').capacity(cap);
    const s1 = chNew([1], dtype.f32, 's1_out').capacity(cap);
    const s2 = chNew([1], dtype.f32, 's2_out').capacity(cap);
    const lane1 = chFrom([0, 1], dtype.u32, 'embed_indptr');
    const positions = chFrom([n], dtype.u32, 'positions');
    const pages = chFrom(range(4), dtype.u32, 'pages');
    const pageIndptr = chFrom([0, divCeil(n + 1, PAGE)], dtype.u32, 'page_indptr');
    const wSlot = chFrom([Math.floor(n / PAGE)], dtype.u32, 'w_slot');
    const wOff = chFrom([n % PAGE], dtype.u32, 'w_off');
    const kvLen = chFrom([n + 1], dtype.u32, 'kv_len');
    const b = new Builder(VOCAB, PAGE);
    bindGeometry(b, [
      [Port.EMBED_TOKENS, tokIn],
      [Port.EMBED_INDPTR, lane1],
      [Port.KV_LEN, kvLen],
      [Port.PAGES, pages],
      [Port.PAGE_INDPTR, pageIndptr],
      [Port.W_SLOT, wSlot],
      [Port.W_OFF, wOff],
      [Port.POSITIONS, positions],
    ]);
    b.stage(Stage.EPILOGUE, () => {
      const length = kvLen.take();
      const r = rngCh.take();
      const logits = intrinsics.logits();
      const scaled = logits.div(temperature);
      const token = gumbelMax(scaled, r);
      const rNext = r.add(iota(2));
      const nextLength = length.add(1);
      const pageCount = nextLength.divCeil(PAGE);
      tokIn.putTensor(token);
      kvLen.putTensor(nextLength);
      positions.putTensor(length);
      wSlot.putTensor(length.div(PAGE));
      wOff.putTensor(length.rem(PAGE));
      pageIndptr.putTensor(indptr(1, pageCount));
      tokOut.putTensor(token);
      const mirror = reshape(cast(token, dtype.f32), [1]);
      s1.putTensor(mirror);
      s2.putTensor(mirror);
      rngCh.putTensor(rNext);
    });
    tokOut.noteHostTake();
    s1.noteHostTake();
    s2.noteHostTake();
    check('naive_decode', b.build());
  });

  it('coverage', () => {
    const k = 8;
    const tok = chFrom([1], dtype.i32, 'tok');
    const ind = chFrom([0, 1], dtype.u32, 'indptr');
    const rngCh = chFrom([1, 2], dtype.u32, 'rng');
    const topP = chFrom([0.9], dtype.f32, 'top_p');
    const bias = chFrom([0.0, -1.5, 2.25, 0.0], dtype.f32, 'bias');
    const out = chNew([1], dtype.i32, 'out');
    const stat = chNew([1], dtype.f32, 'stat');
    const stat2 = chNew([4], dtype.f32, 'stat2');
    const flag = chNew([1], dtype.bool, 'flag');
    const b = new Builder(VOCAB, PAGE);
    b.bindPort(Port.EMBED_TOKENS, tok);
    b.bindPort(Port.EMBED_INDPTR, ind);
    b.stage(Stage.EPILOGUE, () => {
      const logits = intrinsics.logits();
      const r = rngCh.take();
      const p = softmax(logits);
      const lp = logSoftmax(logits);
      const h = entropy(p);
      const h2 = entropyFromLogprobs(p, lp);
      const [tv, ti] = topK(logits, k);
      const keep = pivotThreshold(p, cummassLe(topP.read()));
      const keep2 = pivotThreshold(p, rankLe(Tensor.constant(40, Dtype.U32)));
      const keep3 = pivotThreshold(p, probGe(0.01));
      const both = and(and(keep, keep2), not(keep3));
      const masked = maskApply(logits, or(both, lt(logits, 0.0)));
      const t1 = nucleusSample(masked, topP.read(), r);
      const t2 = maskedArgmax(logits, keep);
      const t3 = gumbelMax(logits, r);
      const g = gather(logits, cast(ti, dtype.u32));
      const g2 = scalarGather(logits, cast(t1, dtype.u32));
      const ssum = reduceSum(tv).add(reduceMax(tv)).sub(reduceMin(tv));
      const cs = cumsum(tv).mul(cumprod(exp(tv)));
      const [sv] = sortDesc(logits);
      const l2 = l2norm(reshape(sv, [VOCAB]));
      const m = matmul(reshape(tv, [1, k]), transpose(reshape(tv, [1, k])));
      const sel = select(gt(t1, t2), t1, t3);
      const sc = scatterSet(logits, cast(t2, dtype.u32), -1.0);
      const sa = scatterAdd(sc, cast(t3, dtype.u32), 1.0);
      const ge_ = ge(sa, 0.5);
      const cm = causalMask(iota(4), 8);
      const sw = slidingWindowMask(iota(4), 8, 3);
      const sk = sinkWindowMask(iota(4), 8, 1, 3);
      const mem = rowMembership(reshape(iota(8), [2, 4]), iota(3));
      const u = rng(r, [4]);
      const bb = bias.read().add(u);
      const extra = abs(recip(sign(neg(bb))))
        .add(log(exp(bb)))
        .add(maxElem(bb, 1.0))
        .sub(minElem(bb, 2.0))
        .add(rem(bb, 3.0));
      const flagV = reduceSum(cast(cm, dtype.u32))
        .add(reduceSum(cast(sw, dtype.u32)))
        .add(reduceSum(cast(sk, dtype.u32)))
        .add(reduceSum(cast(mem, dtype.u32)))
        .add(reduceSum(cast(ge_, dtype.u32)));
      const total = h
        .add(h2)
        .add(ssum)
        .add(reduceSum(cs))
        .add(reduceSum(l2))
        .add(reduceSum(reshape(m, [1])))
        .add(reduceSum(g))
        .add(g2)
        .add(reduceSum(extra))
        .add(cast(flagV, dtype.f32))
        .add(cast(eq(t1, t2), dtype.f32))
        .add(cast(ne(t1, t3), dtype.f32))
        .add(cast(le(t2, t3), dtype.f32));
      stat.putTensor(reshape(total, [1]));
      stat2.putTensor(extra);
      flag.putTensor(reshape(gt(total, 0.0), [1]));
      out.putTensor(reshape(sel, [1]));
      rngCh.putTensor(r.add(iota(2)));
    });
    out.noteHostTake();
    stat.noteHostTake();
    stat2.noteHostTake();
    flag.noteHostTake();
    check('coverage', b.build());
  });

  it('sinks', () => {
    const tok = chFrom([1], dtype.i32, 'tok');
    const ind = chFrom([0, 1], dtype.u32, 'indptr');
    const a = chNew([2, 4, 8], dtype.f32, 'a');
    const bch = chNew([2, 8, 4], dtype.f32, 'b');
    const out = chNew([1], dtype.i32, 'out');
    hostPut(a, new Array(64).fill(0.0), dtype.f32);
    hostPut(bch, new Array(64).fill(0.0), dtype.f32);
    const b = new Builder(VOCAB, PAGE);
    b.bindPort(Port.EMBED_TOKENS, tok);
    b.bindPort(Port.EMBED_INDPTR, ind);
    b.stage(Stage.PROLOGUE, () => {
      intrinsics.kernel.lora(a.read(), bch.read(), Tensor.constant(1 | 4, Dtype.U32));
      intrinsics.kernel.attnPageMask(iota(4));
    });
    b.stage(Stage.ON_ATTN_PROJ, () => {
      intrinsics.query(16);
      intrinsics.kernel.envelopeDot(4);
      intrinsics.kernel.attnPageMask(cast(gt(intrinsics.kernel.envelopeDot(4), 0.0), dtype.u32).add(intrinsics.layer()));
    });
    b.stage(Stage.EPILOGUE, () => {
      out.putTensor(reshape(reduceArgmax(intrinsics.logits()), [1]));
    });
    out.noteHostTake();
    check('sinks', b.build());
  });

  it('refuses a bulk constant', () => {
    const tok = chFrom([1], dtype.i32, 'tok');
    const ind = chFrom([0, 1], dtype.u32, 'indptr');
    const out = chNew([4], dtype.f32, 'out');
    const b = new Builder(VOCAB, PAGE);
    b.bindPort(Port.EMBED_TOKENS, tok);
    b.bindPort(Port.EMBED_INDPTR, ind);
    b.stage(Stage.EPILOGUE, () => out.putTensor(Tensor.constant([1.0, 2.0, 5.0, 7.0], dtype.f32).add(0.0)));
    out.noteHostTake();
    expect(() => b.build()).toThrow(TraceError);
  });

  it('readiness lint', () => {
    const tok = chFrom([1], dtype.i32, 'tok');
    const ind = chFrom([0, 1], dtype.u32, 'indptr');
    const never = chNew([1], dtype.f32, 'never');
    const out = chNew([1], dtype.f32, 'out');
    const b = new Builder(VOCAB, PAGE);
    b.bindPort(Port.EMBED_TOKENS, tok);
    b.bindPort(Port.EMBED_INDPTR, ind);
    b.stage(Stage.EPILOGUE, () => out.putTensor(never.take().add(1.0)));
    out.noteHostTake();
    expect(() => b.build()).toThrow(/never produced/);
  });
});

describe('comparison methods', () => {
  const epilogueBytes = (body: (l: Tensor) => Tensor) => {
    const vocab = 32_000;
    const tok = chNew([1], dtype.i32, 'tok');
    const ind = chFrom([0, 1], dtype.u32, 'indptr');
    const out = chNew([vocab], dtype.bool, 'out');
    hostPut(tok, [1], dtype.i32);
    const b = new Builder(vocab, 16);
    b.bindPort(Port.EMBED_TOKENS, tok);
    b.bindPort(Port.EMBED_INDPTR, ind);
    b.stage(Stage.EPILOGUE, () => out.putTensor(body(intrinsics.logits())));
    return Buffer.from(b.build().encode()).toString('hex');
  };

  it('emit exactly what the free functions emit', () => {
    expect(epilogueBytes((l) => l.lt(0.5))).toBe(epilogueBytes((l) => lt(l, 0.5)));
    expect(epilogueBytes((l) => l.ge(0.5))).toBe(epilogueBytes((l) => ge(l, 0.5)));
    expect(epilogueBytes((l) => l.eq(0.5))).toBe(epilogueBytes((l) => eq(l, 0.5)));
    expect(epilogueBytes((l) => l.ne(0.5))).toBe(epilogueBytes((l) => ne(l, 0.5)));
    expect(epilogueBytes((l) => l.gt(0.5))).toBe(epilogueBytes((l) => gt(l, 0.5)));
    expect(epilogueBytes((l) => l.le(0.5))).toBe(epilogueBytes((l) => le(l, 0.5)));
  });
});
