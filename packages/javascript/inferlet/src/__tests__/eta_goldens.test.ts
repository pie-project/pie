// Byte-identity of the JavaScript ETA port against the Rust `inferlet` crate.
//
// `packages/python/inferlet/tests/goldens/eta_containers.txt` holds, per program,
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
import { entropyBoundAccept, stableAndConfident } from '../eta/diffusion.js';
import { Axis, Dtype, OutputDecl, Port, Stage, dtype, fnv1a64, shapeOf } from '../eta/ir.js';
import { TraceError } from '../eta/trace.js';
import {
  ConstData,
  Tensor,
  abs,
  add,
  and,
  broadcast,
  cast,
  causalMask,
  constData,
  cos,
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
  mul,
  ne,
  neg,
  normal,
  not,
  nucleusSample,
  or,
  packElems,
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
  rsqrt,
  rowMembership,
  scalarGather,
  scatterAdd,
  scatterSet,
  select,
  sign,
  sin,
  sinkWindowMask,
  slidingWindowMask,
  softmax,
  sortDesc,
  sqrt,
  topK,
  transpose,
} from '../eta/value.js';

const VOCAB = 151_936;
const PAGE = 32;

const decl = (name: string, axis: Axis, width: number, readout: boolean): OutputDecl => ({
  name,
  axis,
  width,
  dtype: Dtype.F32,
  layers: 0,
  readout,
});
const logitsDecl = (vocab: number): OutputDecl[][] => [[decl('logits', Axis.ROW, vocab, true)]];

const here = dirname(fileURLToPath(import.meta.url));
const goldensPath = join(here, '..', '..', '..', '..', 'python', 'inferlet', 'tests', 'goldens', 'eta_containers.txt');
const GOLDENS = new Map<string, { hash: bigint; bytes: Uint8Array }>();
for (const line of readFileSync(goldensPath, 'utf8').split('\n')) {
  if (!line.trim()) continue;
  const [name, h, hex] = line.split(' ');
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < bytes.length; i++) bytes[i] = parseInt(hex.slice(2 * i, 2 * i + 2), 16);
  GOLDENS.set(name, { hash: BigInt(h), bytes });
}

const chFrom = (v: Iterable<number | boolean> | number[], dt: Dtype) => DslChannel.fromConst(constData(v, dt));
const chNew = (shape: number[], dt: Dtype) => DslChannel.new(shape, dt);
const hostPut = (ch: DslChannel, v: Iterable<number | boolean>, dt?: Dtype) => {
  constData(v, dt);
  ch.noteHostPut();
};
const range = (a: number, b?: number) => {
  const [lo, hi] = b === undefined ? [0, a] : [a, b];
  return Array.from({ length: hi - lo }, (_, i) => lo + i);
};

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
    const tok = chNew([1], dtype.i32);
    const ind = chFrom([0, 1], dtype.u32);
    const out = chNew([1], dtype.i32);
    const mask = chNew([vocab], dtype.bool);
    const len = chFrom([1], dtype.u32);
    const rngCh = chFrom([7, 0], dtype.u32);
    hostPut(tok, [1], dtype.i32);
    const b = new Builder(vocab, 16);
    b.setOutputs(logitsDecl(vocab));
    b.bindPort(Port.EMBED_TOKENS, tok);
    b.bindPort(Port.EMBED_INDPTR, ind);
    b.bindPort(Port.KV_LEN, len);
    b.stage(Stage.EPILOGUE, () => {
      const logits = intrinsics.output(0, 'logits');
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
    expect(traced.identityHash()).toBe(GOLDENS.get('s3')!.hash);
    check('s3', traced);
  });

  it('text_completion_decode', () => {
    const tokIn = chFrom([42], dtype.i32);
    const embedIndptr = chFrom([0, 1], dtype.u32);
    const fold = chFrom([0xffff_ffff], dtype.u32);
    const tokOut = chNew([1], dtype.i32);
    const b = new Builder(VOCAB, PAGE);
    b.setOutputs(logitsDecl(VOCAB));
    bindGeometry(b, [
      [Port.EMBED_TOKENS, tokIn],
      [Port.EMBED_INDPTR, embedIndptr],
      [Port.FOLD, fold],
    ]);
    b.stage(Stage.EPILOGUE, () => {
      tokOut.putTensor(reshape(reduceArgmax(intrinsics.output(0, 'logits')), [1]));
    });
    tokOut.noteHostTake();
    check('text_completion_decode', b.build());
  });

  it('naive_decode', () => {
    const temperature = 0.7;
    const cap = 8;
    const tokIn = chFrom([3], dtype.i32);
    const rngCh = chFrom([0x7ce1 ^ 0x5bd1, 0], dtype.u32);
    const tokOut = chNew([1], dtype.i32).capacity(cap);
    const s1 = chNew([1], dtype.f32).capacity(cap);
    const s2 = chNew([1], dtype.f32).capacity(cap);
    const lane1 = chFrom([0, 1], dtype.u32);
    const fold = chFrom([0xffff_ffff], dtype.u32);
    const b = new Builder(VOCAB, PAGE);
    b.setOutputs(logitsDecl(VOCAB));
    bindGeometry(b, [
      [Port.EMBED_TOKENS, tokIn],
      [Port.EMBED_INDPTR, lane1],
      [Port.FOLD, fold],
    ]);
    b.stage(Stage.EPILOGUE, () => {
      const r = rngCh.take();
      const logits = intrinsics.output(0, 'logits');
      const scaled = logits.div(temperature);
      const token = gumbelMax(scaled, r);
      const rNext = r.add(iota(2));
      tokIn.putTensor(token);
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
    const tok = chFrom([1], dtype.i32);
    const ind = chFrom([0, 1], dtype.u32);
    const rngCh = chFrom([1, 2], dtype.u32);
    const topP = chFrom([0.9], dtype.f32);
    const bias = chFrom([0.0, -1.5, 2.25, 0.0], dtype.f32);
    const out = chNew([1], dtype.i32);
    const stat = chNew([1], dtype.f32);
    const stat2 = chNew([4], dtype.f32);
    const flag = chNew([1], dtype.bool);
    const b = new Builder(VOCAB, PAGE);
    b.setOutputs(logitsDecl(VOCAB));
    b.bindPort(Port.EMBED_TOKENS, tok);
    b.bindPort(Port.EMBED_INDPTR, ind);
    b.stage(Stage.EPILOGUE, () => {
      const logits = intrinsics.output(0, 'logits');
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
    const tok = chFrom([1], dtype.i32);
    const ind = chFrom([0, 1], dtype.u32);
    const a = chNew([2, 4, 8], dtype.f32);
    const bch = chNew([2, 8, 4], dtype.f32);
    const out = chNew([1], dtype.i32);
    hostPut(a, new Array(64).fill(0.0), dtype.f32);
    hostPut(bch, new Array(64).fill(0.0), dtype.f32);
    const b = new Builder(VOCAB, PAGE);
    b.setOutputs(logitsDecl(VOCAB));
    b.bindPort(Port.EMBED_TOKENS, tok);
    b.bindPort(Port.EMBED_INDPTR, ind);
    b.stage(Stage.PROLOGUE, () => {
      intrinsics.kernel.lora(a.read(), bch.read(), Tensor.constant(1 | 4, Dtype.U32));
      intrinsics.kernel.attnPageMask(iota(4));
    });
    b.stage(Stage.EPILOGUE, () => {
      out.putTensor(reshape(reduceArgmax(intrinsics.output(0, 'logits')), [1]));
    });
    out.noteHostTake();
    check('sinks', b.build());
  });

  it('refuses a bulk constant', () => {
    const tok = chFrom([1], dtype.i32);
    const ind = chFrom([0, 1], dtype.u32);
    const out = chNew([4], dtype.f32);
    const b = new Builder(VOCAB, PAGE);
    b.setOutputs(logitsDecl(VOCAB));
    b.bindPort(Port.EMBED_TOKENS, tok);
    b.bindPort(Port.EMBED_INDPTR, ind);
    b.stage(Stage.EPILOGUE, () => out.putTensor(Tensor.constant([1.0, 2.0, 5.0, 7.0], dtype.f32).add(0.0)));
    out.noteHostTake();
    expect(() => b.build()).toThrow(TraceError);
  });

  it('readiness lint', () => {
    const tok = chFrom([1], dtype.i32);
    const ind = chFrom([0, 1], dtype.u32);
    const never = chNew([1], dtype.f32);
    const out = chNew([1], dtype.f32);
    const b = new Builder(VOCAB, PAGE);
    b.setOutputs(logitsDecl(VOCAB));
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
    const tok = chNew([1], dtype.i32);
    const ind = chFrom([0, 1], dtype.u32);
    const out = chNew([vocab], dtype.bool);
    hostPut(tok, [1], dtype.i32);
    const b = new Builder(vocab, 16);
    b.setOutputs(logitsDecl(vocab));
    b.bindPort(Port.EMBED_TOKENS, tok);
    b.bindPort(Port.EMBED_INDPTR, ind);
    b.stage(Stage.EPILOGUE, () => out.putTensor(body(intrinsics.output(0, 'logits'))));
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

describe('diffusion golden', () => {
  it('diffusion_step matches inferlet_goldens.rs', () => {
    const length = 8;
    const taps = 4;
    const base = 5;
    const end = base + length;
    const bound = 0.5;
    const confidence = 0.1;
    const toks = chFrom(range(length).map((i) => (i * 7919) % 1000), dtype.i32);
    const embedIndptr = chFrom([0, length], dtype.u32);
    const positions = chFrom(range(base, end), dtype.u32);
    const kvLen = chFrom([end], dtype.u32);
    const readout = chFrom(range(length), dtype.u32);
    const temp = chFrom([1.0], dtype.f32);
    const rngState = chFrom([7, 0], dtype.u32);
    const history = chFrom(new Array(length).fill(-1), dtype.i32);
    const canvasOut = chNew([length], dtype.i32);
    const argmaxOut = chNew([length], dtype.i32);
    const stop = chNew([1], dtype.bool);
    const meanOut = chNew([1], dtype.f32);
    const tapIdsOut = chNew([length, taps], dtype.u32);
    const tapWeightsOut = chNew([length, taps], dtype.f32);

    const b = new Builder(VOCAB, PAGE);
    b.setOutputs(logitsDecl(VOCAB));
    bindGeometry(b, [
      [Port.EMBED_TOKENS, toks], [Port.EMBED_INDPTR, embedIndptr], [Port.KV_LEN, kvLen],
      [Port.POSITIONS, positions], [Port.READOUT, readout],
    ]);
    b.stage(Stage.EPILOGUE, () => {
      positions.putTensor(positions.take());
      const r = rngState.take();
      const t = reshape(temp.read(), []);
      const logits = intrinsics.output(0, 'logits');
      const scaled = logits.div(t);
      const probs = softmax(scaled);
      const h = entropy(probs);
      const sampled = gumbelMax(scaled, r);
      const argmax = reduceArgmax(scaled);
      const accept = entropyBoundAccept(h, bound);
      const rNoise = r.add(iota(2));
      const noise = cast(rng(rNoise, [length]).mul(VOCAB), dtype.i32);
      const next = select(accept, sampled, noise);
      const previous = history.take();
      history.putTensor(argmax);
      const done = stableAndConfident(argmax, previous, h, confidence);
      const [tapWeights, tapIds] = topK(probs, taps);
      tapIdsOut.putTensor(tapIds);
      tapWeightsOut.putTensor(tapWeights);
      canvasOut.putTensor(next);
      argmaxOut.putTensor(argmax);
      stop.putTensor(reshape(done, [1]));
      meanOut.putTensor(reshape(reduceSum(h).div(length), [1]));
      rngState.putTensor(rNoise.add(iota(2)));
    });
    for (const c of [canvasOut, argmaxOut, stop, meanOut, tapIdsOut, tapWeightsOut]) c.noteHostTake();
    check('diffusion_step', b.build());
  });
});

describe('beam golden', () => {
  it('beam_step matches inferlet_goldens.rs', () => {
    const B = 2;
    const poolLen = 8 * PAGE;
    const v = VOCAB;
    const initMask: boolean[] = [];
    for (let b = 0; b < B; b++) for (let p = 0; p < poolLen; p++) initMask.push(p === 0);
    const mask = DslChannel.fromConst(new ConstData(shapeOf([B, poolLen]), Dtype.BOOL, packElems(initMask, Dtype.BOOL)));
    const scores = chFrom([0.0, -Infinity], dtype.f32);
    const toks = chFrom(new Array(B).fill(1), dtype.i32);
    const pos = chFrom(new Array(B).fill(0), dtype.u32);
    const fill = chFrom([1], dtype.u32);
    const klen = chFrom(new Array(B).fill(1), dtype.u32);
    const slots = chFrom(new Array(B).fill(0), dtype.u32);
    const lanesB = chFrom(range(B + 1), dtype.u32);
    const out = chNew([B], dtype.i32).capacity(8);
    const outPar = chNew([B], dtype.u32).capacity(8);
    const outScr = chNew([B], dtype.f32).capacity(8);
    const outGreedy = chNew([B], dtype.i32).capacity(8);

    const b = new Builder(VOCAB, PAGE);
    b.setOutputs(logitsDecl(VOCAB));
    bindGeometry(b, [
      [Port.KV_LEN, klen], [Port.POSITIONS, pos], [Port.SLOTS, slots], [Port.ATTN_MASK, mask],
      [Port.EMBED_TOKENS, toks], [Port.EMBED_INDPTR, lanesB],
    ]);
    b.stage(Stage.EPILOGUE, () => {
      const logits = reshape(intrinsics.output(0, 'logits'), [B, v]);
      const cand = broadcast(reshape(scores.take(), [B, 1]), [B, v]).add(logSoftmax(logits));
      const [s, i] = topK(reshape(cand, [B * v]), B);
      const parent = i.div(v);
      const tokI = cast(i.rem(v), dtype.i32);
      const base = fill.take();
      const lane = iota(B);
      const baseB = broadcast(reshape(base, [1]), [B]);
      const wpos = baseB.add(lane);
      const inherited = gather(mask.take(), parent);
      const col = broadcast(reshape(iota(poolLen), [1, poolLen]), [B, poolLen]);
      const wposB = broadcast(reshape(wpos, [B, 1]), [B, poolLen]);
      const newpos = eq(col, wposB);
      mask.putTensor(or(inherited, newpos));
      slots.putTensor(wpos);
      const filled = base.add(B);
      klen.putTensor(broadcast(reshape(filled, [1]), [B]));
      pos.putTensor(pos.take().add(1));
      fill.putTensor(filled);
      scores.putTensor(s);
      toks.putTensor(tokI);
      out.putTensor(tokI);
      outPar.putTensor(parent);
      outScr.putTensor(s);
      outGreedy.putTensor(reshape(reduceArgmax(logits), [B]));
    });
    check('beam_step', b.build());
  });
});

describe('sampler golden', () => {
  // `inferlet_goldens.rs::latent_step` — the image sampler's epilogue: the
  // `velocity` output, an `N(0, 1)` draw, and the `sin`/`cos`/`sqrt`/`rsqrt`
  // ops. The one golden carrying tags 0x08..0x0b, `RngKind.NORMAL` and
  // the `velocity` output, so a port that drifts on any of them fails here.
  it('latent_step', () => {
    const rows = 8;
    const channels = 16;
    const tok = chFrom([1], dtype.i32);
    const indptrCh = chFrom([0, 1], dtype.u32);
    const readout = chFrom(range(rows), dtype.u32);
    const kvLen = chFrom([rows], dtype.u32);
    const positions = chFrom(range(rows), dtype.u32);
    const latent = chNew([rows, channels], dtype.f32);
    const dsigma = chFrom([-0.25], dtype.f32);
    const rngCh = chFrom([9, 0], dtype.u32);
    const out = chNew([rows, channels], dtype.f32);
    const normOut = chNew([rows], dtype.f32);
    hostPut(latent, new Array(rows * channels).fill(0), dtype.f32);

    const b = new Builder(VOCAB, PAGE);
    b.setOutputs([[decl('velocity', Axis.ROW, channels, false)]]);
    bindGeometry(b, [
      [Port.EMBED_TOKENS, tok], [Port.EMBED_INDPTR, indptrCh], [Port.KV_LEN, kvLen],
      [Port.POSITIONS, positions], [Port.READOUT, readout],
    ]);
    b.stage(Stage.EPILOGUE, () => {
      positions.putTensor(positions.take());
      const v = intrinsics.outputRows(0, 'velocity', rows);
      const x = latent.take();
      const d = reshape(dsigma.read(), []);

      const square = mul(v, v);
      const energy = reduceSum(square);
      const norm = sqrt(energy);
      const guarded = add(energy, 1.0e-12);
      const inverse = rsqrt(guarded);
      const column = reshape(inverse, [rows, 1]);
      const spread = broadcast(column, [rows, channels]);
      const unit = mul(v, spread);

      const ramp = cast(iota(channels), dtype.f32);
      const rampRow = reshape(ramp, [1, channels]);
      const rampPlane = broadcast(rampRow, [rows, channels]);
      const angle = mul(rampPlane, d);
      const sine = sin(angle);
      const cosine = cos(angle);
      const embedding = add(sine, cosine);

      const r = rngCh.take();
      const z = normal(r, [rows, channels]);
      const drift = add(unit, embedding);
      const step = mul(drift, d);
      const jitter = mul(z, d);
      const moved = add(x, step);
      const stepped = add(moved, jitter);

      latent.putTensor(stepped);
      out.putTensor(stepped);
      normOut.putTensor(norm);
      rngCh.putTensor(add(r, iota(2)));
    });
    out.noteHostTake();
    normOut.noteHostTake();
    check('latent_step', b.build());
  });

  // `inferlet_goldens.rs::vae_readback`: the one golden reading a voxel output.
  it('vae_readback matches the Rust golden', () => {
    const rows = 16;
    const rgb = 3;
    const out = chNew([rows, rgb], dtype.f32);
    const b = new Builder(VOCAB, PAGE);
    b.setOutputs([[decl('pixels', Axis.VOXEL, rgb, false)]]);
    b.stage(Stage.EPILOGUE, () => {
      const px = intrinsics.outputRows(0, 'pixels', rows);
      const shifted = add(px, 1.0);
      const unit = mul(shifted, 0.5);
      out.putTensor(unit);
    });
    out.noteHostTake();
    check('vae_readback', b.build());
  });
  // `inferlet_goldens.rs::noised_input`: the one golden carrying an input table.
  it('noised_input matches the Rust golden', () => {
    const rows = 4;
    const width = 8;
    const rngCh = chFrom([5, 0], dtype.u32);
    const latent = chNew([rows, width], dtype.f32);
    const timestep = chFrom([0.5], dtype.f32);
    const b = new Builder(VOCAB, PAGE);
    b.setOutputs(logitsDecl(VOCAB));
    b.bindInput(latent);
    b.bindInput(timestep);
    b.stage(Stage.PROLOGUE, () => {
      const r = rngCh.take();
      latent.putTensor(normal(r, [rows, width]));
      rngCh.putTensor(add(r, iota(2)));
    });
    check('noised_input', b.build());
  });
});
