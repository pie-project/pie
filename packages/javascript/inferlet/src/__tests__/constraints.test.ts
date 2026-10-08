// The grammar / mask / tools / media wrappers, against the stubs.

import { describe, expect, it } from 'vitest';

import * as grammar from '../grammar.js';
import * as mask from '../mask.js';
import * as media from '../media.js';
import { ForwardPass, InferletError } from '../eta/bridge.js';
import * as tools from '../tools.js';

describe('mask', () => {
  it('indexes word and bit', () => {
    const m = [0b101];
    expect(mask.bitAllowed(m, 0)).toBe(true);
    expect(mask.bitAllowed(m, 1)).toBe(false);
    expect(mask.bitAllowed(m, 2)).toBe(true);
    expect(mask.bitAllowed([0, 0b10], 33)).toBe(true);
    expect(mask.bitAllowed([0, 0b10], 32)).toBe(false);
  });

  it('refuses tokens past the mask', () => {
    const m = mask.packAllowed(151_669, [7, 151_668]);
    expect(m.length).toBe(4740);
    expect(mask.bitAllowed(m, 7)).toBe(true);
    expect(mask.bitAllowed(m, 151_668)).toBe(true);
    for (const j of [151_680, 151_935]) expect(mask.bitAllowed(m, j)).toBe(false);
  });

  it('pack/unpack round-trips', () => {
    const m = mask.packAllowed(40, [0, 2, 33, 99]);
    expect(m.length).toBe(2);
    const set = mask.unpackMask(m, 40).flatMap((b, i) => (b ? [i] : []));
    expect(set).toEqual([0, 2, 33]);
    expect(mask.unpackMask([], 5)).toEqual([true, true, true, true, true]);
    expect(Array.from(mask.allAllowed(33))).toEqual([0xffff_ffff, 0xffff_ffff]);
    expect(mask.applyMaskArgmax([0.1, 9.0, 3.0], [0b101])).toBe(2);
  });
});

describe('grammar', () => {
  it('walks and terminates', () => {
    const g = grammar.Grammar.fromJsonSchema('{"type":"object"}');
    expect(g.toString()).toBe('{"type":"object"}');
    const m = new grammar.Matcher(g);
    expect(Array.from(m.mask())).toEqual([0b101]);
    m.acceptTokens([1, 2]);
    expect(m.isTerminated()).toBe(false);
    const f = m.fork();
    f.acceptTokens([3]);
    expect(f.isTerminated()).toBe(true);
    expect(m.isTerminated()).toBe(false);
    m.rollback(1);
    expect(m.rollbackCapacity()).toBe(1);
  });

  it('surfaces host refusals as GrammarError', () => {
    expect(() => grammar.Grammar.fromJsonSchema('bad')).toThrow(grammar.GrammarError);
    const m = new grammar.Matcher(grammar.Grammar.json());
    expect(() => m.acceptTokens([999])).toThrow(/999/);
  });
});

describe('tools', () => {
  it('equips, answers, formats', () => {
    expect(Array.from(tools.equip(['a', 'b']))).toEqual([10, 2]);
    expect(Array.from(tools.answer('f', 'xyz'))).toEqual([11, 1, 3]);
    expect(tools.format([])).toBeUndefined();
    expect(tools.format(['a'])!.toString()).toBe('tools');
    expect(Array.from(tools.createMatcher(['a']).mask())).toEqual([0b101]);
  });

  it('decodes events', () => {
    const d = new tools.Decoder();
    expect(d.feed(Uint32Array.from([1]))).toEqual({ type: 'start' });
    expect(d.feed(Uint32Array.from([2]))).toEqual({ type: 'call', call: { name: 'lookup', argumentsJson: '{"q": 1}' } });
  });
});

describe('media', () => {
  it('wraps an image', () => {
    const img = media.Image.fromBytes(new Uint8Array([1]));
    expect(Array.from(img.tokens())).toEqual([5, 6, 7]);
    expect(img.grid()).toEqual({ t: 1, h: 2, w: 3 });
    expect(() => media.Image.fromBytes(new Uint8Array())).toThrow(media.MediaError);
  });

  it('ForwardPass.media hands the pass the WIT resources, tagged', () => {
    const img = media.Image.fromBytes(new Uint8Array([1]));
    const aud = media.Audio.fromBytes(new Uint8Array([2]));
    const fwd = new ForwardPass();
    fwd.media([img, aud]);
    expect((fwd.wit as unknown as { spans: unknown[] }).spans).toEqual([
      { tag: 'image', val: img.handle },
      { tag: 'audio', val: aud.handle },
    ]);
    expect(() => fwd.media([new Uint8Array() as unknown as media.Image])).toThrow(TypeError);
  });
});

describe('ForwardPass per-kind binders', () => {
  it('refuse the wrong pass kind before touching any state', () => {
    const hybrid = new ForwardPass('hybrid');
    expect(() => hybrid.attention(undefined as never, undefined as never)).toThrow(InferletError);
    expect(() => hybrid.bindRecurrent([], {})).toThrow(/bindRecurrent binds/);
    const attention = new ForwardPass('attention');
    expect(() => attention.bindHybrid(undefined, [], {})).toThrow(/bindHybrid binds/);
  });
});

describe('diffusion pass surface', () => {
  it('binds the canvas reading and stages self-conditioning taps', () => {
    const fwd = new ForwardPass('diffusion');
    fwd.canvas('denoise');
    fwd.selfConditioning([1, 2, 3, 4], [0.5, 0.25, 0.125, 0.125]);
    const w = fwd.wit as unknown as { mode: string; selfCond: [number[], number[]] };
    expect(w.mode).toBe('denoise');
    expect(w.selfCond).toEqual([[1, 2, 3, 4], [0.5, 0.25, 0.125, 0.125]]);
    expect(() => new ForwardPass('attention').canvas('encode')).toThrow(/canvas binds/);
  });
});
