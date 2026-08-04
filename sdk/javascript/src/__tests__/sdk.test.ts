// Unit tests for the inferlet SDK's hand-written layer.
//
// SCOPE NOTE: these cover the non-forward interfaces only. That is not an
// omission -- it is the SDK's entire surface right now. The forward-pass
// interfaces (`forward` / `forward-recurrent` / `forward-hybrid` plus
// `channel`, `working-set`, `pipeline`) have no JavaScript counterpart,
// because the guest now ships PTIR container bytes and the encoder that
// produces them exists only in Rust. `scripts/check-sdk-interfaces.sh` fails
// on exactly that, deliberately, and blocks publishing until it is fixed.
//
// This file did not exist before: `package.json` ran `vitest run src/__tests__`
// against a directory that was never created, so `npm test` passed by testing
// nothing while the SDK imported a deleted interface.

import { beforeEach, describe, expect, it } from 'vitest';

import * as model from '../model.js';
import * as tokenizer from '../tokenizer.js';
import * as session from '../session.js';
import * as chat from '../chat.js';
import * as reasoning from '../reasoning.js';
import * as sdk from '../index.js';
import { chatScript, reasoningScript, sessionSpy } from './stubs.js';

describe('package surface', () => {
  it('exports only the live modules', () => {
    expect(Object.keys(sdk).sort()).toEqual([
      'chat',
      'model',
      'reasoning',
      'session',
      'tokenizer',
    ]);
  });

  it('does not re-export the removed forward surface', () => {
    // A stub that throws on use would be worse than nothing: it would make
    // scripts/check-sdk-interfaces.sh green while the SDK still cannot run a
    // model.
    for (const gone of ['Context', 'Forward', 'Generator', 'Sampler', 'Adapter']) {
      expect(sdk).not.toHaveProperty(gone);
    }
  });
});

describe('model', () => {
  it('reports identity', () => {
    expect(model.name()).toBe('mock-model');
    expect(model.architecture()).toBe('qwen3_5');
    expect(model.defaultSystemSpeculation()).toBe(false);
  });

  it('keeps passKind and isLinear consistent', () => {
    // `isLinear() === (passKind() !== 'attention')` is a documented invariant
    // of the WIT interface, and the reason both exist.
    expect(model.passKind()).toBe('hybrid');
    expect(model.isLinear()).toBe(model.passKind() !== 'attention');
  });

  it('exposes memory-shaping capabilities', () => {
    expect(model.kvPageSize()).toBe(16);
    expect(model.frameSize()).toBe(1);
    expect(model.channelCapacity()).toBe(8);
    expect(model.maxEmbedLength()).toBe(2048);
    expect(model.arenaBlockSize()).toBe(8192n);
  });

  it('exposes recurrent capabilities', () => {
    expect(model.rsStateSize()).toBe(4096n);
    expect(model.rsBufferPageSize()).toBe(64);
    expect(model.rsFoldGranularity()).toBe(1);
  });

  it('reports an output vocab that may exceed the tokenizer vocab', () => {
    // Different numbers on purpose -- logits are padded. Use outputVocabSize
    // for anything logits-shaped.
    const [ids] = tokenizer.vocabs();
    expect(model.outputVocabSize()).toBeGreaterThan(ids.length);
  });

  it('no longer carries the tokenizer surface', () => {
    // It lives in `tokenizer` now; an alias on `model` would hide the
    // interface split from anyone reading the SDK.
    for (const gone of ['encode', 'decode', 'vocabs', 'splitRegex', 'specialTokens']) {
      expect(model).not.toHaveProperty(gone);
    }
  });
});

describe('tokenizer', () => {
  it('round-trips encode/decode', () => {
    expect(Array.from(tokenizer.encode('Hi'))).toEqual([72, 105]);
    expect(tokenizer.decode(Uint32Array.from([72, 105]))).toBe('Hi');
  });

  it('returns vocabs and special tokens', () => {
    const [ids, bytes] = tokenizer.vocabs();
    expect(Array.from(ids)).toEqual([0, 1]);
    expect(bytes).toHaveLength(2);

    const [sids] = tokenizer.specialTokens();
    expect(Array.from(sids)).toEqual([2]);
  });

  it('returns the split regex', () => {
    expect(tokenizer.splitRegex()).toBe('\\w+');
  });
});

describe('session', () => {
  beforeEach(() => sessionSpy.reset());

  it('sends strings verbatim', () => {
    session.send('plain text');
    expect(sessionSpy.sent).toEqual(['plain text']);
  });

  it('JSON-stringifies anything else', () => {
    session.send({ event: 'tick', n: 3 });
    session.send([1, 2, 3]);
    expect(JSON.parse(sessionSpy.sent[0]!)).toEqual({ event: 'tick', n: 3 });
    expect(JSON.parse(sessionSpy.sent[1]!)).toEqual([1, 2, 3]);
  });

  it('receives text', async () => {
    sessionSpy.toReceive.push('hello');
    await expect(session.receive()).resolves.toBe('hello');
  });

  it('resolves to undefined once the client has closed', async () => {
    // Unlike the Python SDK, which raises, this surface reports end-of-stream
    // as a value. Pinned because the two SDKs differ here on purpose.
    await expect(session.receive()).resolves.toBeUndefined();
    await expect(session.receiveFile()).resolves.toBeUndefined();
  });

  it('round-trips files', async () => {
    session.sendFile(new Uint8Array([0, 1]));
    expect(Array.from(sessionSpy.sentFiles[0]!)).toEqual([0, 1]);

    sessionSpy.filesToReceive.push(new Uint8Array([2]));
    const got = await session.receiveFile();
    expect(Array.from(got!)).toEqual([2]);
  });
});

describe('chat fillers', () => {
  it('returns token sequences', () => {
    expect(chat.system('s')[0]).toBe(1);
    expect(chat.user('u')[0]).toBe(3);
    expect(chat.assistant('a')[0]).toBe(5);
    expect(Array.from(chat.cue())).toEqual([6]);
    expect(Array.from(chat.seal())).toEqual([7]);
    expect(Array.from(chat.stopTokens())).toEqual([8, 9]);
  });
});

describe('chat decoder', () => {
  beforeEach(() => {
    chatScript.events = [];
  });

  const decoder = (events: { tag: string; val?: unknown }[]) => {
    chatScript.events = events;
    return new chat.Decoder();
  };

  it('maps delta and done', () => {
    const d = decoder([
      { tag: 'delta', val: 'he' },
      { tag: 'delta', val: 'llo' },
      { tag: 'done', val: 'hello' },
    ]);
    expect(d.feed(Uint32Array.from([1]))).toEqual(chat.Event.Delta('he'));
    expect(d.feed(Uint32Array.from([2]))).toEqual(chat.Event.Delta('llo'));
    expect(d.feed(Uint32Array.from([3]))).toEqual(chat.Event.Done('hello'));
  });

  it('turns an empty delta into idle', () => {
    // So callers never need an `if (text)` guard around a Delta branch.
    const d = decoder([{ tag: 'delta', val: '' }]);
    expect(d.feed(Uint32Array.from([1]))).toEqual(chat.Event.Idle());
  });

  it('surfaces interrupt raw', () => {
    const d = decoder([{ tag: 'interrupt', val: 42 }]);
    expect(d.feed(Uint32Array.from([1]))).toEqual(chat.Event.Interrupt(42));
  });

  it('forwards tokens and resets to the host decoder', () => {
    const d = decoder([{ tag: 'delta', val: 'x' }]);
    d.feed(Uint32Array.from([7, 8]));
    expect(chatScript.last!.fed).toEqual([[7, 8]]);
    d.reset();
    expect(chatScript.last!.resets).toBe(1);
  });
});

describe('reasoning decoder', () => {
  beforeEach(() => {
    reasoningScript.events = [];
  });

  const decoder = (events: { tag: string; val?: unknown }[]) => {
    reasoningScript.events = events;
    return new reasoning.Decoder();
  };

  it('maps start, delta and complete', () => {
    const d = decoder([
      { tag: 'start' },
      { tag: 'delta', val: 'think' },
      { tag: 'complete', val: 'think' },
    ]);
    expect(d.feed(Uint32Array.from([1]))).toEqual(reasoning.Event.Start());
    expect(d.feed(Uint32Array.from([2]))).toEqual(reasoning.Event.Delta('think'));
    // The WIT case is `complete`; the SDK calls it `End`. The rename is the
    // reason this mapping is worth pinning.
    expect(d.feed(Uint32Array.from([3]))).toEqual(reasoning.Event.End('think'));
  });

  it('turns an empty delta into idle', () => {
    const d = decoder([{ tag: 'delta', val: '' }]);
    expect(d.feed(Uint32Array.from([1]))).toEqual(reasoning.Event.Idle());
  });
});
