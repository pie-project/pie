// Forward — single forward-pass primitive with explicit KV descriptors.
//
// `ctx.forward()` returns a builder. Attach inputs, samplers, probes, masks,
// adapters, and optional media spans, then `await forward.execute()`.

import {
  ForwardPass as _ForwardPass,
} from 'pie:core/inference';
import type {
  Audio,
  Image,
  Sampler as WitSampler,
  Brle,
  Output as WitOutput,
  SlotOutput,
} from 'pie:core/inference';

import type { Adapter } from './adapter.js';
import type { Context } from './context.js';
import {
  type Probe,
  type ProbeKind,
  type ProbeKindOf,
  type Sampler,
  _probeAccessorKind,
  _probeToWit,
} from './sample.js';

// =============================================================================
// Slot handles
// =============================================================================

/** Reference to a sampler slot. Pass to `output.token()` / `output.tokensAt()`. */
export interface SampleHandle {
  readonly slot: number;
  readonly arity: number;
}

/** Reference to a probe slot. The phantom `K` tag selects output accessors. */
export interface ProbeHandle<K extends ProbeKind = ProbeKind> {
  readonly slot: number;
  readonly kind: K;
}

// =============================================================================
// Slot specs (internal)
// =============================================================================

interface _SampleSlot {
  readonly type: 'sample';
  readonly indices: Uint32Array;
  readonly sampler: Sampler;
}

interface _ProbeSlot {
  readonly type: 'probe';
  readonly index: number;
  readonly wit: WitSampler;
}

type _Slot = _SampleSlot | _ProbeSlot;

// =============================================================================
// Forward
// =============================================================================

/** Single forward pass. Construct via `ctx.forward()`. */
export class Forward {
  readonly #ctx: Context;
  #autoInputs: number[] = [];
  #explicitInputs: Array<[Uint32Array, Uint32Array]> = [];
  #slots: _Slot[] = [];
  #nextSlot = 0;
  #mask: Brle | undefined;
  #attnMask: Brle[] | undefined;
  #adapter: Adapter | undefined;
  #zoSeed: number | undefined;
  #images: Array<[Image, number]> = [];
  #audios: Array<[Audio, number]> = [];
  #deferCommit = false;

  constructor(ctx: Context) {
    this.#ctx = ctx;
  }

  // ── Position accessors ──────────────────────────────────────────────────

  /** Position the first auto-input token will occupy. */
  startPosition(): number {
    return this.#ctx._seqLen;
  }

  /** Page size of the owning context. */
  pageSize(): number {
    return this.#ctx._pageSize;
  }

  // ── Inputs ──────────────────────────────────────────────────────────────

  /** Append `tokens` at positions starting at the context's current seqLen. */
  input(tokens: Iterable<number>): this {
    for (const t of tokens) this.#autoInputs.push(t);
    return this;
  }

  /** Feed `tokens` at caller-supplied `positions` for scoring/probing. */
  inputAt(tokens: Uint32Array, positions: Uint32Array): this {
    if (tokens.length !== positions.length) {
      throw new Error('tokens and positions must be the same length');
    }
    this.#explicitInputs.push([tokens, positions]);
    return this;
  }

  /** Splice an encoded visual span at `anchor`. */
  inputImage(image: Image, anchor: number): this {
    this.#images.push([image, anchor]);
    return this;
  }

  /** Splice an encoded audio clip at `anchor`. */
  inputAudio(audio: Audio, anchor: number): this {
    this.#audios.push([audio, anchor]);
    return this;
  }

  /** Run the pass but leave the context cursor unchanged. */
  deferCommit(): this {
    this.#deferCommit = true;
    return this;
  }

  // ── Slot attach ─────────────────────────────────────────────────────────

  /** Attach a sampler at one or more query indices. */
  sample(indices: Iterable<number>, sampler: Sampler): SampleHandle {
    const idxArr = indices instanceof Uint32Array
      ? new Uint32Array(indices)
      : Uint32Array.from(indices);
    const arity = idxArr.length;
    const h: SampleHandle = { slot: this.#nextSlot, arity };
    this.#slots.push({ type: 'sample', indices: idxArr, sampler });
    this.#nextSlot += arity;
    return h;
  }

  /** Attach a probe at a single query index. */
  probe<P extends Probe>(index: number, probe: P): ProbeHandle<ProbeKindOf<P>> {
    const h: ProbeHandle<ProbeKindOf<P>> = {
      slot: this.#nextSlot,
      kind: _probeAccessorKind(probe) as ProbeKindOf<P>,
    };
    this.#slots.push({ type: 'probe', index, wit: _probeToWit(probe) });
    this.#nextSlot += 1;
    return h;
  }

  // ── Decoration ──────────────────────────────────────────────────────────

  /** Set a static logit mask (BRLE) applied at every sampled position. */
  mask(brle: Brle): this {
    this.#mask = brle;
    return this;
  }

  /** Set per-query-position attention masks. */
  attentionMask(masks: Brle[]): this {
    this.#attnMask = masks;
    return this;
  }

  /** Apply an adapter (LoRA, etc.) for this forward pass. */
  adapter(adapter: Adapter): this {
    this.#adapter = adapter;
    return this;
  }

  /** Set a zo (Evolution Strategies) seed for this forward pass. */
  zoSeed(seed: number): this {
    this.#zoSeed = seed;
    return this;
  }

  // ── Execute ─────────────────────────────────────────────────────────────

  /** Run the forward pass and advance the context cursor for new tail KV. */
  async execute(): Promise<Output> {
    const ctx = this.#ctx;
    const nAuto = this.#autoInputs.length;
    const nExplicit = this.#explicitInputs.reduce((a, [t]) => a + t.length, 0);
    let softTokens = 0;
    for (const [image] of this.#images) softTokens += image.tokenCount();
    for (const [audio] of this.#audios) softTokens += audio.tokenCount();
    const nWrite = nAuto + softTokens;

    if (
      nAuto + nExplicit + this.#images.length + this.#audios.length === 0 &&
      this.#slots.length === 0
    ) {
      throw new Error(
        'Forward.execute() called with no inputs and no slots. ' +
          'Attach at least one input (`forward.input(...)`) or slot ' +
          '(`forward.sample(...)` / `forward.probe(...)`) before executing.',
      );
    }

    const fwd = new _ForwardPass();
    if (nWrite > 0) {
      ctx._attachKv(fwd, ctx._prepareWrite(nWrite));
    } else {
      ctx._attachFullContext(fwd);
    }

    for (const [image, anchor] of this.#images) fwd.inputImage(image, anchor);
    for (const [audio, anchor] of this.#audios) fwd.inputAudio(audio, anchor);

    if (this.#adapter !== undefined) {
      fwd.adapter(this.#adapter._handle);
    }
    if (this.#zoSeed !== undefined) {
      const zoMod = await import('pie:zo/zo' as any);
      zoMod.adapterSeed(fwd, BigInt(this.#zoSeed));
    }

    if (nAuto > 0) {
      const positions = new Uint32Array(nAuto);
      for (let i = 0; i < nAuto; i++) positions[i] = ctx._seqLen + i;
      fwd.inputTokens(Uint32Array.from(this.#autoInputs), positions);
    }
    for (const [tokens, positions] of this.#explicitInputs) {
      fwd.inputTokens(tokens, positions);
    }

    for (const spec of this.#slots) {
      if (spec.type === 'sample') {
        fwd.sampler(spec.indices, spec.sampler._variant);
      } else {
        fwd.sampler(Uint32Array.of(spec.index), spec.wit);
      }
    }

    if (this.#mask !== undefined) fwd.logitMask(this.#mask);
    if (this.#attnMask !== undefined) fwd.attentionMask(this.#attnMask);

    const raw = await fwd.execute();

    if (nWrite > 0 && !this.#deferCommit) {
      ctx._seqLen += nWrite;
      ctx._history.push(...this.#autoInputs);
    }
    if (softTokens > 0) {
      ctx._snapshottable = false;
    }

    return new Output(raw);
  }
}

// =============================================================================
// Output
// =============================================================================

/** Result of one forward-pass execution. */
export class Output {
  readonly #raw: WitOutput;

  /** Generator-accepted tokens this step; empty for raw `Forward.execute()`. */
  readonly tokens: Uint32Array;

  /** Handle for the Generator's auto-attached sampler, if any. */
  readonly autoSampler: SampleHandle | undefined;

  constructor(
    raw: WitOutput,
    tokens: Uint32Array = new Uint32Array(),
    autoSampler?: SampleHandle,
  ) {
    this.#raw = raw;
    this.tokens = tokens;
    this.autoSampler = autoSampler;
  }

  /** Underlying WIT output. */
  get raw(): WitOutput { return this.#raw; }

  // ── Sampler accessors ─────────────────────────────────────────────────

  /** First token from a single-index sampler slot. */
  token(h: SampleHandle): number | undefined {
    const slot = this.#slot(h.slot);
    return slot?.tag === 'token' ? slot.val : undefined;
  }

  /** Tokens at the slot range a multi-index sampler covers. */
  tokensAt(h: SampleHandle): Uint32Array {
    const out: number[] = [];
    for (let i = 0; i < h.arity; i++) {
      const slot = this.#slot(h.slot + i);
      if (slot?.tag === 'token') out.push(slot.val);
      else break;
    }
    return Uint32Array.from(out);
  }

  // ── Probe accessors ───────────────────────────────────────────────────

  /** Distribution as `[ids, probs]` for a `Distribution` probe. */
  distribution(h: ProbeHandle<'distribution'>): [Uint32Array, Float32Array] | undefined {
    const slot = this.#slot(h.slot);
    return slot?.tag === 'distribution' ? slot.val : undefined;
  }

  /** Raw logits bytes for a `Logits` probe. */
  logits(h: ProbeHandle<'logits'>): Uint8Array | undefined {
    const slot = this.#slot(h.slot);
    return slot?.tag === 'logits' ? slot.val : undefined;
  }

  /** Logprob list for a `Logprob` / `Logprobs` probe. */
  logprobs(h: ProbeHandle<'logprobs'>): Float32Array | undefined {
    const slot = this.#slot(h.slot);
    return slot?.tag === 'logprobs' ? slot.val : undefined;
  }

  /** Entropy for an `Entropy` probe. */
  entropy(h: ProbeHandle<'entropy'>): number | undefined {
    const slot = this.#slot(h.slot);
    return slot?.tag === 'entropy' ? slot.val : undefined;
  }

  #slot(idx: number): SlotOutput | undefined {
    return idx >= 0 && idx < this.#raw.slots.length
      ? this.#raw.slots[idx]
      : undefined;
  }
}

export type { Brle };
