// Forward — single forward-pass primitive with auto page management.
//
// `ctx.forward()` returns a `Forward` builder. Attach inputs, samplers,
// probes, masks, then `await forward.execute()`.
//
//     const fwd = ctx.forward();
//     fwd.input(promptTokens);
//     const h = fwd.sample([promptTokens.length - 1], Sampler.argmax());
//     const out = await fwd.execute();
//     const token = out.token(h);
//
// For prefill / scoring / custom decode loops. The `Generator` layer is
// built on top of this for the common token-generation case.

import {
  ForwardPass as _ForwardPass,
} from 'pie:core/inference';
import type {
  Sampler as WitSampler,
  Brle,
  Output as WitOutput,
  SlotOutput,
} from 'pie:core/inference';

import { awaitFuture } from './_async.js';
import type { Adapter } from './adapter.js';
import type { Context } from './context.js';
import type { Audio, Image } from './media.js';
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

/** Reference to a sampler slot. Pass to `output.token()` /
 *  `output.tokensAt()` to read the result. */
export interface SampleHandle {
  readonly slot: number;
  readonly arity: number;
}

/** Reference to a probe slot. The phantom `K` tag selects which
 *  `output.*` accessor compiles. */
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
  readonly probe: Probe;
}

type _Slot = _SampleSlot | _ProbeSlot;

// =============================================================================
// Forward
// =============================================================================

/**
 * Single forward pass. Construct via `ctx.forward()`.
 *
 * Builder methods return `this` so chains compose. `await forward.execute()`
 * runs the host call, commits any newly-filled pages, and returns an
 * `Output`.
 */
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

  constructor(ctx: Context) {
    this.#ctx = ctx;
  }

  // ── Position accessors ──────────────────────────────────────────────────

  /** Position the *first* auto-input token will occupy. Equal to the
   *  owning context's `seqLen` at the time `forward()` was called. The
   *  sampler at index `i` (when `forward.sample([i], ...)`) lands at
   *  `startPosition() + i`. */
  startPosition(): number {
    return this.#ctx._seqLen;
  }

  // ── Inputs ──────────────────────────────────────────────────────────────

  /** Append `tokens` at positions starting at the context's current
   *  sequence length. Multiple calls accumulate. After `execute()` these
   *  tokens occupy KV slots and `seqLen` advances. */
  input(tokens: Iterable<number>): this {
    for (const t of tokens) this.#autoInputs.push(t);
    return this;
  }

  /** Feed `tokens` at caller-supplied `positions`. Use for scoring at
   *  arbitrary positions (e.g. multi-candidate evaluation). These tokens
   *  are NOT auto-committed — caller manages page bookkeeping if positions
   *  overlap or extend beyond `seqLen`. */
  inputAt(tokens: Uint32Array, positions: Uint32Array): this {
    if (tokens.length !== positions.length) {
      throw new Error('tokens and positions must be the same length');
    }
    this.#explicitInputs.push([tokens, positions]);
    return this;
  }

  /** Splice an encoded image span at absolute sequence position `anchor`.
   *  Caller manages page reservation / commit when using raw `Forward`. */
  inputImage(image: Image, anchor: number): this {
    this.#images.push([image, anchor]);
    return this;
  }

  /** Splice an encoded audio span at absolute sequence position `anchor`.
   *  Caller manages page reservation / commit when using raw `Forward`. */
  inputAudio(audio: Audio, anchor: number): this {
    this.#audios.push([audio, anchor]);
    return this;
  }

  // ── Slot attach ─────────────────────────────────────────────────────────

  /** Attach a sampler at one or more `indices` (0-based into the auto-input
   *  window). Returns a handle for reading the sampled token(s) on the
   *  resulting `Output`.
   *
   *  A multi-arity sampler produces `indices.length` Token slots in the
   *  output, so the next slot index advances by that count — any subsequent
   *  `sample` / `probe` call sees the right offset. */
  sample(indices: Iterable<number>, sampler: Sampler): SampleHandle {
    const idxArr = indices instanceof Uint32Array
      ? indices
      : new Uint32Array(indices);
    const arity = idxArr.length;
    const h: SampleHandle = { slot: this.#nextSlot, arity };
    this.#slots.push({ type: 'sample', indices: idxArr, sampler });
    this.#nextSlot += arity;
    return h;
  }

  /** Attach a probe at a single `index`. Returns a typed handle whose
   *  `kind` selects which `output.*` accessor decodes the result. */
  probe<P extends Probe>(index: number, probe: P): ProbeHandle<ProbeKindOf<P>> {
    const h: ProbeHandle<ProbeKindOf<P>> = {
      slot: this.#nextSlot,
      kind: _probeAccessorKind(probe) as ProbeKindOf<P>,
    };
    this.#slots.push({ type: 'probe', index, probe });
    this.#nextSlot += 1;
    return h;
  }

  // ── Decoration ──────────────────────────────────────────────────────────

  /** Set a static logit mask (BRLE) applied at every sampled position. */
  mask(brle: Brle): this {
    this.#mask = brle;
    return this;
  }

  /** Set per-query-position attention masks. Length must match the total
   *  number of query positions across all `input` / `inputAt` calls. */
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

  /**
   * Run the forward pass. Reserves working pages for any auto-inputs,
   * submits all attached inputs and slots, awaits the host, commits any
   * newly-filled pages, and updates the context's cached state.
   *
   * Throws if no inputs and no slots are attached — a vacuous Forward
   * almost always indicates a missed `input(...)` or `sample(...)` call.
   */
  async execute(): Promise<Output> {
    const ctx = this.#ctx;
    const nAuto = this.#autoInputs.length;
    const nExplicit = this.#explicitInputs.reduce((a, [t]) => a + t.length, 0);
    const nTotal = nAuto + nExplicit;

    if (
      nTotal === 0 &&
      this.#slots.length === 0 &&
      this.#images.length === 0 &&
      this.#audios.length === 0
    ) {
      throw new Error(
        'Forward.execute() called with no inputs and no slots. ' +
          'Attach at least one input (`forward.input(...)`) or slot ' +
          '(`forward.sample(...)` / `forward.probe(...)`) before executing.',
      );
    }

    // Reserve pages for auto-inputs (they occupy KV and commit on the way
    // out). Explicit inputs are scoring-only — caller manages their pages.
    if (nAuto > 0) {
      const totalAfter = ctx._workingTokens + nAuto;
      const pagesNeeded = Math.ceil(totalAfter / ctx._pageSize);
      const additional = Math.max(0, pagesNeeded - ctx._workingPages);
      if (additional > 0) {
        ctx._handle.reserveWorkingPages(additional);
        ctx._workingPages = pagesNeeded;
      }
    }

    // Build forward pass.
    const fwd = new _ForwardPass(ctx._handle.model());
    fwd.context(ctx._handle);
    for (const [image, anchor] of this.#images) {
      fwd.inputImage(image._handle, anchor);
    }
    for (const [audio, anchor] of this.#audios) {
      fwd.inputAudio(audio._handle, anchor);
    }
    if (this.#adapter !== undefined) {
      fwd.adapter(this.#adapter._handle);
    }
    if (this.#zoSeed !== undefined) {
      const zoMod = await import('pie:zo/zo' as any);
      zoMod.adapterSeed(fwd, this.#zoSeed);
    }

    if (nAuto > 0) {
      const positions = new Uint32Array(nAuto);
      for (let i = 0; i < nAuto; i++) positions[i] = ctx._seqLen + i;
      fwd.inputTokens(new Uint32Array(this.#autoInputs), positions);
    }
    for (const [tokens, positions] of this.#explicitInputs) {
      fwd.inputTokens(tokens, positions);
    }

    // Slot attaches go in declaration order — slot indices match what we
    // handed back via SampleHandle / ProbeHandle.
    for (const spec of this.#slots) {
      if (spec.type === 'sample') {
        fwd.sampler(spec.indices, spec.sampler._variant);
      } else {
        fwd.sampler(new Uint32Array([spec.index]), _probeToWit(spec.probe));
      }
    }

    if (this.#mask !== undefined) fwd.logitMask(this.#mask);
    if (this.#attnMask !== undefined) fwd.attentionMask(this.#attnMask);

    const raw = await awaitFuture(fwd.execute(), 'Forward.execute failed');

    // Commit pages that auto-input tokens fully filled.
    if (nAuto > 0) {
      const newWorking = ctx._workingTokens + nAuto;
      const toCommit = Math.floor(newWorking / ctx._pageSize);
      if (toCommit > 0) ctx._handle.commitWorkingPages(toCommit);
      ctx._committedPages += toCommit;
      ctx._workingPages -= toCommit;
      ctx._workingTokens = newWorking % ctx._pageSize;
      ctx._seqLen += nAuto;
    }

    return new Output(raw);
  }
}

// =============================================================================
// Output
// =============================================================================

/**
 * Result of one forward-pass execution — produced by both
 * `Forward.execute()` and `GenStep.execute()`.
 *
 * **Common path** (Generator): read `output.tokens` for the accepted
 * tokens this step (post stop / max-tokens truncation).
 *
 * **Raw Forward**: read sampler slots via `token()` / `tokensAt()` using
 * handles returned at attach time. The `tokens` field is empty.
 *
 * **Probes** (both paths): `distribution()` / `logits()` / `logprobs()` /
 * `entropy()` take a typed `ProbeHandle`.
 */
export class Output {
  readonly #raw: WitOutput;

  /** Generator-accepted tokens this step, post stop / max-tokens
   *  truncation. Empty for raw `Forward.execute()` (no Generator state). */
  readonly tokens: Uint32Array;

  /** Handle for the Generator's auto-attached sampler. `undefined` for raw
   *  Forward results and for steps where `clearSampler()` was called. */
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

  /** Underlying WIT output (slot list + speculative side channel). */
  get raw(): WitOutput { return this.#raw; }

  // ── Sampler accessors ─────────────────────────────────────────────────

  /** First token from a single-index sampler slot. */
  token(h: SampleHandle): number | undefined {
    const slot = this.#slot(h.slot);
    return slot?.tag === 'token' ? slot.val : undefined;
  }

  /** Tokens at the slot range a multi-index sampler covers. In speculative
   *  mode the array may be shorter than `arity` if the verifier rejected
   *  drafts. */
  tokensAt(h: SampleHandle): Uint32Array {
    const out: number[] = [];
    for (let i = 0; i < h.arity; i++) {
      const slot = this.#slot(h.slot + i);
      if (slot?.tag === 'token') out.push(slot.val);
      else break;
    }
    return new Uint32Array(out);
  }

  // ── Probe accessors ───────────────────────────────────────────────────

  /** Distribution as `[ids, probs]` for a `Distribution` probe. */
  distribution(h: ProbeHandle<'distribution'>): [Uint32Array, Float32Array] | undefined {
    const slot = this.#slot(h.slot);
    return slot?.tag === 'distribution' ? slot.val : undefined;
  }

  /** Raw logits bytes for a `Logits` probe (length `vocab_size * 4`,
   *  native-endian f32). */
  logits(h: ProbeHandle<'logits'>): Uint8Array | undefined {
    const slot = this.#slot(h.slot);
    return slot?.tag === 'logits' ? slot.val : undefined;
  }

  /** Logprob list for a `Logprob` / `Logprobs` probe. Length 1 for a
   *  single-token query, K for a list query. */
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
