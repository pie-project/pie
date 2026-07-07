// Generator — multi-step token-generation state machine.

import {
  ForwardPass as _ForwardPass,
  type Output as WitOutput,
  type Brle,
} from 'pie:core/inference';

import * as _chat from './chat.js';
import { Output, type ProbeHandle, type SampleHandle } from './forward.js';
import * as _model from './model.js';
import {
  type Constraint,
  type Schema,
  StaticMaskConstraint,
  _brleAndMany,
  anyJson,
  jsonSchema,
} from './grammar.js';
import {
  _probeAccessorKind,
  _probeToWit,
  type Probe,
  type ProbeKindOf,
  type Sampler,
} from './sample.js';

import type { Adapter } from './adapter.js';
import type { Context } from './context.js';
import type { Speculator } from './spec.js';

function _samplerIsArgmax(sampler: Sampler): boolean {
  const variant = sampler._variant;
  return variant.tag === 'top-p' && variant.val[0] === 0.0;
}

// =============================================================================
// Generator options
// =============================================================================

/** Options for `ctx.generate()` / `new Generator()`. */
export interface GenerateOptions {
  /** Hard cap on tokens generated across all steps. */
  maxTokens?: number;
  /** Stop tokens. Generation halts when any of these is sampled. */
  stop?: Iterable<number>;
  /** A Schema, a Constraint, or a list of either. */
  constrain?: Schema | Constraint | Array<Schema | Constraint>;
  /** Static BRLE mask applied every step. */
  logitMask?: Brle;
  /** Custom speculative-decoding drafter. */
  speculator?: Speculator;
  /** Host-driven system speculation. */
  systemSpeculation?: boolean;
  /** Adapter (LoRA, etc.) applied on every forward pass. */
  adapter?: Adapter;
  /** Evolution Strategies seed for every forward pass. */
  zoSeed?: number;
  /** Expected output length for budget planning. */
  horizon?: number;
}

// =============================================================================
// Generator
// =============================================================================

/** Builder + async iterator for token generation. */
export class Generator implements AsyncIterable<GenStep> {
  /** @internal */ readonly _ctx: Context;
  /** @internal */ readonly _sampler: Sampler;
  /** @internal */ _stop: number[];
  /** @internal */ _maxTokens: number | undefined;
  /** @internal */ _horizon: number | undefined;
  /** @internal */ _constraints: Constraint[] = [];
  /** @internal */ _constraintPending: number[] = [];
  /** @internal */ _speculator: Speculator | undefined;
  /** @internal */ _useSystemSpec: boolean;
  /** @internal */ _specDrafts: [Uint32Array, Uint32Array] = [
    new Uint32Array(),
    new Uint32Array(),
  ];
  /** @internal */ _adapter: Adapter | undefined;
  /** @internal */ _zoSeed: number | undefined;
  /** @internal */ _stepProbes: Array<[number, Probe]> = [];
  /** @internal */ _tokensGenerated = 0;
  /** @internal */ _done = false;

  constructor(
    ctx: Context,
    sampler: Sampler,
    options: GenerateOptions = {},
  ) {
    this._ctx = ctx;
    this._sampler = sampler;
    this._stop = options.stop !== undefined ? Array.from(options.stop) : [];
    this._maxTokens = options.maxTokens;
    this._horizon = options.horizon;
    this._adapter = options.adapter;
    this._zoSeed = options.zoSeed;

    if (options.constrain !== undefined) {
      const items = Array.isArray(options.constrain)
        ? options.constrain
        : [options.constrain];
      for (const c of items) this._addConstraint(c);
    }
    if (options.logitMask !== undefined) {
      this._constraints.push(new StaticMaskConstraint(options.logitMask));
    }

    if (options.systemSpeculation && options.speculator !== undefined) {
      throw new Error('speculator and systemSpeculation are mutually exclusive');
    }
    this._speculator = options.speculator;
    this._useSystemSpec = options.systemSpeculation ??
      (options.speculator === undefined &&
        _samplerIsArgmax(sampler) &&
        _model.defaultSystemSpeculation());
  }

  /** @internal */
  _addConstraint(c: Schema | Constraint): void {
    if ('buildConstraint' in c && typeof c.buildConstraint === 'function') {
      this._constraints.push(c.buildConstraint());
    } else if ('step' in c && typeof c.step === 'function') {
      this._constraints.push(c);
    } else {
      throw new TypeError(
        'constrain must be a Schema (with buildConstraint) ' +
          'or a Constraint (with step)',
      );
    }
  }

  // ── Chain methods ──────────────────────────────────────────────────────

  /** Hard cap on tokens generated. */
  maxTokens(n: number): this { this._maxTokens = n; return this; }

  /** Stop tokens. */
  stop(tokens: Iterable<number>): this {
    this._stop = Array.from(tokens);
    return this;
  }

  /** Append to the stop set. */
  addStop(tokens: Iterable<number>): this {
    this._stop.push(...tokens);
    return this;
  }

  /** Opt out of the default system drafter for this generator. */
  disableSystemSpeculation(): this {
    this._useSystemSpec = false;
    this._specDrafts = [new Uint32Array(), new Uint32Array()];
    return this;
  }

  /** Attach a constraint. */
  constrain(c: Schema | Constraint): this {
    this._addConstraint(c);
    return this;
  }

  /** Hint expected output length for budget planning. */
  horizon(n: number): this { this._horizon = n; return this; }

  /** Apply an adapter on every forward pass. */
  adapter(a: Adapter): this { this._adapter = a; return this; }

  /** Set zo (Evolution Strategies) seed on every forward pass. */
  zoSeed(seed: number): this { this._zoSeed = seed; return this; }

  /** Attach a probe to every step at `index`. */
  probeEachStep<P extends Probe>(index: number, probe: P): ProbeHandle<ProbeKindOf<P>> {
    const slot = 1 + this._stepProbes.length;
    this._stepProbes.push([index, probe]);
    return { slot, kind: _probeAccessorKind(probe) as ProbeKindOf<P> };
  }

  // ── Iteration ──────────────────────────────────────────────────────────

  /** Tokens generated so far across all steps. */
  get tokensGenerated(): number { return this._tokensGenerated; }

  /** Whether generation has terminated (max-tokens or stop hit). */
  get isDone(): boolean {
    return this._done || (
      this._maxTokens !== undefined &&
      this._tokensGenerated >= this._maxTokens
    );
  }

  /** Async iterator entry. */
  [Symbol.asyncIterator](): AsyncIterator<GenStep> {
    const self = this;
    return {
      async next(): Promise<IteratorResult<GenStep>> {
        if (self.isDone) return { done: true, value: undefined };
        return { done: false, value: self._buildStep() };
      },
    };
  }

  /** @internal */
  _buildStep(): GenStep {
    const pending = Uint32Array.from(this._ctx._buffer);
    this._ctx._buffer = [];

    let drafts: Uint32Array;
    let draftPositions: Uint32Array;
    if (this._useSystemSpec) {
      [drafts, draftPositions] = this._specDrafts;
      this._specDrafts = [new Uint32Array(), new Uint32Array()];
    } else if (this._speculator !== undefined) {
      [drafts, draftPositions] = this._speculator.draft();
    } else {
      drafts = new Uint32Array();
      draftPositions = new Uint32Array();
    }

    let mask: Brle | undefined;
    if (this._constraints.length > 0) {
      const advance = Uint32Array.from(this._constraintPending);
      this._constraintPending = [];
      const masks = this._constraints
        .map(c => c.step(advance))
        .filter(m => m.length > 0);
      if (masks.length > 0) mask = _brleAndMany(masks);
    }

    return new GenStep(this, pending, drafts, draftPositions, mask);
  }

  // ── User-sampled mode ──────────────────────────────────────────────────

  /** Register manually-sampled tokens with the generator. */
  accept(tokens: Iterable<number>): Uint32Array {
    const arr = Array.from(tokens);
    if (arr.length === 0) return new Uint32Array();

    for (let i = 0; i < arr.length; i++) {
      if (this._stop.includes(arr[i]!)) {
        arr.length = i;
        this._done = true;
        break;
      }
    }
    if (this._maxTokens !== undefined) {
      const remaining = this._maxTokens - this._tokensGenerated;
      if (arr.length > remaining) {
        arr.length = remaining;
        this._done = true;
      }
    }
    if (arr.length === 0) return new Uint32Array();

    this._ctx._buffer.push(...arr);
    this._constraintPending.push(...arr);
    this._tokensGenerated += arr.length;
    if (this._speculator !== undefined) {
      this._speculator.accept(Uint32Array.from(arr));
    }
    return Uint32Array.from(arr);
  }

  // ── Terminal sugar ─────────────────────────────────────────────────────

  /** Drain to completion; return the full token stream. */
  async collectTokens(): Promise<Uint32Array> {
    const all: number[] = [];
    for await (const step of this) {
      const out = await step.execute();
      all.push(...out.tokens);
    }
    return Uint32Array.from(all);
  }

  /** Drain, decode through a chat decoder, return the response text. */
  async collectText(): Promise<string> {
    const decoder = new _chat.Decoder();
    const parts: string[] = [];
    for await (const step of this) {
      const out = await step.execute();
      const ev = decoder.feed(out.tokens);
      if (ev.type === 'delta') parts.push(ev.text);
      else if (ev.type === 'done') return ev.text;
    }
    return parts.join('');
  }

  /** Generate JSON-constrained output and parse it. */
  async collectJson<T = unknown>(opts: {
    schema?: string;
    parse?: (value: unknown) => T;
  } = {}): Promise<T> {
    this._addConstraint(opts.schema !== undefined ? jsonSchema(opts.schema) : anyJson());
    const text = await this.collectText();
    const value = JSON.parse(text);
    return opts.parse ? opts.parse(value) : (value as T);
  }
}

// =============================================================================
// GenStep — short-lived per-iteration handle
// =============================================================================

/** Configuration handle for the upcoming forward pass. */
export class GenStep {
  readonly #gen: Generator;
  readonly #pending: Uint32Array;
  readonly #drafts: Uint32Array;
  readonly #draftPositions: Uint32Array;
  readonly #mask: Brle | undefined;
  #extraProbes: Array<[number, Probe]> = [];
  #userClearedSampler = false;

  /** @internal */
  constructor(
    gen: Generator,
    pending: Uint32Array,
    drafts: Uint32Array,
    draftPositions: Uint32Array,
    mask: Brle | undefined,
  ) {
    this.#gen = gen;
    this.#pending = pending;
    this.#drafts = drafts;
    this.#draftPositions = draftPositions;
    this.#mask = mask;
  }

  /** Drop the generator's auto-attached sampler. */
  clearSampler(): this {
    this.#userClearedSampler = true;
    return this;
  }

  /** Attach an extra probe at `index` for this iteration only. */
  probe<P extends Probe>(index: number, probe: P): ProbeHandle<ProbeKindOf<P>> {
    const base = this.#userClearedSampler ? 0 : 1;
    const slot = base + this.#gen._stepProbes.length + this.#extraProbes.length;
    this.#extraProbes.push([index, probe]);
    return { slot, kind: _probeAccessorKind(probe) as ProbeKindOf<P> };
  }

  /** Run the forward pass and fold the result into the generator's state. */
  async execute(): Promise<Output> {
    const gen = this.#gen;
    const ctx = gen._ctx;
    const nPending = this.#pending.length;
    const nDrafted = this.#drafts.length;

    if (nPending === 0 && nDrafted === 0) {
      gen._done = true;
      return new Output({
        slots: [],
        specTokens: new Uint32Array(),
        specPositions: new Uint32Array(),
      });
    }

    const isCustomSpec = !gen._useSystemSpec && gen._speculator !== undefined;
    const doSdkVerify = isCustomSpec && nDrafted > 0 && nPending > 0;
    const nWrite = doSdkVerify ? nPending + nDrafted : nPending;

    const fwd = new _ForwardPass();
    if (nWrite > 0) {
      ctx._attachKv(fwd, ctx._prepareWrite(nWrite));
    } else {
      ctx._attachFullContext(fwd);
    }

    if (gen._adapter !== undefined) fwd.adapter(gen._adapter._handle);
    if (gen._zoSeed !== undefined) {
      const zoMod = await import('pie:zo/zo' as any);
      zoMod.adapterSeed(fwd, BigInt(gen._zoSeed));
    }

    if (doSdkVerify) {
      const allTokens = new Uint32Array(nPending + nDrafted);
      allTokens.set(this.#pending);
      allTokens.set(this.#drafts, nPending);
      const allPositions = new Uint32Array(nPending + nDrafted);
      for (let i = 0; i < nPending; i++) allPositions[i] = ctx._seqLen + i;
      allPositions.set(this.#draftPositions, nPending);
      fwd.inputTokens(allTokens, allPositions);
    } else {
      if (nPending > 0) {
        const positions = new Uint32Array(nPending);
        for (let i = 0; i < nPending; i++) positions[i] = ctx._seqLen + i;
        fwd.inputTokens(this.#pending, positions);
      }
      if (nDrafted > 0) {
        fwd.inputSpeculativeTokens(this.#drafts, this.#draftPositions);
      }
    }

    const remaining = gen._maxTokens === undefined
      ? undefined
      : gen._maxTokens - gen._tokensGenerated;
    if (gen._useSystemSpec && (remaining === undefined || remaining > 1)) {
      fwd.outputSpeculativeTokens(true);
    }
    if (remaining !== undefined && remaining <= nDrafted + 1) {
      fwd.passSpeculation(false);
    }

    const sampleIdx = nPending > 0 ? nPending - 1 : 0;
    if (!this.#userClearedSampler) {
      if (doSdkVerify) {
        const indices = new Uint32Array(nDrafted + 1);
        for (let i = 0; i < indices.length; i++) indices[i] = sampleIdx + i;
        fwd.sampler(indices, gen._sampler._variant);
      } else {
        fwd.sampler(Uint32Array.of(sampleIdx), gen._sampler._variant);
      }
    }

    for (const [idx, probe] of gen._stepProbes) {
      fwd.sampler(Uint32Array.of(idx), _probeToWit(probe));
    }
    for (const [idx, probe] of this.#extraProbes) {
      fwd.sampler(Uint32Array.of(idx), _probeToWit(probe));
    }

    if (this.#mask !== undefined) fwd.logitMask(this.#mask);

    const raw = await fwd.execute();

    const accepted = this.#acceptedTokens(raw, doSdkVerify);
    if (!this.#userClearedSampler && accepted.length === 0) {
      throw new Error('GenStep.execute: auto-sampler returned no token');
    }

    if (gen._useSystemSpec) {
      gen._specDrafts = [raw.specTokens, raw.specPositions];
    } else if (gen._speculator !== undefined) {
      gen._speculator.accept(Uint32Array.from(accepted));
    }

    const nVerifiedDrafts = nDrafted > 0 ? Math.max(0, accepted.length - 1) : 0;
    if (nDrafted > 0) {
      const nRejected = nDrafted - nVerifiedDrafts;
      if (nRejected > 0 && gen._speculator !== undefined) {
        gen._speculator.rollback(nRejected);
      }
    }

    const nKv = nPending + nVerifiedDrafts;
    if (nKv > 0) {
      ctx._seqLen += nKv;
      ctx._history.push(...this.#pending);
      if (nVerifiedDrafts > 0) {
        ctx._history.push(...this.#drafts.slice(0, nVerifiedDrafts));
      }
    }

    if (gen._constraints.length > 0) {
      gen._constraintPending.push(...accepted);
    }

    const tokens = accepted.slice();
    for (let i = 0; i < tokens.length; i++) {
      if (gen._stop.includes(tokens[i]!)) {
        tokens.length = i;
        gen._done = true;
        break;
      }
    }
    if (gen._maxTokens !== undefined) {
      const maxRemaining = gen._maxTokens - gen._tokensGenerated;
      if (tokens.length > maxRemaining) {
        tokens.length = maxRemaining;
        gen._done = true;
      }
    }
    gen._tokensGenerated += tokens.length;
    if (tokens.length > 0) {
      ctx._buffer.push(tokens[tokens.length - 1]!);
    }

    const autoSampler: SampleHandle | undefined = this.#userClearedSampler
      ? undefined
      : { slot: 0, arity: 1 };

    return new Output(raw, Uint32Array.from(tokens), autoSampler);
  }

  #acceptedTokens(raw: WitOutput, doSdkVerify: boolean): number[] {
    if (this.#userClearedSampler) return [];
    if (doSdkVerify) {
      const nPicks = this.#drafts.length + 1;
      const picks: number[] = [];
      for (let i = 0; i < nPicks; i++) {
        const slot = raw.slots[i];
        if (slot?.tag === 'token') picks.push(slot.val);
      }
      if (picks.length !== nPicks) {
        throw new Error(
          `GenStep.execute verify: expected ${nPicks} Token slots, got ${picks.length}`,
        );
      }
      const accepted = [picks[0]!];
      for (let k = 0; k < this.#drafts.length; k++) {
        if (picks[k] !== this.#drafts[k]) break;
        accepted.push(picks[k + 1]!);
      }
      return accepted;
    }

    const accepted: number[] = [];
    for (const slot of raw.slots) {
      if (slot.tag === 'token') accepted.push(slot.val);
      else break;
    }
    return accepted;
  }
}
