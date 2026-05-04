// Generator — multi-step token-generation state machine.
//
// Configure with options on `ctx.generate(sampler, options)` (or chain
// methods on the returned `Generator`) and iterate with `for await`:
//
//     const g = ctx.generate(Sampler.topP(0.6, 0.95), {
//       maxTokens: 256,
//       constrain: jsonSchema(schemaStr),
//     });
//
//     for await (const step of g) {
//       const out = await step.execute();
//       // out.tokens, out.distribution(probeHandle), ...
//     }
//
// For the common case, terminal sugars cover everything in one line:
//
// * `Generator.collectText()`   — drains, decodes through chat, returns text.
// * `Generator.collectTokens()` — drains, returns all tokens.
// * `Generator.collectJson()`   — adds JSON-schema constraint, drains, parses.
//
// For per-step control (custom sampling, watermarking), iterate manually
// with `for await (const step of gen)`: each step is a `GenStep` you can
// tweak (clear sampler, add probes) before `step.execute()`. Use
// `Generator.accept()` to register a manually-sampled token.

import {
  ForwardPass as _ForwardPass,
  type Output as WitOutput,
  type Brle,
} from 'pie:core/inference';

import { awaitFuture } from './_async.js';
import * as _chat from './chat.js';
import { Output, type ProbeHandle, type SampleHandle } from './forward.js';
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

// =============================================================================
// Generator options
// =============================================================================

/** Options for `ctx.generate()` / `new Generator()`. */
export interface GenerateOptions {
  /** Hard cap on tokens generated across all steps. */
  maxTokens?: number;
  /** Stop tokens. Generation halts when any of these is sampled. */
  stop?: Iterable<number>;
  /** A Schema, a Constraint, or a list of either. Multiple constraints
   *  compose by AND-ing their per-step BRLE masks. */
  constrain?: Schema | Constraint | Array<Schema | Constraint>;
  /** Static BRLE mask applied every step. Composes with `constrain`. */
  logitMask?: Brle;
  /** Custom speculative-decoding drafter. Mutually exclusive with
   *  `systemSpeculation`. */
  speculator?: Speculator;
  /** If true, the runtime drives drafts via its built-in NGRAM/etc.
   *  drafter. Mutually exclusive with `speculator`. */
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

/** Builder + async iterator for token generation.
 *
 *  Construct via `ctx.generate(sampler, options)` — never directly. */
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

    // Constraints — accept Schema, Constraint, or array.
    if (options.constrain !== undefined) {
      const items = Array.isArray(options.constrain)
        ? options.constrain
        : [options.constrain];
      for (const c of items) this._addConstraint(c);
    }
    if (options.logitMask !== undefined) {
      this._constraints.push(new StaticMaskConstraint(options.logitMask));
    }

    // Speculation.
    if (options.systemSpeculation && options.speculator !== undefined) {
      throw new Error(
        'speculator and systemSpeculation are mutually exclusive',
      );
    }
    this._speculator = options.speculator;
    this._useSystemSpec = options.systemSpeculation === true;
  }

  /** @internal */
  _addConstraint(c: Schema | Constraint): void {
    if ('buildConstraint' in c && typeof c.buildConstraint === 'function') {
      this._constraints.push(c.buildConstraint(this._ctx._model));
    } else if ('step' in c && typeof c.step === 'function') {
      this._constraints.push(c);
    } else {
      throw new TypeError(
        'constrain must be a Schema (with buildConstraint) ' +
          'or a Constraint (with step)',
      );
    }
  }

  // ── Chain methods (alternative to options-object) ──────────────────────

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

  /** Attach a constraint. Multiple calls compose by AND-ing per-step
   *  BRLE masks. */
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

  /** Attach a probe to every step at `index`. Returns a typed handle
   *  reusable across each `Output`.
   *
   *  Note: the slot index assumes the auto-sampler is attached. Calling
   *  `step.clearSampler()` on a particular step shifts every per-generator
   *  probe slot down by one for that step only. */
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
    // Drain context buffer (filled by `system / user / cue / ...`).
    const pending = this._ctx._pendingTokens;
    this._ctx._pendingTokens = new Uint32Array();

    // Pull drafts from the speculator.
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

    // Compose constraint masks.
    let mask: Brle | undefined;
    if (this._constraints.length > 0) {
      const advance = new Uint32Array(this._constraintPending);
      this._constraintPending = [];
      const masks = this._constraints
        .map(c => c.step(advance))
        .filter(m => m.length > 0);
      if (masks.length > 0) mask = _brleAndMany(masks);
    }

    return new GenStep(this, pending, drafts, draftPositions, mask);
  }

  // ── User-sampled mode ──────────────────────────────────────────────────

  /** Register manually-sampled tokens with the generator. Use after
   *  `step.clearSampler()` when the inferlet sampled by hand off a probe
   *  — the generator updates max-tokens / stop / constraint counters
   *  and seeds the next iteration's input. */
  accept(tokens: Iterable<number>): Uint32Array {
    const arr = Array.from(tokens);
    if (arr.length === 0) return new Uint32Array();

    // Stop-token truncation.
    for (let i = 0; i < arr.length; i++) {
      if (this._stop.includes(arr[i]!)) {
        arr.length = i;
        this._done = true;
        break;
      }
    }
    // Max-tokens enforcement.
    if (this._maxTokens !== undefined) {
      const remaining = this._maxTokens - this._tokensGenerated;
      if (arr.length > remaining) {
        arr.length = remaining;
        this._done = true;
      }
    }
    if (arr.length === 0) return new Uint32Array();

    // Stage for next forward pass via the buffer; advance counters.
    const result = new Uint32Array(arr);
    const merged = new Uint32Array(this._ctx._pendingTokens.length + result.length);
    merged.set(this._ctx._pendingTokens);
    merged.set(result, this._ctx._pendingTokens.length);
    this._ctx._pendingTokens = merged;

    this._constraintPending.push(...arr);
    this._tokensGenerated += arr.length;
    if (this._speculator !== undefined) this._speculator.accept(result);
    return result;
  }

  // ── Terminal sugar ─────────────────────────────────────────────────────

  /** Drain to completion; return the full token stream. */
  async collectTokens(): Promise<Uint32Array> {
    const all: number[] = [];
    for await (const step of this) {
      const out = await step.execute();
      for (let i = 0; i < out.tokens.length; i++) all.push(out.tokens[i]!);
    }
    return new Uint32Array(all);
  }

  /** Drain, decode through a chat decoder, return the response text.
   *
   *  Returns the chat decoder's `Done.text` if the model emits a clean
   *  end-of-turn (the expected case); otherwise concatenates every
   *  `Delta` chunk. */
  async collectText(): Promise<string> {
    const decoder = new _chat.Decoder(this._ctx._model);
    const parts: string[] = [];
    for await (const step of this) {
      const out = await step.execute();
      const ev = decoder.feed(out.tokens);
      if (ev.type === 'delta') parts.push(ev.text);
      else if (ev.type === 'done') return ev.text;
    }
    return parts.join('');
  }

  /** Generate JSON-constrained output and parse it.
   *
   *  Three calling conventions:
   *
   *  * `gen.collectJson({ schema })`           — returns parsed `unknown`.
   *  * `gen.collectJson({ schema, parse })`    — runs `parse(json)`,
   *                                              returns typed `T`.
   *  * `gen.collectJson()` (no opts)           — falls back to anyJson.
   */
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

/** Configuration handle for the upcoming forward pass. Yielded by
 *  iterating a `Generator`. Pre-populated with the generator's pending
 *  fills, configured sampler, constraint mask, and any speculator drafts.
 *
 *  Tweak (call `probe()`, `clearSampler()`) before `execute()`. */
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

  /** Drop the generator's auto-attached sampler. The caller must read the
   *  distribution off a probe and register their own pick via
   *  `gen.accept()` after `execute()`. */
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

    // Truly nothing to do — no input, no auto-sampler, no extra probes.
    if (
      nPending === 0 &&
      nDrafted === 0 &&
      this.#userClearedSampler &&
      this.#extraProbes.length === 0
    ) {
      gen._done = true;
      return new Output({
        slots: [],
        specTokens: new Uint32Array(),
        specPositions: new Uint32Array(),
      });
    }

    // Reserve pages for pending + drafts.
    const nTotal = nPending + nDrafted;
    if (nTotal > 0) {
      const totalAfter = ctx._workingTokens + nTotal;
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
    if (gen._adapter !== undefined) fwd.adapter(gen._adapter._handle);
    if (gen._zoSeed !== undefined) {
      const zoMod = await import('pie:zo/zo' as any);
      zoMod.adapterSeed(fwd, gen._zoSeed);
    }

    if (nPending > 0) {
      const positions = new Uint32Array(nPending);
      for (let i = 0; i < nPending; i++) positions[i] = ctx._seqLen + i;
      fwd.inputTokens(this.#pending, positions);
    }
    if (nDrafted > 0) {
      fwd.inputSpeculativeTokens(this.#drafts, this.#draftPositions);
    }
    if (gen._useSystemSpec) {
      fwd.outputSpeculativeTokens(true);
    }

    // Sampler at last input position (or 0 if drafts only / no input).
    const sampleIdx = nPending > 0 ? nPending - 1 : 0;
    if (!this.#userClearedSampler) {
      fwd.sampler(new Uint32Array([sampleIdx]), gen._sampler._variant);
    }

    // Per-generator step probes.
    for (const [idx, probe] of gen._stepProbes) {
      fwd.sampler(new Uint32Array([idx]), _probeToWit(probe));
    }
    // Per-step extra probes.
    for (const [idx, probe] of this.#extraProbes) {
      fwd.sampler(new Uint32Array([idx]), _probeToWit(probe));
    }

    if (this.#mask !== undefined) fwd.logitMask(this.#mask);

    const raw = await awaitFuture(fwd.execute(), 'GenStep.execute failed');

    // Collect accepted tokens off slot 0 (and following Token slots in
    // spec mode — verifier produces a sequence).
    let accepted: number[] = [];
    if (!this.#userClearedSampler) {
      for (const slot of raw.slots) {
        if (slot.tag === 'token') accepted.push(slot.val);
        else break;
      }
    }

    // Stash next-iter system drafts; let custom speculators see accepted.
    if (gen._useSystemSpec) {
      gen._specDrafts = [raw.specTokens, raw.specPositions];
    } else if (gen._speculator !== undefined) {
      gen._speculator.accept(new Uint32Array(accepted));
    }

    // Truncate rejected drafts.
    if (nDrafted > 0) {
      const nVerified = Math.max(0, accepted.length - 1);
      const nRejected = nDrafted - nVerified;
      if (nRejected > 0) {
        ctx._handle.truncateWorkingPageTokens(nRejected);
        if (gen._speculator !== undefined) gen._speculator.rollback(nRejected);
      }
    }

    // Commit pages: pending always commit (real KV); verified drafts too.
    const nVerifiedDrafts = nDrafted > 0 ? Math.max(0, accepted.length - 1) : 0;
    const nKv = nPending + nVerifiedDrafts;
    if (nKv > 0) {
      const newWorking = ctx._workingTokens + nKv;
      const toCommit = Math.floor(newWorking / ctx._pageSize);
      if (toCommit > 0) ctx._handle.commitWorkingPages(toCommit);
      ctx._committedPages += toCommit;
      ctx._workingPages -= toCommit;
      ctx._workingTokens = newWorking % ctx._pageSize;
      ctx._seqLen += nKv;
    } else if (nDrafted > 0 && accepted.length === 0) {
      // All drafts rejected with no anchor — re-sync from host.
      ctx._committedPages = ctx._handle.committedPageCount();
      ctx._workingPages = ctx._handle.workingPageCount();
      ctx._workingTokens = ctx._handle.workingPageTokenCount();
      ctx._seqLen = ctx._committedPages * ctx._pageSize + ctx._workingTokens;
    }

    // Advance constraint state with accepted tokens.
    if (gen._constraints.length > 0) {
      gen._constraintPending.push(...accepted);
    }

    // Apply stop / max truncation, accumulate counters, seed buffer.
    let tokens = accepted.slice();
    for (let i = 0; i < tokens.length; i++) {
      if (gen._stop.includes(tokens[i]!)) {
        tokens.length = i;
        gen._done = true;
        break;
      }
    }
    if (gen._maxTokens !== undefined) {
      const remaining = gen._maxTokens - gen._tokensGenerated;
      if (tokens.length > remaining) {
        tokens.length = remaining;
        gen._done = true;
      }
    }
    gen._tokensGenerated += tokens.length;
    if (tokens.length > 0) {
      const last = tokens[tokens.length - 1]!;
      const merged = new Uint32Array(ctx._pendingTokens.length + 1);
      merged.set(ctx._pendingTokens);
      merged[ctx._pendingTokens.length] = last;
      ctx._pendingTokens = merged;
    }

    const autoSampler: SampleHandle | undefined = this.#userClearedSampler
      ? undefined
      : { slot: 0, arity: 1 };

    return new Output(raw, new Uint32Array(tokens), autoSampler);
  }
}
