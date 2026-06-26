// Context — host-managed conversation state.
//
// Wraps `pie:core/context`. Buffers tokens via `system / user / assistant /
// cue / seal / append`, drains via `flush()` or by handing the buffer to a
// `Forward` / `Generator`.
//
//     const ctx = new Context();
//     ctx.system("You are helpful.").user("Tell me a joke.");
//
//     // Auto-flushed by `generate(...)` — no explicit cue/flush needed.
//     const text = await ctx
//       .generate(Sampler.topP(0.6, 0.95), { maxTokens: 256 })
//       .collectText();
//
// `forward()` / `generate()` read from the same `_pendingTokens` buffer the
// fillers append to, so a single call sequence — `ctx.system(...).user(...)
// .generate(...)` — flows from text to tokens to KV without an intermediate
// step the user has to remember.

import { Context as _Context } from 'pie:core/context';

import * as _chat from './chat.js';
import { Forward } from './forward.js';
import { Generator, type GenerateOptions } from './generation.js';
import * as _model from './model.js';

import type { Sampler } from './sample.js';

// =============================================================================
// Context
// =============================================================================

/** Host-managed conversation context.
 *
 *  Construct an empty context, fill via chat methods (or `append`), then
 *  either drain explicitly with `flush` or let `generate` / `forward`
 *  drain on demand. */
export class Context implements Disposable {
  /** @internal */ _handle: _Context;
  /** @internal Pending tokens buffered by fillers, drained by Forward / Generator. */
  _pendingTokens: Uint32Array = new Uint32Array();
  /** @internal Cached page-bookkeeping state — synced from host on construction
   *  and after `truncate()` / failed-spec re-syncs; mutated in lockstep by
   *  Forward / Generator after each successful `execute()`. */
  _pageSize = 0;
  /** @internal */ _seqLen = 0;
  /** @internal */ _committedPages = 0;
  /** @internal */ _workingPages = 0;
  /** @internal */ _workingTokens = 0;

  // ── Construction / lifecycle ─────────────────────────────────────────
  //
  // Public form: `new Context()` opens a fresh empty context.
  // Internal form: `new Context(rawHandle)` adopts an existing handle
  // (used by `open` / `take` / `fork`). The one-arg overload is marked
  // @internal — caller must own the handle.

  constructor();
  /** @internal */
  constructor(handle: _Context);
  constructor(handle?: _Context) {
    this._handle = handle ?? _Context.create();
    this._syncFromHost();
  }

  /** @internal */
  _syncFromHost(): void {
    this._pageSize = this._handle.tokensPerPage();
    this._committedPages = this._handle.committedPageCount();
    this._workingPages = this._handle.workingPageCount();
    this._workingTokens = this._handle.workingPageTokenCount();
    this._seqLen = this._committedPages * this._pageSize + this._workingTokens;
  }

  /** Open a saved snapshot (implicit fork — snapshot stays immutable). */
  static open(name: string): Context | undefined {
    const raw = _Context.open(name);
    return raw === undefined ? undefined : new Context(raw);
  }

  /** Take ownership of a snapshot (snapshot is deleted). */
  static take(name: string): Context | undefined {
    const raw = _Context.take(name);
    return raw === undefined ? undefined : new Context(raw);
  }

  /** Delete a saved snapshot by name. */
  static delete(name: string): void {
    _Context.delete(name);
  }

  /** Fork into a new anonymous context (working pages copied). */
  fork(): Context {
    const obj = new Context(this._handle.fork());
    // Carry pending tokens forward.
    obj._pendingTokens = new Uint32Array(this._pendingTokens);
    return obj;
  }

  /** Save this context with a name. */
  save(name: string): void {
    this._handle.save(name);
  }

  /** Anonymous save — returns a runtime-generated name. */
  snapshot(): string {
    return this._handle.snapshot();
  }

  /** Force-destroy this context immediately, releasing its KV resources. */
  release(): void {
    this._handle.destroy();
  }

  /** Disposable protocol — calls `release()`. */
  [Symbol.dispose](): void {
    this.release();
  }

  // ── Accessors ────────────────────────────────────────────────────────

  /** Tokens per KV page. */
  get pageSize(): number { return this._pageSize; }

  /** Total committed + working tokens (excludes the pending buffer). */
  get seqLen(): number { return this._seqLen; }

  /** Pending (buffered but not yet flushed) tokens. */
  buffer(): Uint32Array {
    return new Uint32Array(this._pendingTokens);
  }

  // ── Chat fillers ─────────────────────────────────────────────────────
  //
  // All return `this` for chaining: ctx.system("...").user("...").

  /** Fill a system-role message. */
  system(message: string): this {
    this._appendPending(_chat.system(message));
    return this;
  }

  /** Fill a user-role message. */
  user(message: string): this {
    this._appendPending(_chat.user(message));
    return this;
  }

  /** Fill an assistant-role message (history replay). */
  assistant(message: string): this {
    this._appendPending(_chat.assistant(message));
    return this;
  }

  /** Cue the model to generate (fills the generation header). */
  cue(): this {
    this._appendPending(_chat.cue());
    return this;
  }

  /** Seal the current turn (insert stop token). */
  seal(): this {
    this._appendPending(_chat.seal());
    return this;
  }

  /** Append raw tokens to the buffer directly. */
  append(tokens: Iterable<number>): this {
    this._appendPending(
      tokens instanceof Uint32Array ? tokens : new Uint32Array(tokens),
    );
    return this;
  }

  /** Append text — encodes via the model's tokenizer. */
  appendText(text: string): this {
    this._appendPending(_model.encode(text));
    return this;
  }

  // ── Flush / truncate ─────────────────────────────────────────────────

  /** Drain buffered tokens through a forward pass and commit pages.
   *  After flush, the buffer is empty and `seqLen` reflects all consumed
   *  tokens. */
  async flush(): Promise<void> {
    if (this._pendingTokens.length === 0) return;
    const fwd = new Forward(this);
    fwd.input(this._pendingTokens);
    this._pendingTokens = new Uint32Array();
    await fwd.execute();
  }

  /** Drop the trailing `n` working-page tokens. Use after a speculative
   *  pass to roll back the rejected suffix. `n` counts only working-page
   *  tokens — pages already committed cannot be truncated through this
   *  API. */
  truncate(n: number): void {
    if (n === 0) return;
    this._handle.truncateWorkingPageTokens(n);
    this._syncFromHost();
  }

  // ── Forward (single forward-pass primitive) ──────────────────────────

  /** Build a single `Forward` — a forward pass with auto page reservation,
   *  position derivation, and post-execute commit. Use for prefill,
   *  scoring, custom decode loops, and anywhere `generate()` is too
   *  high-level. */
  forward(): Forward {
    return new Forward(this);
  }

  // ── Generate (multi-step loop) ───────────────────────────────────────

  /** Build a `Generator` for token generation.
   *
   *  **Auto-flush**: when `autoFlush` is `true` (default), this method
   *  appends `cue()` tokens to the buffer before returning the Generator
   *  and defaults `stop` to the model's chat stop tokens. The first
   *  `execute()` call drains the buffer through a forward pass — no
   *  separate flush call needed. Pass `autoFlush: false` to inspect the
   *  buffer before generation starts (or call `cue()` yourself). */
  generate(
    sampler: Sampler,
    options: GenerateOptions & { autoFlush?: boolean } = {},
  ): Generator {
    const { autoFlush = true, ...rest } = options;
    if (autoFlush) {
      this.cue();
      if (rest.stop === undefined) {
        rest.stop = _chat.stopTokens();
      }
    }
    return new Generator(this, sampler, rest);
  }

  // ── Internal ─────────────────────────────────────────────────────────

  /** @internal Append tokens to the pending buffer. */
  _appendPending(tokens: Uint32Array): void {
    if (tokens.length === 0) return;
    const merged = new Uint32Array(this._pendingTokens.length + tokens.length);
    merged.set(this._pendingTokens);
    merged.set(tokens, this._pendingTokens.length);
    this._pendingTokens = merged;
  }
}
