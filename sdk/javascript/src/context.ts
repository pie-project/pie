// Context — host-managed conversation state.
//
// Wraps `pie:core/context`. Buffers tokens via `system / user / assistant /
// cue / seal / append`, drains via `flush()` or by handing the buffer to a
// `Forward` / `Generator`.
//
//     const ctx = new Context(model);
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
import { ForwardPass as _ForwardPass } from 'pie:core/inference';
import * as _scheduling from 'pie:core/scheduling';

import { awaitFuture } from './_async.js';
import * as _chat from './chat.js';
import { Forward } from './forward.js';
import { Generator, type GenerateOptions } from './generation.js';
import type { Audio, Image, Video } from './media.js';
import { Model } from './model.js';

import type { Sampler } from './sample.js';

// =============================================================================
// Bid math (internal — matches Rust SDK)
// =============================================================================

function _computeBid(
  balance: number,
  pages: number,
  mu: number,
  cv2: number,
  pageSize: number,
  dividend: number,
): number {
  mu = Math.max(mu, 1.0);
  const numerator = balance / mu + dividend;
  const denominator = pages + (mu * (1.0 + cv2)) / (2.0 * pageSize);
  return denominator > 0 ? numerator / denominator : numerator;
}

// =============================================================================
// Context
// =============================================================================

/** Host-managed conversation context.
 *
 *  Construct with a model, fill via chat methods (or `append`), then either
 *  drain explicitly with `flush` or let `generate` / `forward` drain on
 *  demand. */
export class Context implements Disposable {
  /** @internal */ _handle: _Context;
  /** @internal */ _model: Model;
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
  // Public form: `new Context(model)` opens a fresh empty context.
  // Internal form: `new Context(rawHandle, model)` adopts an existing
  // handle (used by `open` / `take` / `fork`). The two-arg overload is
  // marked @internal — caller must own the handle.

  constructor(model: Model);
  /** @internal */
  constructor(handle: _Context, model: Model);
  constructor(modelOrHandle: Model | _Context, model?: Model) {
    if (model === undefined) {
      // Public form.
      this._model = modelOrHandle as Model;
      this._handle = _Context.create(this._model._handle);
    } else {
      // Internal form — adopt the handle.
      this._handle = modelOrHandle as _Context;
      this._model = model;
    }
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
  static open(model: Model, name: string): Context | undefined {
    const raw = _Context.open(model._handle, name);
    return raw === undefined ? undefined : new Context(raw, model);
  }

  /** Take ownership of a snapshot (snapshot is deleted). */
  static take(model: Model, name: string): Context | undefined {
    const raw = _Context.take(model._handle, name);
    return raw === undefined ? undefined : new Context(raw, model);
  }

  /** Delete a saved snapshot by name. */
  static delete(model: Model, name: string): void {
    _Context.delete(model._handle, name);
  }

  /** Fork into a new anonymous context (working pages copied). */
  fork(): Context {
    const obj = new Context(this._handle.fork(), this._model);
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

  /** The model this context is tied to. */
  get model(): Model { return this._model; }

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
    this._appendPending(_chat.system(this._model, message));
    return this;
  }

  /** Fill a user-role message. */
  user(message: string): this {
    this._appendPending(_chat.user(this._model, message));
    return this;
  }

  /** Fill an assistant-role message (history replay). */
  assistant(message: string): this {
    this._appendPending(_chat.assistant(this._model, message));
    return this;
  }

  /** Cue the model to generate (fills the generation header). */
  cue(): this {
    this._appendPending(_chat.cue(this._model));
    return this;
  }

  /** Seal the current turn (insert stop token). */
  seal(): this {
    this._appendPending(_chat.seal(this._model));
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
    this._appendPending(this._model.tokenizer().encode(text));
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

  // ── Multimodal spans ───────────────────────────────────────────────

  /** Splice an encoded image into the context and commit its soft-token KV. */
  async appendImage(image: Image): Promise<void> {
    const prefix = image.prefixTokens();
    const suffix = image.suffixTokens();
    if (prefix.length > 0) this.append(prefix);
    await this.flush();

    const nTokens = image.tokenCount();
    if (nTokens === 0) {
      if (suffix.length > 0) this.append(suffix);
      return;
    }

    this._reserveMediaTokens(nTokens, 'appendImage');
    const fwd = new _ForwardPass(this._model._handle);
    fwd.context(this._handle);
    fwd.inputImage(image._handle, this._seqLen);
    await awaitFuture(fwd.execute(), 'appendImage: forward pass failed');
    this._commitMediaTokens(nTokens, 'appendImage');

    if (suffix.length > 0) this.append(suffix);
  }

  /** Splice adjacent images into one forward when delimiters allow it. */
  async appendImages(images: Iterable<Image>): Promise<void> {
    const items = Array.from(images);
    if (items.length === 0) return;
    const needsDelimiters = items.some(im =>
      im.prefixTokens().length > 0 || im.suffixTokens().length > 0
    );
    if (items.length === 1 || needsDelimiters) {
      for (const image of items) await this.appendImage(image);
      return;
    }

    await this.flush();
    const totalTokens = items.reduce((n, im) => n + im.tokenCount(), 0);
    if (totalTokens === 0) return;

    this._reserveMediaTokens(totalTokens, 'appendImages');
    const fwd = new _ForwardPass(this._model._handle);
    fwd.context(this._handle);
    let anchor = this._seqLen;
    for (const image of items) {
      fwd.inputImage(image._handle, anchor);
      anchor += image.tokenCount();
    }
    await awaitFuture(fwd.execute(), 'appendImages: forward pass failed');
    this._commitMediaTokens(totalTokens, 'appendImages');
  }

  /** Splice an encoded audio clip into the context and commit its soft-token KV. */
  async appendAudio(audio: Audio): Promise<void> {
    const prefix = audio.prefixTokens();
    const suffix = audio.suffixTokens();
    if (prefix.length > 0) this.append(prefix);
    await this.flush();

    const nTokens = audio.tokenCount();
    if (nTokens === 0) {
      if (suffix.length > 0) this.append(suffix);
      return;
    }

    this._reserveMediaTokens(nTokens, 'appendAudio');
    const fwd = new _ForwardPass(this._model._handle);
    fwd.context(this._handle);
    fwd.inputAudio(audio._handle, this._seqLen);
    await awaitFuture(fwd.execute(), 'appendAudio: forward pass failed');
    this._commitMediaTokens(nTokens, 'appendAudio');

    if (suffix.length > 0) this.append(suffix);
  }

  /** Splice adjacent audio clips into one forward when possible. */
  async appendAudios(audios: Iterable<Audio>): Promise<void> {
    const items = Array.from(audios);
    if (items.length === 0) return;
    if (items.length === 1) return this.appendAudio(items[0]!);

    const prefix = items[0]!.prefixTokens();
    const suffix = items[items.length - 1]!.suffixTokens();
    if (prefix.length > 0) this.append(prefix);
    await this.flush();

    const totalTokens = items.reduce((n, audio) => n + audio.tokenCount(), 0);
    if (totalTokens === 0) {
      if (suffix.length > 0) this.append(suffix);
      return;
    }

    this._reserveMediaTokens(totalTokens, 'appendAudios');
    const fwd = new _ForwardPass(this._model._handle);
    fwd.context(this._handle);
    let anchor = this._seqLen;
    for (const audio of items) {
      fwd.inputAudio(audio._handle, anchor);
      anchor += audio.tokenCount();
    }
    await awaitFuture(fwd.execute(), 'appendAudios: forward pass failed');
    this._commitMediaTokens(totalTokens, 'appendAudios');

    if (suffix.length > 0) this.append(suffix);
  }

  /** Splice each sampled video frame, preceded by a generic timestamp marker. */
  async appendVideo(video: Video): Promise<void> {
    const n = video.frameCount();
    for (let i = 0; i < n; i++) {
      const secs = Math.max(0, Math.floor(video.timestamp(i)));
      const marker = ` ${String(Math.floor(secs / 60)).padStart(2, '0')}:` +
        `${String(secs % 60).padStart(2, '0')} `;
      this.appendText(marker);
      await this.appendImage(video.frame(i));
    }
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
        rest.stop = _chat.stopTokens(this._model);
      }
    }
    return new Generator(this, sampler, rest);
  }

  // ── Bidding / scheduling ─────────────────────────────────────────────

  /** Override the auto-computed bid (willingness to pay per page per step).
   *  Most callers should NOT use this — the `Generator` auto-bids each
   *  step. */
  setBid(value: number): void {
    this._handle.bid(value);
  }

  /** Mark this context as idle: drop the bid to zero so other contexts
   *  can take its pages under contention. Returns a `Disposable` that
   *  restores the truthful generation bid on `[Symbol.dispose]`:
   *
   *      using _ = ctx.idle();
   *      const result = await fetch(url);
   *      // bid restored when `_` goes out of scope
   *
   *  On an uncontended device the runtime charges zero rent anyway —
   *  `idle()` is a no-op cost-wise but still safe to call. Under load,
   *  it yields priority to other workloads for the duration. */
  idle(): Disposable {
    const pages = this._committedPages + this._workingPages;
    let saved = 0.0;
    if (pages > 0) {
      const balance = _scheduling.balance(this._model._handle);
      const dividend = _scheduling.dividend(this._model._handle);
      saved = _computeBid(balance, pages, 4096.0, 1.0, this._pageSize, dividend);
    }
    this._handle.bid(0.0);
    const handle = this._handle;
    return {
      [Symbol.dispose]() {
        handle.bid(saved);
      },
    };
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

  /** @internal */
  _reserveMediaTokens(numTokens: number, label: string): void {
    const totalAfter = this._workingTokens + numTokens;
    const pagesNeeded = Math.ceil(totalAfter / this._pageSize);
    const additional = Math.max(0, pagesNeeded - this._workingPages);
    if (additional > 0) {
      try {
        this._handle.reserveWorkingPages(additional);
      } catch (e) {
        throw new Error(`${label}: reserve pages: ${String(e)}`);
      }
      this._workingPages = pagesNeeded;
    }
  }

  /** @internal */
  _commitMediaTokens(numTokens: number, label: string): void {
    const newWorking = this._workingTokens + numTokens;
    const toCommit = Math.floor(newWorking / this._pageSize);
    if (toCommit > 0) {
      try {
        this._handle.commitWorkingPages(toCommit);
      } catch (e) {
        throw new Error(`${label}: commit pages: ${String(e)}`);
      }
    }
    this._committedPages += toCommit;
    this._workingPages -= toCommit;
    this._workingTokens = newWorking % this._pageSize;
    this._seqLen += numTokens;
  }
}
