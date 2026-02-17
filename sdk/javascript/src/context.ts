// Context, TokenStream, EventStream, and Decoder — wraps pie:core/context
// and pie:instruct/* WIT interfaces.

import {
  Context as _Context,
} from 'pie:core/context';
import {
  ForwardPass as _ForwardPass,
} from 'pie:core/inference';
import type { Sampler, Brle, Output } from 'pie:core/inference';
import type { Model as _Model } from 'pie:core/model';
import type { Adapter as _Adapter } from 'pie:core/adapter';

import * as _chat from 'pie:instruct/chat';
import * as _reasoning from 'pie:instruct/reasoning';
import * as _toolUse from 'pie:instruct/tool-use';

import { awaitFuture } from './_async.js';
import type { Model } from './model.js';
import type { Adapter } from './adapter.js';


// ─── Event Types ────────────────────────────────────────────────────────

/** A text chunk was generated. */
export interface EventText {
  readonly type: 'text';
  readonly text: string;
}

/** A reasoning/thinking text chunk. */
export interface EventThinking {
  readonly type: 'thinking';
  readonly text: string;
}

/** Reasoning is complete (full reasoning text). */
export interface EventThinkingDone {
  readonly type: 'thinking-done';
  readonly text: string;
}

/** A tool call was detected. */
export interface EventToolCallStart {
  readonly type: 'tool-call-start';
}

/** A complete tool call: name + JSON arguments. */
export interface EventToolCall {
  readonly type: 'tool-call';
  readonly name: string;
  readonly arguments: string;
}

/** Generation is done (full accumulated text). */
export interface EventDone {
  readonly type: 'done';
  readonly text: string;
}

export type Event =
  | EventText
  | EventThinking
  | EventThinkingDone
  | EventToolCallStart
  | EventToolCall
  | EventDone;

// ─── Decoder ────────────────────────────────────────────────────────────

/** Options for creating a `Decoder`. */
export interface DecoderOptions {
  /** Enable reasoning detection. Default: false. */
  reasoning?: boolean;
  /** Enable tool-use detection. Default: false. */
  toolUse?: boolean;
}

/**
 * Unified event decoder that multiplexes chat, reasoning, and tool-use
 * decoders from the instruct WIT interfaces.
 *
 * Event priority: reasoning > tool-use > chat.
 */
export class Decoder {
  private readonly _chatDecoder: _chat.Decoder;
  private readonly _reasoningDecoder?: _reasoning.Decoder;
  private readonly _toolUseDecoder?: _toolUse.Decoder;
  private _inReasoning = false;

  constructor(model: _Model, options: DecoderOptions = {}) {
    this._chatDecoder = _chat.createDecoder(model);
    if (options.reasoning) {
      this._reasoningDecoder = _reasoning.createDecoder(model);
    }
    if (options.toolUse) {
      this._toolUseDecoder = _toolUse.createDecoder(model);
    }
  }

  /**
   * Feed token IDs to all active decoders and return the highest-priority
   * event. Priority: reasoning > tool-use > chat.
   *
   * All sub-decoders are always fed to maintain their internal state.
   */
  feed(tokens: Uint32Array): Event {
    // Always feed the chat decoder
    const chatEvent = this._chatDecoder.feed(tokens);

    // Reasoning decoder (highest priority)
    if (this._reasoningDecoder) {
      const ev = this._reasoningDecoder.feed(tokens);
      switch (ev.tag) {
        case 'start':
          this._inReasoning = true;
          return { type: 'thinking', text: '' };
        case 'delta':
          return { type: 'thinking', text: ev.val };
        case 'complete':
          this._inReasoning = false;
          return { type: 'thinking-done', text: ev.val };
      }
    }

    // Tool-use decoder (second priority)
    if (this._toolUseDecoder) {
      const ev = this._toolUseDecoder.feed(tokens);
      switch (ev.tag) {
        case 'start':
          return { type: 'tool-call-start' };
        case 'call':
          return { type: 'tool-call', name: ev.val[0], arguments: ev.val[1] };
      }
    }

    // Chat decoder (always active, lowest priority)
    switch (chatEvent.tag) {
      case 'delta':
        return { type: 'text', text: chatEvent.val };
      case 'done':
        return { type: 'done', text: chatEvent.val };
      case 'interrupt':
        // Align with Python SDK: interrupts are mapped to Done("")
        return { type: 'done', text: '' };
    }
  }

  /** Reset all decoders to initial state. */
  reset(): void {
    this._chatDecoder.reset();
    this._reasoningDecoder?.reset();
    this._toolUseDecoder?.reset();
    this._inReasoning = false;
  }
}

// ─── Reserve-and-run helper ─────────────────────────────────────────────

/**
 * Reserve pages, run a forward pass, commit pages, and update cursor.
 *
 * If `sampler` is provided, sampling is applied at the last position.
 * Returns the Output (or undefined if no sampler was given and the
 * output is not tokens).
 *
 * @internal
 */
async function _reserveAndRun(
  ctxHandle: _Context,
  tokens: Uint32Array,
  sampler?: Sampler,
  logitMask?: Brle,
  adapter?: _Adapter,
): Promise<Output> {
  const numTokens = tokens.length;
  const pageSize = ctxHandle.tokensPerPage();
  const cursor = ctxHandle.cursor();

  // Reserve pages (runtime deduplicates against existing uncommitted)
  const totalTokens = cursor + numTokens;
  const pagesNeeded = Math.ceil(totalTokens / pageSize);
  if (pagesNeeded > 0) {
    ctxHandle.reservePages(pagesNeeded);
  }

  // Build forward pass
  const pass = new _ForwardPass(ctxHandle.model());
  pass.context(ctxHandle);

  const lastPos = ctxHandle.lastPosition();
  const seqStart = lastPos !== undefined ? lastPos + 1 : 0;
  const positions = new Uint32Array(numTokens);
  for (let i = 0; i < numTokens; i++) {
    positions[i] = seqStart + i;
  }
  pass.inputTokens(tokens, positions);

  if (sampler !== undefined) {
    pass.sampler(new Uint32Array([numTokens - 1]), sampler);
  }
  if (logitMask !== undefined) {
    pass.logitMask(logitMask);
  }
  if (adapter !== undefined) {
    pass.adapter(adapter);
  }

  // Execute
  const output = await awaitFuture(pass.execute(), 'Forward pass failed');

  // Commit pages and update cursor
  const newCursorAbs = cursor + numTokens;
  const pagesToCommit = Math.floor(newCursorAbs / pageSize);
  if (pagesToCommit > 0) {
    const indices = new Uint32Array(pagesToCommit);
    for (let i = 0; i < pagesToCommit; i++) {
      indices[i] = i;
    }
    ctxHandle.commitPages(indices);
  }
  ctxHandle.setCursor(newCursorAbs % pageSize);

  return output;
}

// ─── TokenStream ────────────────────────────────────────────────────────

/** Options for `generate()`. */
export interface GenerateOptions {
  /** Sampler configuration. */
  sampler: Sampler;
  /** Maximum number of tokens to generate. */
  maxTokens: number;
  /** Stop token IDs. If omitted, uses the model's default stop tokens. */
  stopTokens?: Iterable<number>;
  /** LoRA adapter to use. */
  adapter?: Adapter;
  /** Logit mask (BRLE-encoded). */
  logitMask?: Brle;
  /** Decode events instead of raw tokens. */
  decode?: DecoderOptions;
}

/**
 * An async iterator that generates tokens one step at a time.
 *
 * Each iteration returns a `Uint32Array` of accepted token IDs.
 *
 * ```js
 * for await (const tokens of stream) { ... }
 * const allText = await stream.text();
 * const allTokens = await stream.tokens();
 * ```
 */
export class TokenStream implements AsyncIterable<Uint32Array> {
  private readonly _ctx: Context;
  private readonly _sampler: Sampler;
  private readonly _maxTokens: number;
  private readonly _stopTokens: Set<number>;
  private readonly _adapter?: _Adapter;
  private readonly _logitMask?: Brle;
  private _generated: number = 0;
  private _done: boolean = false;

  /** @internal */
  constructor(ctx: Context, options: GenerateOptions) {
    this._ctx = ctx;
    this._sampler = options.sampler;
    this._maxTokens = options.maxTokens;
    this._adapter = options.adapter?._handle;
    this._logitMask = options.logitMask;

    // Auto-detect stop tokens from model if not provided
    if (options.stopTokens !== undefined) {
      this._stopTokens = new Set(options.stopTokens);
    } else {
      this._stopTokens = new Set(_chat.stopTokens(ctx._handle.model()));
    }
  }

  /** Execute one forward pass and return generated tokens. */
  async step(): Promise<Uint32Array | undefined> {
    if (this._done || this._generated >= this._maxTokens) {
      this._done = true;
      return undefined;
    }

    const ctxHandle = this._ctx._handle;
    const buffered = ctxHandle.bufferedTokens();
    if (buffered.length === 0) {
      this._done = true;
      return undefined;
    }

    const output = await _reserveAndRun(
      ctxHandle,
      buffered,
      this._sampler,
      this._logitMask,
      this._adapter,
    );

    // Process output
    if (output.tag === 'tokens') {
      let tokens = output.val;

      // Truncate at the first stop token (exclude it)
      for (let i = 0; i < tokens.length; i++) {
        if (this._stopTokens.has(tokens[i])) {
          tokens = tokens.subarray(0, i);
          this._done = true;
          break;
        }
      }

      if (tokens.length === 0) {
        this._done = true;
        return undefined;
      }

      this._generated += tokens.length;

      // Seed the next step with only the LAST generated token
      ctxHandle.setBufferedTokens(new Uint32Array([tokens[tokens.length - 1]]));
      return tokens;
    }

    this._done = true;
    return undefined;
  }

  /** Consume the stream and return the decoded text. */
  async text(): Promise<string> {
    const tokenizer = this._ctx._handle.model().tokenizer();
    const chunks: string[] = [];
    for await (const tokens of this) {
      chunks.push(tokenizer.decode(tokens));
    }
    return chunks.join('');
  }

  /** Consume the stream and return all token batches. */
  async tokens(): Promise<Uint32Array[]> {
    const result: Uint32Array[] = [];
    for await (const tokens of this) {
      result.push(tokens);
    }
    return result;
  }

  async *[Symbol.asyncIterator](): AsyncIterableIterator<Uint32Array> {
    while (!this._done) {
      const tokens = await this.step();
      if (tokens !== undefined) {
        yield tokens;
      }
    }
  }
}

// ─── EventStream ────────────────────────────────────────────────────────

/**
 * An async iterator that yields decoded `Event` objects.
 *
 * Supports both `for await...of` iteration and callback-based `.on().run()`.
 *
 * ```js
 * // Iterator style:
 * for await (const event of stream) { ... }
 *
 * // Callback style:
 * await stream
 *   .on('text', text => session.send(text))
 *   .on('done', text => console.log('finished:', text))
 *   .run();
 * ```
 */
export class EventStream implements AsyncIterable<Event> {
  private readonly _stream: TokenStream;
  private readonly _decoder: Decoder;
  // eslint-disable-next-line @typescript-eslint/no-unsafe-function-type
  private _listeners: Map<string, Function[]> = new Map();

  /** @internal */
  constructor(stream: TokenStream, decoder: Decoder) {
    this._stream = stream;
    this._decoder = decoder;
  }

  /**
   * Register a callback for a specific event type.
   * Returns `this` for chaining.
   */
  on(type: 'text', cb: (text: string) => void): this;
  on(type: 'thinking', cb: (text: string) => void): this;
  on(type: 'thinking-done', cb: (text: string) => void): this;
  on(type: 'tool-call-start', cb: () => void): this;
  on(type: 'tool-call', cb: (name: string, args: string) => void): this;
  on(type: 'done', cb: (text: string) => void): this;
  // eslint-disable-next-line @typescript-eslint/no-unsafe-function-type
  on(type: string, cb: Function): this {
    let list = this._listeners.get(type);
    if (!list) {
      list = [];
      this._listeners.set(type, list);
    }
    list.push(cb);
    return this;
  }

  /** Drive the stream to completion, invoking registered callbacks. */
  async run(): Promise<void> {
    for await (const event of this) {
      const listeners = this._listeners.get(event.type);
      if (!listeners) continue;
      for (const cb of listeners) {
        switch (event.type) {
          case 'text':
          case 'thinking':
          case 'thinking-done':
          case 'done':
            cb(event.text);
            break;
          case 'tool-call-start':
            cb();
            break;
          case 'tool-call':
            cb(event.name, event.arguments);
            break;
        }
      }
    }
  }

  async *[Symbol.asyncIterator](): AsyncIterableIterator<Event> {
    for await (const tokens of this._stream) {
      yield this._decoder.feed(tokens);
    }
  }
}

// ─── Context ────────────────────────────────────────────────────────────

/**
 * A conversation / generation context.
 *
 * Wraps the `pie:core/context.Context` WIT resource and provides
 * high-level methods for chat formatting, generation, and forking.
 *
 * ```js
 * const ctx = Context.create(model);
 * ctx.system('You are helpful.');
 * ctx.user('Hello!');
 * const text = ctx.generateText({ sampler: Sampler.topP(0.6, 0.95), maxTokens: 128 });
 * ```
 */
export class Context implements Disposable {
  /** @internal */
  readonly _handle: _Context;

  private constructor(handle: _Context) {
    this._handle = handle;
  }

  /** Create a new anonymous context. Name is NOT needed. */
  static create(model: Model): Context {
    return new Context(_Context.create(model._handle));
  }

  /** Open a saved (named) context. */
  static open(model: Model, name: string): Context | undefined {
    const handle = _Context.open(model._handle, name);
    return handle !== undefined ? new Context(handle) : undefined;
  }

  /** Look up an existing context by name. Alias for open(). */
  static lookup(model: Model, name: string): Context | undefined {
    return Context.open(model, name);
  }

  /** Fork this context into a new anonymous one. */
  fork(): Context {
    return new Context(this._handle.fork());
  }

  /** Save this context with a name, making it persistent. */
  save(name: string): void {
    this._handle.save(name);
  }

  /** Destroy the context and release its KV resources. */
  destroy(): void {
    this._handle.destroy();
  }

  /** Disposable protocol — calls `destroy()`. */
  [Symbol.dispose](): void {
    this.destroy();
  }

  // ── Text-level buffer fill ──

  /** Fill the context buffer with text (encodes via the model's tokenizer). */
  fill(text: string): void {
    const tokenizer = this._handle.model().tokenizer();
    const tokens = tokenizer.encode(text);
    this._handle.appendBufferedTokens(tokens);
  }

  /** Fill the context buffer with raw token IDs. */
  fillTokens(tokens: Uint32Array): void {
    this._handle.appendBufferedTokens(tokens);
  }

  // ── Chat formatting (pie:instruct/chat) ──

  /** Fill a system message. */
  system(message: string): void {
    _chat.system(this._handle, message);
  }

  /** Fill a user message. */
  user(message: string): void {
    _chat.user(this._handle, message);
  }

  /** Fill an assistant message (for history replay). */
  assistant(message: string): void {
    _chat.assistant(this._handle, message);
  }

  /** Cue the model to generate (fills generation header). */
  cue(): void {
    _chat.cue(this._handle);
  }

  /** Seal the current turn (inserts stop token). */
  seal(): void {
    _chat.seal(this._handle);
  }

  /** Returns the stop token IDs for this context's model. */
  stopTokens(): Uint32Array {
    return _chat.stopTokens(this._handle.model());
  }

  // ── Tool use (pie:instruct/tool-use) ──

  /** Register available tools (list of JSON schema strings). */
  equipTools(tools: string[]): void {
    _toolUse.equip(this._handle, tools);
  }

  /** Provide a tool result after a tool call. */
  answerTool(name: string, value: string): void {
    _toolUse.answer(this._handle, name, value);
  }

  // ── Low-level context operations ──

  /** Number of tokens per KV page. */
  tokensPerPage(): number {
    return this._handle.tokensPerPage();
  }

  /** Get the underlying model handle. */
  model(): _Model {
    return this._handle.model();
  }

  /** Get the buffered (pending) token IDs. */
  bufferedTokens(): Uint32Array {
    return this._handle.bufferedTokens();
  }

  /** Set the buffered token IDs. */
  setBufferedTokens(tokens: Uint32Array): void {
    this._handle.setBufferedTokens(tokens);
  }

  /** Append tokens to the buffer. */
  appendBufferedTokens(tokens: Uint32Array): void {
    this._handle.appendBufferedTokens(tokens);
  }

  /** Get the last committed position, or undefined if empty. */
  lastPosition(): number | undefined {
    return this._handle.lastPosition();
  }

  /** Flush buffered tokens by executing a fill forward pass (no sampling).
   *  Keeps the last token as seed for the next generation step. */
  async flush(): Promise<void> {
    const buffered = this._handle.bufferedTokens();
    if (buffered.length <= 1) return;

    // Flush all but the last token
    const tokensToFlush = buffered.slice(0, -1);
    const lastToken = buffered[buffered.length - 1];

    await _reserveAndRun(this._handle, tokensToFlush);

    // Keep last token as seed
    this._handle.setBufferedTokens(new Uint32Array([lastToken]));
  }

  // ── Generation ──

  /**
   * Start token generation, returning a `TokenStream` or `EventStream`.
   *
   * Automatically calls `flush()` and `cue()` before generating.
   * Stop tokens default to the model's stop tokens if not specified.
   */
  async generate(options: GenerateOptions & { decode: DecoderOptions }): Promise<EventStream>;
  async generate(options: GenerateOptions): Promise<TokenStream>;
  async generate(options: GenerateOptions): Promise<TokenStream | EventStream> {
    await this.flush();
    this.cue();

    const stream = new TokenStream(this, options);

    if (options.decode) {
      const decoder = new Decoder(this._handle.model(), options.decode);
      return new EventStream(stream, decoder);
    }

    return stream;
  }

  /**
   * Generate and return the full response text.
   *
   * Convenience method that calls `generate()` + `text()` internally.
   *
   * ```js
   * const reply = ctx.generateText({
   *   sampler: Sampler.topP(0.6, 0.95),
   *   maxTokens: 256,
   * });
   * ```
   */
  async generateText(options: Omit<GenerateOptions, 'decode'>): Promise<string> {
    return (await this.generate(options)).text();
  }
}
