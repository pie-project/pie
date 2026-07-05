// Context — SDK-owned conversation state over kv-working-set.
//
// The runtime no longer owns an opaque context resource. This facade owns the
// semantic state (token buffer, sequence cursor, replay history) and describes
// every forward pass with explicit KV read/write descriptors over a dense
// `KvWorkingSet` page-slot array.
//
// Single-model runtime: the working set binds to the one bound model
// implicitly (`new KvWorkingSet()` takes no handle), and model metadata is
// reached through the global `pie:core/model` functions.

import { ForwardPass as _ForwardPass } from 'pie:core/inference';
import type { Audio, Image } from 'pie:core/inference';
import { KvWorkingSet as _KvWorkingSet } from 'pie:core/working-set';
import * as _chatWit from 'pie:instruct/chat';
import * as _fsPreopens from 'wasi:filesystem/preopens@0.2.4';
import type { Descriptor as _FsDescriptor } from 'wasi:filesystem/types@0.2.4';

import * as _chat from './chat.js';
import { Forward } from './forward.js';
import { Generator, type GenerateOptions } from './generation.js';
import * as _model from './model.js';
import * as _runtime from './runtime.js';

import type { Sampler } from './sample.js';

declare const TextEncoder: {
  new(): { encode(input?: string): Uint8Array };
};
declare const TextDecoder: {
  new(label?: string): { decode(input?: Uint8Array): string };
};

// =============================================================================
// Snapshot blob
// =============================================================================

const SNAPSHOT_VERSION = 1;
let SNAPSHOT_COUNTER = 0;

interface _SnapshotManifest {
  version: number;
  pageSize: number;
  seqLen: number;
  tokens: number[];
  buffer: number[];
  pendingSystem: string | null;
  casHashes: number[];
}

function _snapshotPath(name: string): string {
  return `${name}.pie-snapshot`;
}

function _snapshotRoot(): _FsDescriptor {
  const dirs = _fsPreopens.getDirectories();
  // Prefer the runtime's `/scratch` mount (the inferlet's writable scratch dir,
  // matching the Rust/Python SDK convention); fall back to cwd / first preopen.
  const preferred =
    dirs.find(([, path]) => path === '/scratch') ??
    dirs.find(([, path]) => path === '.' || path === '');
  const entry = preferred ?? dirs[0];
  if (entry === undefined) {
    throw new Error('Context snapshot I/O: no WASI filesystem preopen available');
  }
  return entry[0];
}

function _writeFile(path: string, bytes: Uint8Array): void {
  if (path.startsWith('/')) {
    throw new Error(`Context snapshot I/O: absolute paths are not supported in JS SDK snapshots: ${path}`);
  }
  const file = _snapshotRoot().openAt(
    {},
    path,
    { create: true, truncate: true },
    { write: true },
  );
  let offset = 0n;
  while (offset < BigInt(bytes.length)) {
    const written = file.write(bytes.subarray(Number(offset)), offset);
    if (written <= 0n) {
      throw new Error(`Context snapshot I/O: short write to '${path}'`);
    }
    offset += written;
  }
  file.syncData();
}

function _readFile(path: string): Uint8Array {
  if (path.startsWith('/')) {
    throw new Error(`Context snapshot I/O: absolute paths are not supported in JS SDK snapshots: ${path}`);
  }
  const file = _snapshotRoot().openAt({}, path, {}, { read: true });
  const size = file.stat().size;
  if (size > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error(`Context snapshot I/O: '${path}' is too large to read`);
  }
  const chunks: Uint8Array[] = [];
  let offset = 0n;
  let total = 0;
  while (offset < size) {
    const [chunk, eof] = file.read(size - offset, offset);
    if (chunk.length > 0) {
      chunks.push(chunk);
      offset += BigInt(chunk.length);
      total += chunk.length;
    }
    if (eof) break;
    if (chunk.length === 0) {
      throw new Error(`Context snapshot I/O: short read from '${path}'`);
    }
  }
  const out = new Uint8Array(total);
  let at = 0;
  for (const chunk of chunks) {
    out.set(chunk, at);
    at += chunk.length;
  }
  return out;
}

function _deleteFileBestEffort(path: string): void {
  try {
    if (!path.startsWith('/')) _snapshotRoot().unlinkFileAt(path);
  } catch {
    // Missing snapshots are a no-op, matching the Rust facade.
  }
}

function _readManifest(name: string): _SnapshotManifest {
  const path = _snapshotPath(name);
  let text: string;
  try {
    text = new TextDecoder('utf-8').decode(_readFile(path));
  } catch (e) {
    throw new Error(`snapshot '${name}': read: ${String(e)}`);
  }

  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (e) {
    throw new Error(`snapshot '${name}': parse: ${String(e)}`);
  }
  if (raw === null || typeof raw !== 'object') {
    throw new Error(`snapshot '${name}': parse: manifest is not an object`);
  }
  const obj = raw as Partial<_SnapshotManifest> & {
    page_size?: number;
    seq_len?: number;
    pending_system?: string | null;
    cas_hashes?: number[];
  };
  const version = obj.version;
  if (version !== SNAPSHOT_VERSION) {
    throw new Error(
      `snapshot '${name}': version ${String(version)} unsupported (expected ${SNAPSHOT_VERSION})`,
    );
  }
  const pageSize = obj.pageSize ?? obj.page_size;
  const seqLen = obj.seqLen ?? obj.seq_len;
  const pendingSystem = obj.pendingSystem ?? obj.pending_system ?? null;
  const casHashes = obj.casHashes ?? obj.cas_hashes ?? [];
  if (
    typeof pageSize !== 'number' ||
    typeof seqLen !== 'number' ||
    !Array.isArray(obj.tokens) ||
    !Array.isArray(obj.buffer) ||
    !(pendingSystem === null || typeof pendingSystem === 'string') ||
    !Array.isArray(casHashes)
  ) {
    throw new Error(`snapshot '${name}': parse: invalid manifest shape`);
  }
  return {
    version,
    pageSize,
    seqLen,
    tokens: obj.tokens.map(Number),
    buffer: obj.buffer.map(Number),
    pendingSystem,
    casHashes: casHashes.map(Number),
  };
}

export interface PreparedKvWrite {
  generation: number;
  indices: number[];
  validLens: number[];
  ctxPages: number;
}

// =============================================================================
// Context
// =============================================================================

/** High-level inference context backed by a KV working set. */
export class Context implements Disposable {
  /** @internal Dense runtime KV page-slot array. */
  _kv: _KvWorkingSet;
  /** @internal */ _pageSize: number;
  /** @internal SDK-side token buffer filled by instruct operations. */
  _buffer: number[] = [];
  /** @internal Deferred system prompt text. */
  _pendingSystem: string | null = null;
  /** @internal Materialized KV token count. */
  _seqLen = 0;
  /** @internal Replayable materialized text-token log. */
  _history: number[] = [];
  /** @internal False once non-replayable soft-token KV is materialized. */
  _snapshottable = true;

  constructor() {
    this._kv = new _KvWorkingSet();
    this._pageSize = this._kv.pageSize();
  }

  /** Open a saved snapshot (implicit fork — snapshot stays on disk). */
  static async open(name: string): Promise<Context> {
    return Context._fromManifest(_readManifest(name));
  }

  /** Take ownership of a snapshot (snapshot is deleted after replay). */
  static async take(name: string): Promise<Context> {
    const ctx = await Context.open(name);
    _deleteFileBestEffort(_snapshotPath(name));
    return ctx;
  }

  /** Delete a saved snapshot by name. Missing snapshots are a no-op. */
  static delete(name: string): void {
    _deleteFileBestEffort(_snapshotPath(name));
  }

  /** @internal */
  static async _fromManifest(manifest: _SnapshotManifest): Promise<Context> {
    const ctx = new Context();
    if (manifest.tokens.length > 0) {
      ctx._buffer = manifest.tokens.slice();
      await ctx.flush();
    }
    ctx._buffer = manifest.buffer.slice();
    ctx._pendingSystem = manifest.pendingSystem;
    return ctx;
  }

  /** Fork into a new anonymous context sharing KV pages via lazy CoW. */
  fork(): Context {
    const obj = Object.create(Context.prototype) as Context;
    obj._kv = this._kv.fork();
    obj._pageSize = this._pageSize;
    obj._buffer = this._buffer.slice();
    obj._pendingSystem = this._pendingSystem;
    obj._seqLen = this._seqLen;
    obj._history = this._history.slice();
    obj._snapshottable = this._snapshottable;
    return obj;
  }

  /** Save this context with a name. */
  save(name: string): void {
    if (!this._snapshottable) {
      throw new Error(
        'Context.save: multimodal contexts are not snapshottable in v1 ' +
          '(soft-token KV cannot be replayed from a token log)',
      );
    }
    const manifest: _SnapshotManifest = {
      version: SNAPSHOT_VERSION,
      pageSize: this._pageSize,
      seqLen: this._seqLen,
      tokens: this._history.slice(),
      buffer: this._buffer.slice(),
      pendingSystem: this._pendingSystem,
      casHashes: [],
    };
    const bytes = new TextEncoder().encode(JSON.stringify(manifest));
    _writeFile(_snapshotPath(name), bytes);
  }

  /** Anonymous save — returns a runtime-generated name. */
  snapshot(): string {
    const name = `anon-${_runtime.instanceId()}-${SNAPSHOT_COUNTER++}`;
    this.save(name);
    return name;
  }

  /** Force-release this context. KV resources are owned by the WIT resource. */
  release(): void {
    // No explicit destroy exists on kv-working-set; dropping the resource releases it.
  }

  /** Disposable protocol — calls `release()`. */
  [Symbol.dispose](): void {
    this.release();
  }

  // ── Accessors ────────────────────────────────────────────────────────

  /** Tokens per KV page. */
  get pageSize(): number { return this._pageSize; }

  /** Total materialized tokens (excludes the pending buffer). */
  get seqLen(): number { return this._seqLen; }

  /** Pending (buffered but not yet flushed) tokens. */
  buffer(): Uint32Array {
    return Uint32Array.from(this._buffer);
  }

  /** Escape hatch: the underlying KV working set (power users). */
  workingSet(): _KvWorkingSet {
    return this._kv;
  }

  // ── Chat fillers ─────────────────────────────────────────────────────

  /** Fill a system-role message. */
  system(message: string): this {
    this._flushPendingSystem();
    this._pendingSystem = message;
    return this;
  }

  /** Fill a user-role message. */
  user(message: string): this {
    let tokens: Uint32Array;
    if (this._pendingSystem !== null) {
      const system = this._pendingSystem;
      this._pendingSystem = null;
      tokens = _chatWit.systemUser(system, message);
    } else if (this._isFirstChatFill()) {
      tokens = _chatWit.firstUser(message);
    } else {
      tokens = _chat.user(message);
    }
    this._appendPendingRaw(tokens);
    return this;
  }

  /** Fill an assistant-role message (history replay). */
  assistant(message: string): this {
    this._flushPendingSystem();
    this._appendPendingRaw(_chat.assistant(message));
    return this;
  }

  /** Cue the model to generate (fills the generation header). */
  cue(): this {
    this._flushPendingSystem();
    this._appendPendingRaw(_chat.cue());
    return this;
  }

  /** Seal the current turn (insert stop token). */
  seal(): this {
    this._flushPendingSystem();
    this._appendPendingRaw(_chat.seal());
    return this;
  }

  /** Append raw tokens to the buffer directly. */
  append(tokens: Iterable<number>): this {
    this._flushPendingSystem();
    this._appendPendingRaw(tokens);
    return this;
  }

  /** Append text — encodes via the bound model's tokenizer. */
  appendText(text: string): this {
    this._flushPendingSystem();
    this._appendPendingRaw(_model.encode(text));
    return this;
  }

  // ── Sequence / page bookkeeping ──────────────────────────────────────

  /** Drop the trailing `n` materialized tokens and free pages that become empty. */
  truncate(n: number): void {
    n = Math.min(Math.max(0, Math.floor(n)), this._seqLen);
    if (n === 0) return;
    this._seqLen -= n;
    this._history.length = Math.max(0, this._history.length - n);

    const livePages = Math.ceil(this._seqLen / this._pageSize);
    const have = this._kv.size();
    if (have > livePages) {
      const drop: number[] = [];
      for (let i = livePages; i < have; i++) drop.push(i);
      try {
        this._kv.free(Uint32Array.from(drop));
      } catch {
        // Best-effort: stale trailing pages are overwritten by the next forward.
      }
    }
  }

  /** @internal Build KV write descriptors for `n` new tail tokens. */
  _prepareWrite(n: number): PreparedKvWrite {
    const p = this._pageSize;
    const firstWritePage = Math.floor(this._seqLen / p);
    const totalAfter = this._seqLen + n;
    const totalPages = Math.ceil(totalAfter / p);
    const have = this._kv.size();
    if (totalPages > have) {
      this._kv.alloc(totalPages - have);
    }
    const generation = this._kv.generation();
    const indices: number[] = [];
    const validLens: number[] = [];
    for (let pg = firstWritePage; pg < totalPages; pg++) {
      indices.push(pg);
      validLens.push(Math.min(totalAfter - pg * p, p));
    }
    return { generation, indices, validLens, ctxPages: firstWritePage };
  }

  /** @internal Attach KV read/write descriptors returned by `_prepareWrite`. */
  _attachKv(pass: _ForwardPass, write: PreparedKvWrite): void {
    if (write.ctxPages > 0) {
      pass.kvContext({
        set: this._kv,
        start: 0,
        len: write.ctxPages,
        validTokens: write.ctxPages * this._pageSize,
      });
    }
    pass.kvOutput({
      set: this._kv,
      generation: write.generation,
      indices: Uint32Array.from(write.indices),
      perPageValidLens: Uint32Array.from(write.validLens),
    });
  }

  /** @internal Attach a read-only descriptor covering all materialized KV. */
  _attachFullContext(pass: _ForwardPass): void {
    const ctxPages = Math.ceil(this._seqLen / this._pageSize);
    if (ctxPages > 0) {
      pass.kvContext({
        set: this._kv,
        start: 0,
        len: ctxPages,
        validTokens: this._seqLen,
      });
    }
  }

  // ── Flush / multimodal append ────────────────────────────────────────

  /** Drain buffered tokens through a forward pass and materialize KV pages. */
  async flush(): Promise<void> {
    this._flushPendingSystem();
    if (this._buffer.length === 0) return;

    const tokens = this._buffer;
    this._buffer = [];
    const n = tokens.length;
    const positions = new Uint32Array(n);
    for (let i = 0; i < n; i++) positions[i] = this._seqLen + i;

    const write = this._prepareWrite(n);
    const pass = new _ForwardPass();
    this._attachKv(pass, write);
    pass.inputTokens(Uint32Array.from(tokens), positions);
    await pass.execute();

    this._history.push(...tokens);
    this._seqLen += n;
  }

  /** Splice an encoded image/video span into the context. */
  async appendImage(image: Image): Promise<void> {
    const prefix = image.prefixTokens();
    if (prefix.length > 0) this.append(prefix);
    await this.flush();

    const numTokens = image.tokenCount();
    if (numTokens > 0) {
      const write = this._prepareWrite(numTokens);
      const pass = new _ForwardPass();
      this._attachKv(pass, write);
      pass.inputImage(image, this._seqLen);
      await pass.execute();
      this._seqLen += numTokens;
      this._snapshottable = false;
    }

    const suffix = image.suffixTokens();
    if (suffix.length > 0) this.append(suffix);
  }

  /** Splice an encoded audio clip into the context. */
  async appendAudio(audio: Audio): Promise<void> {
    const prefix = audio.prefixTokens();
    if (prefix.length > 0) this.append(prefix);
    await this.flush();

    const numTokens = audio.tokenCount();
    if (numTokens > 0) {
      const write = this._prepareWrite(numTokens);
      const pass = new _ForwardPass();
      this._attachKv(pass, write);
      pass.inputAudio(audio, this._seqLen);
      await pass.execute();
      this._seqLen += numTokens;
      this._snapshottable = false;
    }

    const suffix = audio.suffixTokens();
    if (suffix.length > 0) this.append(suffix);
  }

  // ── Forward / Generate ───────────────────────────────────────────────

  /** Build a single forward pass with automatic page allocation. */
  forward(): Forward {
    this._flushPendingSystem();
    return new Forward(this);
  }

  /** Build a token generator. */
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
    } else {
      this._flushPendingSystem();
    }
    return new Generator(this, sampler, rest);
  }

  // ── Internal ─────────────────────────────────────────────────────────

  /** @internal */
  _flushPendingSystem(): void {
    if (this._pendingSystem === null) return;
    const system = this._pendingSystem;
    this._pendingSystem = null;
    this._appendPendingRaw(_chat.system(system));
  }

  /** @internal */
  _isFirstChatFill(): boolean {
    return this._seqLen === 0 && this._buffer.length === 0;
  }

  /** @internal Append tokens to the pending buffer without flushing system. */
  _appendPendingRaw(tokens: Iterable<number>): void {
    for (const t of tokens) this._buffer.push(t);
  }
}
