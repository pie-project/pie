// Stub WIT bindings for unit-testing the SDK's hand-written layer outside a
// jco build.
//
// `pie:inferlet/*` specifiers only resolve to real host functions inside a
// component, so vitest aliases them here (see vitest.config.ts). These stubs
// stand in for the imports the hand-written layer actually uses, and no more.
//
// SCOPE: the non-forward interfaces only, which is the SDK's entire surface
// today. The forward-pass interfaces have no JavaScript counterpart -- see
// src/index.ts and scripts/check-sdk-interfaces.sh. When they land, their
// stubs belong here.
//
// These are stubs, not a simulator, and a stub can be made to agree with a
// world that no longer exists. What proves the SDK matches the REAL world is
// `npm run generate-bindings && npm run build`: tsc resolves every specifier
// through the generated `paths` map, so a member the host stopped offering is
// a compile error.

// ── pie:inferlet/model ──────────────────────────────────────────────────────

export type ForwardKind = 'attention' | 'recurrent' | 'hybrid' | 'diffusion';

export const modelStub = {
  name: () => 'mock-model',
  architecture: () => 'qwen3_5',
  defaultSystemSpeculation: () => false,
  isLinear: () => true,
  passKind: (): ForwardKind => 'hybrid',
  outputVocabSize: () => 151936,
  kvPageSize: () => 16,
  frameSize: () => 1,
  channelCapacity: () => 8,
  maxEmbedLength: () => 2048,
  prefillChunkHint: () => 2048,
  rsStateSize: () => 4096n,
  rsBufferPageSize: () => 64,
  rsFoldGranularity: () => 1,
  arenaBlockSize: () => 8192n,
  mtpDepth: () => 0,
  submitDeadlineUs: () => 50_000n,
  runAheadWindow: () => 4,
  draftBlock: () => undefined,
  canvas: () => ({ length: 32, hidden: 2560, selfCondTaps: 4 }),
};

// ── pie:inferlet/tokenizer ──────────────────────────────────────────────────

// Byte-identity codec: keeps assertions readable while still exercising the
// Uint32Array marshalling.
export const tokenizerStub = {
  encode: (text: string) =>
    Uint32Array.from([...text].map((c) => c.charCodeAt(0))),
  decode: (tokens: Uint32Array) =>
    String.fromCharCode(...Array.from(tokens)),
  vocabs: () => [
    { id: 0, bytes: new Uint8Array([97]) },
    { id: 1, bytes: new Uint8Array([98]) },
  ],
  splitRegex: () => '\\w+',
  specialTokens: () => [{ id: 2, bytes: new Uint8Array([60, 62]) }],
  tokenBytes: (tokens: Uint32Array) => Array.from(tokens, (t) => new Uint8Array([t])),
  tokensWithPrefix: (prefix: Uint8Array) => Uint32Array.from(prefix),
};

// ── pie:inferlet/session ────────────────────────────────────────────────────

export const sessionSpy = {
  sent: [] as string[],
  sentFiles: [] as Uint8Array[],
  toReceive: [] as (string | undefined)[],
  filesToReceive: [] as (Uint8Array | undefined)[],
  reset() {
    this.sent = [];
    this.sentFiles = [];
    this.toReceive = [];
    this.filesToReceive = [];
  },
};

export const sessionStub = {
  send: (message: string) => {
    sessionSpy.sent.push(message);
  },
  receive: async () => sessionSpy.toReceive.shift(),
  receiveBlocking: () => sessionSpy.toReceive.shift(),
  sendFile: (data: Uint8Array) => {
    sessionSpy.sentFiles.push(data);
  },
  receiveFile: async () => sessionSpy.filesToReceive.shift(),
  receiveFileBlocking: () => sessionSpy.filesToReceive.shift(),
};

// ── the forward-pass surface ────────────────────────────────────────────────
//
// Enough for `eta/bridge.ts` to import and for its host-side plumbing to be
// exercised without a device: a channel remembers what was put and hands it
// back on `takeBlocking`; a pass keeps the container bytes it was given.

export class ChannelStub {
  static created: ChannelStub[] = [];
  cells: Uint8Array[] = [];
  constructor(
    public shape: Uint32Array,
    public dtype: string,
    public capacity: number,
  ) {
    ChannelStub.created.push(this);
  }
  put(value: Uint8Array) {
    this.cells.push(new Uint8Array(value));
  }
  set(value: Uint8Array) {
    if (!this.cells.length) throw Object.assign(new Error('set on an empty channel'), { payload: 'set on an empty channel' });
    this.cells[0] = new Uint8Array(value);
  }
  takeBlocking(): Uint8Array {
    const v = this.cells.shift();
    if (!v) throw Object.assign(new Error('take on an empty channel (stub)'), { payload: 'take on an empty channel (stub)' });
    return v;
  }
  readBlocking(): Uint8Array {
    if (!this.cells.length) throw Object.assign(new Error('read on an empty channel (stub)'), { payload: 'read on an empty channel (stub)' });
    return this.cells[0];
  }
}

export class PipelineStub {
  closed = false;
  close() {
    this.closed = true;
  }
}

export class KvWorkingSetStub {
  pages = 0;
  pageLen() {
    return this.pages;
  }
  reserve(pages: number) {
    const r = { start: this.pages, len: pages };
    this.pages += pages;
    return r;
  }
  fork() {
    return new KvWorkingSetStub();
  }
}

export class RsWorkingSetStub {
  bufferSize() {
    return 0;
  }
}

export class ForwardPassStub {
  static submitted: [unknown, unknown[]][] = [];
  embedded: unknown[] | null = null;
  attentionArgs: unknown[] | null = null;
  programBytes: Uint8Array | null = null;
  programChannels: unknown[] | null = null;
  spans: unknown[] | null = null;
  embed(tokens: unknown, indptr: unknown) {
    this.embedded = [tokens, indptr];
  }
  readout() {}
  attention(...args: unknown[]) {
    this.attentionArgs = args;
  }
  setMaxLayers() {}
  setDraftingBlock() {}
  media(spans: unknown[]) {
    this.spans = [...spans];
  }
  mode: string | null = null;
  selfCond: [number[], number[]] | null = null;
  canvas(mode: string) {
    this.mode = mode;
  }
  selfConditioning(rows: Uint32Array, weights: Float32Array) {
    this.selfCond = [Array.from(rows), Array.from(weights)];
  }
  program(bytes: Uint8Array, channels: unknown[]) {
    this.programBytes = new Uint8Array(bytes);
    this.programChannels = [...channels];
  }
}

export const forwardStub = {
  ForwardPass: ForwardPassStub,
  submit: (on: unknown, slots: unknown[]) => {
    ForwardPassStub.submitted.push([on, [...slots]]);
  },
  park: () => {},
};

// ── pie:inferlet/chat and pie:inferlet/reasoning ────────────────────────────
//
// jco lowers a WIT variant to `{ tag, val }`, and both decoders switch on
// `tag` -- so the stub has to keep that shape exactly.

export interface Variant {
  tag: string;
  val?: unknown;
}

class ScriptedDecoder {
  static script: Variant[] = [];
  fed: number[][] = [];
  resets = 0;
  #queue: Variant[];

  constructor(script: Variant[]) {
    this.#queue = [...script];
  }

  feed(tokens: Uint32Array): Variant {
    this.fed.push(Array.from(tokens));
    const next = this.#queue.shift();
    // WIT declares `feed: func(...) -> result<event, error>`, which jco lowers
    // to "returns an event or throws". There is no empty return, so a test
    // that runs off the end of its script is a test bug, not an idle event.
    if (next === undefined) {
      throw new Error('decoder stub: script exhausted');
    }
    return next;
  }

  reset(): void {
    this.resets += 1;
  }
}

export const chatScript: { events: Variant[]; last?: ScriptedDecoder } = {
  events: [],
};

export const chatStub = {
  prefix: () => Uint32Array.from([0]),
  systemUser: (s: string, u: string) => Uint32Array.from([4, s.length, u.length]),
  system: (m: string) => Uint32Array.from([1, ...[...m].map((c) => c.charCodeAt(0))]),
  firstUser: (m: string) => Uint32Array.from([2, ...[...m].map((c) => c.charCodeAt(0))]),
  user: (m: string) => Uint32Array.from([3, ...[...m].map((c) => c.charCodeAt(0))]),
  systemUser: () => Uint32Array.from([4]),
  assistant: (m: string) => Uint32Array.from([5, ...[...m].map((c) => c.charCodeAt(0))]),
  cue: () => Uint32Array.from([6]),
  seal: () => Uint32Array.from([7]),
  stopTokens: () => Uint32Array.from([8, 9]),
  Decoder: class extends ScriptedDecoder {
    constructor() {
      super(chatScript.events);
      chatScript.last = this;
    }
  },
};

export const reasoningScript: { events: Variant[]; last?: ScriptedDecoder } = {
  events: [],
};

export const reasoningStub = {
  Decoder: class extends ScriptedDecoder {
    constructor() {
      super(reasoningScript.events);
      reasoningScript.last = this;
    }
  },
};


// ── pie:inferlet/grammar, tools, media ──────────────────────────────────────
//
// Enough for the wrappers to import and marshal; a grammar "matches"
// everything and terminates after 3 tokens.

const witErr = (payload: string) => Object.assign(new Error(payload), { payload });

export class GrammarStub {
  constructor(readonly source: string) {}
  static fromJsonSchema(schema: string) {
    if (schema === 'bad') throw witErr('not a schema');
    return new GrammarStub(schema);
  }
  static json() {
    return new GrammarStub('json');
  }
  static fromRegex(p: string) {
    return new GrammarStub(p);
  }
  static fromEbnf(e: string) {
    return new GrammarStub(e);
  }
  toString() {
    return this.source;
  }
}

export class MatcherStub {
  accepted: number[] = [];
  constructor(readonly grammar: GrammarStub) {}
  acceptTokens(ids: Uint32Array) {
    if (Array.from(ids).includes(999)) throw witErr('token 999 is not allowed');
    this.accepted.push(...ids);
  }
  mask() {
    return Uint32Array.from([0b101]);
  }
  isTerminated() {
    return this.accepted.length >= 3;
  }
  reset() {
    this.accepted = [];
  }
  fork() {
    const m = new MatcherStub(this.grammar);
    m.accepted = [...this.accepted];
    return m;
  }
  rollback(n: number) {
    this.accepted.length -= n;
  }
  rollbackCapacity() {
    return this.accepted.length;
  }
}

export const grammarStub = { Grammar: GrammarStub, Matcher: MatcherStub };

export const toolsStub = {
  equip: (tools: string[]) => Uint32Array.from([10, tools.length]),
  answer: (name: string, value: string) => Uint32Array.from([11, name.length, value.length]),
  format: (tools: string[]) => (tools.length ? new GrammarStub('tools') : undefined),
  createMatcher: () => new MatcherStub(new GrammarStub('tools')),
  Decoder: class {
    n = 0;
    feed() {
      this.n += 1;
      if (this.n === 1) return { tag: 'start' };
      return { tag: 'call', val: { name: 'lookup', argumentsJson: '{"q": 1}' } };
    }
    reset() {
      this.n = 0;
    }
  },
};

class ImageStub {
  static fromBytes(data: Uint8Array) {
    if (data.length === 0) throw witErr('empty image');
    return new ImageStub();
  }
  tokens() {
    return Uint32Array.from([5, 6, 7]);
  }
  digest() {
    return new Uint8Array([1, 2]);
  }
  tokenCount() {
    return 3;
  }
  positionSpan() {
    return 1;
  }
  grid() {
    return { t: 1, h: 2, w: 3 };
  }
  prefixTokens() {
    return Uint32Array.from([5]);
  }
  suffixTokens() {
    return Uint32Array.from([7]);
  }
}

export const mediaStub = { Image: ImageStub, Audio: ImageStub, Video: ImageStub };
