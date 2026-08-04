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

export type ForwardKind = 'attention' | 'recurrent' | 'hybrid';

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
  rsStateSize: () => 4096n,
  rsBufferPageSize: () => 64,
  rsFoldGranularity: () => 1,
  arenaBlockSize: () => 8192n,
};

// ── pie:inferlet/tokenizer ──────────────────────────────────────────────────

// Byte-identity codec: keeps assertions readable while still exercising the
// Uint32Array marshalling.
export const tokenizerStub = {
  encode: (text: string) =>
    Uint32Array.from([...text].map((c) => c.charCodeAt(0))),
  decode: (tokens: Uint32Array) =>
    String.fromCharCode(...Array.from(tokens)),
  vocabs: (): [Uint32Array, Uint8Array[]] => [
    Uint32Array.from([0, 1]),
    [new Uint8Array([97]), new Uint8Array([98])],
  ],
  splitRegex: () => '\\w+',
  specialTokens: (): [Uint32Array, Uint8Array[]] => [
    Uint32Array.from([2]),
    [new Uint8Array([60, 62])],
  ],
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
  sendFile: (data: Uint8Array) => {
    sessionSpy.sentFiles.push(data);
  },
  receiveFile: async () => sessionSpy.filesToReceive.shift(),
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
  system: (m: string) => Uint32Array.from([1, ...[...m].map((c) => c.charCodeAt(0))]),
  firstUser: (m: string) => Uint32Array.from([2, ...[...m].map((c) => c.charCodeAt(0))]),
  user: (m: string) => Uint32Array.from([3, ...[...m].map((c) => c.charCodeAt(0))]),
  systemUser: () => Uint32Array.from([4]),
  assistant: (m: string) => Uint32Array.from([5, ...[...m].map((c) => c.charCodeAt(0))]),
  cue: () => Uint32Array.from([6]),
  seal: () => Uint32Array.from([7]),
  stopTokens: () => Uint32Array.from([8, 9]),
  createDecoder: () => {
    const d = new ScriptedDecoder(chatScript.events);
    chatScript.last = d;
    return d;
  },
};

export const reasoningScript: { events: Variant[]; last?: ScriptedDecoder } = {
  events: [],
};

export const reasoningStub = {
  createDecoder: () => {
    const d = new ScriptedDecoder(reasoningScript.events);
    reasoningScript.last = d;
    return d;
  },
};
