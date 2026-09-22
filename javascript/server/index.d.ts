import type { PieClient } from '@pie-project/client';

/** The native addon file this platform loads, or null when none is built. */
export function addonPath(): string | null;

/** `pie serve`'s config: the file's shape, as an object or TOML text. */
export type Config = string | Record<string, unknown>;

export interface StartOptions {
  /** What to serve: a Hugging Face repo, a path, an imported artifact's name. In the browser: the artifact's URL. */
  model: string;
  /** The rest of the config (server, engine, sandbox, …), as an object; TOML text is accepted for a bare `start(config)`. */
  config?: Record<string, unknown> | string;
  /** Browser only: download progress of a lazy boot. */
  onProgress?: (served: number, total: number, from: 'network' | 'cache') => void;
}

/** A running pie engine in this process. The same class serves in Node (native engines) and in the browser (WebGPU). */
export class Server {
  /**
   * Boot the engine; resolves once it is serving, with every registered
   * language component (`import "@pie-project/language-python"`) installed.
   */
  static start(options: StartOptions | Config): Promise<Server>;
  /** `ws://host:port` the engine listens on; `pie://local` in the browser. */
  readonly url: string;
  /** Node: `http://host:port`, where the OpenAI / Anthropic / Gemini routes live. */
  readonly httpUrl?: string;
  /** True until `shutdown()` resolves. */
  readonly running: boolean;
  /** A language component (`python`, `javascript`) from bytes or a URL. */
  installLanguage(language: 'python' | 'javascript' | string, source: Uint8Array | ArrayBuffer | URL | string): Promise<string>;
  /** An inferlet from its component bytes (or URL) and manifest text, replacing an installed version; resolves to `name@version`. */
  install(source: Uint8Array | ArrayBuffer | URL | string, manifest: string): Promise<string>;
  /** A `PieClient` connected to this engine; closed by `shutdown()`. */
  connect(options?: { identity?: string }): Promise<PieClient>;
  /** Stop the engine and release it. Idempotent. */
  shutdown(): Promise<void>;
}
