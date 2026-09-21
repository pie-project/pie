import type { PieClient } from '@pie-project/client';

/** The native addon file this platform loads, or null when none is built. */
export function addonPath(): string | null;

/** A running pie engine in this process. */
export class Server {
  /**
   * Boot the engine from the TOML text `pie serve --config` reads, or an
   * object of the same shape. Resolves once the engine is serving.
   */
  static start(config: string | Record<string, unknown>): Promise<Server>;
  /** `ws://host:port` the engine listens on. */
  readonly url: string;
  /** `http://host:port`, where the OpenAI / Anthropic / Gemini routes live. */
  readonly httpUrl: string;
  /** True until `shutdown()` resolves. */
  readonly running: boolean;
  /** A `PieClient` connected to this engine; closed by `shutdown()`. */
  connect(options?: { identity?: string }): Promise<PieClient>;
  /** Stop the engine and release it. Idempotent. */
  shutdown(): Promise<void>;
}
