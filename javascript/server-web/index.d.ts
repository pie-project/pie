import { PieClient } from "@pie-project/client";

export { PieClient };

export interface LoadInfo {
  worker: boolean;
}

export interface LoadOptions {
  worker?: boolean;
  log?: string;
  wasmUrl?: string;
  fresh?: boolean;
}

export interface BootSummary {
  model: string;
  sku: string;
  trace: string;
  weight_bytes: number;
  kv_pages: number;
  [key: string]: unknown;
}

export function load(options?: LoadOptions): Promise<LoadInfo>;

export function bootLazy(
  url: string,
  config?: string,
  onProgress?: (served: number, total: number, from: "network" | "cache") => void,
): Promise<BootSummary>;

export function awaitCache(): Promise<void>;

export function install(wasmBytes: Uint8Array | ArrayBuffer, manifestToml: string): Promise<string>;

export function client(): PieClient;

export function connect(): Promise<PieClient>;

export function memoryBytes(): Promise<number>;

export { registerLanguage, registeredLanguages, attachLanguages } from "./src/languages.js";

export interface StartOptions extends LoadOptions {
  /** The URL of the `.wgpu.zt` artifact, served with Range support. */
  model: string;
  /** The boot config: an object, or TOML text. */
  config?: Record<string, unknown> | string;
  onProgress?: (served: number, total: number, from: "network" | "cache") => void;
}

/** One running pie in this tab, with the surface of the Node host. */
export class Server {
  static start(options: StartOptions): Promise<Server>;
  readonly summary: BootSummary;
  readonly url: string;
  readonly running: boolean;
  installLanguage(language: "python" | "javascript" | string, source: Uint8Array | ArrayBuffer | URL | string): Promise<string>;
  install(source: Uint8Array | ArrayBuffer | URL | string, manifest: string): Promise<string>;
  connect(): Promise<PieClient>;
  memoryBytes(): Promise<number>;
  awaitCache(): Promise<void>;
  shutdown(): Promise<void>;
}
