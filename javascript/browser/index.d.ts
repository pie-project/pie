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
