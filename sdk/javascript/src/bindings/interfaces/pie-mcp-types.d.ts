/** @module Interface pie:mcp/types **/
/**
 * Opaque JSON carried as a string. Never inspected by the runtime.
 */
export type Json = string;
/**
 * Transport / JSON-RPC error. Mirrors the JSON-RPC 2.0 error shape.
 * This represents protocol-level failures (server unreachable, method
 * not found, etc.). Tool-level failures (`isError: true` in a
 * `tools/call` response) arrive as a successful `json` payload that
 * SDKs should inspect.
 */
export interface Error {
  code: number,
  message: string,
  data?: Json,
}
