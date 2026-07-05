/** @module Interface pie:core/http **/
/**
 * One HTTP request, host-side + async, response fully buffered. Does NOT
 * follow redirects (the SDK resolves `location` and re-requests). There is
 * no guest-side wasi:io pollable.
 */
export function fetch(request: Request): Promise<Response>;
export type Error = import('./pie-core-types.js').Error;
/**
 * A single buffered HTTP request.
 */
export interface Request {
  method: string,
  url: string,
  headers: Array<[string, string]>,
  body?: Uint8Array,
}
/**
 * A fully-buffered HTTP response. HTTP status (incl. 4xx/5xx) is a normal
 * response; `error` is reserved for transport/host failures.
 */
export interface Response {
  status: number,
  headers: Array<[string, string]>,
  body: Uint8Array,
}
