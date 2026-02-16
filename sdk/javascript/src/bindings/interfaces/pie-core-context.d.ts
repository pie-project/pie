/** @module Interface pie:core/context **/
export type Pollable = import('./wasi-io-poll.js').Pollable;
export type Error = import('./pie-core-types.js').Error;
export type FutureBool = import('./pie-core-types.js').FutureBool;
export type Model = import('./pie-core-model.js').Model;
export type PageId = number;

export class Context {
  /**
   * This type does not have a public constructor.
   */
  private constructor();
  /**
  * Creates a new context, optionally with a name
  */
  static create(model: Model, name: string, fill: Uint32Array | undefined): Context;
  destroy(): void;
  /**
  * Retrieves an existing context by name
  */
  static lookup(model: Model, name: string): Context | undefined;
  /**
  * Forks this context into a new one with the given name
  */
  fork(newName: string): Context;
  /**
  * Acquires a lock on the context, returning a lock result
  */
  acquireLock(): FutureBool;
  /**
  * Releases the lock
  */
  releaseLock(): void;
  /**
  * Number of tokens per page
  */
  tokensPerPage(): number;
  model(): Model;
  /**
  * Number of committed KV pages in the context
  */
  committedPageCount(): number;
  /**
  * Number of uncommitted KV pages in the context
  */
  uncommittedPageCount(): number;
  /**
  * Commit the KV pages to the context
  */
  commitPages(pageIndices: Uint32Array): void;
  /**
  * Reserve KV pages for the context
  */
  reservePages(numPages: number): void;
  /**
  * Release KV pages from the context
  */
  releasePages(numPages: number): void;
  cursor(): number;
  setCursor(cursor: number): void;
  bufferedTokens(): Uint32Array;
  setBufferedTokens(tokens: Uint32Array): void;
  appendBufferedTokens(tokens: Uint32Array): void;
  lastPosition(): number | undefined;
}
