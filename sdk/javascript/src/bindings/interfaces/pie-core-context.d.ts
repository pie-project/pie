/** @module Interface pie:core/context **/
export type Error = import('./pie-core-types.js').Error;
export type PageId = number;

export class Context {
  /**
   * This type does not have a public constructor.
   */
  private constructor();
  /**
  * Creates a fresh empty context
  */
  static create(): Context;
  /**
  * Opens a snapshot (implicit fork — snapshot stays immutable, working pages are copied)
  */
  static open(name: string): Context;
  /**
  * Takes ownership of a snapshot (snapshot is deleted, GPU pages transfer without copy)
  */
  static take(name: string): Context;
  /**
  * Deletes a saved snapshot by name
  */
  static 'delete'(name: string): void;
  /**
  * Forks into a new anonymous context (working pages are copied via GPU D2D or CPU H2D)
  */
  fork(): Context;
  /**
  * Named save — snapshots committed chain + working pages under a user-chosen name
  */
  save(name: string): void;
  /**
  * Anonymous save — snapshots committed chain + working pages, returns runtime-generated name
  */
  snapshot(): string;
  /**
  * Force-destroys immediately
  */
  destroy(): void;
  /**
  * Number of tokens per page
  */
  tokensPerPage(): number;
  /**
  * Number of committed KV pages in the context
  */
  committedPageCount(): number;
  /**
  * Number of working KV pages (based on buffered tokens in the resource handle)
  */
  workingPageCount(): number;
  /**
  * Commit working KV pages to the context
  */
  commitWorkingPages(numPages: number): void;
  /**
  * Reserve additional working GPU pages for the context
  */
  reserveWorkingPages(numPages: number): void;
  /**
  * Release working GPU pages from the context
  */
  releaseWorkingPages(numPages: number): void;
  /**
  * Number of tokens in working pages (filled via forward pass but not yet committed)
  */
  workingPageTokenCount(): number;
  /**
  * Remove the last N tokens from working pages (for rollback, e.g. speculative rejection)
  */
  truncateWorkingPageTokens(numTokens: number): void;
  /**
  * Suspend this context (release GPU pages, offload to CPU).
  * Restoration is system-driven under FCFS: suspended contexts are
  * restored oldest-launched first as memory frees up.
  */
  suspend(): void;
}
