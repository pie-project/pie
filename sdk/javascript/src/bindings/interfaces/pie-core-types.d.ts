/** @module Interface pie:core/types **/
export type Pollable = import('./wasi-io-poll.js').Pollable;
export type Error = string;
export type Blob = Uint8Array;

export class FutureBlob {
  /**
   * This type does not have a public constructor.
   */
  private constructor();
  pollable(): Pollable;
  get(): Blob | undefined;
}

export class FutureString {
  /**
   * This type does not have a public constructor.
   */
  private constructor();
  /**
  * Pollable to check readiness
  */
  pollable(): Pollable;
  /**
  * Retrieves the message if available; None if not ready
  */
  get(): string | undefined;
}
