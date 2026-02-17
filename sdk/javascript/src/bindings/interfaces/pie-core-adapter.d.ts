/** @module Interface pie:core/adapter **/
export type Pollable = import('./wasi-io-poll.js').Pollable;
export type Error = import('./pie-core-types.js').Error;
export type FutureBool = import('./pie-core-types.js').FutureBool;
export type Model = import('./pie-core-model.js').Model;

export class Adapter {
  /**
   * This type does not have a public constructor.
   */
  private constructor();
  static create(model: Model, name: string): Adapter;
  destroy(): void;
  static open(model: Model, name: string): Adapter | undefined;
  fork(name: string): Adapter;
  acquireLock(): FutureBool;
  releaseLock(): void;
  load(path: string): void;
  save(path: string): void;
}
