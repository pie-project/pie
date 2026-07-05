/** @module Interface pie:core/adapter **/
export type Error = import('./pie-core-types.js').Error;

export class Adapter {
  /**
   * This type does not have a public constructor.
   */
  private constructor();
  static create(name: string): Adapter;
  destroy(): void;
  static open(name: string): Adapter | undefined;
  fork(name: string): Adapter;
  load(path: string): void;
  save(path: string): void;
}
