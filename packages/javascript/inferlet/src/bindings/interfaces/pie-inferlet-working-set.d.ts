/** @module Interface pie:inferlet/working-set@0.3.0 **/
export type Error = import('./pie-inferlet-types.js').Error;
export type Pipeline = import('./pie-inferlet-pipeline.js').Pipeline;

export class WorkingSet {
  constructor()
  /**
  * Let passes write slots below `end`: needed only when a pass's slots
  * are computed on the device.
  */
  reserve(end: number): void;
  /**
  * The frozen length; while a device-counted fold is in flight, the
  * length before it.
  */
  folded(): number;
  /**
  * Slots mapped in from state the runtime already held: a prompt's
  * cached tokens.
  */
  cachedTokens(): number;
  /**
  * A child that shares the frozen part and gets a copy of the buffer,
  * ordered on `on`.
  */
  fork(on: Pipeline): WorkingSet;
}
