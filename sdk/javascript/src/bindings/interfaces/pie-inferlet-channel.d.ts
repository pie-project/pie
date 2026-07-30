/** @module Interface pie:inferlet/channel@0.3.0 **/
export type Error = import('./pie-inferlet-types.js').Error;
export type Shape = import('./pie-inferlet-types.js').Shape;
export type Dtype = import('./pie-inferlet-types.js').Dtype;
export type Data = import('./pie-inferlet-types.js').Data;

export class Channel {
  constructor(shape: Shape, dtype: Dtype, capacity: number)
  /**
  * Hand a value to the device. On a `seeded` channel before the first
  * fire this is the seed; on a host-writer channel it is the next
  * staged cell (empty -> fill; full -> back-pressure). `bool` payloads
  * are dtype-native (1 byte per bool); the wire packs to bits.
  */
  put(value: Data): void;
  /**
  * Atomically replace the committed front cell without changing queue
  * occupancy. This is a fused take+put: it never exposes an empty slot,
  * never enqueues, and leaves any later put queued for the next fire
  * untouched. Errors when the channel is empty, poisoned/closed, or
  * its front is currently claimed by an in-flight fire.
  */
  set(value: Data): void;
  /**
  * Move a committed value out to the host (host-reader channels). Full
  * -> value + empty the cell. Empty -> BLOCKS by awaiting the pass's
  * in-flight fires (submit order) until the cell fills; errors when no
  * in-flight fire remains (nothing will ever fill it) or the channel
  * is poisoned — a fire that feeds it failed, and the error carries
  * that fire's failure (under run-ahead, poison IS the error channel).
  */
  take(): Promise<Data>;
  /**
  * Copy a committed value to the host, leaving the cell full. Same
  * await/poison discipline as `take`.
  */
  read(): Promise<Data>;
}
