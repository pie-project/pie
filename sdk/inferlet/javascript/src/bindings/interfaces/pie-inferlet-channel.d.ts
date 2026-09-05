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
  * `take` for a guest whose toolchain cannot lower an `async func`
  * (componentize-js / StarlingMonkey today: "imported functions can
  * only be synchronous pending component-model-level async
  * support"). Same contract and same host path as `take`, but the
  * guest's task BLOCKS in the call instead of yielding — nothing else
  * in that instance runs while it waits, which is exactly what a
  * single-pipeline decode loop wants and what a guest juggling two
  * pipelines from one task must not use.
  */
  takeBlocking(): Data;
  /**
  * `read` for the same guests; see `take-blocking`.
  */
  readBlocking(): Data;
}
