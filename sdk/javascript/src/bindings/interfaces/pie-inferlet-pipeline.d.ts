/** @module Interface pie:inferlet/pipeline@0.3.0 **/

export class Pipeline {
  constructor()
  /**
  * End the stream and tear down its ordering domain. The scheduler
  * stops awaiting this pipeline immediately; every fire already
  * submitted (queued, preparing, or dispatched) still runs to
  * settlement in submission order and its outputs remain take-able.
  * Later submissions fail. Work never submitted is simply absent.
  * Resource drop has exactly these semantics.
  */
  close(): void;
}
