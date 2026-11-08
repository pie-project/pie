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
  /**
  * The host's advice to a program that trades compute for latency
  * (speculation): whether no other pipeline is active beside this
  * one, so its fires batch alone. A pipeline is active from its
  * making, and again from each submit, until it parks or closes.
  * Advice, not a count; it may change from one frame to the next.
  */
  speculationAdvised(): boolean;
}
