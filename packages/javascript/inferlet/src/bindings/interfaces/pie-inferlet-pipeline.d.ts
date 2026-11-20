/** @module Interface pie:inferlet/pipeline@0.3.0 **/
export type Error = import('./pie-inferlet-types.js').Error;

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
  * Take back what the last `fires` fires submitted on this closed
  * pipeline wrote: a program that ran ahead of its stop (EOS, a stop
  * string) hands back the rows past it. Returns once every fire
  * submitted on it has landed; each working set those fires wrote
  * then ends where the first of them began writing it, and its pages
  * past that are released. Their outputs stay take-able. Refused on
  * an open pipeline, or past the fires submitted. A set whose rows
  * cannot be taken back (a recurrent state folded them, the device
  * placed them, a fold sealed the page they end in) fails the
  * pipeline by that name: the set is refused wherever it is used
  * next.
  */
  retract(fires: number): void;
  /**
  * The host's advice to a program that trades compute for latency
  * (speculation): whether no other pipeline is active beside this
  * one, so its fires batch alone. A pipeline is active from its
  * making, and again from each submit, until it parks or closes.
  * Advice, not a count; it may change from one frame to the next.
  */
  speculationAdvised(): boolean;
}
