// Reasoning / thinking-block decoder.
//
// Wraps the host's `pie:instruct/reasoning.decoder`. Emits `Event.Start`
// when the model enters a thinking block, `Event.Delta` for each chunk
// of reasoning text, and `Event.End` when the block closes (with the
// full accumulated reasoning text).
//
// Compose with `chat.Decoder` by feeding the same token batch to both —
// the reasoning decoder's events are independent of chat's (no implicit
// suppression). The chat decoder handles its own filtering so visible
// text and reasoning text don't overlap.

import * as _reasoning from 'pie:instruct/reasoning';

import type { Model } from './model.js';

// =============================================================================
// Events
// =============================================================================

/** Discriminated union of reasoning-decoder events. */
export type Event =
  | EventIdle
  | EventStart
  | EventDelta
  | EventEnd;

/** No reasoning boundary crossed in this batch. */
export interface EventIdle {
  readonly type: 'idle';
}

/** The model entered a reasoning block. No text yet. */
export interface EventStart {
  readonly type: 'start';
}

/** Streamed chunk of reasoning text (post-detokenization). Always
 *  non-empty. */
export interface EventDelta {
  readonly type: 'delta';
  readonly text: string;
}

/** The block closed — `text` is the full accumulated reasoning text
 *  from `Start` to `End`. */
export interface EventEnd {
  readonly type: 'end';
  readonly text: string;
}

// ─── Constructors ─────────────────────────────────────────────────────────

export const Event = {
  Idle:  ():                EventIdle  => ({ type: 'idle' }),
  Start: ():                EventStart => ({ type: 'start' }),
  Delta: (text: string):    EventDelta => ({ type: 'delta', text }),
  End:   (text: string):    EventEnd   => ({ type: 'end', text }),
} as const;

// =============================================================================
// Decoder
// =============================================================================

/** Stateful reasoning decoder. Feed token batches in order; one event
 *  per call. */
export class Decoder {
  readonly #inner: _reasoning.Decoder;

  constructor(model: Model) {
    this.#inner = _reasoning.createDecoder(model._handle);
  }

  /** Feed a token batch and get back the event that fired. Returns
   *  `Event.Idle()` when the batch landed outside any reasoning block,
   *  or inside one but on tokens that produced no visible reasoning
   *  text. */
  feed(tokens: Uint32Array): Event {
    const ev = this.#inner.feed(tokens);
    switch (ev.tag) {
      case 'start':    return Event.Start();
      case 'delta':    return ev.val ? Event.Delta(ev.val) : Event.Idle();
      case 'complete': return Event.End(ev.val);
    }
  }

  /** Reset to initial state. */
  reset(): void {
    this.#inner.reset();
  }
}
