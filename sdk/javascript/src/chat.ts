// Chat-template templating + parsing.
//
// Two halves:
//
// 1. **Fillers** (`system`, `user`, `assistant`, `cue`, `seal`,
//    `stopTokens`) produce token sequences for the model's chat template.
//    The `Context` calls them through its `system` / `user` / `cue` /
//    `seal` methods; for inferlets that build prompts manually (no
//    Context buffering), these are the public entry points.
//
// 2. **Decoder** (`Decoder`, `Event`) parses the model's generated tokens
//    back into visible text + structural events.
//
// Both halves wrap the host's `pie:instruct/chat` interface — chat
// template knowledge lives in the Pie runtime, not in the SDK.

import * as _chat from 'pie:instruct/chat';

import type { Model } from './model.js';

// =============================================================================
// Template fillers
// =============================================================================

/** Token sequence for a system-role message. */
export function system(model: Model, message: string): Uint32Array {
  return _chat.system(model._handle, message);
}

/** Token sequence for a user-role message. */
export function user(model: Model, message: string): Uint32Array {
  return _chat.user(model._handle, message);
}

/** Token sequence for an assistant-role message (history replay). */
export function assistant(model: Model, message: string): Uint32Array {
  return _chat.assistant(model._handle, message);
}

/** Token sequence for the generation cue (tells the model "your turn"). */
export function cue(model: Model): Uint32Array {
  return _chat.cue(model._handle);
}

/** Token sequence that seals the current turn (inserts a stop token). */
export function seal(model: Model): Uint32Array {
  return _chat.seal(model._handle);
}

/** Stop-token IDs for `model`'s chat template — pass to
 *  `Generator.stop()` for explicit termination control. */
export function stopTokens(model: Model): Uint32Array {
  return _chat.stopTokens(model._handle);
}

// =============================================================================
// Events
// =============================================================================
//
// Per `feed()`, exactly one event fires. `Event.Idle` is the no-op
// signal — the batch was consumed but didn't cross a semantic boundary
// worth surfacing (e.g. landed on a token whose visible text is empty,
// or inside a region this decoder doesn't report on like a reasoning
// block).

/** Discriminated union of chat-decoder events. Match on `event.type`. */
export type Event =
  | EventIdle
  | EventDelta
  | EventDone
  | EventInterrupt;

/** No semantic boundary crossed in this batch. */
export interface EventIdle {
  readonly type: 'idle';
}

/** Streamed text chunk (post-detokenization, post-template-strip).
 *  Always non-empty. */
export interface EventDelta {
  readonly type: 'delta';
  readonly text: string;
}

/** End-of-turn reached — `text` is the full accumulated text since the
 *  last `reset()`. */
export interface EventDone {
  readonly type: 'done';
  readonly text: string;
}

/** The model emitted a special / control token that the chat template
 *  recognized but didn't lower to visible text. The id is surfaced raw
 *  so the caller can decide what to do.
 *
 *  Common cases this fires for:
 *  - Tool-call boundary markers (e.g. `<|tool_call|>` in some templates)
 *    — useful as an early-stop hint when you don't have `tools.Decoder`
 *    attached.
 *  - Custom control tokens injected by fine-tuned models.
 *  - Format markers (turn boundaries, role separators) the host template
 *    chose to expose rather than swallow.
 *
 *  Most callers ignore this branch. */
export interface EventInterrupt {
  readonly type: 'interrupt';
  readonly token: number;
}

// ─── Constructors ─────────────────────────────────────────────────────────

export const Event = {
  Idle:      ():                       EventIdle      => ({ type: 'idle' }),
  Delta:     (text: string):           EventDelta     => ({ type: 'delta', text }),
  Done:      (text: string):           EventDone      => ({ type: 'done', text }),
  Interrupt: (token: number):          EventInterrupt => ({ type: 'interrupt', token }),
} as const;

// =============================================================================
// Decoder
// =============================================================================

/** Stateful chat decoder. Feed token batches in order; one event per call.
 *  `reset()` returns the decoder to its initial state. */
export class Decoder {
  readonly #inner: _chat.Decoder;

  constructor(model: Model) {
    this.#inner = _chat.createDecoder(model._handle);
  }

  /** Feed a token batch and get back the event that fired (one per call).
   *  Returns `Event.Idle()` when nothing semantically happened — e.g. the
   *  batch landed on a token whose visible text is empty, or inside a
   *  region this decoder doesn't report on. */
  feed(tokens: Uint32Array): Event {
    const ev = this.#inner.feed(tokens);
    switch (ev.tag) {
      case 'delta':
        // Empty delta means tokens consumed produced no visible character —
        // surface as Idle, not Delta(''), so the user doesn't need an
        // `if (text)` guard.
        return ev.val ? Event.Delta(ev.val) : Event.Idle();
      case 'done':
        return Event.Done(ev.val);
      case 'interrupt':
        return Event.Interrupt(ev.val);
    }
  }

  /** Reset to initial state. */
  reset(): void {
    this.#inner.reset();
  }
}
