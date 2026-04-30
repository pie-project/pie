// Optional helpers for tool calling.
//
// `inferlet` does not bake a tool-call loop into the `Generator` surface —
// the right loop shape varies a lot between agents (ReAct, CodeAct,
// JSON-call, native-grammar) and we'd rather give you the pieces than a
// framework.
//
// This module exposes the host's tool-template capability so callers that
// *do* want the model's native format can reach for it explicitly:
//
// * `equipPrefix(model, tools)` — token sequence that registers tool
//   schemas in the chat template (model-specific). Append before your
//   user message via `ctx.append(...)`.
// * `answerPrefix(model, name, value)` — token sequence that frames a
//   tool result for the next turn.
// * `nativeMatcher(model, tools)` — a grammar matcher that constrains
//   output to well-formed tool calls (or `undefined` if the model has
//   no enforceable format). Wrap with `GrammarConstraint` and pass to
//   `Generator.constrain(...)` to enforce well-formed output.
// * `Decoder` — streaming detector for tool calls inside generated
//   text. Feed each step's tokens; collect `Event.Call` events.
//
// For agents that hand-roll their own format (e.g. `agent-react`'s
// `Action: ToolName[input]` parsing), none of these are required.

import * as _tool from 'pie:instruct/tool-use';

import { Matcher } from './grammar.js';
import type { Model } from './model.js';

// =============================================================================
// Templating
// =============================================================================

/** Token sequence that registers `tools` (each a JSON Schema string or
 *  object) in the chat template. */
export function equipPrefix(
  model: Model,
  tools: Array<string | object>,
): Uint32Array {
  const strs = tools.map(t => typeof t === 'string' ? t : JSON.stringify(t));
  return _tool.equip(model._handle, strs);
}

/** Token sequence that frames a tool result for the next turn. `name`
 *  matches the call the model made; `value` is typically a
 *  JSON-serializable result (object/array auto-stringified). */
export function answerPrefix(
  model: Model,
  name: string,
  value: string | object,
): Uint32Array {
  const s = typeof value === 'string' ? value : JSON.stringify(value);
  return _tool.answer(model._handle, name, s);
}

// =============================================================================
// Native matcher (for grammar enforcement)
// =============================================================================

/** Build a grammar `Matcher` that enforces the model's native tool-call
 *  format. Returns `undefined` if the model has no enforceable format —
 *  caller should fall through to free-form generation + their own parser.
 *
 *  Pair with `GrammarConstraint` to pass to `Generator.constrain()`:
 *
 *      const matcher = tools.nativeMatcher(model, schemas);
 *      if (matcher) {
 *        gen = gen.constrain(new GrammarConstraint(matcher));
 *      }
 */
export function nativeMatcher(
  model: Model,
  tools: Array<string | object>,
): Matcher | undefined {
  const strs = tools.map(t => typeof t === 'string' ? t : JSON.stringify(t));
  // The host's `format()` returns Grammar | undefined; pass to
  // create_matcher only if non-undefined.
  const grammar = _tool.format(model._handle, strs);
  if (grammar === undefined) return undefined;
  // create_matcher always succeeds when the model has a format.
  const raw = _tool.createMatcher(model._handle, strs);
  return Matcher._fromHandle(raw);
}

// =============================================================================
// Events
// =============================================================================

/** Discriminated union of tool-decoder events. Per `feed()`, exactly one
 *  event fires.
 *
 *  `Start` fires while a tool-call structure is being assembled but the
 *  arguments haven't closed yet — it's both "boundary entered" and the
 *  no-meaningful-event signal during accumulation. Most callers can
 *  ignore it and only act on `Call`. */
export type Event = EventStart | EventCall;

/** A tool call is in progress — keep feeding. */
export interface EventStart {
  readonly type: 'start';
}

/** Complete tool call. `args` is JSON-encoded. */
export interface EventCall {
  readonly type: 'call';
  readonly name: string;
  readonly args: string;
}

// ─── Constructors ─────────────────────────────────────────────────────────

export const Event = {
  Start: ():                                 EventStart => ({ type: 'start' }),
  Call:  (name: string, args: string):       EventCall  => ({ type: 'call', name, args }),
} as const;

// =============================================================================
// Decoder
// =============================================================================

/** Stateful tool-call decoder. Feed each step's tokens; collect
 *  `Event.Call` events when complete tool calls are detected. */
export class Decoder {
  readonly #inner: _tool.Decoder;

  constructor(model: Model) {
    this.#inner = _tool.createDecoder(model._handle);
  }

  /** Feed a token batch and get back the event that fired. `Event.Start`
   *  indicates an in-progress tool call; `Event.Call` fires once when the
   *  arguments close. */
  feed(tokens: Uint32Array): Event {
    const ev = this.#inner.feed(tokens);
    switch (ev.tag) {
      case 'start': return Event.Start();
      case 'call':  return Event.Call(ev.val[0], ev.val[1]);
    }
  }

  /** Reset to initial state. */
  reset(): void {
    this.#inner.reset();
  }
}
