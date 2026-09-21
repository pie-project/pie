// Tool calling — `pie:inferlet/tools`.
//
// Equip tool specs into the prompt, constrain generation to well-formed
// calls, and parse the calls back out.

import * as _tools from 'pie:inferlet/tools@0.3.0';

import { Grammar, GrammarError, Matcher } from './grammar.js';

export interface ToolCall {
  readonly name: string;
  readonly argumentsJson: string;
}

export interface EventStart {
  readonly type: 'start';
}

export interface EventCall {
  readonly type: 'call';
  readonly call: ToolCall;
}

export type Event = EventStart | EventCall;

/** Token sequence declaring `tools` (JSON specs) to the model. */
export function equip(tools: string[]): Uint32Array {
  try {
    return _tools.equip(tools);
  } catch (e: unknown) {
    const payload = (e as { payload?: unknown })?.payload;
    throw new GrammarError(`tools.equip: ${typeof payload === 'string' ? payload : String(e)}`);
  }
}

/** Token sequence returning a tool's result to the model. */
export function answer(name: string, value: string): Uint32Array {
  return _tools.answer(name, value);
}

/** The grammar of a well-formed call to one of `tools`, if the template has one. */
export function format(tools: string[]): Grammar | undefined {
  const g = _tools.format(tools);
  return g ? new Grammar(g) : undefined;
}

/** A matcher over `format`'s grammar. */
export function createMatcher(tools: string[]): Matcher {
  return new Matcher(_tools.createMatcher(tools));
}

/** Parses generated tokens into tool-call events. */
export class Decoder {
  readonly #inner = new _tools.Decoder();

  feed(tokens: ArrayLike<number>): Event {
    let ev: _tools.Event;
    try {
      ev = this.#inner.feed(Uint32Array.from(tokens));
    } catch (e: unknown) {
      const payload = (e as { payload?: unknown })?.payload;
      throw new GrammarError(`tools decoder: ${typeof payload === 'string' ? payload : String(e)}`);
    }
    if (ev.tag === 'call') return { type: 'call', call: { name: ev.val.name, argumentsJson: ev.val.argumentsJson } };
    return { type: 'start' };
  }

  reset(): void {
    this.#inner.reset();
  }
}
