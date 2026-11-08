// Tool calling — `pie:inferlet/tools`.
//
// Open a conversation with tool specs (`chat.open`), constrain generation to
// well-formed calls, parse the calls back out, and answer them.

import * as _tools from 'pie:inferlet/tools@0.3.0';

import { Grammar, GrammarError } from './grammar.js';

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

/** Tool results `[name, value]` that follow one another, as the template
 *  writes them together; one result is a list of one. */
export function answers(results: Array<[string, string]>): Uint32Array {
  return _tools.answers(results);
}

/**
 * An earlier assistant turn: `message`, then the `calls` it made in the
 * template's call syntax. `current` marks a turn after the conversation's
 * last user query, which some templates replay with its reasoning.
 */
export function assistantCalls(message: string, calls: ToolCall[], current: boolean): Uint32Array {
  return _tools.assistantCalls(message, [...calls], current);
}

/** The grammar of a well-formed call to one of `tools`, if the template has one. */
export function format(tools: string[]): Grammar | undefined {
  const g = _tools.format(tools);
  return g ? new Grammar(g) : undefined;
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
