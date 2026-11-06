// Grammar compilation + incremental matching — `pie:inferlet/grammar`.
//
// `Grammar` compiles a JSON Schema / regex / EBNF source once for the bound
// model's vocabulary; `Matcher` walks one generation against it, exposing
// the packed allowed-token bitmask that `mask.ts` interprets.

import * as _grammar from 'pie:inferlet/grammar@0.3.0';
import * as _session from 'pie:inferlet/session@0.3.0';

/** A grammar the host refused to compile, or a token the matcher refused. */
export class GrammarError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'GrammarError';
  }
}

function wit<T>(what: string, f: () => T): T {
  try {
    return f();
  } catch (e: unknown) {
    const payload = (e as { payload?: unknown })?.payload;
    throw new GrammarError(`${what}: ${typeof payload === 'string' ? payload : String(e)}`);
  }
}

/** A compiled constraint grammar. */
export class Grammar {
  /** @internal */
  constructor(readonly inner: _grammar.Grammar) {}

  /** A grammar for JSON values conforming to `schema`. */
  static fromJsonSchema(schema: string): Grammar {
    return new Grammar(wit('grammar from JSON schema', () => _grammar.Grammar.fromJsonSchema(schema)));
  }

  /** The grammar of any JSON value. */
  static json(): Grammar {
    return new Grammar(_grammar.Grammar.json());
  }

  static fromRegex(pattern: string): Grammar {
    return new Grammar(wit('grammar from regex', () => _grammar.Grammar.fromRegex(pattern)));
  }

  static fromEbnf(ebnf: string): Grammar {
    return new Grammar(wit('grammar from EBNF', () => _grammar.Grammar.fromEbnf(ebnf)));
  }

  toString(): string {
    return this.inner.toString();
  }
}

/** An incremental match of one generation against a `Grammar`. */
export class Matcher {
  private readonly inner: _grammar.Matcher;

  constructor(grammar: Grammar | _grammar.Matcher) {
    this.inner = grammar instanceof Grammar ? new _grammar.Matcher(grammar.inner) : grammar;
  }

  /** The handle the launch moved in under `name`, once. */
  static take(name: string): Matcher {
    const inner = _session.takeMatcher(name);
    if (inner === undefined) throw new GrammarError(`the launch moved in no matcher \`${name}\``);
    return new Matcher(inner);
  }

  /** Hands this matcher back to the client under the launch's `name`. Unusable after. */
  give(name: string): void {
    wit('give', () => _session.giveMatcher(name, this.inner));
  }

  /** Advance the match by `tokenIds`; a forbidden token throws `GrammarError`. */
  acceptTokens(tokenIds: ArrayLike<number>): void {
    wit('accept tokens', () => this.inner.acceptTokens(Uint32Array.from(tokenIds)));
  }

  /** The packed allowed-token bitmask for the next position (see `mask.ts`). */
  mask(): Uint32Array {
    return this.inner.mask();
  }

  isTerminated(): boolean {
    return this.inner.isTerminated();
  }

  reset(): void {
    this.inner.reset();
  }

  /** An independent copy at the current position (branching). */
  fork(): Matcher {
    return new Matcher(this.inner.fork());
  }

  rollback(numTokens: number): void {
    this.inner.rollback(numTokens);
  }

  rollbackCapacity(): number {
    return this.inner.rollbackCapacity();
  }
}
