// Grammar, Matcher, GrammarConstraint, and Schema for constrained decoding.
//
// The common path is declarative: pass a Schema to
// `ctx.generate({ ..., constrain })` (or any object implementing the
// `Schema` interface), and the SDK compiles it into a stateful matcher
// and drives it per generated token.
//
//     const text = await ctx.generate(Sampler.argmax(), {
//       maxTokens: 512,
//       constrain: ebnf(grammar),
//     }).collectText();
//
// Built-in implementors:
//
// * `jsonSchema(schemaStr)` — JSON conforming to a JSON Schema string
// * `anyJson()`             — any valid JSON
// * `regex(pattern)`        — strings matching a regex
// * `ebnf(source)`          — custom EBNF grammar
//
// User code can implement the `Schema` interface on any class with a
// `buildConstraint()` method — duck-typed, no inheritance required.
//
// For custom logic that isn't a grammar (banned tokens, learned
// constraints, etc.), implement the `Constraint` interface and pass it
// directly.

import {
  Grammar as _Grammar,
  Matcher as _Matcher,
} from 'pie:core/inference';
import type { Brle } from 'pie:core/inference';

// =============================================================================
// Grammar / Matcher (raw resource wrappers)
// =============================================================================

/** Describes the structure that LLM output must conform to. Wraps the
 *  `pie:core/inference.Grammar` WIT resource. */
export class Grammar {
  /** @internal */ readonly _handle: _Grammar;

  private constructor(handle: _Grammar) { this._handle = handle; }

  /** Construct from a JSON Schema string. */
  static fromJsonSchema(schema: string): Grammar {
    return new Grammar(_Grammar.fromJsonSchema(schema));
  }

  /** Construct a built-in free-form JSON grammar (any valid JSON). */
  static json(): Grammar {
    return new Grammar(_Grammar.json());
  }

  /** Construct from a regular expression pattern. */
  static fromRegex(pattern: string): Grammar {
    return new Grammar(_Grammar.fromRegex(pattern));
  }

  /** Construct from an EBNF grammar string. */
  static fromEbnf(ebnf: string): Grammar {
    return new Grammar(_Grammar.fromEbnf(ebnf));
  }
}

/** Stateful grammar matcher producing token masks. Most callers should
 *  reach for `GrammarConstraint` instead. */
export class Matcher {
  /** @internal */ _handle: _Matcher;

  /** Build from a compiled `Grammar`. */
  constructor(grammar: Grammar) {
    this._handle = new _Matcher(grammar._handle);
  }

  /** @internal Wrap a pre-existing host matcher (used by `tools.nativeMatcher`). */
  static _fromHandle(handle: _Matcher): Matcher {
    const m = Object.create(Matcher.prototype) as Matcher;
    m._handle = handle;
    return m;
  }

  acceptTokens(tokenIds: Uint32Array): void {
    this._handle.acceptTokens(tokenIds);
  }

  nextTokenLogitMask(): Brle {
    return this._handle.nextTokenLogitMask();
  }

  isTerminated(): boolean {
    return this._handle.isTerminated();
  }

  reset(): void {
    this._handle.reset();
  }
}

// =============================================================================
// Constraint interface + GrammarConstraint
// =============================================================================

/** Stateful sampling constraint.
 *
 *  On each generation step, the `Generator` passes any newly accepted
 *  tokens (or empty on the first step) and gets back the BRLE-encoded
 *  logit mask for the next position. Returning an empty mask means
 *  "no restriction" and is treated as transparent during composition. */
export interface Constraint {
  step(accepted: Uint32Array): Brle;
}

/** Grammar-driven `Constraint` backed by a host `Matcher`. Most callers
 *  should reach for a `Schema` implementor instead — `GrammarConstraint`
 *  is the lower-level type for callers that want to keep a constraint
 *  instance around (e.g., for use with `tools.nativeMatcher`). */
export class GrammarConstraint implements Constraint {
  readonly #matcher: Matcher;

  constructor(matcher: Matcher) {
    this.#matcher = matcher;
  }

  /** Build from a pre-compiled grammar (compile once, reuse). */
  static fromGrammar(grammar: Grammar): GrammarConstraint {
    return new GrammarConstraint(new Matcher(grammar));
  }

  /** Build from a JSON Schema string. */
  static fromJsonSchema(schema: string): GrammarConstraint {
    return GrammarConstraint.fromGrammar(Grammar.fromJsonSchema(schema));
  }

  /** Build a constraint that accepts any valid JSON. */
  static json(): GrammarConstraint {
    return GrammarConstraint.fromGrammar(Grammar.json());
  }

  /** Build from a regular expression pattern. */
  static fromRegex(pattern: string): GrammarConstraint {
    return GrammarConstraint.fromGrammar(Grammar.fromRegex(pattern));
  }

  /** Build from an EBNF grammar string. */
  static fromEbnf(ebnf: string): GrammarConstraint {
    return GrammarConstraint.fromGrammar(Grammar.fromEbnf(ebnf));
  }

  step(accepted: Uint32Array): Brle {
    if (accepted.length > 0) this.#matcher.acceptTokens(accepted);
    return this.#matcher.nextTokenLogitMask();
  }
}

/** @internal Wraps a static BRLE mask as a `Constraint` (returned every step). */
export class StaticMaskConstraint implements Constraint {
  readonly #mask: Brle;
  constructor(mask: Brle) { this.#mask = mask; }
  step(_: Uint32Array): Brle { return this.#mask; }
}

// =============================================================================
// Schema interface + built-in implementors
// =============================================================================

/** Declarative description of a constraint.
 *
 *  Implementations are passed to `ctx.generate({ constrain: ... })` (or
 *  `Generator.constrain(...)`) and compiled into a `GrammarConstraint`.
 *
 *  User code can implement this interface on any object — duck-typed, no
 *  inheritance required:
 *
 *      class MyLark {
 *        constructor(public readonly source: string) {}
 *        buildConstraint(): GrammarConstraint {
 *          const g = compileLarkToPieGrammar(this.source);
 *          return GrammarConstraint.fromGrammar(g);
 *        }
 *      }
 *
 *      ctx.generate(Sampler.argmax(), { constrain: new MyLark(grammar) });
 */
export interface Schema {
  buildConstraint(): GrammarConstraint;
}

/** JSON conforming to a JSON Schema string. */
export function jsonSchema(schema: string): Schema {
  return {
    buildConstraint: () => GrammarConstraint.fromJsonSchema(schema),
  };
}

/** Any valid JSON value. */
export function anyJson(): Schema {
  return {
    buildConstraint: () => GrammarConstraint.json(),
  };
}

/** Strings matching a regular expression pattern. */
export function regex(pattern: string): Schema {
  return {
    buildConstraint: () => GrammarConstraint.fromRegex(pattern),
  };
}

/** A custom EBNF grammar. */
export function ebnf(source: string): Schema {
  return {
    buildConstraint: () => GrammarConstraint.fromEbnf(source),
  };
}

/** Wrap a pre-compiled `Grammar` as a `Schema` (compile once, reuse). */
export function grammar(g: Grammar): Schema {
  return {
    buildConstraint: () => GrammarConstraint.fromGrammar(g),
  };
}

// =============================================================================
// BRLE intersection (for composing constraint masks)
// =============================================================================

/** @internal AND two BRLE-encoded masks of equal length. */
export function _brleAnd(a: Brle, b: Brle): Brle {
  if (a.length === 0 || b.length === 0) return new Uint32Array();
  const out: number[] = [];
  let aIdx = 0, bIdx = 0;
  let aLeft = a[0]!, bLeft = b[0]!;
  let aValue = false, bValue = false;
  let wantValue = false;
  let accum = 0;
  while (true) {
    const take = Math.min(aLeft, bLeft);
    const result = aValue && bValue;
    if (result === wantValue) {
      accum += take;
    } else {
      out.push(accum);
      accum = take;
      wantValue = !wantValue;
    }
    aLeft -= take;
    bLeft -= take;
    if (aLeft === 0) {
      aIdx++;
      if (aIdx === a.length) break;
      aLeft = a[aIdx]!;
      aValue = !aValue;
    }
    if (bLeft === 0) {
      bIdx++;
      if (bIdx === b.length) break;
      bLeft = b[bIdx]!;
      bValue = !bValue;
    }
  }
  out.push(accum);
  return new Uint32Array(out);
}

/** @internal Reduce a list of BRLE masks via AND. */
export function _brleAndMany(masks: Brle[]): Brle {
  if (masks.length === 0) return new Uint32Array();
  if (masks.length === 1) return masks[0]!;
  let acc = masks[0]!;
  for (let i = 1; i < masks.length; i++) {
    acc = _brleAnd(acc, masks[i]!);
  }
  return acc;
}
