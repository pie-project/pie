// Grammar and Matcher wrappers — wraps pie:core/inference WIT resources.

import {
    Grammar as _Grammar,
    Matcher as _Matcher,
} from 'pie:core/inference';
import type { Brle } from 'pie:core/inference';
import type { Tokenizer } from './model.js';

/**
 * Describes the structure that LLM output must conform to.
 *
 * Wraps the `pie:core/inference.Grammar` WIT resource.
 */
export class Grammar {
    /** @internal */
    readonly _handle: _Grammar;

    private constructor(handle: _Grammar) {
        this._handle = handle;
    }

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

/**
 * Stateful matcher that walks a grammar automaton as tokens are accepted.
 *
 * Wraps the `pie:core/inference.Matcher` WIT resource.
 */
export class Matcher {
    /** @internal */
    readonly _handle: _Matcher;

    constructor(grammar: Grammar, tokenizer: Tokenizer) {
        this._handle = new _Matcher(grammar._handle, tokenizer._handle);
    }

    /** Accept one or more decoded tokens, advancing the matcher state. */
    acceptTokens(tokenIds: Uint32Array): void {
        this._handle.acceptTokens(tokenIds);
    }

    /** Returns a BRLE bitmask of allowed next tokens at the current position. */
    nextTokenLogitMask(): Brle {
        return this._handle.nextTokenLogitMask();
    }

    /** Check whether the matcher has reached a terminal state. */
    isTerminated(): boolean {
        return this._handle.isTerminated();
    }

    /** Reset the matcher to its initial state for reuse. */
    reset(): void {
        this._handle.reset();
    }
}
