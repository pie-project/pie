"""Measure how far a real vocabulary collapses into terminal-sequence groups.

The zero-CPU design rests on one number. If `allowed(token)` factors as

    lexer_ok(lexer_state, token)  AND  parser_ok(stack_top, terminals(token))

then the runtime never needs a per-configuration token row: it needs, per lexer
state, the tokens grouped by the terminal sequence they emit. The parser check
is then one ACTION lookup per *group*, and the mask is the union of the
admitted groups' precomputed bitsets — all of which is compile-time data.

That only works if the group count is small. This scans a real vocabulary
through a JSON lexer and counts the groups. Hundreds is a win; tens of
thousands means the per-group lookup degenerates back to O(V).
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

STRUCTURAL = {
    ord("{"): "LBRACE",
    ord("}"): "RBRACE",
    ord("["): "LBRACKET",
    ord("]"): "RBRACKET",
    ord(","): "COMMA",
    ord(":"): "COLON",
}
WHITESPACE = set(b" \t\n\r")
DIGITS = set(b"0123456789")
NUMBER_BODY = DIGITS | set(b"+-.eE")

# Lexer states of a JSON scanner. `STRING` and `ESCAPE` are inside a string
# literal; `NUMBER` is mid-number; `VALUE` is between lexemes.
VALUE, STRING, ESCAPE, NUMBER, LITERAL = range(5)
STATE_NAMES = {VALUE: "value", STRING: "string", ESCAPE: "escape", NUMBER: "number", LITERAL: "literal"}


def scan(token: bytes, state: int) -> tuple[tuple[str, ...], int] | None:
    """Run one token through the lexer. Returns (terminals, next state)."""
    emitted: list[str] = []
    for byte in token:
        if state == ESCAPE:
            state = STRING
            continue
        if state == STRING:
            if byte == ord("\\"):
                state = ESCAPE
            elif byte == ord('"'):
                emitted.append("STRING_END")
                state = VALUE
            elif byte < 0x20:
                return None
            continue
        if state == NUMBER:
            if byte in NUMBER_BODY:
                continue
            emitted.append("NUMBER")
            state = VALUE
            # fall through and re-dispatch this byte in VALUE state
        if state == LITERAL:
            if 0x61 <= byte <= 0x7A:
                continue
            emitted.append("LITERAL")
            state = VALUE
        if state == VALUE:
            if byte in WHITESPACE:
                continue
            if byte in STRUCTURAL:
                emitted.append(STRUCTURAL[byte])
                continue
            if byte == ord('"'):
                emitted.append("STRING_BEGIN")
                state = STRING
                continue
            if byte in DIGITS or byte in b"-":
                state = NUMBER
                continue
            if 0x61 <= byte <= 0x7A:
                state = LITERAL
                continue
            return None
    return tuple(emitted), state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizers", nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("results/token-groups.json"))
    args = parser.parse_args()

    from transformers import AutoTokenizer

    report = {}
    for name in args.tokenizers:
        tokenizer = AutoTokenizer.from_pretrained(name)
        vocab_size = len(tokenizer)
        tokens: list[bytes] = []
        for token_id in range(vocab_size):
            piece = tokenizer.convert_ids_to_tokens(token_id)
            if piece is None:
                tokens.append(b"")
                continue
            try:
                tokens.append(
                    tokenizer.convert_tokens_to_string([piece]).encode("utf-8")
                )
            except Exception:  # noqa: BLE001
                tokens.append(b"")

        print(f"\n=== {name} (vocab {vocab_size}) ===")
        per_state = {}
        for state in (VALUE, STRING, NUMBER):
            groups: collections.Counter = collections.Counter()
            rejected = 0
            for token in tokens:
                if not token:
                    rejected += 1
                    continue
                result = scan(token, state)
                if result is None:
                    rejected += 1
                    continue
                groups[result] += 1
            sizes = sorted(groups.values(), reverse=True)
            covered = sum(sizes)
            top10 = sum(sizes[:10]) / covered * 100 if covered else 0.0
            per_state[STATE_NAMES[state]] = {
                "groups": len(groups),
                "accepted_tokens": covered,
                "rejected_tokens": rejected,
                "largest_group": sizes[0] if sizes else 0,
                "top10_coverage": top10,
            }
            print(
                f"  {STATE_NAMES[state]:8s} groups={len(groups):6d} "
                f"accepted={covered:7d} rejected={rejected:7d} "
                f"largest={sizes[0] if sizes else 0:7d} "
                f"top10 covers {top10:5.1f}%"
            )
        report[name] = {"vocab_size": vocab_size, "states": per_state}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
