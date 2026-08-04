"""Which corpus schemas do the two engines lower to the *same* language.

Everything measured so far compares engines across schemas they lower
differently: 73.9% of ours are relaxations, XGrammar relaxes `oneOf` and
`type`, and the divergence probe found that where two documents part it is
almost always because our mask admitted a token theirs forbids. That confounds
every other comparison - a throughput number over schemas whose grammars
differ is partly a measurement of the grammars.

So this finds the subset where they do not differ. For each schema it walks
random paths through the *intersection* of the two masks, comparing them token
by token, and keeps only the schemas where they never disagree. The result is
a set on which any remaining difference cannot be semantic.

    python -m engrain_lab.rigor.agreement --schemas 200 --walks 8

Writes `results/agreement.json`, a list of schema indices, which
`rigor.e2e --agreed-only` then runs both engines on.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

RESULTS = Path("results")
OUT = RESULTS / "agreement.json"


# The keywords a schema must avoid for both engines to lower it as written:
# an open object shadows for us, `oneOf` and `anyOf` merge or lose siblings,
# `uniqueItems` is not context-free, and a required set past the subset budget
# stops being counted. What is left is closed objects over scalars.
_TYPES = [
    {"type": "string"},
    {"type": "integer"},
    {"type": "boolean"},
    {"type": "number"},
    {"type": "string", "maxLength": 12},
    {"type": "array", "items": {"type": "string"}, "maxItems": 3},
]


def _synthetic(count: int, seed: int) -> list[dict]:
    """Schemas both engines should lower to the same language."""
    rng = np.random.default_rng(seed)
    words = [
        "name", "size", "colour", "count", "active", "label", "kind", "score",
        "owner", "tags", "note", "level", "state", "code", "title", "path",
    ]
    schemas = []
    for index in range(count):
        width = int(rng.integers(1, 5))
        names = list(rng.choice(words, size=width, replace=False))
        properties = {
            name: dict(_TYPES[int(rng.integers(len(_TYPES)))]) for name in names
        }
        needed = names[: int(rng.integers(1, width + 1))]
        schema = {
            "type": "object",
            "properties": properties,
            "required": sorted(needed),
            "additionalProperties": False,
        }
        if index % 4 == 3:
            # One level of nesting, closed the same way.
            schema = {
                "type": "object",
                "properties": {"inner": schema, "id": {"type": "integer"}},
                "required": ["inner"],
                "additionalProperties": False,
            }
        schemas.append(schema)
    return schemas


def _allowed(mask: np.ndarray, token: int) -> bool:
    return bool(int(mask[token // 32]) & (1 << (token % 32)))


def main() -> int:
    import torch
    import xgrammar as xg
    from transformers import AutoTokenizer

    from engrain.internals import Compiler

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--schemas", type=int, default=200)
    parser.add_argument("--walks", type=int, default=8)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument(
        "--synthetic",
        type=int,
        default=0,
        help="instead of the corpus, generate this many schemas that both "
        "engines should lower exactly: closed objects, scalar and bounded "
        "types, no oneOf or anyOf, required inside the subset budget. The "
        "corpus has no such schema - all 200 of its first entries differ - so "
        "this is the only way to ask what the engines do when the grammars "
        "agree.",
    )
    parser.add_argument("--max-digits", type=int, default=None)
    parser.add_argument(
        "--no-whitespace",
        action="store_true",
        help="forbid whitespace on both sides, which is the only way to ask "
        "the question at all. The two engines disagree about whitespace in "
        "both directions - we allow a carriage return and RFC 8259 says we are "
        "right, and they allow whitespace after the document where we "
        "deliberately do not - so with it on, every schema differs at the "
        "first token and nothing else can be compared.",
    )
    parser.add_argument(
        "--with-cr",
        action="store_true",
        help="count a carriage return as a disagreement. It is one, and it is "
        "XGrammar's: RFC 8259 says JSON whitespace is space, tab, line feed "
        "and carriage return, and their `any_whitespace` omits the last. "
        "Left in, every schema disagrees at the first token and nothing else "
        "can be seen, so by default the tokens that carry one are set aside "
        "and the count below says how many.",
    )
    arguments = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(arguments.model)
    vocabulary = []
    for token_id in range(len(tokenizer)):
        piece = tokenizer.convert_ids_to_tokens(token_id)
        try:
            vocabulary.append(tokenizer.convert_tokens_to_string([piece]).encode())
        except Exception:  # noqa: BLE001
            vocabulary.append(b"")
    carries_cr = np.array([b"\r" in piece for piece in vocabulary])
    ours = Compiler(vocabulary)
    info = xg.TokenizerInfo.from_huggingface(tokenizer, vocab_size=len(tokenizer))
    xgc = xg.GrammarCompiler(info)

    if arguments.synthetic:
        instances = [
            {"schema": json.dumps(schema)}
            for schema in _synthetic(arguments.synthetic, arguments.seed)
        ]
    else:
        instances = json.loads(
            (RESULTS / "jsonschemabench-instances.json").read_text()
        )["instances"][: arguments.schemas]

    rng = np.random.default_rng(arguments.seed)
    disagreements: collections.Counter[str] = collections.Counter()
    theirs_only: collections.Counter[str] = collections.Counter()
    agreed: list[int] = []
    verdicts = {
        "agree": 0,
        "we admit more": 0,
        "they admit more": 0,
        "each admits something the other does not": 0,
        "one refused the schema": 0,
        "relaxed, so not compared": 0,
    }
    for index, instance in enumerate(instances):
        source = instance["schema"]
        try:
            grammar = ours.compile_json_schema(
                source,
                max_digits=arguments.max_digits,
                max_whitespace=0 if arguments.no_whitespace else None,
            )
        except Exception:  # noqa: BLE001
            verdicts["one refused the schema"] += 1
            continue
        # A declared relaxation is a difference by construction, so there is
        # nothing to learn from walking it.
        if grammar.relaxations:
            verdicts["relaxed, so not compared"] += 1
            continue
        try:
            compiled = xgc.compile_json_schema(
                source,
                any_whitespace=not arguments.no_whitespace,
                any_order=True,
            )
        except Exception:  # noqa: BLE001
            verdicts["one refused the schema"] += 1
            continue

        wider = narrower = False
        for _ in range(arguments.walks):
            matcher = grammar.matcher(0)
            theirs = xg.GrammarMatcher(compiled, terminate_without_stop_token=True)
            mine = torch.zeros(grammar.bitset_words, dtype=torch.int32)
            yours = xg.allocate_token_bitmask(1, len(tokenizer))
            for _ in range(arguments.steps):
                mine.zero_()
                matcher.fill_bitmask(mine)
                theirs.fill_next_token_bitmask(yours, 0)
                ours_bits = mine.numpy()
                their_bits = np.asarray(yours).reshape(-1)
                width = min(ours_bits.size, their_bits.size)
                mine_set = np.unpackbits(
                    ours_bits[:width].view(np.uint8), bitorder="little"
                )[: len(tokenizer)].astype(bool)
                yours_set = np.unpackbits(
                    their_bits[:width].view(np.uint8), bitorder="little"
                )[: len(tokenizer)].astype(bool)
                if not arguments.with_cr:
                    mine_set &= ~carries_cr
                    yours_set &= ~carries_cr
                wider |= bool((mine_set & ~yours_set).any())
                narrower |= bool((yours_set & ~mine_set).any())
                if wider or narrower:
                    for token in np.flatnonzero(mine_set & ~yours_set)[:2]:
                        disagreements[repr(tokenizer.decode([int(token)]))] += 1
                    for token in np.flatnonzero(yours_set & ~mine_set)[:2]:
                        theirs_only[repr(tokenizer.decode([int(token)]))] += 1
                    break
                # Both agree, so walk on either.
                choices = np.flatnonzero(mine_set)
                if choices.size == 0:
                    break
                token = int(choices[rng.integers(choices.size)])
                if not matcher.accept_token(token) or not theirs.accept_token(token):
                    break
            if wider or narrower:
                break

        if wider and narrower:
            verdicts["each admits something the other does not"] += 1
        elif wider:
            verdicts["we admit more"] += 1
        elif narrower:
            verdicts["they admit more"] += 1
        else:
            verdicts["agree"] += 1
            agreed.append(index)

    print(f"{len(instances)} schemas walked {arguments.walks} times each")
    for reason, count in verdicts.items():
        print(f"  {count:4d}  {reason}")
    RESULTS.mkdir(exist_ok=True)
    OUT.write_text(
        json.dumps(
            {"walks": arguments.walks, "steps": arguments.steps, "indices": agreed},
            indent=2,
        )
    )
    if disagreements:
        print("  what we admit and they do not:")
        for piece, count in disagreements.most_common(6):
            print(f"    {count:4d}  {piece}")
    if theirs_only:
        print("  what they admit and we do not:")
        for piece, count in theirs_only.most_common(6):
            print(f"    {count:4d}  {piece}")
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
