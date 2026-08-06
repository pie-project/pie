"""Regex and EBNF, where a referee exists and the schema front end does not.

Everything measured so far went through JSON Schema, which is one front end and
the one both engines have argued about hardest. A regex is different in a way
that matters: `re.fullmatch` is an adjudicator neither engine wrote, so
"admits too much" and "admits too little" are both decidable without appeal to
a lowering. An EBNF has no such referee, so it is judged by cases and by the
two engines against each other.

Three questions, in the order that makes the later ones worth asking:

**Does it compile?** A pattern one engine refuses is a pattern no comparison
can use, and refusals are a result in themselves.

**Is the mask right?** Walk it, and hand what comes out to `re.fullmatch`.
Then the mirror: take strings the pattern *does* match and check the mask
admits them, which is the half a generative test cannot see.

**What does it cost?** Compile, and the mask at batch.

    python -m engrain_lab.rigor.nonjson --walks 40
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any

from .harness import BYTE_VOCABULARY

# vLLM fills XGrammar's rows on a thread pool only above this many structured
# requests in a step, and serially below it.
_VLLM_THREADS_ABOVE = 128


def _fill_rows(matchers, bitmask, rows) -> None:
    for row in rows:
        matchers[row].fill_next_token_bitmask(bitmask, row)


# Patterns of the shape structured output actually asks for, plus the
# constructs that separate a regex engine from a toy: bounded repetition,
# alternation under repetition, negated classes, and nesting.
PATTERNS: list[tuple[str, str, list[str], list[str]]] = [
    ("date", r"\d{4}-\d{2}-\d{2}", ["2026-08-05"], ["2026-8-05", "20260805"]),
    (
        "datetime",
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",
        ["2026-08-05T13:45:01"],
        ["2026-08-05 13:45:01"],
    ),
    (
        "uuid",
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        ["3f2504e0-4f89-11d3-9a0c-0305e82c3301"],
        ["3f2504e0-4f89-11d3-9a0c-0305e82c33"],
    ),
    ("ipv4", r"(\d{1,3}\.){3}\d{1,3}", ["192.168.0.1"], ["192.168.0"]),
    ("semver", r"\d+\.\d+\.\d+(-[a-z]+)?", ["1.20.3", "1.0.0-beta"], ["1.20"]),
    ("hex colour", r"#[0-9a-fA-F]{6}", ["#a3F09b"], ["#a3F09"]),
    ("identifier", r"[A-Za-z_][A-Za-z0-9_]*", ["_a1", "Name"], ["1a"]),
    ("currency", r"-?\d+\.\d{2}", ["-12.50", "0.00"], ["12.5"]),
    ("percent", r"(100|[1-9]?[0-9])%", ["0%", "42%", "100%"], ["101%"]),
    ("phone", r"\+?[0-9]{1,3}-[0-9]{3}-[0-9]{4}", ["+82-010-1234"], ["82-01-1234"]),
    ("mac", r"([0-9A-F]{2}:){5}[0-9A-F]{2}", ["0A:1B:2C:3D:4E:5F"], ["0A:1B:2C:3D:4E"]),
    (
        "http line",
        r"(GET|POST|PUT|DELETE) /[a-z/]* HTTP/1\.[01]",
        ["GET /a/b HTTP/1.1", "DELETE / HTTP/1.0"],
        ["PATCH / HTTP/1.1"],
    ),
    ("csv field", r"[^,\n]+(,[^,\n]+)*", ["a,b,c", "one"], ["a,,b"]),
    ("bounded repeat", r"(ab|cd){2,4}", ["abab", "abcdab"], ["ab"]),
    ("nested group", r"((x|y)+z){1,2}", ["xyz", "xzyz"], ["z"]),
    ("optional chain", r"a?b?c?d", ["d", "abcd", "bd"], ["abc"]),
    ("class union", r"[a-z0-9_.-]+@[a-z0-9-]+\.[a-z]{2,4}", ["a.b-c@x-y.com"], ["a@b"]),
    ("digits bounded", r"[0-9]{3,6}", ["123", "123456"], ["12", "1234567"]),
    ("escaped meta", r"\$\d+\.\d{2}", ["$5.00"], ["5.00"]),
    ("anchored", r"^[a-z]+$", ["abc"], ["aBc"]),
]

# EBNF has no outside referee, so each grammar carries its own cases and the
# two engines are also compared with each other. Written in the `::=` dialect
# both accept, with `root` as the entry, which is what XGrammar requires.
GRAMMARS: list[tuple[str, str, list[str], list[str]]] = [
    (
        "list",
        'root ::= "[" items "]"\nitems ::= item (", " item)*\nitem ::= [a-z]+\n',
        ["[a]", "[ab, cd]"],
        ["[]", "[a,b]"],
    ),
    (
        "nested parens",
        'root ::= expr\nexpr ::= "(" expr ")" | "x"\n',
        ["x", "(x)", "((x))"],
        ["(x", "()"],
    ),
    (
        "arithmetic",
        'root ::= sum\nsum ::= product (("+" | "-") product)*\n'
        'product ::= atom (("*" | "/") atom)*\natom ::= [0-9]+ | "(" sum ")"\n',
        ["1", "1+2*3", "(1+2)*3"],
        ["1+", "*2"],
    ),
    (
        "key value",
        'root ::= pair (";" pair)*\npair ::= key "=" value\n'
        'key ::= [a-z]+\nvalue ::= [0-9]+ | "\\"" [a-z]* "\\""\n',
        ["a=1", 'a=1;b="xy"'],
        ["a=", "=1"],
    ),
    (
        "choice of shapes",
        'root ::= yes | no | count\nyes ::= "yes"\nno ::= "no"\ncount ::= [0-9]+\n',
        ["yes", "no", "42"],
        ["maybe", ""],
    ),
]


# The cases above are the ones a user writes, and both engines get every one of
# them right - which is the result, and also why they cannot tell the engines
# apart. These are the ones that separate a parser with a stack from a
# pushdown automaton with a budget: deep recursion, an ambiguity that needs
# more than one token of lookahead, an alternation wider than a table, and a
# repetition wider than a counter.
def _deep(depth: int) -> str:
    return (
        "root ::= e0\n"
        + "".join(f'e{i} ::= "(" e{i + 1} ")" | "x"\n' for i in range(depth))
        + f'e{depth} ::= "x"\n'
    )


HARD: list[tuple[str, str, list[str], list[str]]] = [
    (
        "recursion, self",
        'root ::= "(" root ")" | "x"',
        ["x", "(" * 19 + "x" + ")" * 19],
        ["(x"],
    ),
    ("recursion, 64 rules", _deep(64), ["x", "(x)", "((x))"], ["(x"]),
    (
        "ambiguous tail",
        'root ::= a | b\na ::= [a-z]+ "!"\nb ::= [a-z]+ "?"\n',
        ["abc!", "abc?"],
        ["abc"],
    ),
    (
        "common long prefix",
        'root ::= x | y\nx ::= "aaaaaaaaaaaaaaaaaaaa" "1"\n'
        'y ::= "aaaaaaaaaaaaaaaaaaaa" "2"\n',
        ["aaaaaaaaaaaaaaaaaaaa1", "aaaaaaaaaaaaaaaaaaaa2"],
        ["aaaaaaaaaaaaaaaaaaaa3"],
    ),
    (
        "alternation of 512",
        "root ::= " + " | ".join(f'"w{i:03d}"' for i in range(512)),
        ["w000", "w511"],
        ["w512"],
    ),
    (
        "left recursion",
        'root ::= list\nlist ::= list "," item | item\nitem ::= [a-z]\n',
        ["a", "a,b,c"],
        ["a,"],
    ),
    (
        "balanced pairs",
        'root ::= s\ns ::= "a" s "b" | ""\n',
        ["", "ab", "aaabbb"],
        ["aab", "ba"],
    ),
]

HARD_PATTERNS: list[tuple[str, str, list[str], list[str]]] = [
    ("repeat 1..64", r"[0-9]{1,64}", ["1", "1" * 64], ["1" * 65]),
    ("repeat 200", r"a{200}", ["a" * 200], ["a" * 199]),
    ("wide class", r"[\u0100-\u2000]+", ["\u0101\u1fff"], ["a"]),
    (
        "alternation of 256",
        "|".join(f"t{i:03d}" for i in range(256)),
        ["t000", "t255"],
        ["t256"],
    ),
    ("nested quantifier", r"((a|b){2,3}c){2,3}", ["aacaac", "abcabcabc"], ["ac"]),
    ("backtrack bait", r"(a+)+b", ["aab", "ab"], ["aa"]),
]

# UTF-8 well-formedness, which a byte-level mask has to enforce and which a
# walk finds only by luck. An overlong encoding is a valid *codepoint* spelled
# in more bytes than it needs, and RFC 3629 forbids it - it is the classic way
# past a filter that checks the decoded string.
UTF8: list[tuple[str, bytes, bool]] = [
    ("valid 2-byte U+0101", "\u0101".encode(), True),
    ("valid 3-byte U+1FFF", "\u1fff".encode(), True),
    ("truncated 3-byte", b"\xe1\x80", False),
    ("lone continuation", b"\x80", False),
    ("overlong U+0101", b"\xe0\x84\x81", False),
    ("3-byte with bad tail", b"\xe1\x80\x41", False),
    ("surrogate U+D800", b"\xed\xa0\x80", False),
    ("outside the class", "\u2001".encode(), False),
]
UTF8_PATTERN = r"[\u0100-\u2000]+"

MAX_WALK = 400
STOP_CHANCE = 0.2


def _walk(matcher, allowed_of, done, rng) -> tuple[str, bool]:
    out = bytearray()
    for _ in range(MAX_WALK):
        if done(matcher) and rng.random() < STOP_CHANCE:
            return out.decode("utf-8", "replace"), True
        allowed = allowed_of(matcher)
        if not allowed:
            return out.decode("utf-8", "replace"), done(matcher)
        chosen = rng.choice(allowed)
        matcher.accept_token(chosen)
        out.append(chosen)
    return out.decode("utf-8", "replace"), done(matcher)


def _engines():
    import torch
    import xgrammar as xg

    import engrain.internals as E

    ours = E.Compiler(BYTE_VOCABULARY)
    info = xg.TokenizerInfo(
        BYTE_VOCABULARY, xg.VocabType.RAW, vocab_size=256, stop_token_ids=[]
    )
    theirs = xg.GrammarCompiler(info, cache_enabled=False)
    bitmask = xg.allocate_token_bitmask(1, 256)
    bits = torch.arange(32, dtype=torch.int32)

    def their_allowed(matcher):
        matcher.fill_next_token_bitmask(bitmask, 0)
        words = bitmask[0].to(torch.int32)
        flags = ((words.unsqueeze(1) >> bits) & 1).to(torch.bool).reshape(-1)[:256]
        return flags.nonzero().flatten().tolist()

    return ours, theirs, their_allowed, xg


def _accepts(make, allowed_of, done, text: bytes) -> bool:
    matcher = make()
    for byte in text:
        if byte not in allowed_of(matcher):
            return False
        matcher.accept_token(byte)
    return done(matcher)


def measure(walks: int, seed: int) -> dict[str, Any]:
    ours, theirs, their_allowed, xg = _engines()
    report: dict[str, Any] = {"regex": [], "ebnf": [], "compile_ms": {}}
    compile_ms: dict[str, list[float]] = {"engrain": [], "xgrammar": []}

    for kind, cases in (
        ("regex", PATTERNS + HARD_PATTERNS),
        ("ebnf", GRAMMARS + HARD),
    ):
        for name, source, good, bad in cases:
            row: dict[str, Any] = {"name": name, "kind": kind}
            built: dict[str, Any] = {}

            started = time.perf_counter()
            try:
                grammar = (
                    ours.compile_regex(source)
                    if kind == "regex"
                    else ours.compile_ebnf(source, "root")
                )
                compile_ms["engrain"].append(1000 * (time.perf_counter() - started))
                built["engrain"] = (
                    lambda g=grammar: g.matcher(),
                    lambda m: m.allowed_tokens(),
                    lambda m: m.can_terminate(),
                )
            except Exception as error:  # noqa: BLE001
                row["engrain refused"] = str(error)[:100]

            started = time.perf_counter()
            try:
                compiled = (
                    theirs.compile_regex(source)
                    if kind == "regex"
                    else theirs.compile_grammar(source)
                )
                compile_ms["xgrammar"].append(1000 * (time.perf_counter() - started))
                built["xgrammar"] = (
                    lambda c=compiled: xg.GrammarMatcher(
                        c, terminate_without_stop_token=True
                    ),
                    their_allowed,
                    lambda m: m.is_completed(),
                )
            except Exception as error:  # noqa: BLE001
                row["xgrammar refused"] = str(error)[:100]

            for engine, (make, allowed_of, done) in built.items():
                # Under-acceptance: a string the pattern matches must be
                # reachable. This is the half a walk cannot see, and the half a
                # narrowing lowering fails.
                admits = sum(
                    1
                    for text in good
                    if _accepts(make, allowed_of, done, text.encode())
                )
                refuses = sum(
                    1
                    for text in bad
                    if not _accepts(make, allowed_of, done, text.encode())
                )
                # Over-acceptance: generate, and let the referee decide. Only
                # regex has one that neither engine wrote.
                loose = walked = 0
                rng = random.Random(seed)
                for _ in range(walks):
                    text, finished = _walk(make(), allowed_of, done, rng)
                    if not finished:
                        continue
                    walked += 1
                    if kind == "regex" and not re.fullmatch(source, text, re.DOTALL):
                        loose += 1
                row[engine] = {
                    "accepts valid": f"{admits}/{len(good)}",
                    "rejects invalid": f"{refuses}/{len(bad)}",
                    "walked": walked,
                    "over-accepted": loose,
                }
            # Both engines are sound on these, so anything one generates is a
            # string the source really allows - which makes it a test case the
            # other has to accept. A differential under-acceptance test with no
            # third party needed, and the half a walk cannot see.
            if len(built) == 2:
                for source_engine, other in (
                    ("engrain", "xgrammar"),
                    ("xgrammar", "engrain"),
                ):
                    make, allowed_of, done = built[source_engine]
                    omake, oallowed, odone = built[other]
                    rng = random.Random(seed + 1)
                    refused = generated = 0
                    for _ in range(walks):
                        text, finished = _walk(make(), allowed_of, done, rng)
                        if not finished:
                            continue
                        # A refusal only counts against the refuser if the
                        # string is one the source really allows. Otherwise the
                        # *generator* was loose and the refuser was right - and
                        # the first run of this had it backwards, reporting a
                        # win for us as a loss.
                        if kind == "regex" and not re.fullmatch(
                            source, text, re.DOTALL
                        ):
                            continue
                        generated += 1
                        if not _accepts(omake, oallowed, odone, text.encode()):
                            refused += 1
                    row[f"{other} refuses {source_engine}'s"] = f"{refused}/{generated}"
            report[kind].append(row)

    # The byte-level question, asked directly.
    utf8: dict[str, list[str]] = {}
    for engine, build in (
        ("engrain", lambda: ours.compile_regex(UTF8_PATTERN)),
        ("xgrammar", lambda: theirs.compile_regex(UTF8_PATTERN)),
    ):
        compiled = build()
        if engine == "engrain":
            make = (
                lambda g=compiled: g.matcher(),
                lambda m: m.allowed_tokens(),
                lambda m: m.can_terminate(),
            )
        else:
            make = (
                lambda c=compiled: xg.GrammarMatcher(
                    c, terminate_without_stop_token=True
                ),
                their_allowed,
                lambda m: m.is_completed(),
            )
        utf8[engine] = [
            name for name, raw, want in UTF8 if _accepts(*make, raw) is not want
        ]
    report["utf8_wrong"] = utf8

    for engine, values in compile_ms.items():
        values.sort()
        report["compile_ms"][engine] = {
            "n": len(values),
            "p50": statistics.median(values) if values else None,
            "max": max(values) if values else None,
        }
    return report


def timing(
    batches: list[int], steps: int, repeats: int, model: str
) -> list[dict[str, Any]]:
    """What a mask costs at batch, on grammars that are not JSON Schema.

    Same protocol as `rigor.lockstep`: identical states by construction, both
    engines driven along the same string, and XGrammar threaded exactly where
    vLLM threads it - above 128 rows in a step, in chunks of sixteen.

    Against a real tokenizer, not the byte vocabulary the correctness half
    uses. Over 256 tokens a per-row fill is 256 bits and both engines are
    measuring their own fixed costs; the vocabulary is most of the work here
    and leaving it out measures nothing anyone runs.
    """
    import functools
    from concurrent.futures import ThreadPoolExecutor

    import torch
    import xgrammar as xg
    from transformers import AutoTokenizer

    import engrain.internals as E

    from .harness import load_vocabulary

    tokenizer = AutoTokenizer.from_pretrained(model)
    vocabulary = load_vocabulary(model)
    ours = E.Compiler(vocabulary)
    theirs = xg.GrammarCompiler(
        xg.TokenizerInfo.from_huggingface(tokenizer, vocab_size=len(tokenizer)),
        cache_enabled=False,
    )
    pairs = []
    for _name, source, good, _bad in PATTERNS + HARD_PATTERNS:
        longest = max(good, key=len)
        text = tokenizer(longest, add_special_tokens=False).input_ids
        if len(text) < 4:
            continue
        try:
            grammar = ours.compile_regex(source)
            compiled = theirs.compile_regex(source)
        except Exception:  # noqa: BLE001
            continue
        pairs.append((grammar, compiled, text))
    print(f"{len(pairs)} regexes with a string long enough to drive")

    pool = E.DeviceGrammar(max_configs=8)
    ids = [pool.admit(grammar) for grammar, _, _ in pairs]
    rows = []
    for batch in batches:
        chosen = [(pairs[i % len(pairs)], ids[i % len(ids)]) for i in range(batch)]
        device = pool.new_batch(batch)
        device.set_grammars([identifier for _, identifier in chosen])
        mine = [grammar.matcher(0) for (grammar, _, _), _ in chosen]
        yours = [
            xg.GrammarMatcher(compiled, terminate_without_stop_token=True)
            for (_, compiled, _), _ in chosen
        ]
        bitmask = xg.allocate_token_bitmask(batch, len(tokenizer))
        threads = (
            ThreadPoolExecutor(max_workers=8) if batch > _VLLM_THREADS_ABOVE else None
        )
        ourtimes, theirtimes = [], []
        for step in range(steps + repeats):
            for row, ((_, _, text), _) in enumerate(chosen):
                token = text[step % len(text)]
                mine[row].accept_token(token)
                yours[row].accept_token(token)
            if step < steps:
                continue
            torch.cuda.synchronize()
            started = time.perf_counter()
            device.set_matchers(mine)
            device.fill_mask()
            torch.cuda.synchronize()
            ourtimes.append(time.perf_counter() - started)
            started = time.perf_counter()
            if threads is None:
                for row, matcher in enumerate(yours):
                    matcher.fill_next_token_bitmask(bitmask, row)
            else:
                chunks = [
                    list(range(i, min(i + 16, batch))) for i in range(0, batch, 16)
                ]
                list(threads.map(functools.partial(_fill_rows, yours, bitmask), chunks))
            theirtimes.append(time.perf_counter() - started)
        if threads is not None:
            threads.shutdown()
        rows.append(
            {
                "batch": batch,
                "engrain_us": 1e6 * statistics.median(ourtimes),
                "xgrammar_us": 1e6 * statistics.median(theirtimes),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walks", type=int, default=40)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", default="results/rigor-nonjson.json")
    parser.add_argument("--batches", type=int, nargs="*", default=[8, 32, 128, 512])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    arguments = parser.parse_args()

    report = measure(arguments.walks, arguments.seed)
    for kind in ("regex", "ebnf"):
        print(f"\n=== {kind}")
        print(
            f"{'':<22} {'engrain':^32} {'xgrammar':^32} {'cross-refusals':^17}\n"
            f"{'':<22} {'valid':>7} {'invalid':>8} {'loose':>7} {'walks':>6} "
            f"{'valid':>7} {'invalid':>8} {'loose':>7} {'walks':>6} "
            f"{'we refuse':>9} {'they refuse':>11}"
        )
        for row in report[kind]:
            line = f"{row['name']:<22}"
            for engine in ("engrain", "xgrammar"):
                if engine in row:
                    cell = row[engine]
                    line += (
                        f" {cell['accepts valid']:>7} {cell['rejects invalid']:>8}"
                        f" {cell['over-accepted']:>8} {cell['walked']:>7} "
                    )
                else:
                    line += f" {'REFUSED':>30} "
            line += (
                f" {row.get(chr(101) + 'ngrain refuses xgrammar' + chr(39) + 's', '-'):>9}"
                f" {row.get('xgrammar refuses engrain' + chr(39) + 's', '-'):>11}"
            )
            print(line)
    print("\nUTF-8 well-formedness on a wide class, cases answered wrongly:")
    for engine, wrong in report["utf8_wrong"].items():
        print(f"  {engine:<10} {len(wrong)} of {len(UTF8)}   {wrong or ''}")
    print("\ncompile:", json.dumps(report["compile_ms"]))

    if arguments.batches:
        print()
        report["timing"] = timing(arguments.batches, 6, 9, arguments.model)
        print(f"{'batch':>6} {'engrain':>10} {'xgrammar':>10} {'x':>7}")
        for row in report["timing"]:
            share = row["xgrammar_us"] / row["engrain_us"]
            print(
                f"{row['batch']:>6} {row['engrain_us']:>9.1f}u "
                f"{row['xgrammar_us']:>9.1f}u {share:>7.2f}"
            )

    path = Path(arguments.out)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(f"written to {path}")


if __name__ == "__main__":
    main()
