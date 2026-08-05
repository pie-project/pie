"""What does residency cost, and what does it cost to get there?

Device residency is the thesis, and it is paid for in two currencies a serving
operator actually budgets.

**Memory (q16, q17).** A resident artifact competes with the KV cache for the
same HBM, so the honest unit is not megabytes but tokens of context given up.
The comparison against XGrammar is not like-for-like and should not be
presented as if it were: XGrammar keeps a compact automaton on the host and
recomputes the token mapping every step, which is exactly the trade being
made. What can be compared fairly is the total, the growth with vocabulary
size, and the concurrency at which the trade stops paying.

**Compile time (q18).** Schemas arrive per request in any real deployment, so
a compiler that is a hundred times slower is unusable however fast its steps
are. Both engines compile the same corpus, cold, and the distribution is
reported rather than the total - one pathological schema is a different
problem from a uniformly slow compiler.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

from .harness import (
    Answer,
    Distribution,
    TOKENIZERS,
    load_corpus,
    load_vocabulary,
    write_report,
)

# An A100 80GB running a 7B model in bf16 keeps roughly this much per token of
# KV cache, at 32 layers x 2 x 8 heads x 128 dims x 2 bytes. Used only to state
# the memory cost in a unit an operator budgets in.
KV_BYTES_PER_TOKEN = 32 * 2 * 8 * 128 * 2


def measure(
    model: str, limit: int | None, lexer_states: int, corpus: str | None = None
) -> dict[str, Any]:
    """Compile the corpus in both engines, timing and weighing each schema."""
    import engrain
    import engrain.internals
    import xgrammar as xg
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    vocabulary = load_vocabulary(model)
    instances = load_corpus(Path(corpus)) if corpus else load_corpus()
    if limit:
        instances = instances[:limit]

    our_compiler = engrain.internals.Compiler(vocabulary)
    info = xg.TokenizerInfo.from_huggingface(tokenizer, vocab_size=len(tokenizer))
    their_compiler = xg.GrammarCompiler(info, cache_enabled=False)

    ours_ms: list[float] = []
    theirs_ms: list[float] = []
    resident: list[int] = []
    groups: list[int] = []
    our_failures = 0
    their_failures = 0

    for instance in instances:
        started = time.perf_counter()
        try:
            grammar = our_compiler.compile_json_schema(instance["schema"], lexer_states)
            ours_ms.append((time.perf_counter() - started) * 1e3)
            resident.append(grammar.resident_bytes)
            groups.append(grammar.num_groups)
        except Exception:  # noqa: BLE001
            our_failures += 1

        started = time.perf_counter()
        try:
            their_compiler.compile_json_schema(instance["schema"])
            theirs_ms.append((time.perf_counter() - started) * 1e3)
        except Exception:  # noqa: BLE001
            their_failures += 1

    return {
        "model": model,
        "vocabulary": len(vocabulary),
        "schemas": len(instances),
        "ours_compiled": len(ours_ms),
        "theirs_compiled": len(theirs_ms),
        "our_failures": our_failures,
        "their_failures": their_failures,
        "ours_compile_ms": Distribution.of(ours_ms).__dict__,
        "theirs_compile_ms": Distribution.of(theirs_ms).__dict__,
        "resident_total_mb": sum(resident) / 1e6,
        "resident_per_schema_bytes": Distribution.of(
            [float(value) for value in resident]
        ).__dict__,
        "groups": Distribution.of([float(value) for value in groups]).__dict__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(TOKENIZERS.values()))
    parser.add_argument("--schemas", type=int, default=120)
    parser.add_argument("--lexer-states", type=int, default=4000)
    parser.add_argument(
        "--corpus",
        default=None,
        help="which corpus to compile. `results/corpus-exact.json` is the "
        "fragment this engine enforces with nothing left over.",
    )
    arguments = parser.parse_args()

    per_model = []
    for model in arguments.models:
        try:
            per_model.append(
                measure(
                    model, arguments.schemas, arguments.lexer_states, arguments.corpus
                )
            )
        except Exception as error:  # noqa: BLE001
            per_model.append({"model": model, "error": str(error)[:200]})
        latest = per_model[-1]
        if "error" in latest:
            print(f"{model}: {latest['error']}")
            continue
        print(
            f"{model}: vocab {latest['vocabulary']}, "
            f"{latest['ours_compiled']}/{latest['schemas']} compiled, "
            f"{latest['resident_total_mb']:.1f} MB resident, "
            f"compile p50 {latest['ours_compile_ms']['p50']:.1f} ms "
            f"vs xgrammar {latest['theirs_compile_ms']['p50']:.1f} ms"
        )

    usable = [row for row in per_model if "error" not in row]
    answers = []

    if usable:
        first = usable[0]
        per_schema = first["resident_per_schema_bytes"]["p50"]
        tokens = per_schema / KV_BYTES_PER_TOKEN
        answers.append(
            Answer(
                "q16-mem",
                f"median schema costs {per_schema / 1e6:.2f} MB resident, "
                f"about {tokens:.0f} tokens of KV cache for a 7B model; "
                f"{first['resident_total_mb']:.0f} MB for all "
                f"{first['ours_compiled']} compiled schemas",
                detail={
                    "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
                    "resident_per_schema_bytes": first["resident_per_schema_bytes"],
                    "groups": first["groups"],
                    "note": (
                        "Not comparable to XGrammar's cache: it keeps an "
                        "automaton on the host and recomputes the token "
                        "mapping every step. That recomputation is the cost "
                        "this memory buys out, and quoting the two side by "
                        "side as if they were the same quantity would be "
                        "dishonest."
                    ),
                },
            )
        )
        answers.append(
            Answer(
                "q17-vocab",
                "; ".join(
                    f"{row['model'].split('/')[-1]} vocab {row['vocabulary']}: "
                    f"{row['resident_total_mb']:.0f} MB over "
                    f"{row['ours_compiled']} schemas"
                    for row in usable
                ),
                detail={"per_model": usable},
            )
        )
        answers.append(
            Answer(
                "q18-compile",
                "; ".join(
                    f"{row['model'].split('/')[-1]}: ours p50 "
                    f"{row['ours_compile_ms']['p50']:.1f} ms p99 "
                    f"{row['ours_compile_ms']['p99']:.0f} ms, xgrammar p50 "
                    f"{row['theirs_compile_ms']['p50']:.1f} ms p99 "
                    f"{row['theirs_compile_ms']['p99']:.0f} ms"
                    for row in usable
                ),
                detail={
                    "note": (
                        "XGrammar's compiler cache is disabled so that both "
                        "engines are measured cold, which is the case that "
                        "matters when schemas arrive per request."
                    )
                },
            )
        )
    else:
        for question in ("q16-mem", "q17-vocab", "q18-compile"):
            answers.append(Answer(question, "", unanswered="no tokenizer loaded"))

    write_report("cost", answers, {"per_model": per_model})


if __name__ == "__main__":
    main()
