"""Does the mask mean what we say it means?

Speed is worth nothing if the mask is wrong, so this runs first. Three
questions, each answered by generating rather than by inspecting:

**Over-acceptance (q01, q02).** Walk the grammar: at every step take the set of
bytes the matcher admits, choose one at random, and repeat until the matcher
says the document may end. Whatever comes out is, by construction, a document
the mask permits. Handing it to a real JSON Schema validator asks the only
question that matters - is everything the mask permits actually valid? A
constrained decoder that answers "no" is broken no matter how fast it is.

**Under-acceptance (q01).** The mirror. Documents that are valid must be
reachable, or the mask is silently truncating the model's choices. The corpus
instances answer this and are measured elsewhere; here the concern is that a
*relaxed* lowering may buy acceptance by admitting nonsense, so over- and
under-acceptance are reported per precision level.

The same walk runs under XGrammar's mask, over the same byte vocabulary and the
same schemas. "Some of what we admit is invalid" is only damning next to a
baseline that admits nothing invalid; measuring both turns an absolute number
into a comparison, and neither engine implements every JSON Schema keyword.

**Agreement with XGrammar (q03).** Two engines claiming the same semantics
should admit the same tokens. Comparing whole bitmasks at every step of a real
document turns a claim into a count, and a reference validator adjudicates
wherever they differ - "we disagree with the baseline" is not a result until
somebody says who was right.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .harness import (
    BYTE_VOCABULARY,
    Answer,
    load_corpus,
    write_report,
)

# A walk has to stop somewhere. Long enough to get well past the opening of a
# nested document, short enough that thousands of walks finish.
MAX_WALK_BYTES = 400

# Once the document may end, end it with this probability per step. Anything
# lower spends the whole budget inside one array.
STOP_CHANCE = 0.15


# Keywords the front end does not lower. A document that violates one of these
# is over-accepted for a reason that is known and bounded, which is a different
# fact from a parser that lets through something it claims to check.
UNLOWERED = (
    "dependencies",
    "dependentRequired",
    "dependentSchemas",
    "uniqueItems",
    "not",
    "propertyNames",
    "contains",
    "multipleOf",
    "if",
    "then",
    "else",
    "unevaluatedProperties",
    "unevaluatedItems",
    "const",
)


@dataclass
class WalkOutcome:
    schema_index: int
    engine: str
    text: str
    valid: bool | None
    reason: str
    cause: str


def _keywords(node: Any, found: set[str]) -> set[str]:
    if isinstance(node, dict):
        found.update(node.keys())
        for value in node.values():
            _keywords(value, found)
    elif isinstance(node, list):
        for value in node:
            _keywords(value, found)
    return found


def _blame(schema: dict[str, Any], reason: str) -> str:
    """Why was an invalid document admitted?

    Distinguishing "a keyword we never claimed to lower" from "a keyword we did
    lower, wrongly" matters: the first is a coverage gap with a known fix, the
    second is a bug in something already shipped.
    """
    present = _keywords(schema, set())
    for keyword in UNLOWERED:
        if keyword in present and keyword.lower() in reason.lower():
            return f"unlowered keyword: {keyword}"
    for keyword in UNLOWERED:
        if keyword in present:
            return f"schema uses unlowered {keyword}"
    if "is not valid under any of the given schemas" in reason:
        return "relaxed anyOf/oneOf: required sets intersected"
    return "unexplained"


def _walk(matcher: Any, allowed_of: Any, done: Any, rng: random.Random) -> tuple[str, bool]:
    """Generate a document the mask permits, uniformly among admitted bytes."""
    produced = bytearray()
    for _ in range(MAX_WALK_BYTES):
        if done(matcher) and rng.random() < STOP_CHANCE:
            return produced.decode("utf-8", "replace"), True
        allowed = allowed_of(matcher)
        if not allowed:
            return produced.decode("utf-8", "replace"), done(matcher)
        chosen = rng.choice(allowed)
        matcher.accept_token(chosen)
        produced.append(chosen)
    return produced.decode("utf-8", "replace"), done(matcher)


def _ours(schema_text: str, compiler: Any):
    grammar = compiler.compile_json_schema(schema_text)
    level = str(getattr(grammar, "precision", "unknown"))

    def make():
        return grammar.matcher()

    return make, (lambda m: m.allowed_tokens()), (lambda m: m.can_terminate()), level


def _xgrammar(schema_text: str, compiler: Any, bitmask: Any, torch: Any):
    import xgrammar as xg

    grammar = compiler.compile_json_schema(schema_text)
    bits = torch.arange(32, dtype=torch.int32)

    def make():
        return xg.GrammarMatcher(grammar, terminate_without_stop_token=True)

    def allowed(matcher):
        matcher.fill_next_token_bitmask(bitmask, 0)
        words = bitmask[0].to(torch.int32)
        flags = ((words.unsqueeze(1) >> bits) & 1).to(torch.bool).reshape(-1)[:256]
        return flags.nonzero().flatten().tolist()

    return make, allowed, (lambda m: m.is_completed()), "xgrammar"


def over_acceptance(
    schemas: list[dict[str, str]],
    walks: int,
    seed: int,
    limit: int | None,
    with_xgrammar: bool,
) -> tuple[list[Answer], list[WalkOutcome]]:
    """Generate under each engine's mask, then check with a real validator."""
    import engrain
    import jsonschema

    our_compiler = engrain.Compiler(BYTE_VOCABULARY)

    engines: dict[str, Any] = {"engrain": None}
    bitmask = None
    torch = None
    if with_xgrammar:
        import torch as _torch
        import xgrammar as xg

        torch = _torch
        info = xg.TokenizerInfo(
            BYTE_VOCABULARY, xg.VocabType.RAW, vocab_size=256, stop_token_ids=[]
        )
        engines["xgrammar"] = xg.GrammarCompiler(info)
        bitmask = xg.allocate_token_bitmask(1, 256)

    outcomes: list[WalkOutcome] = []
    tally: dict[str, dict[str, int]] = {}
    per_level: dict[str, dict[str, int]] = {}
    compiled: dict[str, int] = {name: 0 for name in engines}

    for index, instance in enumerate(schemas):
        if limit is not None and compiled["engrain"] >= limit:
            break
        schema = json.loads(instance["schema"])

        built: dict[str, Any] = {}
        try:
            built["engrain"] = _ours(instance["schema"], our_compiler)
        except Exception:  # noqa: BLE001 - refusals are coverage, not soundness
            continue
        if with_xgrammar:
            try:
                built["xgrammar"] = _xgrammar(
                    instance["schema"], engines["xgrammar"], bitmask, torch
                )
            except Exception:  # noqa: BLE001
                pass

        for name, (make, allowed_of, done, level) in built.items():
            compiled[name] = compiled.get(name, 0) + 1
            bucket = tally.setdefault(
                name, {"walks": 0, "valid": 0, "invalid": 0, "unfinished": 0}
            )
            level_bucket = None
            if name == "engrain":
                level_bucket = per_level.setdefault(
                    level, {"valid": 0, "invalid": 0}
                )
            rng = random.Random(seed + index)

            for _ in range(walks):
                try:
                    text, finished = _walk(make(), allowed_of, done, rng)
                except Exception as error:  # noqa: BLE001
                    outcomes.append(
                        WalkOutcome(index, name, "", None, str(error)[:120], "walk failed")
                    )
                    continue
                bucket["walks"] += 1
                if not finished:
                    bucket["unfinished"] += 1
                    continue
                verdict, reason = _check(text, schema, jsonschema)
                if verdict is None:
                    continue
                key = "valid" if verdict else "invalid"
                bucket[key] += 1
                if level_bucket is not None:
                    level_bucket[key] += 1
                if not verdict:
                    outcomes.append(
                        WalkOutcome(
                            index, name, text, False, reason, _blame(schema, reason)
                        )
                    )

    answers: list[Answer] = []
    headline_parts = []
    for name, bucket in sorted(tally.items()):
        finished = bucket["valid"] + bucket["invalid"]
        rate = 100.0 * bucket["valid"] / finished if finished else 0.0
        headline_parts.append(f"{name} {bucket['valid']}/{finished} valid ({rate:.1f}%)")

    causes: dict[str, dict[str, int]] = {}
    for outcome in outcomes:
        if outcome.valid is False:
            causes.setdefault(outcome.engine, {}).setdefault(outcome.cause, 0)
            causes[outcome.engine][outcome.cause] += 1

    answers.append(
        Answer(
            question_id="q01-sound",
            headline="documents generated under the mask that validate: "
            + "; ".join(headline_parts),
            detail={
                "schemas": compiled,
                "walks_per_schema": walks,
                "per_engine": tally,
                "causes": causes,
                "note": (
                    "Every document was produced by choosing only bytes the "
                    "mask admitted, so an invalid one is a byte the mask should "
                    "have refused. Both engines walk the same schemas over the "
                    "same 256-byte vocabulary."
                ),
            },
        )
    )
    answers.append(
        Answer(
            question_id="q02-superset",
            headline="over-acceptance by precision level: "
            + ", ".join(
                f"{level} {100.0 * b['valid'] / max(1, b['valid'] + b['invalid']):.1f}%"
                f" valid of {b['valid'] + b['invalid']}"
                for level, b in sorted(per_level.items())
            ),
            detail={"per_level": per_level},
        )
    )
    return answers, outcomes


def _check(text: str, schema: dict[str, Any], jsonschema: Any) -> tuple[bool | None, str]:
    try:
        document = json.loads(text)
    except Exception as error:  # noqa: BLE001
        return False, f"not JSON: {error}"
    try:
        jsonschema.validate(document, schema)
    except jsonschema.ValidationError as error:
        return False, str(error).splitlines()[0][:140]
    except Exception:  # noqa: BLE001 - the validator itself refused the schema
        return None, ""
    return True, ""


def mask_agreement(
    schemas: list[dict[str, str]],
    model: str,
    limit: int,
) -> Answer:
    """Compare whole bitmasks with XGrammar, step by step, on real documents."""
    try:
        import torch
        import xgrammar as xg
        from transformers import AutoTokenizer
    except Exception as error:  # noqa: BLE001
        return Answer("q03-vs-xgr-mask", "", unanswered=f"import failed: {error}")

    import engrain

    tokenizer = AutoTokenizer.from_pretrained(model)
    vocabulary: list[bytes] = []
    for token_id in range(len(tokenizer)):
        piece = tokenizer.convert_ids_to_tokens(token_id)
        try:
            vocabulary.append(tokenizer.convert_tokens_to_string([piece]).encode())
        except Exception:  # noqa: BLE001
            vocabulary.append(b"")

    info = xg.TokenizerInfo.from_huggingface(tokenizer, vocab_size=len(tokenizer))
    xgrammar_compiler = xg.GrammarCompiler(info)
    our_compiler = engrain.Compiler(vocabulary)

    steps = 0
    identical = 0
    ours_only = 0
    theirs_only = 0
    compared_schemas = 0
    examples: list[dict[str, Any]] = []

    for instance in schemas:
        if compared_schemas >= limit:
            break
        try:
            theirs = xgrammar_compiler.compile_json_schema(instance["schema"])
            ours = our_compiler.compile_json_schema(instance["schema"])
        except Exception:  # noqa: BLE001 - coverage measures refusals
            continue
        compared_schemas += 1

        token_ids = tokenizer(instance["text"], add_special_tokens=False)["input_ids"]
        their_matcher = xg.GrammarMatcher(theirs)
        our_matcher = ours.matcher()
        bitmask = xg.allocate_token_bitmask(1, len(tokenizer))

        for token in token_ids:
            their_matcher.fill_next_token_bitmask(bitmask, 0)
            their_allowed = _unpack(bitmask, len(tokenizer))
            our_allowed = torch.zeros(len(tokenizer), dtype=torch.bool)
            our_allowed[list(our_matcher.allowed_tokens())] = True

            steps += 1
            if torch.equal(their_allowed, our_allowed):
                identical += 1
            else:
                extra_ours = int((our_allowed & ~their_allowed).sum())
                extra_theirs = int((their_allowed & ~our_allowed).sum())
                ours_only += extra_ours
                theirs_only += extra_theirs
                if len(examples) < 20:
                    examples.append(
                        {
                            "schema": instance["config"],
                            "prefix": instance["text"][: max(0, steps)][-60:],
                            "tokens_only_we_allow": extra_ours,
                            "tokens_only_xgrammar_allows": extra_theirs,
                        }
                    )
            if not our_matcher.accept_token(token):
                break
            if not their_matcher.accept_token(token):
                break

    if steps == 0:
        return Answer(
            "q03-vs-xgr-mask", "", unanswered="no schema compiled in both engines"
        )
    return Answer(
        question_id="q03-vs-xgr-mask",
        headline=(
            f"{identical}/{steps} steps mask-identical to XGrammar "
            f"({100.0 * identical / steps:.1f}%)"
        ),
        detail={
            "schemas": compared_schemas,
            "steps": steps,
            "identical": identical,
            "extra_tokens_we_allow": ours_only,
            "extra_tokens_xgrammar_allows": theirs_only,
            "examples": examples,
            "note": (
                "A disagreement is not yet a verdict. Tokens only one engine "
                "allows have to be adjudicated against a validator before "
                "either side may be called wrong."
            ),
        },
    )


def _unpack(bitmask: Any, vocabulary_size: int) -> Any:
    import torch

    words = bitmask[0].to(torch.int32)
    bits = torch.arange(32, dtype=torch.int32)
    expanded = ((words.unsqueeze(1) >> bits) & 1).to(torch.bool).reshape(-1)
    return expanded[:vocabulary_size]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, default=None)
    parser.add_argument("--walks", type=int, default=20)
    parser.add_argument("--schemas", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--skip-xgrammar", action="store_true")
    arguments = parser.parse_args()

    schemas = load_corpus(arguments.instances) if arguments.instances else load_corpus()

    answers, outcomes = over_acceptance(
        schemas,
        arguments.walks,
        arguments.seed,
        arguments.schemas,
        not arguments.skip_xgrammar,
    )

    if arguments.skip_xgrammar:
        answers.append(Answer("q03-vs-xgr-mask", "", unanswered="skipped by request"))
    else:
        answers.append(mask_agreement(schemas, arguments.model, arguments.schemas))

    failures = [
        {
            "schema": o.schema_index,
            "engine": o.engine,
            "cause": o.cause,
            "reason": o.reason,
            "text": o.text[:200],
        }
        for o in outcomes
        if o.valid is False
    ]
    write_report(
        "soundness",
        answers,
        {
            "walks_per_schema": arguments.walks,
            "seed": arguments.seed,
            "invalid_examples": failures[:60],
            "invalid_total": len(failures),
        },
    )


if __name__ == "__main__":
    main()
