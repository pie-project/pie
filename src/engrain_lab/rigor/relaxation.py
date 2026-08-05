"""How relaxed is a relaxed grammar, and does the list say so?

`CompiledGrammar.relaxations` names every keyword this engine stopped
enforcing, points at it with a JSON pointer, and gives the edit that would
enforce it. That is a claim, and it has two halves worth measuring separately:

**Is it complete?** Walk the mask, validate what comes out, and ask of every
rejection whether the constraint that rejected it is one the list named. A
caller who reads the list and re-checks exactly those keywords is safe only if
the answer is always yes. An over-acceptance the list does not cover is the
failure this file exists to find - the mask widened and nobody said where.

**How much does each entry cost?** A list of ten findings is useless if nine of
them never fire. Attributing each invalid document to the entry responsible
turns the list from a disclaimer into a ranking, so an author fixes the one
that matters. `required` and `additionalProperties` are known to dominate; this
puts a number on the rest.

The unit is a document, not a language: two grammars can differ on infinitely
many strings a model would never write. A random walk over the mask is not a
model either - it is uniform where a model is not - so the rate here is an
upper bound on what serving sees, and it is comparable across keywords, which
is what the ranking needs.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from typing import Any

from .harness import BYTE_VOCABULARY, load_corpus, write_report
from .soundness import MAX_WALK_BYTES, STOP_CHANCE


def _walk(matcher: Any, rng: random.Random) -> tuple[str, bool]:
    """Generate a document the mask permits, uniformly among admitted bytes."""
    produced = bytearray()
    for _ in range(MAX_WALK_BYTES):
        if matcher.can_terminate() and rng.random() < STOP_CHANCE:
            return produced.decode("utf-8", "replace"), True
        allowed = matcher.allowed_tokens()
        if not allowed:
            return produced.decode("utf-8", "replace"), matcher.can_terminate()
        chosen = rng.choice(allowed)
        matcher.accept_token(chosen)
        produced.append(chosen)
    return produced.decode("utf-8", "replace"), matcher.can_terminate()


def _blamed(error: Any) -> list[tuple[str, str]]:
    """Every keyword a validator's rejection implicates, outermost last.

    `absolute_schema_path` ends with the keyword the error was raised on, and
    the pointer is everything before it - which is the object the relaxation
    walk pointed at. But a validator reports the *deepest* best-match error, so
    a document failing a root `oneOf` comes back as `#/oneOf/0/anyOf`. Failing
    the inner `anyOf` is what makes branch 0 fail, which is what makes the
    root `oneOf` fail, so an entry naming either one explains this document.
    Walking back up the path recovers all of them.
    """
    path = [str(part) for part in error.absolute_schema_path]
    if not path:
        return [(str(error.validator), "#")]
    found = []
    for cut in range(len(path), 0, -1):
        if path[cut - 1].isdigit():
            continue
        found.append(
            (path[cut - 1], "#" + "".join(f"/{part}" for part in path[: cut - 1]))
        )
    return found


def _aliases(schema: Any) -> dict[str, str]:
    """Where a `$ref` sits, and what it points at.

    A validator reports the path it took through the *resolved* schema, so a
    relaxation declared at `#/definitions/operation` comes back as `#/items`.
    Without this the two never match and the coverage number blames the engine
    for a difference in how the two sides spell the same place.
    """
    found: dict[str, str] = {}
    stack = [(schema, "#")]
    while stack:
        node, at = stack.pop()
        if isinstance(node, dict):
            target = node.get("$ref")
            if isinstance(target, str) and target.startswith("#"):
                found[at] = target.rstrip("/")
            for key, value in node.items():
                stack.append((value, f"{at}/{key}"))
        elif isinstance(node, list):
            for index, item in enumerate(node):
                stack.append((item, f"{at}/{index}"))
    return found


def _spellings(at: str, aliases: dict[str, str]) -> set[str]:
    """Every pointer naming the same place.

    Two rewrites, to a fixpoint. A `$ref` is followed, because a validator
    reports the path through the resolved schema. And an `/allOf/<n>` segment
    is dropped, because an `allOf` branch is merged into the object above it
    before anything is lowered - so a constraint the validator locates in a
    branch is one the engine reports on the parent, and they are the same
    place under two names.
    """
    found = {at}
    for _ in range(6):
        grown = set(found)
        for pointer in found:
            for site, target in aliases.items():
                if pointer == site or pointer.startswith(site + "/"):
                    grown.add(target + pointer[len(site) :])
            collapsed = re.sub(r"/allOf/\d+", "", pointer)
            if collapsed != pointer:
                grown.add(collapsed or "#")
        if grown == found:
            break
        found = grown
    return found


def _covers(note: dict[str, str], keyword: str, at: str) -> bool:
    """Does this entry account for a rejection at that keyword and place?

    An entry covers its own object and everything under it. A relaxation of
    `additionalProperties` on an object is exactly the statement that the
    values under it are unchecked, so a rejection deeper down is the entry
    firing rather than a second, undeclared one.
    """
    if note["keyword"] != keyword and not (
        note["keyword"] == "additionalProperties"
        and keyword in ("type", "enum", "const")
    ):
        return False
    return at == note["at"] or at.startswith(note["at"].rstrip("#") + "/")


def _covered(notes: list[dict[str, str]], keyword: str, places: set[str]):
    for note in notes:
        if any(_covers(note, keyword, at) for at in places):
            return note
    return None


def measure(
    schemas: list[dict[str, str]],
    walks: int,
    seed: int,
    limit: int,
) -> dict[str, Any]:
    import engrain.internals
    import jsonschema

    compiler = engrain.internals.Compiler(BYTE_VOCABULARY)

    declared: dict[str, int] = defaultdict(int)
    fired: dict[str, int] = defaultdict(int)
    undeclared: dict[str, int] = defaultdict(int)
    examples: list[dict[str, str]] = []
    totals = {
        "schemas": 0,
        "relaxed": 0,
        "notes": 0,
        "walks": 0,
        "complete": 0,
        "valid": 0,
        "invalid": 0,
        "attributed": 0,
    }

    for index, instance in enumerate(schemas):
        if totals["schemas"] >= limit:
            break
        try:
            grammar = compiler.compile_json_schema(instance["schema"])
        except Exception:  # noqa: BLE001 - a refusal is coverage, not relaxation
            continue
        schema = json.loads(instance["schema"])
        aliases = _aliases(schema)
        notes = list(grammar.relaxations)
        totals["schemas"] += 1
        totals["relaxed"] += 1 if notes else 0
        totals["notes"] += len(notes)
        for note in notes:
            declared[note["keyword"]] += 1

        rng = random.Random(seed + index)
        hit: set[str] = set()
        for _ in range(walks):
            try:
                text, finished = _walk(grammar.matcher(), rng)
            except Exception:  # noqa: BLE001
                continue
            totals["walks"] += 1
            if not finished:
                continue
            totals["complete"] += 1
            try:
                document = json.loads(text)
            except Exception:  # noqa: BLE001 - an incomplete walk is not a verdict
                continue
            try:
                jsonschema.validate(document, schema)
            except jsonschema.ValidationError as error:
                totals["invalid"] += 1
                covering = None
                keyword, at = "?", "#"
                for keyword, at in _blamed(error):
                    covering = _covered(notes, keyword, _spellings(at, aliases))
                    if covering is not None:
                        break
                if covering is not None:
                    totals["attributed"] += 1
                    hit.add(covering["keyword"])
                else:
                    undeclared[keyword] += 1
                    if len(examples) < 12:
                        examples.append(
                            {
                                "schema": str(index),
                                "keyword": keyword,
                                "at": at,
                                "document": text[:120],
                            }
                        )
            except Exception:  # noqa: BLE001 - the validator refused the schema
                continue
            else:
                totals["valid"] += 1
        for keyword in hit:
            fired[keyword] += 1

    per_keyword = {
        keyword: {
            "grammars declaring it": declared[keyword],
            "grammars where it fired": fired.get(keyword, 0),
            "fires": (
                round(100.0 * fired.get(keyword, 0) / declared[keyword], 1)
                if declared[keyword]
                else 0.0
            ),
        }
        for keyword in sorted(declared, key=lambda k: -declared[k])
    }
    return {
        "totals": totals,
        "attributed": (
            round(100.0 * totals["attributed"] / totals["invalid"], 1)
            if totals["invalid"]
            else 100.0
        ),
        "per_keyword": per_keyword,
        "undeclared": dict(sorted(undeclared.items(), key=lambda item: -item[1])),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walks", type=int, default=20)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="relaxation")
    arguments = parser.parse_args()

    report = measure(load_corpus(), arguments.walks, arguments.seed, arguments.limit)
    totals = report["totals"]

    print(
        f"{totals['schemas']} schemas compiled, {totals['relaxed']} carry a "
        f"relaxation, {totals['notes']} notes in all"
    )
    print(
        f"{totals['complete']} complete walks: {totals['valid']} valid, "
        f"{totals['invalid']} invalid"
    )
    print(
        f"{report['attributed']}% of invalid documents are attributable to a "
        f"declared relaxation"
    )
    print()
    print(f"{'keyword':<24} {'declared':>9} {'fired':>7} {'rate':>7}")
    for keyword, row in report["per_keyword"].items():
        print(
            f"{keyword:<24} {row['grammars declaring it']:>9} "
            f"{row['grammars where it fired']:>7} {row['fires']:>6.1f}%"
        )
    if report["undeclared"]:
        print()
        print("rejected on a keyword no relaxation named:")
        for keyword, count in report["undeclared"].items():
            print(f"  {keyword:<22} {count}")

    write_report(arguments.out, [], report)


if __name__ == "__main__":
    main()
