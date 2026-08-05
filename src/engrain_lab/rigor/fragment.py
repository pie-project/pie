"""A corpus of schemas this engine enforces exactly, and the documents to drive it.

Every measurement so far has compared two engines on schemas neither lowers
faithfully. That is the honest thing to report and it is a bad place to learn
anything: a difference in a mask is partly a difference in what the two masks
*mean*, and `rigor.agreement` found no corpus schema, in any whitespace
configuration, where they mean the same thing.

This builds the other corpus. Each schema is rewritten by applying the remedies
`CompiledGrammar.relaxations` gives - close the objects, move a keyword inside
the branches it constrains, delete what no level lowers - and kept only if the
rewrite lands inside the fragment this engine enforces with nothing left over.
The test is the engine's own: `relaxations == []`, checked, not asserted.

What comes out is small and favourable by construction, and both of those have
to be said out loud. It is favourable because it is exactly the fragment we are
good at. It is *useful* because on it there is nothing to trade: both engines
are exact, the languages agree, and a difference in cost is only a difference
in cost. A comparison where one side is allowed to be wrong is not a comparison
of two solutions to the same problem.

    python -m engrain_lab.rigor.fragment --out results/corpus-exact.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .harness import BYTE_VOCABULARY, load_corpus
from .relaxation import _walk

# Keywords no lowering here reads. Deleting one widens the schema, which is
# the point: what remains is what the mask actually enforces, written down.
UNLOWERED = (
    "multipleOf",
    "uniqueItems",
    "dependencies",
    "dependentRequired",
    "dependentSchemas",
    "not",
    "contains",
    "minContains",
    "maxContains",
    "propertyNames",
    "if",
    "then",
    "else",
    "unevaluatedProperties",
    "unevaluatedItems",
    "contentEncoding",
    "contentMediaType",
    "contentSchema",
    "$recursiveRef",
    "$dynamicRef",
)

# What the parser can carry at once for a closed object. Above it `required`
# stops being enforced, so the rewrite trims to it rather than leaving a
# requirement the mask ignores.
REQUIRED_BUDGET = 6

# Keys that belong to the schema around a choice rather than to the choice.
KEPT_BESIDE = (
    "$schema",
    "$id",
    "$comment",
    "title",
    "description",
    "definitions",
    "$defs",
)

SCHEMA_MAP = (
    "properties",
    "patternProperties",
    "$defs",
    "definitions",
    "dependentSchemas",
)
SCHEMA_LIST = ("allOf", "anyOf", "oneOf", "prefixItems")
SCHEMA_ONE = (
    "items",
    "additionalItems",
    "additionalProperties",
    "propertyNames",
    "contains",
)


def _merge(outer: dict[str, Any], branch: dict[str, Any]) -> dict[str, Any]:
    """The object a choice sits on, pushed into one of its branches.

    This is the remedy the engine gives for a keyword beside a choice - "move
    it inside each branch, where it is lowered with them" - carried out. The
    branch wins wherever both say something, because it is the more specific
    of the two.
    """
    merged = dict(outer)
    for key, value in branch.items():
        if key == "properties" and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        elif key == "required" and isinstance(merged.get(key), list):
            merged[key] = merged[key] + [n for n in value if n not in merged[key]]
        else:
            merged[key] = value
    return merged


def tighten(node: Any) -> Any:
    """Rewrite a schema until it says only what this engine enforces."""
    if isinstance(node, list):
        return [tighten(item) for item in node]
    if not isinstance(node, dict):
        return node

    out = {key: value for key, value in node.items() if key not in UNLOWERED}

    for key in SCHEMA_MAP:
        if isinstance(out.get(key), dict):
            out[key] = {name: tighten(value) for name, value in out[key].items()}
    for key in SCHEMA_LIST:
        if isinstance(out.get(key), list):
            out[key] = [tighten(value) for value in out[key]]
    for key in SCHEMA_ONE:
        if isinstance(out.get(key), (dict, list)):
            out[key] = tighten(out[key])

    # A choice keeps its branches and gives up its siblings, so the siblings go
    # in with them. `oneOf` becomes `anyOf`: exactly-one is not a property of a
    # prefix unless the branches are pinned apart, and a union is the language
    # the parser builds either way - so this makes the schema say what the mask
    # will do rather than leaving a promise the mask cannot keep.
    for key in ("oneOf", "anyOf"):
        branches = out.get(key)
        if not isinstance(branches, list) or not branches:
            continue
        beside = {
            name: value
            for name, value in out.items()
            if name != key and name not in KEPT_BESIDE
        }
        kept = {
            name: value
            for name, value in out.items()
            if name != key and name in KEPT_BESIDE
        }
        out = {
            **kept,
            "anyOf": [
                _merge(beside, branch) if isinstance(branch, dict) else branch
                for branch in branches
            ],
        }
        break

    # Close every object that declares properties. An open one lets a key that
    # spells a declared name read as an additional property, and then the
    # declared type is not enforced.
    if isinstance(out.get("properties"), dict) or out.get("patternProperties"):
        out["additionalProperties"] = False

    out.pop("maxProperties", None)
    if isinstance(out.get("required"), list):
        required = out["required"][:REQUIRED_BUDGET]
        if required:
            out["required"] = required
        else:
            out.pop("required")
        if isinstance(out.get("minProperties"), int):
            out["minProperties"] = min(out["minProperties"], len(required))
            if out["minProperties"] == 0:
                out.pop("minProperties")
    elif "minProperties" in out:
        out.pop("minProperties")
    return out


def prune(document: Any, schema: Any, root: Any) -> Any:
    """Drop from a document whatever the tightened schema no longer admits.

    Closing an object invalidates a corpus instance that carried an extra key,
    and dropping the schema for it would bias the corpus toward the schemas
    whose examples happened to be minimal. Removing the key keeps both.
    """
    if isinstance(schema, dict) and "$ref" in schema:
        target = schema["$ref"]
        if isinstance(target, str) and target.startswith("#/"):
            node = root
            for part in target[2:].split("/"):
                if not isinstance(node, dict) or part not in node:
                    return document
                node = node[part]
            return prune(document, node, root)
    if isinstance(schema, dict) and isinstance(schema.get("anyOf"), list):
        for branch in schema["anyOf"]:
            pruned = prune(document, branch, root)
            if _valid(pruned, {**root, **branch} if branch is schema else branch, root):
                return pruned
        return document
    if isinstance(document, dict) and isinstance(schema, dict):
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return document
        return {
            key: prune(value, properties[key], root)
            for key, value in document.items()
            if key in properties
        }
    if isinstance(document, list) and isinstance(schema, dict):
        items = schema.get("items")
        if isinstance(items, dict):
            return [prune(item, items, root) for item in document]
    return document


def _valid(document: Any, schema: Any, root: Any) -> bool:
    import jsonschema

    try:
        jsonschema.validate(document, schema)
    except Exception:  # noqa: BLE001
        return False
    return True


def over_accepts(grammar: Any, schema: Any, walks: int, seed: int) -> bool:
    """Does anything this mask admits fail the schema?

    The membership test, and it is a measurement rather than a prediction.
    `relaxations` says what the *lowering* gave up, and a schema can reach the
    same place through an interaction nobody modelled - a `$ref` beside an
    `allOf` beside a `oneOf`. Three of the corpus schemas do exactly that: they
    report nothing relaxed and over-accept anyway. Predicting them means
    re-implementing the front end inside the reporter, which is the duplicate
    model that put the bug there. Generating documents and reading them does
    not.
    """
    import random

    import jsonschema

    rng = random.Random(seed)
    for _ in range(walks):
        text, finished = _walk(grammar.matcher(), rng)
        if not finished:
            continue
        try:
            document = json.loads(text)
        except Exception:  # noqa: BLE001 - a walk that ran out of budget
            continue
        try:
            jsonschema.validate(document, schema)
        except jsonschema.ValidationError:
            return True
        except Exception:  # noqa: BLE001 - the validator refused the schema
            continue
    return False


def build(
    instances: list[dict[str, str]],
    limit: int,
    with_xgrammar: bool,
    walks: int = 20,
    seed: int = 0,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    import engrain.internals

    compiler = engrain.internals.Compiler(BYTE_VOCABULARY)
    xgc = None
    if with_xgrammar:
        import xgrammar as xg

        info = xg.TokenizerInfo(
            BYTE_VOCABULARY, xg.VocabType.RAW, vocab_size=256, stop_token_ids=[]
        )
        xgc = xg.GrammarCompiler(info)

    kept: list[dict[str, str]] = []
    why = {
        "seen": 0,
        "unparsable schema": 0,
        "refused": 0,
        "still relaxed": 0,
        "over-accepts anyway": 0,
        "xgrammar refused": 0,
        "no example document": 0,
        "document does not validate": 0,
        "kept": 0,
        "kept with a document": 0,
    }
    for instance in instances:
        if len(kept) >= limit:
            break
        why["seen"] += 1
        try:
            schema = tighten(json.loads(instance["schema"]))
        except Exception:  # noqa: BLE001
            why["unparsable schema"] += 1
            continue
        # 88 of the corpus instances are not JSON - the benchmark's `text`
        # field is a sample, not a fixture. Dropping the schema with the
        # document would lose a sixth of the corpus for a reason that has
        # nothing to do with either engine, so the schema stays and the
        # document is empty. A consumer that needs one skips it.
        try:
            text = json.dumps(prune(json.loads(instance["text"]), schema, schema))
        except Exception:  # noqa: BLE001
            why["no example document"] += 1
            text = ""
        source = json.dumps(schema)
        try:
            grammar = compiler.compile_json_schema(source)
        except Exception:  # noqa: BLE001
            why["refused"] += 1
            continue
        if grammar.relaxations:
            why["still relaxed"] += 1
            continue
        if over_accepts(grammar, schema, walks, seed + why["seen"]):
            why["over-accepts anyway"] += 1
            continue
        if xgc is not None:
            try:
                xgc.compile_json_schema(source, any_whitespace=True, any_order=True)
            except Exception:  # noqa: BLE001
                why["xgrammar refused"] += 1
                continue
        if text and not _valid(json.loads(text), schema, schema):
            why["document does not validate"] += 1
            text = ""
        why["kept"] += 1
        why["kept with a document"] += 1 if text else 0
        kept.append(
            {"config": instance.get("config", ""), "schema": source, "text": text}
        )
    return kept, why


def check(kept: list[dict[str, str]], walks: int, seed: int) -> dict[str, int]:
    """Generate under the mask and validate: an exact fragment over-accepts nothing.

    The claim is that these schemas need no downstream check, and a claim about
    a mask is worth what a document says about it.
    """
    import random

    import jsonschema

    import engrain.internals

    from .relaxation import _walk

    compiler = engrain.internals.Compiler(BYTE_VOCABULARY)
    tally = {"walks": 0, "complete": 0, "valid": 0, "invalid": 0}
    for index, instance in enumerate(kept):
        grammar = compiler.compile_json_schema(instance["schema"])
        schema = json.loads(instance["schema"])
        rng = random.Random(seed + index)
        for _ in range(walks):
            text, finished = _walk(grammar.matcher(), rng)
            tally["walks"] += 1
            if not finished:
                continue
            tally["complete"] += 1
            try:
                document = json.loads(text)
            except Exception:  # noqa: BLE001
                tally["invalid"] += 1
                continue
            try:
                jsonschema.validate(document, schema)
            except jsonschema.ValidationError:
                tally["invalid"] += 1
            except Exception:  # noqa: BLE001
                continue
            else:
                tally["valid"] += 1
    return tally


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--walks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-xgrammar",
        action="store_true",
        help="keep schemas XGrammar refuses. Off by default: a corpus one "
        "engine cannot compile is not a corpus to compare on.",
    )
    parser.add_argument("--out", default="results/corpus-exact.json")
    arguments = parser.parse_args()

    kept, why = build(
        load_corpus(),
        arguments.limit,
        not arguments.no_xgrammar,
        arguments.walks,
        arguments.seed,
    )
    for reason, count in why.items():
        print(f"{reason:<20} {count}")

    # A second, independent pass: different seeds, so the corpus is not just
    # the schemas that survived the first ones.
    tally = check(kept, arguments.walks * 2, arguments.seed + 10_000)
    print()
    print(
        f"{tally['complete']} complete walks under the mask: "
        f"{tally['valid']} valid, {tally['invalid']} invalid"
    )

    path = Path(arguments.out)
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        json.dumps({"instances": kept, "built": why, "walked": tally}, indent=2)
    )
    print(f"\n{len(kept)} schemas written to {path}")


if __name__ == "__main__":
    main()
