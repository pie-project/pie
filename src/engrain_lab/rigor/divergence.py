"""Why two engines given the same model and seed write different documents.

The end-to-end comparison shows us emitting 8-36% more tokens than XGrammar
and validating fewer documents, and that only means anything if the two are
being asked the same question. They are: same model, same prompts, same
sampling parameters, one seed per request. So either the masks differ, or the
run is not reproducible - and those want different fixes.

    python -m engrain_lab.rigor.divergence --backend engrain --tag a
    python -m engrain_lab.rigor.divergence --backend engrain --tag b
    python -m engrain_lab.rigor.divergence --backend xgrammar --tag a
    python -m engrain_lab.rigor.divergence --compare

`--compare` reports two things. First whether a backend repeats itself, which
separates nondeterminism from everything else. Then, for the first token at
which two backends part, whether each one's choice was in the *other* one's
mask - which says whether they parted because the grammars disagree or because
the sampler happened to land elsewhere inside masks that agreed.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

RESULTS = Path("results")
SEED = 20260802


def _schemas(count: int) -> list[dict]:
    from engrain_lab.rigor.e2e import _agreed_schemas

    corpus = _agreed_schemas()
    return [corpus[index % len(corpus)] for index in range(count)]


def _generate(backend: str, tag: str, count: int, max_tokens: int, memory: float):
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    schemas = _schemas(count)
    prompts = [
        f"Produce one JSON document, number {index}. JSON only."
        for index in range(count)
    ]
    engine = LLM(
        model="Qwen/Qwen3-0.6B",
        gpu_memory_utilization=memory,
        max_model_len=1024,
        structured_outputs_config={"backend": backend},
        seed=SEED,
    )
    params = [
        SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=max_tokens,
            seed=SEED + index,
            structured_outputs=StructuredOutputsParams(json=json.dumps(schema)),
        )
        for index, schema in enumerate(schemas)
    ]
    outputs = engine.generate(prompts, params)
    rows = [
        {
            "index": index,
            "tokens": list(output.outputs[0].token_ids),
            "text": output.outputs[0].text,
        }
        for index, output in enumerate(outputs)
    ]
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"divergence-{backend}-{tag}.json"
    out.write_text(json.dumps({"backend": backend, "rows": rows}, indent=2))
    print(f"written to {out}")


def _load(backend: str, tag: str):
    path = RESULTS / f"divergence-{backend}-{tag}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["rows"]


def _first_difference(left: list[int], right: list[int]) -> int | None:
    for index, (one, other) in enumerate(zip(left, right, strict=False)):
        if one != other:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _compare(count: int, max_tokens: int) -> int:
    import xgrammar as xg
    from transformers import AutoTokenizer

    from engrain.internals import Compiler

    for backend in ("engrain", "xgrammar"):
        one, other = _load(backend, "a"), _load(backend, "b")
        if one and other:
            same = sum(1 for a, b in zip(one, other, strict=True) if a["tokens"] == b["tokens"])
            print(f"{backend}: {same}/{len(one)} requests identical across two runs")

    ours, theirs = _load("engrain", "a"), _load("xgrammar", "a")
    if not ours or not theirs:
        print("need both backends generated at tag 'a'")
        return 1

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    vocabulary = []
    for token_id in range(len(tokenizer)):
        piece = tokenizer.convert_ids_to_tokens(token_id)
        try:
            vocabulary.append(tokenizer.convert_tokens_to_string([piece]).encode())
        except Exception:  # noqa: BLE001
            vocabulary.append(b"")
    compiler = Compiler(vocabulary)
    info = xg.TokenizerInfo.from_huggingface(tokenizer, vocab_size=len(tokenizer))
    xgc = xg.GrammarCompiler(info)

    schemas = _schemas(count)
    identical = 0
    positions: list[int] = []
    blame = {
        "ours admits theirs, theirs admits ours": 0,
        "theirs forbids what we chose": 0,
        "we forbid what they chose": 0,
        "each forbids the other's": 0,
        "one of them ended": 0,
        "could not be replayed": 0,
    }
    import numpy as np
    import torch

    for index, (mine, yours) in enumerate(zip(ours, theirs, strict=True)):
        at = _first_difference(mine["tokens"], yours["tokens"])
        if at is None:
            identical += 1
            continue
        positions.append(at)
        if at >= len(mine["tokens"]) or at >= len(yours["tokens"]):
            blame["one of them ended"] += 1
            continue
        source = json.dumps(schemas[index])
        try:
            grammar = compiler.compile_json_schema(source, max_digits=8)
            matcher = grammar.matcher(0)
            theirs_grammar = xgc.compile_json_schema(source, any_whitespace=True)
            theirs_matcher = xg.GrammarMatcher(
                theirs_grammar, terminate_without_stop_token=True
            )
            prefix = mine["tokens"][:at]
            for token in prefix:
                if not matcher.accept_token(token) or not theirs_matcher.accept_token(
                    token
                ):
                    raise RuntimeError("prefix refused")
            our_mask = torch.zeros(grammar.bitset_words, dtype=torch.int32)
            matcher.fill_bitmask(our_mask)
            their_mask = xg.allocate_token_bitmask(1, len(tokenizer))
            theirs_matcher.fill_next_token_bitmask(their_mask, 0)
        except Exception:  # noqa: BLE001
            blame["could not be replayed"] += 1
            continue

        def allowed(mask, token: int) -> bool:
            word = int(np.asarray(mask).reshape(-1)[token // 32])
            return bool(word & (1 << (token % 32)))

        mine_at, yours_at = mine["tokens"][at], yours["tokens"][at]
        we_allow_theirs = allowed(our_mask, yours_at)
        they_allow_ours = allowed(their_mask, mine_at)
        if we_allow_theirs and they_allow_ours:
            blame["ours admits theirs, theirs admits ours"] += 1
        elif we_allow_theirs and not they_allow_ours:
            blame["theirs forbids what we chose"] += 1
        elif not we_allow_theirs and they_allow_ours:
            blame["we forbid what they chose"] += 1
        else:
            blame["each forbids the other's"] += 1

    print(f"\nengrain against xgrammar: {identical}/{len(ours)} identical")
    if positions:
        print(
            f"  first difference at token p50 {statistics.median(positions):.0f}, "
            f"min {min(positions)}, max {max(positions)}"
        )
    print("  at that token:")
    for reason, number in blame.items():
        if number:
            print(f"    {number:4d}  {reason}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--tag", default="a")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--memory", type=float, default=0.30)
    parser.add_argument("--compare", action="store_true")
    arguments = parser.parse_args()
    if arguments.compare:
        return _compare(arguments.count, arguments.max_tokens)
    if not arguments.backend:
        parser.error("--backend or --compare")
    _generate(
        arguments.backend,
        arguments.tag,
        arguments.count,
        arguments.max_tokens,
        arguments.memory,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
