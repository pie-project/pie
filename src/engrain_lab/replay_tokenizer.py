"""Replay generated JSON through a tokenizer's grammar and record the rows.

Stage two of the workload pipeline. `generate_instances` produced the JSON text
a real model emits under a real constraint; this replays that text through a
given tokenizer, so the recorded states are the ones that tokenizer would have
visited. Running it for several tokenizers isolates the effect of vocabulary
size and tokenizer family from the effect of the schema.

Rows are stored in whichever form is smaller — the allowed list for a narrow
row, the forbidden list for a wide one — because a wide row's allowed list is
about 98% of the table and carries almost no information.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

WIDE_THRESHOLD = 8192


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--instances",
        type=Path,
        default=Path("results/jsonschemabench-instances.json"),
    )
    parser.add_argument("--max-rows", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import xgrammar as xgr
    from transformers import AutoTokenizer

    payload = json.loads(args.instances.read_text(encoding="utf-8"))
    instances = payload["instances"]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    compiler = xgr.GrammarCompiler(info, cache_enabled=True)
    vocab_size = info.vocab_size
    print(f"tokenizer {args.tokenizer}: vocab {vocab_size}")

    widths: list[int] = []
    configs: list[str] = []
    kinds: list[int] = []
    payloads: list[np.ndarray] = []
    schema_of_row: list[int] = []
    step_of_row: list[int] = []
    schemas: list[str] = []
    token_sequences: list[list[int]] = []
    rng = np.random.default_rng(args.seed)
    declared = compiled_count = 0

    for instance in instances:
        declared += 1
        if not instance["text"].strip():
            continue
        try:
            compiled = compiler.compile_json_schema(instance["schema"])
        except Exception:  # noqa: BLE001
            continue
        compiled_count += 1
        ids = tokenizer(instance["text"], add_special_tokens=False).input_ids
        matcher = xgr.GrammarMatcher(compiled)
        mask = xgr.allocate_token_bitmask(1, vocab_size)
        schema_id = len(schemas)
        schemas.append(instance["schema"])
        emitted: list[int] = []

        for step, token in enumerate(ids):
            matcher.fill_next_token_bitmask(mask)
            bits = np.unpackbits(mask.numpy().view(np.uint8), bitorder="little")
            allowed = bits[:vocab_size].astype(bool)
            width = int(allowed.sum())
            if width == 0:
                break
            widths.append(width)
            configs.append(instance["config"])
            if len(payloads) < args.max_rows and rng.random() < 0.5:
                if width > WIDE_THRESHOLD:
                    kinds.append(1)
                    payloads.append(np.flatnonzero(~allowed).astype(np.int32))
                else:
                    kinds.append(0)
                    payloads.append(np.flatnonzero(allowed).astype(np.int32))
                schema_of_row.append(schema_id)
                step_of_row.append(step)
            if not matcher.accept_token(token):
                break
            emitted.append(int(token))
            if matcher.is_terminated():
                break
        token_sequences.append(emitted)

    widths_array = np.asarray(widths, dtype=np.int64)
    print(f"instances replayed   : {compiled_count}/{declared}")
    print(f"decoding steps       : {widths_array.size}")
    print(f"median allowed tokens: {int(np.median(widths_array))}")
    for percentile in (25, 50, 75, 90, 99):
        print(
            f"  p{percentile:<3d}: {int(np.percentile(widths_array, percentile))}"
        )
    print(f"wide share (> {WIDE_THRESHOLD}) : {(widths_array > WIDE_THRESHOLD).mean()*100:.1f}%")
    print(f"forced single token  : {(widths_array == 1).mean()*100:.1f}%")

    print("\nper split:")
    config_array = np.asarray(configs)
    per_split = {}
    for config in sorted(set(configs)):
        subset = widths_array[config_array == config]
        per_split[config] = {
            "steps": int(subset.size),
            "median": int(np.median(subset)),
            "wide_share": float((subset > WIDE_THRESHOLD).mean()),
        }
        print(
            f"  {config:18s} steps={subset.size:6d} "
            f"median={int(np.median(subset)):7d} "
            f"wide={(subset > WIDE_THRESHOLD).mean()*100:5.1f}%"
        )

    indptr = np.zeros(len(payloads) + 1, dtype=np.int32)
    for index, row in enumerate(payloads):
        indptr[index + 1] = indptr[index] + row.size
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        indptr=indptr,
        payload=(
            np.concatenate(payloads).astype(np.int32)
            if payloads
            else np.zeros(0, dtype=np.int32)
        ),
        kinds=np.asarray(kinds, dtype=np.int8),
        widths=widths_array,
        row_schema=np.asarray(schema_of_row, dtype=np.int32),
        row_step=np.asarray(step_of_row, dtype=np.int32),
        vocab_size=np.asarray([vocab_size], dtype=np.int64),
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "tokenizer": args.tokenizer,
                "vocab_size": vocab_size,
                "instances_replayed": compiled_count,
                "steps": int(widths_array.size),
                "per_split": per_split,
                "schemas": schemas,
                "tokens": token_sequences,
            }
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
