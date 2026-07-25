"""Collect real constrained-decoding state trajectories.

JSONSchemaBench ships schemas but no instances, and a hand-written schema is
not a workload. This runs an actual model under an actual XGrammar constraint
over real benchmark schemas and records, for every decoding step, the set of
tokens the grammar allowed. The resulting rows are what the sampler benchmark
replays, so the width distribution it sees is the one a real request produces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--configs", nargs="+", default=None)
    parser.add_argument("--schemas-per-config", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-rows", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("results/jsonschemabench-rows.npz")
    )
    args = parser.parse_args()

    import xgrammar as xgr
    from datasets import get_dataset_config_names, load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    configs = args.configs or get_dataset_config_names(
        "epfl-dlab/JSONSchemaBench"
    )
    print(f"configs: {configs}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16
    ).cuda()
    model.eval()
    info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    compiler = xgr.GrammarCompiler(info, cache_enabled=True)
    vocab_size = info.vocab_size

    rows: list[np.ndarray] = []
    widths: list[int] = []
    row_schema: list[int] = []
    row_step: list[int] = []
    schema_texts: list[str] = []
    token_sequences: list[list[int]] = []
    declared = accepted = generated = 0
    rng = np.random.default_rng(args.seed)

    for config in configs:
        dataset = load_dataset("epfl-dlab/JSONSchemaBench", config)
        split = "test" if "test" in dataset else list(dataset.keys())[0]
        picks = rng.choice(
            len(dataset[split]),
            size=min(args.schemas_per_config, len(dataset[split])),
            replace=False,
        )
        for index in picks:
            schema = dataset[split][int(index)]["json_schema"]
            declared += 1
            try:
                compiled = compiler.compile_json_schema(schema)
            except Exception:  # noqa: BLE001
                continue
            accepted += 1
            schema_id = len(schema_texts)
            schema_texts.append(schema)
            emitted: list[int] = []

            prompt = tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": "Produce one JSON value matching this "
                        f"schema. Reply with JSON only.\n{schema[:2000]}",
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            matcher = xgr.GrammarMatcher(compiled)
            mask = xgr.allocate_token_bitmask(1, vocab_size)
            past = None
            current = ids
            with torch.inference_mode():
                for step in range(args.max_new_tokens):
                    output = model(current, past_key_values=past, use_cache=True)
                    past = output.past_key_values
                    logits = output.logits[0, -1, :vocab_size].float()

                    matcher.fill_next_token_bitmask(mask)
                    bits = np.unpackbits(
                        mask.numpy().view(np.uint8), bitorder="little"
                    )[:vocab_size]
                    allowed = np.flatnonzero(bits).astype(np.int32)
                    if allowed.size == 0:
                        break
                    if len(rows) < args.max_rows:
                        rows.append(allowed)
                        widths.append(int(allowed.size))
                        row_schema.append(schema_id)
                        row_step.append(step)
                    generated += 1

                    device_mask = torch.from_numpy(
                        np.where(bits.astype(bool), 0.0, -np.inf)
                    ).cuda()
                    token = int(torch.argmax(logits + device_mask).item())
                    if not matcher.accept_token(token):
                        break
                    emitted.append(token)
                    if matcher.is_terminated():
                        break
                    current = torch.tensor([[token]], device="cuda")
            token_sequences.append(emitted)
        print(
            f"  {config}: compiled {accepted}/{declared}, "
            f"{generated} decoding steps so far"
        )

    widths_array = np.asarray(widths, dtype=np.int64)
    print()
    print(f"schemas declared      : {declared}")
    print(f"schemas compiled      : {accepted} ({accepted/declared*100:.1f}%)")
    print(f"decoding steps sampled: {widths_array.size}")
    print(f"median allowed tokens : {int(np.median(widths_array))}")
    print(f"mean allowed tokens   : {widths_array.mean():.0f}")
    for percentile in (50, 75, 90, 95, 99):
        print(
            f"  p{percentile:<3d}: {int(np.percentile(widths_array, percentile))}"
        )
    print(f"steps wider than 8192 : {(widths_array > 8192).mean()*100:.1f}%")
    print(f"steps with one token  : {(widths_array == 1).mean()*100:.1f}%")

    indptr = np.zeros(len(rows) + 1, dtype=np.int32)
    for index, row in enumerate(rows):
        indptr[index + 1] = indptr[index] + row.size
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        indptr=indptr,
        indices=np.concatenate(rows).astype(np.int32),
        widths=widths_array,
        vocab_size=np.asarray([vocab_size], dtype=np.int64),
        schemas_declared=np.asarray([declared], dtype=np.int64),
        schemas_compiled=np.asarray([accepted], dtype=np.int64),
        row_schema=np.asarray(row_schema, dtype=np.int32),
        row_step=np.asarray(row_step, dtype=np.int32),
    )
    sidecar = args.output.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {"schemas": schema_texts, "tokens": token_sequences}
        ),
        encoding="utf-8",
    )
    print(f"wrote {sidecar}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
