"""Generate constrained JSON instances for JSONSchemaBench schemas.

Content and tokenization are separated on purpose. This script produces the
JSON *text* a real model emits under a real grammar constraint; a second stage
replays that text through several tokenizers. Replaying text is faithful
because the grammar state a matcher reaches is fully determined by the bytes
consumed, so each tokenizer sees exactly the states it would have visited had
the model used it.

Generation is batched, sampled rather than greedy, and long enough to reach
deep nesting, since truncating at a few dozen tokens biases the recorded states
towards object openings.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--configs", nargs="+", default=None)
    parser.add_argument("--schemas-per-config", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("results/jsonschemabench-instances.json")
    )
    args = parser.parse_args()

    import xgrammar as xgr
    from datasets import get_dataset_config_names, load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    configs = args.configs or get_dataset_config_names("epfl-dlab/JSONSchemaBench")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16
    ).cuda()
    model.eval()
    info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    compiler = xgr.GrammarCompiler(info, cache_enabled=True)
    vocab_size = info.vocab_size

    tasks: list[dict] = []
    declared = 0
    compile_seconds = 0.0
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
            started = time.perf_counter()
            try:
                compiled = compiler.compile_json_schema(schema)
            except Exception:  # noqa: BLE001
                continue
            compile_seconds += time.perf_counter() - started
            tasks.append({"config": config, "schema": schema, "compiled": compiled})
        print(f"  {config}: {len(tasks)} compiled so far / {declared} declared")

    print(
        f"\ncompiled {len(tasks)}/{declared} schemas "
        f"({len(tasks)/declared*100:.1f}%) in {compile_seconds:.1f}s"
    )

    results: list[dict] = []
    for start in range(0, len(tasks), args.batch_size):
        group = tasks[start : start + args.batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": "Produce one JSON value matching this schema. "
                        f"Reply with JSON only.\n{task['schema'][:3000]}",
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for task in group
        ]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        matchers = [xgr.GrammarMatcher(task["compiled"]) for task in group]
        batch_matcher = xgr.BatchGrammarMatcher(max_threads=8)
        mask = xgr.allocate_token_bitmask(len(group), vocab_size)
        emitted = [[] for _ in group]
        alive = [True] * len(group)

        past = None
        current = encoded.input_ids
        attention = encoded.attention_mask
        with torch.inference_mode():
            for _ in range(args.max_new_tokens):
                output = model(
                    current,
                    attention_mask=attention,
                    past_key_values=past,
                    use_cache=True,
                )
                past = output.past_key_values
                logits = output.logits[:, -1, :vocab_size].float()

                indices = [i for i, live in enumerate(alive) if live]
                if not indices:
                    break
                batch_matcher.batch_fill_next_token_bitmask(
                    [matchers[i] for i in indices], mask, list(range(len(indices)))
                )
                bits = np.unpackbits(
                    mask.numpy().view(np.uint8), axis=-1, bitorder="little"
                )[:, :vocab_size]
                blocked = torch.from_numpy(
                    np.where(bits[: len(indices)].astype(bool), 0.0, -np.inf)
                ).cuda()
                selected = logits[indices] + blocked
                probs = torch.softmax(selected / args.temperature, dim=-1)
                ordered, order = torch.sort(probs, descending=True, dim=-1)
                cumulative = ordered.cumsum(-1)
                ordered = ordered.masked_fill(
                    (cumulative - ordered) > args.top_p, 0.0
                )
                ordered /= ordered.sum(-1, keepdim=True)
                choice = order.gather(-1, torch.multinomial(ordered, 1))

                tokens = torch.zeros(
                    len(group), 1, dtype=torch.long, device="cuda"
                )
                for position, i in enumerate(indices):
                    token = int(choice[position].item())
                    tokens[i, 0] = token
                    if not matchers[i].accept_token(token):
                        alive[i] = False
                        continue
                    emitted[i].append(token)
                    if matchers[i].is_terminated():
                        alive[i] = False
                current = tokens
                attention = torch.cat(
                    [attention, torch.ones_like(tokens)], dim=1
                )

        for task, tokens in zip(group, emitted, strict=True):
            results.append(
                {
                    "config": task["config"],
                    "schema": task["schema"],
                    "text": tokenizer.decode(tokens, skip_special_tokens=True),
                }
            )
        print(
            f"  generated {len(results)}/{len(tasks)} "
            f"(mean {np.mean([len(t) for t in emitted]):.0f} tokens)"
        )

    lengths = [len(item["text"]) for item in results]
    print(
        f"\ninstances: {len(results)}, mean {np.mean(lengths):.0f} bytes, "
        f"median {int(np.median(lengths))}, max {max(lengths)}"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "model": args.model,
                "schemas_declared": declared,
                "schemas_compiled": len(tasks),
                "compile_seconds": compile_seconds,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
                "instances": results,
            }
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
