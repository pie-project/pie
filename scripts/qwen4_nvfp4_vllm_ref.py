"""vLLM's reading of a modelopt NVFP4 Qwen 3.8 Flash Next snapshot, dumped in
the parity battery's form under its own tag, so pie's W4A16 run of the same
NVFP4 planes can be read against a server that runs them.

    python3 scripts/qwen4_nvfp4_vllm_ref.py <nvfp4-snapshot> OUT [--steps 16] [--tag vllm] \
        [--moe-backend marlin]

Reads `OUT/probes.json` (written by `scripts/qwen4_parity_ref.py`) and writes,
per probe NAME, `OUT/NAME.<tag>.tf.f32`, `OUT/NAME.<tag>.gen.f32` and
`OUT/NAME.<tag>.json`, as every reference does. The rows are vLLM's raw logits
(`logprobs_mode = raw_logits`, the whole vocabulary): the teacher-forced row
of a prompt position is the prompt logits of the next one, the last the first
sampled step's. vLLM's default NVFP4 MoE quantizes the activations to e2m1
under each expert's `input_scale` (W4A4), which pie's W4A16 product does not,
so the two agree as two roundings of one model rather than bit for bit;
`--moe-backend marlin` runs vLLM's W4A16 product instead.
"""

import argparse
import json
import os
import time

import numpy as np
from vllm import LLM, SamplingParams


def rows_of(flat, vocab):
    """Each position of a FlatLogprobs as a dense `[vocab]` float32 row."""
    rows = []
    for start, end in zip(flat.start_indices, flat.end_indices):
        if start == end:
            rows.append(None)
            continue
        row = np.full(vocab, -np.inf, dtype=np.float32)
        row[np.asarray(flat.token_ids[start:end])] = np.asarray(flat.logprobs[start:end], dtype=np.float32)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--tag", default="vllm")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.15)
    ap.add_argument("--moe-backend", default="auto")
    args = ap.parse_args()
    probes = json.load(open(os.path.join(args.out, "probes.json")))["probes"]

    llm = LLM(
        model=args.snapshot,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=1024,
        max_logprobs=-1,
        logprobs_mode="raw_logits",
        enforce_eager=True,
        trust_remote_code=True,
        kernel_config={"moe_backend": args.moe_backend},
    )
    vocab = llm.llm_engine.model_config.get_vocab_size()
    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.steps + 1,
        logprobs=-1,
        prompt_logprobs=-1,
        flat_logprobs=True,
        detokenize=False,
    )
    for probe in probes:
        name, ids = probe["name"], probe["ids"]
        started = time.time()
        out = llm.generate([{"prompt_token_ids": ids}], params, use_tqdm=False)[0]
        prompt = rows_of(out.prompt_logprobs, vocab)
        sampled = rows_of(out.outputs[0].logprobs, vocab)
        produced = list(out.outputs[0].token_ids)
        tf = np.stack(prompt[1:] + [sampled[0]])
        gen_rows = np.stack(sampled[: args.steps + 1])
        tf.tofile(os.path.join(args.out, f"{name}.{args.tag}.tf.f32"))
        gen_rows.tofile(os.path.join(args.out, f"{name}.{args.tag}.gen.f32"))
        json.dump(
            {"ids": ids, "argmax": [int(a) for a in tf.argmax(-1)], "gen": produced[: args.steps], "vocab": vocab},
            open(os.path.join(args.out, f"{name}.{args.tag}.json"), "w"),
        )
        print(f"  {name}: {len(ids)} tokens, gen={produced[:12]}  ({time.time() - started:.1f}s)")


if __name__ == "__main__":
    main()
