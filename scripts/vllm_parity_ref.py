"""vLLM's reading of a probe battery, for checkpoints transformers cannot
dequantize (modelopt FP8 / NVFP4, block-FP8 releases): per probe, the
teacher-forced top-k log-probabilities at every prompt position and the
greedy continuation with its top-k at every step.

    /root/venvs/vllm/bin/python scripts/vllm_parity_ref.py <snapshot> OUT/probes.json OUT \
        [--steps 24] [--top 20] [--kv auto|fp8] [--tag vllm] [--mem 0.3]

Writes `OUT/NAME.<tag>.json`: {"ids", "argmax", "gen", "tf_top", "gen_top"},
`*_top` a list per position of [[token, logprob], ...]. `scripts/vllm_parity_compare.py`
grades pie's dump (`a_family_is_read_against_its_reference.rs`) against it.
"""

import argparse
import json
import os

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def tops(entry):
    if entry is None:
        return []
    return sorted(([int(t), float(lp.logprob)] for t, lp in entry.items()), key=lambda p: -p[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("probes")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--kv", default="auto")
    ap.add_argument("--tag", default="vllm")
    ap.add_argument("--mem", type=float, default=0.3)
    ap.add_argument("--max-len", type=int, default=4096)
    args = ap.parse_args()
    probes = json.load(open(args.probes))["probes"]
    llm = LLM(
        args.snapshot,
        kv_cache_dtype=args.kv,
        gpu_memory_utilization=args.mem,
        max_model_len=args.max_len,
        max_logprobs=args.top,
        enable_prefix_caching=False,
        seed=0,
    )
    params = SamplingParams(temperature=0.0, max_tokens=args.steps, logprobs=args.top, prompt_logprobs=args.top)
    outs = llm.generate([TokensPrompt(prompt_token_ids=p["ids"]) for p in probes], params)
    for probe, out in zip(probes, outs):
        # prompt_logprobs[i] is the distribution of token i given those before:
        # the row pie reads out at position i - 1.
        tf_top = [tops(e) for e in out.prompt_logprobs[1:]]
        gen = list(out.outputs[0].token_ids)
        gen_top = [tops(e) for e in out.outputs[0].logprobs]
        json.dump(
            {
                "ids": probe["ids"],
                "argmax": [t[0][0] if t else -1 for t in tf_top],
                "gen": gen,
                "tf_top": tf_top,
                "gen_top": gen_top,
            },
            open(os.path.join(args.out, f"{probe['name']}.{args.tag}.json"), "w"),
        )
        print(f"{probe['name']}: gen={gen[:12]}")


if __name__ == "__main__":
    main()
