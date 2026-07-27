"""Print vLLM's per-step top-k logprobs for the bench's exact prompt.

Token-exact parity is only a meaningful pass/fail signal when the argmax is
decisive. On the shrunken frontier checkpoints (`shrink_checkpoint.py`) the
logit distribution is close to uniform, so a 1e-3 kernel difference can flip a
token. Run this to see the top1-top2 margin at every step before concluding
that a mismatch is a bug.

    /root/.venv/vllm/bin/python benches/logit_margin.py \
        --model /root/models/kimi26-l1 --prompt "The capital of France is" \
        --max-tokens 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BENCH_PROMPT, BENCH_SYSTEM, hf_chat_token_ids_and_counts  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default=BENCH_PROMPT)
    ap.add_argument("--system", default=BENCH_SYSTEM)
    ap.add_argument(
        "--unique-prompts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match the benches' default prompt rendering, which "
             "appends \"(Request #0)\". Both engines must see the "
             "same string or the dumps are not comparable.",
    )
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--gpu-mem-util", type=float, default=0.55)
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--kv-cache-dtype", default="auto")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    prompt = f"{args.prompt} (Request #0)" if args.unique_prompts else args.prompt
    ids, _ = hf_chat_token_ids_and_counts(args.model, args.system, [prompt])
    kwargs = {}
    if args.kv_cache_dtype != "auto":
        kwargs["kv_cache_dtype"] = args.kv_cache_dtype
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=True,
        **kwargs,
    )
    out = llm.generate(
        [TokensPrompt(prompt_token_ids=ids[0])],
        SamplingParams(
            temperature=0, max_tokens=args.max_tokens, logprobs=args.topk
        ),
    )[0].outputs[0]

    print(f"prompt_token_ids ({len(ids[0])}): {ids[0]}")
    print(f"output_token_ids: {list(out.token_ids)}")
    for step, lp in enumerate(out.logprobs or []):
        ranked = sorted(lp.items(), key=lambda kv: kv[1].logprob, reverse=True)
        top = ranked[0][1].logprob
        second = ranked[1][1].logprob if len(ranked) > 1 else float("-inf")
        entries = " ".join(f"{tid}:{v.logprob:.4f}" for tid, v in ranked)
        print(f"step {step:2d} margin={top - second:.5f}  {entries}")


if __name__ == "__main__":
    main()
