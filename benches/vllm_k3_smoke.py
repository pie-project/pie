#!/usr/bin/env python3
"""Smoke-test that vLLM can actually run the shrunken Kimi-K3 at TP=2."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# TP>1 needs the worker processes; the in-process engine is only for TP=1 hooks.
if os.environ.get("PIE_VLLM_INPROC"):
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from vllm_kimi_k3_register import register  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/root/models/k3-mini")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-tokens", type=int, default=8)
    args = ap.parse_args()

    register()

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=True,
    )
    out = llm.generate([args.prompt],
                       SamplingParams(temperature=0, max_tokens=args.max_tokens))
    print("=== OK ===")
    print("prompt :", repr(out[0].prompt))
    print("output :", repr(out[0].outputs[0].text))
    print("ids    :", out[0].outputs[0].token_ids)


if __name__ == "__main__":
    main()
