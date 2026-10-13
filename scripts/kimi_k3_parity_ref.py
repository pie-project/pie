"""Kimi-K3 reference rows, from vLLM's own `kimi_k3`, so a pie reading of the
same snapshot can be read against it.

    python3 scripts/kimi_k3_parity_ref.py <snapshot-dir> OUT [--steps 16] [--probes probes.json]

Writes `OUT/probes.json` (`{"probes": [{"name", "ids"}]}`; a stated file is
used instead) and, per probe NAME, the three files every reference writes:

    OUT/NAME.ref.tf.f32   float32 [len(ids), vocab]  teacher-forced rows at every position
    OUT/NAME.ref.gen.f32  float32 [steps + 1, vocab] rows at the last prompt position and
                                                       after each greedy step
    OUT/NAME.ref.json     {"ids", "argmax", "gen", "vocab"}

`scripts/dsv4_mini_parity_compare.py OUT --a pie --b ref` reads the pie side
(`crates/engine-cuda/tests/engine_cuda/a_family_is_read_against_its_reference.rs`)
against these.

vLLM hands out log-probabilities, not logits: each row is the logits less
their log-sum-exp, which moves neither an argmax, a KL nor a correlation.
The teacher-forced rows are the prompt's (`prompt_logprobs`), row 0 left at
-inf where vLLM states none; the generated rows are the greedy samples'. A
text-only cut (`KimiK3ForCausalLM`, its trunk under `model.` and its
configuration flat) is served as vLLM's `KimiLinearForCausalLM`, through a
view of the snapshot whose configuration says so, and whose quantization
recipe leaves the latent MoE's bf16 down and up projections unquantized as
the checkpoint stores them (vLLM would otherwise seat them as MXFP4).
"""

import argparse
import json
import math
import os
import tempfile
import time

import numpy as np

PROMPTS = [
    ("capital", "The capital of France is"),
    ("haiku", "Write a haiku about the sea."),
    ("code", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"),
    ("story", "Once upon a time, in a small village by the river,"),
]

def dense(row, vocab):
    out = np.full(vocab, -np.inf, dtype=np.float32)
    for token, lp in row.items():
        value = lp.logprob if hasattr(lp, "logprob") else lp
        if token < vocab and math.isfinite(value):
            out[token] = value
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--probes", default=None)
    ap.add_argument("--memory", type=float, default=0.15)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from vllm import LLM, SamplingParams

    stated = json.load(open(os.path.join(args.snapshot, "config.json")))
    vocab = (stated.get("text_config") or stated)["vocab_size"]
    served = args.snapshot
    if "text_config" not in stated:
        served = tempfile.mkdtemp(prefix="kimi_text_")
        for name in os.listdir(args.snapshot):
            if name != "config.json":
                os.symlink(os.path.join(os.path.abspath(args.snapshot), name), os.path.join(served, name))
        stated.update(model_type="kimi_linear", architectures=["KimiLinearForCausalLM"])
        recipe = stated.get("quantization_config")
        if recipe is not None:
            recipe["ignore"] = list(recipe.get("ignore", [])) + ["re:.*routed_expert_(down|up)_proj.*"]
        json.dump(stated, open(os.path.join(served, "config.json"), "w"))
    llm = LLM(
        model=served,
        trust_remote_code=True,
        max_model_len=512,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        gpu_memory_utilization=args.memory,
        enforce_eager=True,
        max_logprobs=vocab,
        limit_mm_per_prompt={"image": 0} if "text_config" in stated else None,
    )
    tokenizer = llm.get_tokenizer()

    if args.probes:
        probes = json.load(open(args.probes))["probes"]
    else:
        probes = []
        for name, text in PROMPTS:
            ids = tokenizer.encode(text, add_special_tokens=False)
            probes.append({"name": name, "ids": ids})
        json.dump({"probes": probes}, open(os.path.join(args.out, "probes.json"), "w"), indent=1)

    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.steps + 1,
        logprobs=vocab,
        prompt_logprobs=vocab,
    )
    for probe in probes:
        name, ids = probe["name"], probe["ids"]
        started = time.time()
        result = llm.generate([{"prompt_token_ids": ids}], params, use_tqdm=False)[0]
        prompt_rows = result.prompt_logprobs
        sample = result.outputs[0]
        # `prompt_logprobs[i]` scores token i given the tokens before it, so the
        # row read after token i is `prompt_logprobs[i + 1]`, and after the last
        # prompt token it is the first sample's.
        tf = [dense(prompt_rows[i + 1], vocab) for i in range(len(ids) - 1)]
        gen_rows = [dense(row, vocab) for row in sample.logprobs]
        tf.append(gen_rows[0])
        produced = list(sample.token_ids[: args.steps])
        tf = np.stack(tf)
        tf.astype(np.float32).tofile(os.path.join(args.out, f"{name}.ref.tf.f32"))
        np.stack(gen_rows).astype(np.float32).tofile(os.path.join(args.out, f"{name}.ref.gen.f32"))
        json.dump(
            {"ids": ids, "argmax": [int(a) for a in tf.argmax(-1)], "gen": produced, "vocab": vocab},
            open(os.path.join(args.out, f"{name}.ref.json"), "w"),
        )
        print(f"  {name}: {len(ids)} tokens, vocab {vocab}, gen={produced[:12]}  ({time.time() - started:.1f}s)")
        print(f"      {tokenizer.decode(produced)!r}")

if __name__ == "__main__":
    main()
