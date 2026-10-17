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

Today's vLLM seats no compressed-tensors MXFP4 bank for that architecture
(its loader asks for a `w13_weight_scale` the layer never declared), so a
snapshot whose experts are `mxfp4-pack-quantized` is first read out to bf16
(each e2m1 nibble pair, low nibble first, times its e8m0 exponent: exact in
bf16) into a scratch copy beside OUT, served unquantized, and removed.
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parity_ref  # noqa: E402
import shutil
import tempfile
import time

import numpy as np

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def unpacked(snapshot, into):
    """`snapshot` with each `*.weight_packed` MXFP4 bank read out to a bf16
    `*.weight`, its quantization recipe dropped, written to `into`."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    lut = torch.tensor(E2M1)
    index = json.load(open(os.path.join(snapshot, "model.safetensors.index.json")))
    weight_map = {}
    for shard in sorted(set(index["weight_map"].values())):
        out = {}
        with safe_open(os.path.join(snapshot, shard), "pt") as held:
            names = set(held.keys())
            for name in sorted(names):
                if name.endswith(".weight_scale") and name[: -len("_scale")] + "_packed" in names:
                    continue
                t = held.get_tensor(name)
                if name.endswith(".weight_packed"):
                    stem = name[: -len(".weight_packed")]
                    exps = held.get_tensor(stem + ".weight_scale").to(torch.int32)
                    codes = torch.stack([(t & 0xF).long(), (t >> 4).long()], dim=-1).reshape(t.shape[0], -1)
                    scale = torch.pow(2.0, (exps - 127).float()).repeat_interleave(32, dim=-1)
                    name = stem + ".weight"
                    t = (lut[codes] * scale[:, : codes.shape[1]]).to(torch.bfloat16)
                out[name] = t.contiguous()
                weight_map[name] = shard
        save_file(out, os.path.join(into, shard), metadata={"format": "pt"})
    json.dump({"metadata": {}, "weight_map": weight_map},
              open(os.path.join(into, "model.safetensors.index.json"), "w"))
    for name in os.listdir(snapshot):
        held = os.path.join(snapshot, name)
        if os.path.isfile(held) and not name.endswith(".safetensors") and name != "model.safetensors.index.json":
            shutil.copy(held, os.path.join(into, name))
    config = json.load(open(os.path.join(into, "config.json")))
    config.pop("quantization_config", None)
    if "text_config" in config:
        config["text_config"].pop("quantization_config", None)
    json.dump(config, open(os.path.join(into, "config.json"), "w"), indent=1)

def dense(row, vocab):
    out = np.full(vocab, -np.inf, dtype=np.float32)
    for token, lp in row.items():
        value = lp.logprob if hasattr(lp, "logprob") else lp
        if token < vocab and math.isfinite(value):
            out[token] = value
    return out

def main():
    ap = parity_ref.parser()
    ap.add_argument("--memory", type=float, default=0.15)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from vllm import LLM, SamplingParams

    stated = json.load(open(os.path.join(args.snapshot, "config.json")))
    vocab = (stated.get("text_config") or stated)["vocab_size"]
    scratch = None
    recipe = (stated.get("text_config") or stated).get("quantization_config") or {}
    if recipe.get("format") == "mxfp4-pack-quantized":
        scratch = tempfile.mkdtemp(prefix="kimi_bf16_", dir=args.out)
        unpacked(args.snapshot, scratch)
        args.snapshot = scratch
        stated = json.load(open(os.path.join(args.snapshot, "config.json")))
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

    probes = parity_ref.probes(args, lambda text: tokenizer.encode(text, add_special_tokens=False))

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
        parity_ref.dump(args, name, ids, np.stack(tf), gen_rows, produced, started, tokenizer.decode)

    if scratch is not None:
        shutil.rmtree(scratch, ignore_errors=True)

if __name__ == "__main__":
    main()
