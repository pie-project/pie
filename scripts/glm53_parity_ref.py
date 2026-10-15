"""GLM-5.3 reference logits, from `transformers`' own `glm_moe_dsa`
(`GlmMoeDsaForCausalLM`), so a pie reading of the same snapshot can be read
against the official implementation.

    python3 scripts/glm53_parity_ref.py <snapshot-dir> OUT [--steps 16] [--probes probes.json]

Writes `OUT/probes.json` (`{"probes": [{"name", "ids"}]}`; a stated file is
used instead) and, per probe NAME, the three files every reference writes:

    OUT/NAME.ref.tf.f32   float32 [len(ids), vocab]  teacher-forced logits at every position
    OUT/NAME.ref.gen.f32  float32 [steps + 1, vocab] logits at the last prompt position and
                                                       after each greedy step
    OUT/NAME.ref.json     {"ids", "argmax", "gen", "vocab"}

`scripts/dsv4_mini_parity_compare.py OUT --a pie --b ref` reads the pie side
(`crates/engine-cuda/tests/engine_cuda/a_family_is_read_against_its_reference.rs`)
against these.

The checkpoint is FP8 (e4m3, 128x128 blocks, `*_scale_inv`); each weight is
dequantized here as the importer does (f32 product, rounded to bf16), the
experts fused into the `[experts, 2·inter, hidden]` banks the module holds,
and the model run in bf16. `--index-topk` overrides the indexer's budget, so
a short probe exercises the selection its shared layers borrow.
"""

import argparse
import glob
import json
import os
import re
import time

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaForCausalLM

PROMPTS = [
    ("capital", "The capital of France is"),
    ("haiku", "Write a haiku about the sea."),
    ("code", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"),
    ("story", "Once upon a time, in a small village by the river,"),
    (
        "chat",
        "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>What is 17 times 23?<|assistant|><think>",
    ),
]

def dequantized(weight, scale):
    """`weight` (e4m3) times its block factors, in f32, rounded to bf16."""
    rows, cols = weight.shape
    block = 128
    s = scale.float().repeat_interleave(block, 0)[:rows].repeat_interleave(block, 1)[:, :cols]
    return (weight.float() * s).to(torch.bfloat16)

def read_checkpoint(snapshot):
    raw = {}
    for path in sorted(glob.glob(os.path.join(snapshot, "*.safetensors"))):
        with safe_open(path, "pt") as f:
            for k in f.keys():
                raw[k] = f.get_tensor(k)
    out = {}
    for k, v in raw.items():
        if k.endswith("_scale_inv"):
            continue
        if v.dtype == torch.float8_e4m3fn:
            v = dequantized(v, raw[k + "_scale_inv"])
        out[k] = v
    experts = {}
    for k in list(out):
        m = re.match(r"(model\.layers\.\d+\.mlp\.experts)\.(\d+)\.(gate|up|down)_proj\.weight", k)
        if m:
            experts.setdefault(m.group(1), {})[(int(m.group(2)), m.group(3))] = out.pop(k)
    for prefix, parts in experts.items():
        n = 1 + max(e for e, _ in parts)
        out[prefix + ".gate_up_proj"] = torch.stack(
            [torch.cat([parts[(e, "gate")], parts[(e, "up")]], 0) for e in range(n)]
        )
        out[prefix + ".down_proj"] = torch.stack([parts[(e, "down")] for e in range(n)])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--probes", default=None)
    ap.add_argument("--index-topk", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.snapshot)
    config = AutoConfig.from_pretrained(args.snapshot)
    if hasattr(config, "quantization_config"):
        del config.quantization_config
    if args.index_topk is not None:
        config.index_topk = args.index_topk
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    with torch.device(args.device):
        model = GlmMoeDsaForCausalLM._from_config(config, dtype=torch.bfloat16)
    state = read_checkpoint(args.snapshot)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"missing {missing}\nunexpected {unexpected}")
    for name, module in model.named_modules():
        if name.endswith("mlp.gate"):
            module.e_score_correction_bias.data = module.e_score_correction_bias.data.float()
        if name.endswith("indexer.weights_proj"):
            module.float()
    model.eval()

    if args.probes:
        probes = json.load(open(args.probes))["probes"]
    else:
        probes = []
        for name, text in PROMPTS:
            ids = tokenizer.encode(text, add_special_tokens=False)
            probes.append({"name": name, "ids": ids})
        json.dump({"probes": probes}, open(os.path.join(args.out, "probes.json"), "w"), indent=1)

    for probe in probes:
        name, ids = probe["name"], probe["ids"]
        started = time.time()
        input_ids = torch.tensor([ids], device=args.device)
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=True)
            tf = out.logits[0].float().cpu().numpy()
            past = out.past_key_values
            gen_rows = [tf[-1].copy()]
            produced = []
            last = int(tf[-1].argmax())
            for _ in range(args.steps):
                produced.append(last)
                step = model(
                    input_ids=torch.tensor([[last]], device=args.device),
                    past_key_values=past,
                    use_cache=True,
                )
                past = step.past_key_values
                row = step.logits[0, -1].float().cpu().numpy()
                gen_rows.append(row)
                last = int(row.argmax())
        vocab = tf.shape[-1]
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
