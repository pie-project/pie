"""GLM-5.3 reference logits, from `transformers`' own `glm_moe_dsa`
(`GlmMoeDsaForCausalLM`), so a pie reading of the same snapshot can be read
against the official implementation.

    python3 scripts/glm53_parity_ref.py <snapshot-dir> OUT [--steps 16] [--probes probes.json]

Writes the dump `scripts/parity_ref.py` describes.

The checkpoint is FP8 (e4m3, 128x128 blocks, `*_scale_inv`); each weight is
dequantized here as the importer does (f32 product, rounded to bf16), the
experts fused into the `[experts, 2·inter, hidden]` banks the module holds,
and the model run in bf16. `--index-topk` overrides the indexer's budget, so
a short probe exercises the selection its shared layers borrow.
"""

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parity_ref  # noqa: E402
import re
import time

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaForCausalLM

PROMPTS = [
    *parity_ref.PROMPTS,
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
    ap = parity_ref.parser()
    ap.add_argument("--index-topk", type=int, default=None)
    args = ap.parse_args()
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
    missing, unexpected = model.load_state_dict(read_checkpoint(args.snapshot), strict=False)
    if missing or unexpected:
        raise SystemExit(f"missing {missing}\nunexpected {unexpected}")
    for name, module in model.named_modules():
        if name.endswith("mlp.gate"):
            module.e_score_correction_bias.data = module.e_score_correction_bias.data.float()
        if name.endswith("indexer.weights_proj"):
            module.float()
    model.eval()
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)  # noqa: E731
    for probe in parity_ref.probes(args, encode, PROMPTS):
        started = time.time()
        tf, rows, gen = parity_ref.greedy_cached(model, probe["ids"], args.steps, args.device)
        parity_ref.dump(args, probe["name"], probe["ids"], tf, rows, gen, started, tokenizer.decode)


if __name__ == "__main__":
    main()
