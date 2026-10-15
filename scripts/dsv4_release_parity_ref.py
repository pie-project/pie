"""DeepSeek-V4 reference logits for a release-layout snapshot (V4-Pro's or
V4-Flash's `mini-l5-e16`), from `transformers`' own `deepseek_v4`, so a pie
reading of the same snapshot can be read against it. Activations stay bf16 (or
f32): the release's FP8 activation quantization (`act_quant` before every fp8
or fp4 GEMM, and the KV and indexer quant-dequant) is not applied, so this is
the reference for the deployments that serve dequantized weights at bf16
activations.

    python3 scripts/dsv4_release_parity_ref.py <snapshot-dir> OUT [--steps 16] [--probes probes.json]
        [--dtype float32] [--act-quant] [--tag ref]

Writes `OUT/probes.json` (`{"probes": [{"name", "ids"}]}`; a stated file is
used instead) and, per probe NAME, the three files every reference writes:

    OUT/NAME.ref.tf.f32   float32 [len(ids), vocab]  teacher-forced logits at every position
    OUT/NAME.ref.gen.f32  float32 [steps + 1, vocab] logits at the last prompt position and
                                                       after each greedy step
    OUT/NAME.ref.json     {"ids", "argmax", "gen", "vocab"}

`scripts/dsv4_mini_parity_compare.py OUT --a pie --b ref` reads the pie side
(`crates/engine-cuda/tests/engine_cuda/a_family_is_read_against_its_reference.rs`)
against these.

`transformers` reads bf16 planes, and the release stores its routed experts
as MXFP4 codes (two e2m1 nibbles a byte, the low one first) beside `.scale`
e8m0 exponents, one a 32-wide block, and V4-Flash's release its dense planes as
fp8 beside `.scale` e8m0 exponents, one a 128-square block. Each is written out
to bf16, which holds every such value exactly, into `OUT/.bf16-view`, the configuration's
`quantization_config` and `expert_dtype` taken off. The reference runs in
float32 by default: its hyper-connection streams are f32 and its bf16 arm
hands them to bf16 projections unconverted.

`--act-quant` reads the release at DeepSeek's own precision
(`inference/model.py`'s `linear`): every projection the release stores fp8
or fp4 (attention's `wq_a`, `wq_b`, `wkv`, `wo_b`, the indexer's `wq_b`,
the shared and routed experts) quantizes its bf16 activations first, e4m3
under one power-of-two (ue8m0) scale a 1x128 block (`act_quant`), and lands
its product at bf16; a routed expert weighs its activations by the route
before its down leg quantizes them, and its routes and the shared expert
sum in f32. Those products are exact (each code times its scale), so the
quantized activations' f32 matmul is the reference's fp8 and fp4 GEMMs. The
reference's quant-dequant simulations elsewhere (the window KV's fp8 1x64,
the indexer's fp4 after a Hadamard rotation) are not applied, as pie applies
none of them.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer, DeepseekV4ForCausalLM

PROMPTS = [
    ("capital", "The capital of France is"),
    ("haiku", "Write a haiku about the sea."),
    ("code", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"),
    ("story", "Once upon a time, in a small village by the river,"),
]

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])

def read_out(codes, scale):
    codes = codes.view(torch.uint8)
    pairs = torch.stack([codes & 0x0F, (codes >> 4) & 0x0F], dim=-1).reshape(codes.shape[0], -1)
    values = E2M1[pairs.long()].reshape(codes.shape[0], -1, 32)
    values = values * torch.pow(2.0, scale.view(torch.uint8).float() - 127.0).unsqueeze(-1)
    return values.reshape(codes.shape[0], -1).to(torch.bfloat16)

def read_out_fp8(codes, scale):
    """An fp8 plane under its `.scale` e8m0 exponents, one a square block
    (128 wide in V4): each code times two to its block's exponent, which
    bf16 holds exactly."""
    rows, cols = codes.shape
    group = max(-(-rows // scale.shape[0]), -(-cols // scale.shape[1]))
    factors = torch.pow(2.0, scale.view(torch.uint8).float() - 127.0)
    factors = factors.repeat_interleave(group, 0)[:rows].repeat_interleave(group, 1)[:, :cols]
    return (codes.float() * factors).to(torch.bfloat16)

def act_quant(x, block=128):
    """DeepSeek's `act_quant` with `scale_fmt = ue8m0`, read back: bf16
    activations in 1x`block` blocks, each scale `2^ceil(log2(amax / 448))` of
    its block (the amax floored at 1e-4), the codes e4m3."""
    shape = x.shape
    x = x.to(torch.bfloat16).float().reshape(-1, shape[-1] // block, block)
    amax = x.abs().amax(-1, keepdim=True).clamp(min=1e-4)
    s = torch.pow(2.0, torch.ceil(torch.log2(amax * (1.0 / 448.0))))
    q = (x / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
    return (q * s).reshape(shape)


def quantized(linear):
    """`linear` reading its activations as `act_quant` lands them and
    landing its product at bf16."""
    def forward(x):
        return F.linear(act_quant(x), linear.weight).to(torch.bfloat16).to(x.dtype)
    linear.forward = forward


def experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """`inference/model.py`'s `Expert` over the routes: gate and up of the
    quantized activations at bf16, the SwiGLU in f32 times the route's
    weight, then the down leg of those quantized, summed in f32."""
    final = torch.zeros(hidden_states.shape, dtype=torch.float32, device=hidden_states.device)
    for e in range(self.num_experts):
        token_idx, top_k_pos = torch.where(top_k_index == e)
        if token_idx.numel() == 0:
            continue
        x = act_quant(hidden_states[token_idx])
        gate_up = F.linear(x, self.gate_up_proj[e]).to(torch.bfloat16).float()
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        act = F.silu(gate) * up * top_k_weights[token_idx, top_k_pos, None].float()
        down = F.linear(act_quant(act), self.down_proj[e]).to(torch.bfloat16).float()
        final.index_add_(0, token_idx, down)
    return final


def moe_forward(self, hidden_states, input_ids=None):
    batch, seq_len, hidden_dim = hidden_states.shape
    flat = hidden_states.view(-1, hidden_dim)
    if self.is_hash:
        _, weights, indices = self.gate(hidden_states, input_ids)
    else:
        _, weights, indices = self.gate(hidden_states)
    routed = experts_forward(self.experts, flat, indices, weights)
    shared = self.shared_experts(hidden_states).view(-1, hidden_dim).float()
    return (routed + shared).to(torch.bfloat16).to(hidden_states.dtype).view(batch, seq_len, hidden_dim)


def at_release_precision(model, parts=("dense", "experts")):
    """Every fp8 (`dense`) and fp4 (`experts`) projection of `model` read as
    the release reads it."""
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as v4

    for module in model.modules():
        if "dense" not in parts:
            pass
        elif isinstance(module, v4.DeepseekV4Attention):
            for linear in (module.q_a_proj, module.q_b_proj, module.kv_proj, module.o_b_proj):
                quantized(linear)
        elif isinstance(module, v4.DeepseekV4Indexer):
            quantized(module.q_b_proj)
        elif isinstance(module, v4.DeepseekV4MLP):
            for linear in (module.gate_proj, module.up_proj, module.down_proj):
                quantized(linear)
        if "experts" in parts and isinstance(module, v4.DeepseekV4SparseMoeBlock):
            module.forward = moe_forward.__get__(module)


def bf16_view(snapshot, view):
    if os.path.exists(os.path.join(view, "model.safetensors")):
        return
    os.makedirs(view, exist_ok=True)
    index = json.load(open(os.path.join(snapshot, "model.safetensors.index.json")))["weight_map"]
    shards = {}

    def get(name):
        shard = index[name]
        if shard not in shards:
            shards[shard] = safe_open(os.path.join(snapshot, shard), "pt")
        return shards[shard].get_tensor(name)

    state = {}
    for name in index:
        if name.endswith(".scale") and name[: -len(".scale")] + ".weight" in index:
            continue
        t = get(name)
        if ".experts." in name and t.dtype in (torch.int8, torch.uint8):
            t = read_out(t, get(name[: -len(".weight")] + ".scale"))
        elif t.dtype == torch.float8_e4m3fn:
            t = read_out_fp8(t, get(name[: -len(".weight")] + ".scale"))
        state[name] = t.contiguous()
    save_file(state, os.path.join(view, "model.safetensors"), metadata={"format": "pt"})
    config = json.load(open(os.path.join(snapshot, "config.json")))
    config.pop("quantization_config", None)
    config.pop("expert_dtype", None)
    json.dump(config, open(os.path.join(view, "config.json"), "w"), indent=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--probes", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--act-quant", default=None, const="dense,experts", nargs="?")
    ap.add_argument("--tag", default="ref")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.snapshot)
    view = os.path.join(args.out, ".bf16-view")
    bf16_view(args.snapshot, view)
    model = DeepseekV4ForCausalLM.from_pretrained(view, dtype=getattr(torch, args.dtype))
    model.to(args.device).eval()
    if args.act_quant:
        at_release_precision(model, args.act_quant.split(","))

    if args.probes:
        probes = json.load(open(args.probes))["probes"]
    else:
        bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
        probes = []
        for name, text in PROMPTS:
            probes.append({"name": name, "ids": bos + tokenizer.encode(text, add_special_tokens=False)})
        json.dump({"probes": probes}, open(os.path.join(args.out, "probes.json"), "w"), indent=1)

    for probe in probes:
        name, ids = probe["name"], probe["ids"]
        started = time.time()
        with torch.no_grad():
            out = model(input_ids=torch.tensor([ids], device=args.device), use_cache=True)
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
        tf.astype(np.float32).tofile(os.path.join(args.out, f"{name}.{args.tag}.tf.f32"))
        np.stack(gen_rows).astype(np.float32).tofile(os.path.join(args.out, f"{name}.{args.tag}.gen.f32"))
        json.dump(
            {"ids": ids, "argmax": [int(a) for a in tf.argmax(-1)], "gen": produced, "vocab": vocab},
            open(os.path.join(args.out, f"{name}.{args.tag}.json"), "w"),
        )
        print(f"  {name}: {len(ids)} tokens, vocab {vocab}, gen={produced[:12]}  ({time.time() - started:.1f}s)")

if __name__ == "__main__":
    main()
