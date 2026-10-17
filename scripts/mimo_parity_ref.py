"""MiMo-V2.6 reference logits, from the checkpoint's own `modeling_mimo_v2.py`
(`MiMoV2ForCausalLM`, eager attention so the sinks are read), its weights read
out to bf16 here, so a pie reading of the same miniature can be read against
the released implementation.

    python3 scripts/mimo_parity_ref.py <snapshot-dir> OUT [--steps 16] [--probes probes.json]

Writes `OUT/probes.json` (`{"probes": [{"name", "ids"}]}`; a stated file is
used instead) and, per probe NAME, the three files every reference writes:

    OUT/NAME.ref.tf.f32   float32 [len(ids), vocab]  teacher-forced logits at every position
    OUT/NAME.ref.gen.f32  float32 [steps + 1, vocab] logits at the last prompt position and
                                                       after each greedy step
    OUT/NAME.ref.json     {"ids", "argmax", "gen", "vocab"}

`scripts/dsv4_mini_parity_compare.py OUT --a pie --b ref` reads the pie side
(`crates/engine-cuda/tests/engine_cuda/a_family_is_read_against_its_reference.rs`)
against these.

What the release stores and the remote code does not read itself:

* fp8 planes beside `weight_scale_inv` 128x128 block factors. The fused
  `qkv_proj` is stored pre-sharded at `num_key_value_heads` chunks, each its
  own heads' [Q | K | V] scaled apart (the last block of a chunk cut short),
  and the remote code splits one [Q | K | V]: each chunk is read out under
  its own factors and the chunks de-interleaved, as sglang's
  `_deinterleave_qkv_shards` and vLLM's `_shard_fp8_qkv_proj` do.
* the routed experts as MXFP4: `.weight` e2m1 nibble pairs (low nibble
  first) beside `.weight_scale` e8m0 exponents, one per 32 columns.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parity_ref  # noqa: E402
import time

import torch
from safetensors import safe_open

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

PROMPTS = [
    ("capital", "The capital of France is"),
    (
        "chat",
        "<|im_start|>system\nYou are terse.<|im_end|><|im_start|>user\nWhat is 17 times 23?<|im_end|>"
        "<|im_start|>assistant\n",
    ),
]

def long_prompt(tokenizer, target):
    """A text past the 128-token window, so the sliding layers drop keys."""
    text = (
        "The history of the river begins in the mountains, where snow melts each spring and "
        "feeds a hundred small streams. They join in the valley below, pass the old mill and "
        "the stone bridge, and turn east toward the plain. "
    )
    ids = []
    k = 0
    while len(ids) < target:
        ids = tokenizer(text * (k + 1) + "In the end, the river", add_special_tokens=False)["input_ids"]
        k += 1
    return ids[-target:]

def block_dequant(w, s, chunks=1):
    """fp8 codes `w` [rows, cols] times their 128x128 factors `s`, the grid
    restarting every `rows / chunks` rows."""
    rows, cols = w.shape
    per = rows // chunks
    out = []
    for c, (wc, sc) in enumerate(zip(w.split(per), s.split(s.shape[0] // chunks))):
        f = sc.repeat_interleave(128, 0)[:per].repeat_interleave(128, 1)[:, :cols]
        out.append(wc.float() * f.float())
    return torch.cat(out)

def mxfp4_dequant(codes, scales):
    lut = torch.tensor(E2M1, dtype=torch.float32)
    lo = lut[(codes & 0xF).long()]
    hi = lut[(codes >> 4).long()]
    vals = torch.stack([lo, hi], dim=-1).reshape(codes.shape[0], -1)
    exp = torch.pow(2.0, scales.float() - 127.0)
    return vals * exp.repeat_interleave(32, 1)

def load_state(snapshot, config):
    files = sorted(f for f in os.listdir(snapshot) if f.endswith(".safetensors"))
    tensors = {}
    for f in files:
        with safe_open(os.path.join(snapshot, f), framework="pt") as h:
            for k in h.keys():
                tensors[k] = h.get_tensor(k)
    shards = config["num_key_value_heads"]
    state = {}
    for name, t in tensors.items():
        if name.endswith("_scale_inv") or name.endswith(".weight_scale"):
            continue
        if t.dtype == torch.float8_e4m3fn:
            s = tensors[name + "_scale_inv"]
            if name.endswith("qkv_proj.weight"):
                layer = int(name.split(".")[2])
                swa = config["hybrid_layer_pattern"][layer] == 1
                heads = config["swa_num_attention_heads" if swa else "num_attention_heads"]
                kv = config["swa_num_key_value_heads" if swa else "num_key_value_heads"]
                qk = config["swa_head_dim" if swa else "head_dim"]
                vd = config["swa_v_head_dim" if swa else "v_head_dim"]
                dense = block_dequant(t, s, shards)
                q, k, v = heads // shards * qk, kv // shards * qk, kv // shards * vd
                parts = dense.split(q + k + v)
                t = torch.cat([p[:q] for p in parts] + [p[q:q + k] for p in parts] + [p[q + k:] for p in parts])
            else:
                t = block_dequant(t, s)
        elif name + "_scale" in tensors:
            t = mxfp4_dequant(t, tensors[name + "_scale"])
        state[name] = t
    return state

def main():
    ap = parity_ref.parser()
    ap.add_argument("--long", type=int, default=300, help="tokens in the probe past the window")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, trust_remote_code=True)
    raw = json.load(open(os.path.join(args.snapshot, "config.json")))
    config = AutoConfig.from_pretrained(args.snapshot, trust_remote_code=True)
    config._attn_implementation = "eager"
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, dtype=torch.bfloat16)
    state = load_state(args.snapshot, raw)
    model.load_state_dict({k: v.to(torch.bfloat16) for k, v in state.items()}, strict=False, assign=True)
    del state
    missing = [n for n, p in model.named_parameters() if p.is_meta]
    if missing:
        sys.exit(f"the snapshot does not hold {missing[:4]}")
    # the rope tables are not stored; built on the meta device, they are built again
    for module in model.modules():
        if hasattr(module, "inv_freq") and hasattr(module, "rope_init_fn"):
            inv_freq, _ = module.rope_init_fn(module.config, "cpu")
            module.inv_freq = inv_freq
            module.original_inv_freq = inv_freq
    for name, buf in model.named_buffers():
        if buf.is_meta:
            sys.exit(f"buffer {name} was never built")
    model.to(args.device).eval()

    window_probe = [] if args.probes else [{"name": "window", "ids": long_prompt(tokenizer, args.long)}]
    encode = lambda text: tokenizer(text, add_special_tokens=False)["input_ids"]  # noqa: E731
    probes = parity_ref.probes(args, encode, PROMPTS, window_probe)

    window = raw["sliding_window"]

    # The masks the remote code would build, stated here: its mask helpers
    # are written against an older transformers. A window of 128 keeps the
    # token and the 127 before it.
    def masks(n):
        i = torch.arange(n, device=args.device)[:, None]
        j = torch.arange(n, device=args.device)[None, :]
        floor = torch.finfo(torch.bfloat16).min
        additive = lambda keep: torch.where(keep, 0.0, floor).to(torch.bfloat16)[None, None]
        return {
            "full_attention": additive(j <= i),
            "sliding_window_attention": additive((j <= i) & (j > i - window)),
        }

    @torch.no_grad()
    def logits(ids):
        x = torch.tensor([ids], device=args.device)
        out = model(input_ids=x, attention_mask=masks(len(ids)), use_cache=False)
        return out.logits[0].float().cpu().numpy()

    for probe in probes:
        started = time.time()
        tf, rows, gen = parity_ref.greedy(logits, probe["ids"], args.steps)
        parity_ref.dump(args, probe["name"], probe["ids"], tf, rows, gen, started, tokenizer.decode)

if __name__ == "__main__":
    main()
