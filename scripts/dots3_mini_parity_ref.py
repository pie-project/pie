"""Dots3 NOTE reference logits for the `mini-l6-e16` snapshot.

A plain-torch transcription of vLLM's `vllm/models/dots3_note/nvidia/model.py`
(`_forward_note_mla`, `Dots3NoteMoE`, DeepSeek-V3.2's `Indexer`) and its
sliding-window attention backend (`attention.py`: a key at `p` is read by the
query at `q` when `q - 513 < p <= q`), reading the snapshot's safetensors
directly and driven by its own config: `layer_types`, `first_k_dense_replace`,
the full and `swa_*` attention dims.

    python scripts/dots3_mini_parity_ref.py --probes probes.json --out OUT [--steps 16]

Outputs, per probe NAME in `probes.json` (`{"probes": [{"name", "ids"}]}`):
    OUT/NAME.ref.tf.f32   float32 [len(ids), vocab]   teacher-forced logits at every position
    OUT/NAME.ref.gen.f32  float32 [steps + 1, vocab]  logits at the last prompt position and
                                                       after each greedy step
    OUT/NAME.ref.json     {"argmax": [...], "gen": [...]}

The indexer's top-k is exact (no fp8 round-trip of its q and keys), so a
sequence longer than `index_topk` may pick a different key where two scores
tie within fp8's resolution.
"""

import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parity_ref  # noqa: E402
import time

import torch
from safetensors import safe_open


class Checkpoint:
    def __init__(self, snapshot, device, dtype):
        self.config = json.load(open(os.path.join(snapshot, "config.json")))
        self.files = {}
        for path in sorted(glob.glob(os.path.join(snapshot, "*.safetensors"))):
            with safe_open(path, "pt") as f:
                for name in f.keys():
                    self.files[name] = path
        self.device = device
        self.dtype = dtype
        self.cache = {}

    def __call__(self, name, dtype=None):
        if name not in self.cache:
            with safe_open(self.files[name], "pt") as f:
                t = f.get_tensor(name)
            self.cache[name] = t.to(self.device, dtype or self.dtype)
        return self.cache[name]


# The dtype activations are held in between ops: float32, or bfloat16 as the
# served model holds them (`--bf16`).
ACT = [torch.float32]


def rmsnorm(x, w, eps):
    xf = x.float()
    return ((xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)) * w.float()).to(ACT[0])


def rope_interleaved(x, positions, theta):
    """GPT-J rope over the last dim of `x` ([T, ..., d]): pairs (2i, 2i+1)."""
    d = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    ang = positions.float()[:, None] * inv[None, :]
    shape = [x.shape[0]] + [1] * (x.dim() - 2) + [d // 2]
    cos, sin = ang.cos().view(shape), ang.sin().view(shape)
    xf = x.float()
    a, b = xf[..., 0::2], xf[..., 1::2]
    out = torch.empty_like(xf)
    out[..., 0::2] = a * cos - b * sin
    out[..., 1::2] = b * cos + a * sin
    return out.to(ACT[0])


class Reference:
    def __init__(self, ck):
        self.ck = ck
        c = ck.config
        self.c = c
        self.layers = c["num_hidden_layers"]
        self.eps = c["rms_norm_eps"]
        self.hidden = c["hidden_size"]

    def dims(self, l):
        c = self.c
        if self.c["layer_types"][l] == "sliding_attention":
            return dict(
                heads=c["swa_num_attention_heads"], nope=c["swa_qk_nope_head_dim"],
                rope=c["swa_qk_rope_head_dim"], v=c["swa_v_head_dim"],
                q_lora=c["swa_q_lora_rank"], kv_lora=c["swa_kv_lora_rank"],
                theta=c["swa_rope_theta"], window=c["sliding_window_size"])
        return dict(
            heads=c["num_attention_heads"], nope=c["qk_nope_head_dim"],
            rope=c["qk_rope_head_dim"], v=c["v_head_dim"], q_lora=c["q_lora_rank"],
            kv_lora=c["kv_lora_rank"], theta=c["rope_theta"], window=None)

    def indexer_mask(self, l, x, q_c, positions):
        """The keys the indexer keeps for each query: `index_topk` of the ones
        at or before it, by the weighted sum of relu(q . k) over its heads."""
        ck, c = self.ck, self.c
        at = lambda s: f"model.layers.{l}.self_attn.indexer.{s}"
        H, D, R = c["index_n_heads"], c["index_head_dim"], c["qk_rope_head_dim"]
        T = x.shape[0]
        q = (q_c @ ck(at("wq_b.weight")).T).view(T, H, D)
        k = x @ ck(at("wk.weight")).T
        k = torch.nn.functional.layer_norm(
            k.float(), (D,), ck(at("k_norm.weight")).float(), ck(at("k_norm.bias")).float(), 1e-6).to(ACT[0])
        theta = c["rope_theta"]
        q = torch.cat([rope_interleaved(q[..., :R], positions, theta), q[..., R:]], -1)
        k = torch.cat([rope_interleaved(k[:, :R], positions, theta), k[:, R:]], -1)
        w = (x @ ck(at("weights_proj.weight")).T) * (H ** -0.5) * (D ** -0.5)
        score = torch.einsum("thd,sd->tsh", q.float(), k.float()).relu()
        score = (score * w.float()[:, None, :]).sum(-1)
        causal = positions[None, :] <= positions[:, None]
        score = score.masked_fill(~causal, float("-inf"))
        top = c["index_topk"]
        if T <= top:
            return causal
        idx = score.topk(top, dim=-1).indices
        keep = torch.zeros_like(causal)
        keep.scatter_(1, idx, True)
        return keep & causal

    def attention(self, l, x, positions):
        ck, d = self.ck, self.dims(l)
        at = lambda s: f"model.layers.{l}.self_attn.{s}"
        T, H = x.shape[0], d["heads"]
        q_c = rmsnorm(x @ ck(at("q_a_proj.weight")).T, ck(at("q_a_layernorm.weight")), self.eps)
        q_c = q_c * math.sqrt(self.hidden / d["q_lora"])
        kv = x @ ck(at("kv_a_proj_with_mqa.weight")).T
        kv_c, k_pe = kv[:, : d["kv_lora"]], kv[:, d["kv_lora"]:]
        kv_c = rmsnorm(kv_c, ck(at("kv_a_layernorm.weight")), self.eps) * math.sqrt(self.hidden / d["kv_lora"])
        k_pe = rope_interleaved(rmsnorm(k_pe, ck(at("k_rope_only_layernorm.weight")), self.eps), positions, d["theta"])
        q = (q_c @ ck(at("q_b_proj.weight")).T).view(T, H, d["nope"] + d["rope"])
        q_nope, q_pe = q[..., : d["nope"]], rope_interleaved(q[..., d["nope"]:], positions, d["theta"])
        kvb = (kv_c @ ck(at("kv_b_proj.weight")).T).view(T, H, d["nope"] + d["v"])
        k_nope, v = kvb[..., : d["nope"]], kvb[..., d["nope"]:]
        scale = (d["nope"] + d["rope"]) ** -0.5
        s = (torch.einsum("thd,shd->hts", q_nope.float(), k_nope.float())
             + torch.einsum("thd,sd->hts", q_pe.float(), k_pe.float()))
        s = s * scale
        rel = positions[:, None] - positions[None, :]
        if d["window"] is not None:
            keep = (rel >= 0) & (rel < d["window"])
        else:
            keep = self.indexer_mask(l, x, q_c, positions)
        s = s.masked_fill(~keep[None], float("-inf"))
        o = torch.einsum("hts,shd->thd", s.softmax(-1), v.float()).to(ACT[0])
        gate = torch.sigmoid((x @ ck(at("g_proj.weight")).T).float()).to(ACT[0])
        o = (o * gate[..., None]).reshape(T, H * d["v"])
        return o @ ck(at("o_proj.weight")).T

    def mlp(self, l, x):
        ck, c = self.ck, self.c
        at = lambda s: f"model.layers.{l}.mlp.{s}"
        swiglu = lambda p, y: (torch.nn.functional.silu(y @ ck(p + "gate_proj.weight").T)
                               * (y @ ck(p + "up_proj.weight").T)) @ ck(p + "down_proj.weight").T
        if l < c["first_k_dense_replace"]:
            return swiglu(at(""), x)
        E, K = c["n_routed_experts"], c["num_experts_per_tok"]
        scores = torch.sigmoid((x @ ck(at("gate.weight")).T).float())
        choice = scores + ck(at("gate.e_score_correction_bias"), torch.float32)
        idx = choice.topk(K, dim=-1).indices
        w = scores.gather(1, idx)
        if c.get("norm_topk_prob", True):
            w = w / w.sum(-1, keepdim=True)
        w = w * c.get("routed_scaling_factor", 1.0)
        out = swiglu(at("shared_experts."), x)
        for e in range(E):
            rows, slot = (idx == e).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            y = swiglu(at(f"experts.{e}."), x[rows])
            out.index_add_(0, rows, (y.float() * w[rows, slot][:, None]).to(ACT[0]))
        return out

    @torch.no_grad()
    def logits(self, ids):
        ck = self.ck
        dev = ck.device
        positions = torch.arange(len(ids), device=dev)
        h = ck("model.embed_tokens.weight")[torch.tensor(ids, device=dev)]
        for l in range(self.layers):
            at = lambda s: f"model.layers.{l}.{s}"
            x = rmsnorm(h, ck(at("input_layernorm.weight")), self.eps)
            h = h + self.attention(l, x, positions)
            x = rmsnorm(h, ck(at("post_attention_layernorm.weight")), self.eps)
            h = h + self.mlp(l, x)
        x = rmsnorm(h, ck("model.norm.weight"), self.eps)
        return (x @ ck("lm_head.weight").T).float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=os.path.expanduser(
        "~/.cache/huggingface/hub/models--dots-studio--dots3-note-prev/snapshots/mini-l6-e16"))
    ap.add_argument("--probes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="ref")
    ap.add_argument("--bf16", action="store_true", help="hold weights and activations in bfloat16")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ACT[0] = torch.bfloat16 if args.bf16 else torch.float32
    ref = Reference(Checkpoint(args.snapshot, args.device, ACT[0]))
    probes = json.load(open(args.probes))["probes"]
    print(f"reference over {ref.layers} layers: {ref.c['layer_types']}")
    for probe in probes:
        started = time.time()
        tf, rows, gen = parity_ref.greedy(ref.logits, probe["ids"], args.steps)
        parity_ref.dump(args, probe["name"], probe["ids"], tf, rows, gen, started)


if __name__ == "__main__":
    main()
