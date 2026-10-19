"""Nemotron-3 Super reference logits for a release-layout snapshot (its
modelopt mixed-precision export, or any cut of it), written as vLLM's
`nemotron_h.py` computes its text model so a pie reading of the same snapshot
can be read against it: Mamba-2 mixers (their dt unclamped, as vLLM's
`time_step_limit` of (0, inf) leaves it), NoPE GQA attention, and the latent
MoE of non-gated relu2 experts beside a relu2 shared expert. The recurrence is
walked token by token in f32.

Quantized weights read dequantized for bf16 activations: modelopt FP8 codes
times their one `weight_scale`, NVFP4 e2m1 codes (the even element the low
nibble) times their e4m3 scale a 16 times `weight_scale_2`, rounded to bf16.

    python3 scripts/nemotron_h_parity_ref.py <snapshot-dir> OUT [--steps 16] [--probes probes.json]

Writes, per probe NAME, `OUT/NAME.ref.{tf,gen}.f32` and `OUT/NAME.ref.json`
as `scripts/parity_ref.py` states.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parity_ref  # noqa: E402

import torch
import torch.nn.functional as F
from safetensors import safe_open

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


class Weights:
    def __init__(self, snapshot, device):
        self.where = json.load(open(os.path.join(snapshot, "model.safetensors.index.json")))["weight_map"]
        self.snapshot, self.device, self.files, self.memo = snapshot, device, {}, {}

    def raw(self, name):
        shard = self.where[name]
        if shard not in self.files:
            self.files[shard] = safe_open(os.path.join(self.snapshot, shard), "pt", device="cpu")
        return self.files[shard].get_tensor(name).to(self.device)

    def get(self, name):
        """`name` as bf16, dequantized where the export stores it quantized."""
        if name in self.memo:
            return self.memo[name]
        stem = name[: -len(".weight")] if name.endswith(".weight") else None
        if stem and stem + ".weight_scale_2" in self.where:
            codes = self.raw(name)
            table = torch.tensor(E2M1 + [-v for v in E2M1], device=codes.device)
            vals = torch.stack([table[(codes & 0x0F).long()], table[(codes >> 4).long()]], -1).flatten(-2)
            vals = vals * self.raw(stem + ".weight_scale").float().repeat_interleave(16, -1)
            w = (vals * self.raw(stem + ".weight_scale_2").float()).to(torch.bfloat16)
        elif stem and stem + ".weight_scale" in self.where:
            w = (self.raw(name).float() * self.raw(stem + ".weight_scale").float()).to(torch.bfloat16)
        else:
            w = self.raw(name)
        if ".experts." not in name:
            self.memo[name] = w
        return w


def rmsnorm(x, weight, eps):
    y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (weight.float() * y).to(x.dtype)


def mamba(w, c, n, h):
    T = h.shape[0]
    H, P, N, G = c["mamba_num_heads"], c["mamba_head_dim"], c["ssm_state_size"], c["n_groups"]
    inner = H * P
    zxbcdt = h @ w.get(n("in_proj.weight")).T
    z, xbc, dt = zxbcdt.split([inner, inner + 2 * G * N, H], dim=-1)
    k = c["conv_kernel"]
    conv = w.get(n("conv1d.weight")).float().squeeze(1)  # [C, k]
    padded = F.pad(xbc.float().T, (k - 1, 0))  # [C, T + k - 1]
    xbc = F.silu(padded.unfold(1, k, 1).mul(conv[:, None]).sum(-1) + w.get(n("conv1d.bias")).float()[:, None]).T
    xbc = xbc.to(h.dtype)
    x, B, C = xbc.split([inner, G * N, G * N], dim=-1)
    dt = F.softplus(dt.float() + w.get(n("dt_bias")).float())  # [T, H]
    A = -torch.exp(w.get(n("A_log")).float())
    D = w.get(n("D")).float()
    xs = x.float().view(T, H, P)
    Bs = B.float().view(T, G, N).repeat_interleave(H // G, dim=1)  # [T, H, N]
    Cs = C.float().view(T, G, N).repeat_interleave(H // G, dim=1)
    state = torch.zeros(H, P, N, device=h.device)
    ys = []
    for t in range(T):
        state = state * torch.exp(dt[t] * A)[:, None, None] + (dt[t][:, None] * xs[t])[:, :, None] * Bs[t][:, None, :]
        ys.append(torch.einsum("hpn,hn->hp", state, Cs[t]) + D[:, None] * xs[t])
    y = torch.stack(ys).reshape(T, inner).to(h.dtype)
    gated = y.float() * F.silu(z.float())
    group = inner // G
    gv = gated.view(T, G, group)
    normed = (gv * torch.rsqrt(gv.pow(2).mean(-1, keepdim=True) + c["layer_norm_epsilon"])).view(T, inner)
    y = (normed * w.get(n("norm.weight")).float()).to(h.dtype)
    return y @ w.get(n("out_proj.weight")).T


def attention(w, c, n, h):
    T = h.shape[0]
    nh, nkv, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    q = (h @ w.get(n("q_proj.weight")).T).view(T, nh, hd).transpose(0, 1)
    k = (h @ w.get(n("k_proj.weight")).T).view(T, nkv, hd).transpose(0, 1)
    v = (h @ w.get(n("v_proj.weight")).T).view(T, nkv, hd).transpose(0, 1)
    o = F.scaled_dot_product_attention(q[None], k[None], v[None], is_causal=True, enable_gqa=True)[0]
    return o.transpose(0, 1).reshape(T, nh * hd) @ w.get(n("o_proj.weight")).T


def relu2(x):
    return torch.square(F.relu(x))


def moe(w, c, n, h):
    logits = F.linear(h.float(), w.get(n("gate.weight")).float())
    scores = logits.sigmoid()
    choice = scores + w.get(n("gate.e_score_correction_bias")).float()
    top = torch.topk(choice, k=c["num_experts_per_tok"], dim=-1, sorted=False)[1]
    weights = scores.gather(1, top)
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * c["routed_scaling_factor"]
    lat = h @ w.get(n("fc1_latent_proj.weight")).T
    out = torch.zeros(lat.shape, dtype=torch.float32, device=h.device)
    for e in top.unique().tolist():
        rows, slot = torch.where(top == e)
        up = w.get(n(f"experts.{e}.up_proj.weight"))
        down = w.get(n(f"experts.{e}.down_proj.weight"))
        y = relu2(lat[rows] @ up.T) @ down.T
        out.index_add_(0, rows, y.float() * weights[rows, slot, None])
    routed = out.to(h.dtype) @ w.get(n("fc2_latent_proj.weight")).T
    shared = relu2(h @ w.get(n("shared_experts.up_proj.weight")).T) @ w.get(n("shared_experts.down_proj.weight")).T
    return routed + shared


def forward(w, c, ids):
    x = F.embedding(torch.tensor(ids, device=w.device), w.get("backbone.embeddings.weight"))
    mixers = {"M": mamba, "*": attention, "E": moe}
    for l, kind in enumerate(c["hybrid_override_pattern"][: c["num_hidden_layers"]]):
        n = lambda s: f"backbone.layers.{l}.mixer.{s}"  # noqa: E731
        h = rmsnorm(x, w.get(f"backbone.layers.{l}.norm.weight"), c["layer_norm_epsilon"])
        x = x + mixers[kind](w, c, n, h)
    x = rmsnorm(x, w.get("backbone.norm_f.weight"), c["layer_norm_epsilon"])
    return (x @ w.get("lm_head.weight").T).float()


def main():
    args = parity_ref.parser().parse_args()
    c = json.load(open(os.path.join(args.snapshot, "config.json")))
    w = Weights(args.snapshot, args.device)
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(args.snapshot, "tokenizer.json"))
    torch.set_grad_enabled(False)
    for probe in parity_ref.probes(args, lambda text: [c["bos_token_id"]] + tok.encode(text, add_special_tokens=False)):
        started = time.time()
        tf, rows, gen = parity_ref.greedy(lambda ids: forward(w, c, ids).cpu().numpy(), probe["ids"], args.steps)
        parity_ref.dump(args, probe["name"], probe["ids"], tf, rows, gen, started, decode=tok.decode)


if __name__ == "__main__":
    main()
