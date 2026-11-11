#!/usr/bin/env python3
"""landed.py <tiny checkpoint dir> <out .safetensors>: a tiny HF checkpoint's tensors under the names its trace gives
its params, joined as the import contract joins them (q / k / v rows stacked, gate / up rows stacked, gpt-oss's
interleaved gate / up rows apart, Gemma's per-layer table cut a layer), so `wafer-stage --plan .. --landed ..` cuts it
as a load cuts a real checkpoint. The tiny checkpoints have no catalog deployment (their dims are no shipping model's),
so the contract cannot read them; this is the contract's reading for them, nothing more (no value changes)."""
import glob, json, os, struct, sys
import numpy as np

DT = {"BF16": (np.uint16, "BF16"), "F32": (np.float32, "F32"), "U8": (np.uint8, "U8")}


def read(md):
    T = {}
    for f in sorted(glob.glob(md + "/*.safetensors")):
        b = open(f, "rb").read()
        n = struct.unpack("<Q", b[:8])[0]
        h = json.loads(b[8:8 + n])
        for k, v in h.items():
            if k == "__metadata__":
                continue
            s, e = v["data_offsets"]
            T[k] = (v["dtype"], np.frombuffer(b[8 + n + s:8 + n + e], DT[v["dtype"]][0]).reshape(v["shape"]))
    return T


def write(path, out):
    hdr, blobs, off = {}, [], 0
    for k, (dt, a) in out.items():
        a = np.ascontiguousarray(a)
        hdr[k] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off + a.nbytes]}
        blobs.append(a.tobytes())
        off += a.nbytes
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def main():
    md, dst = sys.argv[1:3]
    cfg = json.load(open(os.path.join(md, "config.json")))
    tc = cfg.get("text_config", cfg)
    T = read(md)
    P = "model.language_model." if any(k.startswith("model.language_model.") for k in T) else "model."
    out = {}

    def put(name, key):
        out[name] = T[key]

    def cat(name, keys, axis=0):
        dts = {T[k][0] for k in keys}
        assert len(dts) == 1, (name, keys, dts)
        out[name] = (dts.pop(), np.concatenate([T[k][1] for k in keys], axis))

    put("embed", P + "embed_tokens.weight")
    put("final_norm", P + "norm.weight")
    if "lm_head.weight" in T:
        put("head", "lm_head.weight")
    NL = tc["num_hidden_layers"]
    arch = (cfg.get("architectures") or [""])[0]
    for l in range(NL):
        L = P + f"layers.{l}."
        n = lambda leaf: f"layers.{l}.{leaf}"
        if arch.startswith("GptOss"):
            A, M = L + "self_attn.", L + "mlp."
            put(n("attn_norm"), L + "input_layernorm.weight")
            put(n("mlp_norm"), L + "post_attention_layernorm.weight")
            cat(n("attn.qkv"), [A + f"{x}_proj.weight" for x in "qkv"])
            cat(n("attn.qkv_bias"), [A + f"{x}_proj.bias" for x in "qkv"])
            put(n("attn.sinks"), A + "sinks")
            put(n("attn.o_proj"), A + "o_proj.weight")
            put(n("attn.o_bias"), A + "o_proj.bias")
            put(n("mlp.router"), M + "router.weight")
            put(n("mlp.router_bias"), M + "router.bias")
            for leg, src in (("gate_up", "gate_up_proj_blocks"), ("gate_up.scales", "gate_up_proj_scales"),
                             ("gate_up_bias", "gate_up_proj_bias")):
                dt, a = T[M + "experts." + src]
                out[n("mlp." + leg)] = (dt, np.concatenate([a[:, 0::2], a[:, 1::2]], 1))
            put(n("mlp.down"), M + "experts.down_proj_blocks")
            put(n("mlp.down.scales"), M + "experts.down_proj_scales")
            put(n("mlp.down_bias"), M + "experts.down_proj_bias")
        elif arch.startswith("Qwen3Moe"):
            # Qwen3 MoE: q / k / v rows stacked, each expert's gate / up rows stacked, the experts stacked
            A, M = L + "self_attn.", L + "mlp."
            put(n("attn_norm"), L + "input_layernorm.weight")
            put(n("mlp_norm"), L + "post_attention_layernorm.weight")
            cat(n("qkv"), [A + f"{x}_proj.weight" for x in "qkv"])
            put(n("q_norm"), A + "q_norm.weight")
            put(n("k_norm"), A + "k_norm.weight")
            put(n("o"), A + "o_proj.weight")
            put(n("router"), M + "gate.weight")
            E = tc["num_experts"]
            ex = lambda e, leg: T[M + f"experts.{e}.{leg}_proj.weight"]
            out[n("gate_up")] = ("BF16", np.stack([np.concatenate([ex(e, "gate")[1], ex(e, "up")[1]], 0) for e in range(E)]))
            out[n("down")] = ("BF16", np.stack([ex(e, "down")[1] for e in range(E)]))
        elif arch.startswith("Gemma4"):
            A, M = L + "self_attn.", L + "mlp."
            for name, key in (("attn_norm", "input_layernorm"), ("post_attn_norm", "post_attention_layernorm"),
                              ("pre_ffw_norm", "pre_feedforward_layernorm"),
                              ("post_ffw_norm", "post_feedforward_layernorm")):
                put(n(name), L + key + ".weight")
            put(n("attn.q_norm"), A + "q_norm.weight")
            if l < NL - (tc.get("num_kv_shared_layers") or 0):
                v = A + "v_proj.weight" if A + "v_proj.weight" in T else A + "k_proj.weight"
                cat(n("attn.banks.qkv"), [A + "q_proj.weight", A + "k_proj.weight", v])
                put(n("attn.banks.k_norm"), A + "k_norm.weight")
            else:
                put(n("attn.banks.q_proj"), A + "q_proj.weight")
            put(n("o_proj"), A + "o_proj.weight")
            cat(n("gate_up"), [M + "gate_proj.weight", M + "up_proj.weight"])
            put(n("down"), M + "down_proj.weight")
            ple = tc.get("hidden_size_per_layer_input") or 0
            if ple:
                p = lambda leaf: f"ple.per_layer.{l}.{leaf}"
                dt, tab = T[P + "embed_tokens_per_layer.weight"]
                out[p("table")] = (dt, tab[:, l * ple:(l + 1) * ple])
                put(p("gate"), L + "per_layer_input_gate.weight")
                put(p("proj"), L + "per_layer_projection.weight")
                put(p("norm"), L + "post_per_layer_input_norm.weight")
                put(p("scalar"), L + "layer_scalar")
            else:
                put(n("scalar"), L + "layer_scalar")
        elif "linear_num_key_heads" in tc:
            # Qwen3.5: GDN and gated attention layers
            put(n("mixer_norm"), L + "input_layernorm.weight")
            put(n("mlp_norm"), L + "post_attention_layernorm.weight")
            if L + "linear_attn.in_proj_qkv.weight" in T:
                G = L + "linear_attn."
                cat(n("gdn.in_qkvz"), [G + "in_proj_qkv.weight", G + "in_proj_z.weight"])
                cat(n("gdn.in_ba"), [G + "in_proj_b.weight", G + "in_proj_a.weight"])
                dt, cv = T[G + "conv1d.weight"]
                out[n("gdn.conv")] = (dt, cv[:, 0, :])
                put(n("gdn.dt_bias"), G + "dt_bias")
                put(n("gdn.a_log"), G + "A_log")
                put(n("gdn.norm"), G + "norm.weight")
                put(n("gdn.out_proj"), G + "out_proj.weight")
            else:
                A = L + "self_attn."
                cat(n("attn.qg_proj_k_proj_v_proj"), [A + f"{x}_proj.weight" for x in "qkv"])
                put(n("attn.q_norm"), A + "q_norm.weight")
                put(n("attn.k_norm"), A + "k_norm.weight")
                put(n("attn.o_proj"), A + "o_proj.weight")
            cat(n("mlp.gate_up"), [L + "mlp.gate_proj.weight", L + "mlp.up_proj.weight"])
            put(n("mlp.down"), L + "mlp.down_proj.weight")
        else:
            # dense attention (Qwen3, Llama 3)
            A = L + "self_attn."
            put(n("attn_norm"), L + "input_layernorm.weight")
            put(n("mlp_norm"), L + "post_attention_layernorm.weight")
            cat(n("qkv"), [A + f"{x}_proj.weight" for x in "qkv"])
            if A + "q_norm.weight" in T:
                put(n("q_norm"), A + "q_norm.weight")
                put(n("k_norm"), A + "k_norm.weight")
            put(n("o"), A + "o_proj.weight")
            cat(n("gate_up"), [L + "mlp.gate_proj.weight", L + "mlp.up_proj.weight"])
            put(n("down"), L + "mlp.down_proj.weight")
    if arch.startswith("Gemma4") and (tc.get("hidden_size_per_layer_input") or 0):
        put("ple.model_proj", P + "per_layer_model_projection.weight")
        put("ple.model_norm", P + "per_layer_projection_norm.weight")
    write(dst, out)
    print("landed", len(out), "params ->", dst)


if __name__ == "__main__":
    main()
