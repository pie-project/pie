"""Composable module DSL for the load-parity harness.

A model is *composed* from small module primitives (attention / FFN variants),
not a per-family hand-written builder:

    Recipe(model_type, dims, attn=<module>, ffn_plan=[<module> per layer], ...)
    build(recipe) -> gen.Template   (config.json dict + source TensorSpecs)

`named_recipes()` expresses every supported model_type as a composition (so the
model_type-specific loader transforms — fusion / MLA-join / phi3-split / prefix /
mxfp4 — are exercised). `random_recipe(rng)` generates random valid compositions
(random dims / counts / quant) tagged with a `synthetic_*` model_type, which maps
to the loader's GENERIC ArchProfile and so exercises the generic transform algebra
(direct copy / TP-shard-by-name / FP8 block / MoE). The oracle (oracle.py) derives
the expectation from the source by byte-reconstruction, so either kind is checkable.

Dims are kept TP-safe: every block-FP8 axis (`weight_block_size [128,128]`) that
the loader shards on is a multiple of 128*tp(=2) so its block count stays even.

Coverage note (cpp-refact.md Phase 5): `qwen3_vl_text` and `gemma4_text` are
covered above — both text towers reduce to the existing mha/dense composition
(a plain Qwen3 shape and a PLE/MoE-off Gemma-4 shape respectively). Three
families do NOT fit this composer without materially extending it, and are
intentionally left out rather than faked:
  * `gpt_oss` — its expert tensors are dispatched by NAME PRESENCE, not a
    config/runtime flag (`model/mixtral/gpt_oss.cpp`'s `has_native_mxfp4_experts`
    checks only whether `experts.{gate,up,down}_proj.weight` exist, not their
    dtype), so a synthetic BF16 tensor at those exact names is ambiguous with
    the packed-MXFP4 native path and would need a fourth generic-loader
    concept (arch-specific dispatch-by-presence) to model safely.
  * `nemotron_h` — per-layer type is heterogeneous (mamba / attention / moe
    from `hybrid_override_pattern`) with SSM tensors (`A_log`, `D`, `dt_bias`,
    `conv1d.*`) this DSL's `ffn_plan` (FFN-only per layer) has no primitive for.
  * `gemma3n` — AltUp + Laurel are mandatory (binder throws if
    `altup_num_inputs<=1` or `laurel_rank<=0`), adding a whole new per-layer
    tensor family (`altup.*`, `laurel.*`, per-layer-embedding residuals) with
    no dense/MoE analog to compose from.
Their gate for this phase is the existing CMake-wired parity/compile targets
(`gemma4_audio_full_parity`, `gemma4_vision_full_parity(_bf16)`,
`gemma4_vision_patch_parity`, `qwen3_vl_vision_full_parity` for the
vision/audio adapters; `csm_backbone_parity` / `csm_depth_decoder_parity` /
`csm_generate_parity` / `mimi_decoder_full_parity` for `csm`, which isn't a
causal-LM shape at all and so was never a candidate for this harness).
`nemotron_h` and `gpt_oss` currently have no dedicated harness; that gap is
flagged here for a follow-up pass rather than papered over with a recipe that
would not actually exercise their binder.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gen import Template, TensorSpec

INDEX_HEAD = 128   # DSA lightning-indexer head dim (glm); replicated under TP
INDEX_WP = 32      # DSA weights_proj rows


# --- leaf emitters --------------------------------------------------------- #
def _norm(name: str, dim: int) -> list[TensorSpec]:
    return [TensorSpec(name, [dim], "BF16")]


def _lin(name: str, out_dim: int, in_dim: int, dt: str) -> list[TensorSpec]:
    """A linear weight, plus a FP32 `_scale_inv` sibling when block-FP8."""
    specs = [TensorSpec(name, [out_dim, in_dim], dt)]
    if dt in ("F8_E4M3", "F8_E5M2"):
        sr, sc = -(-out_dim // 128), -(-in_dim // 128)   # ceil-div, block [128,128]
        specs.append(TensorSpec(name.removesuffix(".weight") + ".weight_scale_inv",
                                [sr, sc], "F32"))
    return specs


@dataclass
class Dims:
    hid: int = 256
    n_heads: int = 2
    n_kv: int = 2
    head_dim: int = 128
    inter: int = 256
    vocab: int = 256
    # MLA
    q_lora: int = 256
    kv_lora: int = 256
    qk_nope: int = 128
    qk_rope: int = 128
    v_head: int = 128
    # MoE
    moe_inter: int = 256
    n_exp: int = 4
    n_shared: int = 1
    gate_bias: bool = False   # DeepSeek-V3/GLM sigmoid-router e_score_correction_bias
    eps: float = 1e-5

    @property
    def q_rows(self) -> int: return self.n_heads * self.head_dim
    @property
    def kv_rows(self) -> int: return self.n_kv * self.head_dim
    @property
    def q_b_rows(self) -> int: return self.n_heads * (self.qk_nope + self.qk_rope)
    @property
    def kv_b_rows(self) -> int: return self.n_heads * (self.qk_nope + self.v_head)
    @property
    def o_in_mla(self) -> int: return self.n_heads * self.v_head


# --- attention modules ----------------------------------------------------- #
def _attn_mha(ap: str, d: Dims, dt: str, *, qk_norm: bool = False, **_):
    t = (_lin(ap + "q_proj.weight", d.q_rows, d.hid, dt)
         + _lin(ap + "k_proj.weight", d.kv_rows, d.hid, dt)
         + _lin(ap + "v_proj.weight", d.kv_rows, d.hid, dt)
         + _lin(ap + "o_proj.weight", d.hid, d.q_rows, dt))
    if qk_norm:
        t += _norm(ap + "q_norm.weight", d.head_dim) + _norm(ap + "k_norm.weight", d.head_dim)
    return t


def _attn_fused_qkv(ap: str, d: Dims, dt: str, **_):
    # phi3: single fused qkv_proj (the loader SPLITS it into q/k/v).
    return ([TensorSpec(ap + "qkv_proj.weight", [d.q_rows + 2 * d.kv_rows, d.hid], dt)]
            + _lin(ap + "o_proj.weight", d.hid, d.q_rows, dt))


def _attn_mla(ap: str, d: Dims, dt: str, *, qk_norm: bool = False, indexer: bool = False, **_):
    t = (_lin(ap + "q_a_proj.weight", d.q_lora, d.hid, dt)
         + _norm(ap + "q_a_layernorm.weight", d.q_lora)
         + _lin(ap + "q_b_proj.weight", d.q_b_rows, d.q_lora, dt)
         + _lin(ap + "kv_a_proj_with_mqa.weight", d.kv_lora + d.qk_rope, d.hid, dt)
         + _norm(ap + "kv_a_layernorm.weight", d.kv_lora)
         + _lin(ap + "kv_b_proj.weight", d.kv_b_rows, d.kv_lora, dt)
         + _lin(ap + "o_proj.weight", d.hid, d.o_in_mla, dt))
    if indexer:   # DSA lightning indexer (glm); all replicated under TP
        t += (_lin(ap + "indexer.wq_b.weight", INDEX_HEAD, d.q_lora, dt)
              + _lin(ap + "indexer.wk.weight", INDEX_HEAD, d.hid, dt)
              + [TensorSpec(ap + "indexer.weights_proj.weight", [INDEX_WP, d.hid], "BF16")]
              + _norm(ap + "indexer.k_norm.weight", INDEX_HEAD)
              + _norm(ap + "indexer.k_norm.bias", INDEX_HEAD))
    return t


# --- FFN modules ----------------------------------------------------------- #
def _ffn_dense(p: str, d: Dims, dt: str):
    return (_lin(p + "mlp.gate_proj.weight", d.inter, d.hid, dt)
            + _lin(p + "mlp.up_proj.weight", d.inter, d.hid, dt)
            + _lin(p + "mlp.down_proj.weight", d.hid, d.inter, dt))


def _ffn_fused_gate_up(p: str, d: Dims, dt: str):   # phi3: loader splits gate_up
    return [TensorSpec(p + "mlp.gate_up_proj.weight", [2 * d.inter, d.hid], dt),
            TensorSpec(p + "mlp.down_proj.weight", [d.hid, d.inter], dt)]


def _ffn_moe_qwen(p: str, d: Dims, dt: str):        # gate/up/down experts (+ shared)
    t = [TensorSpec(p + "mlp.gate.weight", [d.n_exp, d.hid], "BF16")]
    if d.gate_bias:
        t += [TensorSpec(p + "mlp.gate.e_score_correction_bias", [d.n_exp], "BF16")]
    for e in range(d.n_exp):
        ep = p + f"mlp.experts.{e}."
        t += (_lin(ep + "gate_proj.weight", d.moe_inter, d.hid, dt)
              + _lin(ep + "up_proj.weight", d.moe_inter, d.hid, dt)
              + _lin(ep + "down_proj.weight", d.hid, d.moe_inter, dt))
    if d.n_shared:
        sp = p + "mlp.shared_experts."
        t += (_lin(sp + "gate_proj.weight", d.moe_inter, d.hid, dt)
              + _lin(sp + "up_proj.weight", d.moe_inter, d.hid, dt)
              + _lin(sp + "down_proj.weight", d.hid, d.moe_inter, dt))
    return t


def _ffn_moe_mixtral(p: str, d: Dims, dt: str):     # w1/w2/w3 experts, no shared
    t = [TensorSpec(p + "block_sparse_moe.gate.weight", [d.n_exp, d.hid], "BF16")]
    for e in range(d.n_exp):
        ep = p + f"block_sparse_moe.experts.{e}."
        t += (_lin(ep + "w1.weight", d.moe_inter, d.hid, dt)
              + _lin(ep + "w2.weight", d.hid, d.moe_inter, dt)
              + _lin(ep + "w3.weight", d.moe_inter, d.hid, dt))
    return t


ATTN = {"mha": _attn_mha, "fused_qkv": _attn_fused_qkv, "mla": _attn_mla}
FFN = {"dense": _ffn_dense, "fused_gate_up": _ffn_fused_gate_up,
       "moe_qwen": _ffn_moe_qwen, "moe_mixtral": _ffn_moe_mixtral}


# --- recipe ---------------------------------------------------------------- #
@dataclass
class Recipe:
    model_type: str
    dims: Dims = field(default_factory=Dims)
    attn: str = "mha"
    ffn_plan: list[str] = field(default_factory=lambda: ["dense", "dense"])
    qk_norm: bool = False
    indexer: bool = False
    prefix: str = ""             # e.g. kimi "language_model."
    tie_embed: bool = False
    gemma_ffn_norms: bool = False
    moe_style: str = ""          # "mixtral" | "qwen" | "deepseek" (config keys)
    weight_dtype: str = "BF16"   # source weight dtype (BF16 or F8_E4M3)
    runtime_quant: str = ""      # "" or "mxfp4" (driver flag)
    quant_method: str = ""       # "" or "fp8"  (config.quantization_config)
    tp_engine_ok: bool = True    # False = engine gates tp>1 (linear-attn hybrids)
    extra_config: dict = field(default_factory=dict)  # verbatim config.json
                                                        # keys a family needs
                                                        # that the generic
                                                        # composer doesn't
                                                        # already derive
                                                        # (e.g. gemma4-text's
                                                        # `layer_types`).
    name: str = ""


def _config(r: Recipe) -> dict:
    d = r.dims
    cfg = {
        "model_type": r.model_type,
        "architectures": [r.model_type],
        "num_hidden_layers": len(r.ffn_plan),
        "hidden_size": d.hid,
        "num_attention_heads": d.n_heads,
        "num_key_value_heads": d.n_kv,
        "head_dim": d.head_dim,
        "intermediate_size": d.inter,
        "vocab_size": d.vocab,
        "rms_norm_eps": d.eps,
        "tie_word_embeddings": r.tie_embed,
        "max_position_embeddings": 4096,
        "rope_theta": 10000.0,
    }
    if r.qk_norm:
        cfg["use_qk_norm"] = True
    if r.attn == "mla":
        cfg.update(q_lora_rank=d.q_lora, kv_lora_rank=d.kv_lora,
                   qk_nope_head_dim=d.qk_nope, qk_rope_head_dim=d.qk_rope,
                   v_head_dim=d.v_head, head_dim=d.qk_rope)
    if any(k.startswith("moe") for k in r.ffn_plan):
        cfg.update(num_experts_per_tok=2, moe_intermediate_size=d.moe_inter,
                   norm_topk_prob=True)
        if r.moe_style == "mixtral":
            cfg["num_local_experts"] = d.n_exp
        elif r.moe_style == "qwen":
            cfg["num_experts"] = d.n_exp
            cfg["n_routed_experts"] = d.n_exp
        else:  # deepseek / glm
            cfg["n_routed_experts"] = d.n_exp
            cfg["n_shared_experts"] = d.n_shared
            cfg["first_k_dense_replace"] = sum(
                1 for k in r.ffn_plan if not k.startswith("moe"))
    if r.gemma_ffn_norms:
        cfg.update(query_pre_attn_scalar=d.head_dim, sliding_window=4096)
    if r.quant_method == "fp8":
        cfg["quantization_config"] = {"quant_method": "fp8", "fmt": "e4m3",
                                      "weight_block_size": [128, 128]}
    cfg.update(r.extra_config)
    return cfg


def build(r: Recipe) -> Template:
    d, pfx = r.dims, r.prefix
    attn_fn = ATTN[r.attn]
    t = [TensorSpec(pfx + "model.embed_tokens.weight", [d.vocab, d.hid], "BF16")]
    for i, ffn_key in enumerate(r.ffn_plan):
        p = f"{pfx}model.layers.{i}."
        t += _norm(p + "input_layernorm.weight", d.hid)
        t += _norm(p + "post_attention_layernorm.weight", d.hid)
        if r.gemma_ffn_norms:
            t += _norm(p + "pre_feedforward_layernorm.weight", d.hid)
            t += _norm(p + "post_feedforward_layernorm.weight", d.hid)
        t += attn_fn(p + "self_attn.", d, r.weight_dtype,
                     qk_norm=r.qk_norm, indexer=r.indexer)
        t += FFN[ffn_key](p, d, r.weight_dtype)
    t += _norm(pfx + "model.norm.weight", d.hid)
    if not r.tie_embed:
        t += [TensorSpec(pfx + "lm_head.weight", [d.vocab, d.hid], "BF16")]
    return Template(name=r.name or r.model_type, model_type=r.model_type,
                    config=_config(r), tensors=t, runtime_quant=r.runtime_quant)


# --- named recipes: the real supported model_types as compositions --------- #
def _dense(mt, **kw):
    return Recipe(mt, attn="mha", ffn_plan=["dense", "dense"], name=mt, **kw)


def _moe(mt, style, *, shared=0, **kw):
    d = Dims(n_shared=shared)
    ffn = "moe_mixtral" if style == "mixtral" else "moe_qwen"
    return Recipe(mt, dims=d, attn="mha", ffn_plan=[ffn, ffn],
                  moe_style=style, name=mt, **kw)


def _mla(mt, *, prefix="", indexer=False, layers=4, dense=1, gate_bias=False, **kw):
    d = Dims(n_shared=1, gate_bias=gate_bias)
    ffn_plan = ["dense"] * dense + ["moe_qwen"] * (layers - dense)
    return Recipe(mt, dims=d, attn="mla", ffn_plan=ffn_plan, moe_style="deepseek",
                  prefix=prefix, indexer=indexer, name=mt, **kw)


def _gemma(mt, **kw):
    return Recipe(mt, attn="mha", ffn_plan=["dense", "dense"],
                  gemma_ffn_norms=True, tie_embed=True, name=mt, **kw)


def named_recipes() -> dict[str, Recipe]:
    r: dict[str, Recipe] = {}
    for mt in ("llama", "llama3", "qwen2", "mistral", "mistral3", "ministral3"):
        r[mt] = _dense(mt)
    for mt in ("qwen3", "olmo2", "olmo3"):
        r[mt] = _dense(mt, qk_norm=True)
    r["qwen3_vl_text"] = _dense("qwen3_vl_text", qk_norm=True)  # HF: text tower
                                                                 # of Qwen3-VL is
                                                                 # standard Qwen3;
                                                                 # the vision tower
                                                                 # is a separate
                                                                 # adapter this
                                                                 # harness doesn't
                                                                 # model.
    for mt in ("qwen3_5", "qwen3_5_text"):
        r[mt] = _dense(mt, qk_norm=True, tp_engine_ok=False)
    r["phi3"] = Recipe("phi3", attn="fused_qkv", ffn_plan=["fused_gate_up"] * 2,
                       name="phi3")
    r["mixtral"] = _moe("mixtral", "mixtral")
    r["qwen3_moe"] = _moe("qwen3_moe", "qwen", qk_norm=True)
    for mt in ("qwen3_5_moe", "qwen3_5_moe_text"):
        r[mt] = _moe(mt, "qwen", qk_norm=True, tp_engine_ok=False)
    for mt in ("deepseek_v2", "deepseek_v3", "deepseek_v4"):
        r[mt] = _mla(mt)
    r["kimi_k2"] = _mla("kimi_k2", prefix="language_model.")
    for mt in ("gemma2",):
        r[mt] = _gemma(mt)
    for mt in ("gemma3", "gemma3_text"):
        r[mt] = _gemma(mt, qk_norm=True)
    # gemma4_text: the dense (non-MoE, no vision/audio) Gemma-4 shape — PLE
    # (`hidden_size_per_layer_input`) and `enable_moe_block` both default off,
    # so the binder only needs the standard mha + 4-norm-per-layer tensors
    # plus `layer_types` (required, but an all-"full_attention" plan keeps
    # the sliding-window / kv-sharing branches inert for this smoke shape).
    r["gemma4_text"] = _gemma("gemma4_text", qk_norm=True,
                              extra_config={"layer_types":
                                            ["full_attention", "full_attention"]})
    r["glm_moe_dsa"] = _mla("glm_moe_dsa", indexer=True, layers=4, dense=3,
                            gate_bias=True, weight_dtype="F8_E4M3",
                            quant_method="fp8", runtime_quant="mxfp4")
    return r


# --- random composition ---------------------------------------------------- #
def random_recipe(rng: np.random.Generator, idx: int = 0) -> Recipe:
    """A random valid composition at random TP-safe shapes.

    The loader's TP slicing is name-based and arch-agnostic, but the *engine*
    only starts a TP group for a known model_type — so each random composition is
    tagged with a TP-enabled "carrier" model_type matching its structure (dense ->
    llama/qwen3, mixtral -> mixtral, qwen-MoE -> qwen3_moe, MLA -> deepseek_v3).
    For dense/MoE those carriers map to the GENERIC ArchProfile, so the loader does
    exactly the generic slicing; the carrier just satisfies the engine gate.

    Dims are TP-safe: heads even (divide tp=2), block-FP8 axes multiples of 256
    (even #128-blocks)."""
    ch = lambda xs: xs[int(rng.integers(0, len(xs)))]   # noqa: E731
    n_heads = ch([2, 4])
    d = Dims(
        hid=ch([256, 512]),
        n_heads=n_heads,
        n_kv=ch([2, n_heads]),          # even, divides n_heads and tp=2
        head_dim=128,
        inter=ch([256, 512]),
        vocab=256,
        moe_inter=ch([256, 512]),
        n_exp=ch([2, 4, 6]),
        n_shared=0,
    )
    n_layers = int(rng.integers(1, 4))
    attn, qk_norm = ch(["mha", "mla"]), False
    if attn == "mla":
        # always >=1 MoE layer: the deepseek_v3 carrier's engine TP setup needs a
        # valid moe_intermediate_size (an all-dense MLA isn't a real deepseek arch).
        dense_n = int(rng.integers(0, n_layers))
        d.n_shared = ch([0, 1])
        ffn_plan = ["dense"] * dense_n + ["moe_qwen"] * (n_layers - dense_n)
        moe_style, carrier = "deepseek", "deepseek_v3"   # MLA path (q_kv_a fused joins)
    else:
        kind = ch(["dense", "moe_mixtral", "moe_qwen"])
        if kind == "dense":
            ffn_plan, moe_style = ["dense"] * n_layers, ""
            qk_norm = bool(rng.integers(0, 2))
            carrier = "qwen3" if qk_norm else "llama"
        elif kind == "moe_mixtral":
            ffn_plan, moe_style = ["moe_mixtral"] * n_layers, "mixtral"
            carrier = "mixtral"
        else:
            ffn_plan, moe_style = ["moe_qwen"] * n_layers, "qwen"
            qk_norm = bool(rng.integers(0, 2))
            carrier = "qwen3_moe"
    dt = "F8_E4M3" if rng.random() < 0.4 else "BF16"
    return Recipe(carrier, dims=d, attn=attn, ffn_plan=ffn_plan, moe_style=moe_style,
                  qk_norm=qk_norm, weight_dtype=dt,
                  quant_method="fp8" if dt == "F8_E4M3" else "",
                  name=f"random{idx}_{carrier}")
