"""Dump vLLM's per-layer activations for a single prefill, in pie's format.

The CUDA driver writes `s<step>_<tag>.npy` files when `PIE_ACT_DUMP_DIR` is set
(see `driver/cuda/src/model/act_dump.hpp`). This script produces the same tags
from vLLM so `act_parity.py` can report the first tag where the two engines
disagree — which turns "the models emit different tokens" into "layer 3's MoE
is wrong".

Tags (all `[tokens, width]`, float32):

    embed                   token embeddings
    hc_expand               DeepSeek-V4 only: embedding broadcast to hc_mult
    L<nn>_post_attn         residual stream after the attention block
    L<nn>_out               residual stream after the whole decoder layer
    hc_head                 DeepSeek-V4 only: streams collapsed back to [N, H]
    final_norm              input to lm_head
    logits                  lm_head output

Usage:
    /root/.venv/vllm/bin/python benches/act_dump_vllm.py \
        --model /root/models/kimi26-mini --out /tmp/act/vllm \
        --prompt "The capital of France is"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The in-process engine is what makes the model object reachable for hooks.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from common import BENCH_PROMPT, BENCH_SYSTEM, hf_chat_token_ids_and_counts  # noqa: E402

try:  # Kimi-K3 is registered from outside vLLM; see the module docstring there.
    from vllm_kimi_k3_register import register as _register_kimi_k3
except ImportError:  # pragma: no cover - the harness may be used standalone
    _register_kimi_k3 = None


def _save(out_dir: Path, tag: str, tensor) -> None:
    import numpy as np

    arr = tensor.detach().float().cpu().numpy()
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    np.save(out_dir / f"s0_{tag}.npy", arr.astype(np.float32))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default=BENCH_PROMPT)
    ap.add_argument("--system", default=BENCH_SYSTEM)
    ap.add_argument(
        "--unique-prompts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match the benches' default prompt rendering, which "
             "appends \"(Request #0)\". Both engines must see the "
             "same string or the dumps are not comparable.",
    )
    ap.add_argument(
        "--prompt-token-ids",
        default=None,
        help="Comma-separated token ids to run instead of rendering a chat "
             "template. The only way to make this dump comparable with pie's "
             "is to feed both engines the same ids, and the two render "
             "templates differently.",
    )
    ap.add_argument("--gpu-mem-util", type=float, default=0.55)
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--kv-cache-dtype", default="auto")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch

    if _register_kimi_k3 is not None:
        try:
            _register_kimi_k3()
        except Exception as exc:  # noqa: BLE001
            # The K3 shim needs config/modelling classes that only newer vLLM
            # builds ship. Failing here would take down dumping for every OTHER
            # model too, which is the opposite of what an optional
            # registration should do -- so report it and carry on. A K3 run
            # will then fail later, on its own, with vLLM's own message.
            print(f"[act-dump] Kimi-K3 registration unavailable: {exc}",
                  file=sys.stderr)

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    if args.prompt_token_ids:
        ids = [[int(t) for t in args.prompt_token_ids.split(",") if t.strip()]]
    else:
        prompt = f"{args.prompt} (Request #0)" if args.unique_prompts else args.prompt
        ids, _ = hf_chat_token_ids_and_counts(args.model, args.system, [prompt])
    kwargs = {}
    if args.kv_cache_dtype != "auto":
        kwargs["kv_cache_dtype"] = args.kv_cache_dtype
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=True,
        **kwargs,
    )

    core = llm.llm_engine.engine_core
    for attr in ("engine_core", "engine"):
        core = getattr(core, attr, core)
    runner = core.model_executor.driver_worker.model_runner
    model = runner.model
    # Unwrap multimodal wrappers (Kimi-K2.6 is `KimiK25ForConditionalGeneration`
    # around a `language_model`) down to the module that owns `.layers`.
    inner = model
    for _ in range(4):
        if hasattr(inner, "layers"):
            break
        for attr in ("model", "language_model", "text_model"):
            nxt = getattr(inner, attr, None)
            if nxt is not None and nxt is not inner:
                inner = nxt
                break
        else:
            break
    if not hasattr(inner, "layers"):
        raise SystemExit(
            f"cannot find decoder layers under {type(model).__name__}; "
            f"attributes: {[a for a in dir(model) if not a.startswith('_')][:40]}"
        )

    n_prompt = len(ids[0])
    seen: dict[str, int] = {}

    def tag_once(tag: str, tensor) -> None:
        # Only the first (prefill) call; vLLM keeps running for the decode
        # steps and the dummy warmup runs would otherwise overwrite it.
        if tag in seen:
            return
        if tensor is None:
            return
        if tensor.shape[0] != n_prompt:
            return  # warmup / decode shapes
        seen[tag] = 1
        _save(out_dir, tag, tensor)

    hooks = []

    def emb_hook(_mod, _inp, out):
        tag_once("embed", out)

    hooks.append(inner.embed_tokens.register_forward_hook(emb_hook))

    def make_layer_hook(idx: int):
        def hook(_mod, _inp, out):
            # DeepSeek-V4 returns (x, residual, post_mix, res_mix) where the
            # residual stream is `residual` [N, hc_mult, H]; the DeepSeek-V2 /
            # GLM families return (hidden_states, residual) where the true
            # stream is hidden_states + residual.
            if isinstance(out, tuple):
                if len(out) == 4:
                    # DeepSeek-V4: `residual` is the stream *after* the
                    # attention post-mix; the FFN post-mix is deferred to the
                    # next layer, so replay it here to get a true
                    # end-of-layer residual comparable with pie's `out`.
                    tag_once(f"L{idx:02d}_res_attn", out[1])
                    tag_once(f"L{idx:02d}_ffn_out", out[0])
                    try:
                        from vllm.model_executor.kernels.mhc.torch import (
                            mhc_post_torch,
                        )

                        tag_once(
                            f"L{idx:02d}_out",
                            mhc_post_torch(out[0], out[1], out[2], out[3]),
                        )
                    except Exception:
                        tag_once(f"L{idx:02d}_out", out[1])
                elif len(out) == 3:
                    # Kimi-K3: `(mlp_out, prefix_sum, block_residual)`. The
                    # running sum is closed by the *next* layer, so replay it
                    # here to get an end-of-layer residual comparable with
                    # pie's `out`. `block_residual` is the [T, B, H] AttnRes
                    # stack, not part of the stream.
                    if out[1] is not None:
                        tag_once(f"L{idx:02d}_out", out[1] + out[0])
                    else:
                        tag_once(f"L{idx:02d}_out", out[0])
                elif len(out) == 2 and out[1] is not None:
                    tag_once(f"L{idx:02d}_out", out[0] + out[1])
                else:
                    tag_once(f"L{idx:02d}_out", out[0])
            else:
                tag_once(f"L{idx:02d}_out", out)

        return hook

    for i, layer in enumerate(inner.layers):
        if hasattr(layer, "register_forward_hook"):
            hooks.append(layer.register_forward_hook(make_layer_hook(i)))
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        if attn is not None:
            def make_attn_hook(idx: int):
                def hook(_mod, _inp, out):
                    t = out[0] if isinstance(out, tuple) else out
                    tag_once(f"L{idx:02d}_attn_out", t)

                return hook

            hooks.append(attn.register_forward_hook(make_attn_hook(i)))

            # Fine-grained MLA internals. These modules live on `self_attn`
            # but are invoked from inside the MLA backend impl; forward hooks
            # still fire because they are called as modules.
            for sub_name, tag_suffix in (
                ("q_a_layernorm", "q_a"),
                ("kv_a_layernorm", "kv_c"),
                ("q_b_proj", "q_b"),
                ("kv_b_proj", "kv_b"),
                # DeepSeek-V4 (`vllm/models/deepseek_v4/attention.py`).
                ("q_norm", "q_a"),
                ("wq_b", "q_b"),
                ("kv_norm", "kv"),
            ):
                sub = getattr(attn, sub_name, None)
                if sub is None or not hasattr(sub, "register_forward_hook"):
                    continue

                def make_sub_hook(idx: int, suffix: str):
                    def hook(_mod, _inp, out):
                        t = out[0] if isinstance(out, tuple) else out
                        tag_once(f"L{idx:02d}_{suffix}", t)

                    return hook

                hooks.append(sub.register_forward_hook(make_sub_hook(i, tag_suffix)))

            # The input to o_proj is the concatenated per-head attention
            # output — pie's `attn_v`. DeepSeek-V4 splits the output
            # projection into wo_a (per-group) + wo_b; wo_a's input is the
            # post-inverse-RoPE attention output (pie's `attn_out`).
            o_proj = getattr(attn, "o_proj", None)
            if o_proj is not None and hasattr(o_proj, "register_forward_pre_hook"):
                def make_oproj_pre_hook(idx: int):
                    def hook(_mod, inp):
                        t = inp[0] if isinstance(inp, tuple) else inp
                        tag_once(f"L{idx:02d}_attn_v", t)

                    return hook

                hooks.append(o_proj.register_forward_pre_hook(make_oproj_pre_hook(i)))

            wo_a = getattr(attn, "wo_a", None)
            if wo_a is not None and hasattr(wo_a, "register_forward_pre_hook"):
                def make_woa_pre_hook(idx: int):
                    def hook(_mod, inp):
                        t = inp[0] if isinstance(inp, tuple) else inp
                        if t is not None and t.dim() == 3:
                            t = t.reshape(t.shape[0], -1)
                        tag_once(f"L{idx:02d}_wo_a_in", t)

                    return hook

                hooks.append(wo_a.register_forward_pre_hook(make_woa_pre_hook(i)))

            # DeepSeek-V4 fuses inverse-RoPE + wo_a + wo_b into `_o_proj`,
            # whose first argument is the raw attention output
            # [num_tokens, padded_heads, head_dim] — pie's `attn_out`.
            o_proj_fn = getattr(attn, "_o_proj", None)
            if callable(o_proj_fn):
                def make_o_proj_wrapper(idx: int, fn, mod):
                    def wrapped(o, positions):
                        t = o
                        if t is not None and t.dim() == 3:
                            t = t.reshape(t.shape[0], -1)
                        tag_once(f"L{idx:02d}_attn_out", t)
                        r = fn(o, positions)
                        rt = r[0] if isinstance(r, tuple) else r
                        tag_once(f"L{idx:02d}_o_proj", rt)
                        return r

                    return wrapped

                attn._o_proj = make_o_proj_wrapper(i, o_proj_fn, attn)

        # The layer's own input_layernorm gives pie's `norm_x`.
        in_ln = getattr(layer, "input_layernorm", None)
        if in_ln is not None and hasattr(in_ln, "register_forward_hook"):
            def make_in_ln_hook(idx: int):
                def hook(_mod, _inp, out):
                    t = out[0] if isinstance(out, tuple) else out
                    tag_once(f"L{idx:02d}_norm_x", t)

                return hook

            hooks.append(in_ln.register_forward_hook(make_in_ln_hook(i)))

        # The fused add-rmsnorm returns (normed, residual + attn_out), so its
        # second output IS the post-attention residual stream — the same thing
        # pie dumps as L<nn>_post_attn.
        post_ln = getattr(layer, "post_attention_layernorm", None) or getattr(
            layer, "ffn_norm", None
        )
        if post_ln is not None and hasattr(post_ln, "register_forward_hook"):
            def make_post_ln_hook(idx: int):
                def hook(_mod, _inp, out):
                    if isinstance(out, tuple) and len(out) > 1:
                        tag_once(f"L{idx:02d}_post_attn", out[1])

                return hook

            hooks.append(post_ln.register_forward_hook(make_post_ln_hook(i)))

    # The final norm's *input* is exactly the hyper-connection head output on
    # DeepSeek-V4, so hooking it gives a direct check on `hc_head`.
    if hasattr(inner, "norm") and hasattr(inner.norm, "register_forward_pre_hook"):
        def norm_pre_hook(_m, args):
            if args:
                tag_once("hc_head", args[0])
        hooks.append(inner.norm.register_forward_pre_hook(norm_pre_hook))
    if hasattr(inner, "norm") and hasattr(inner.norm, "register_forward_hook"):
        def norm_hook(_mod, _inp, out):
            tag_once("final_norm", out[0] if isinstance(out, tuple) else out)

        hooks.append(inner.norm.register_forward_hook(norm_hook))

    logits_holder = {}
    orig_compute = model.compute_logits

    def patched_compute(hidden_states, *a, **kw):
        res = orig_compute(hidden_states, *a, **kw)
        if res is not None and "logits" not in logits_holder:
            logits_holder["logits"] = res.detach().clone()
        return res

    model.compute_logits = patched_compute

    llm.generate(
        [TokensPrompt(prompt_token_ids=ids[0])],
        SamplingParams(temperature=0, max_tokens=1),
    )

    for h in hooks:
        h.remove()
    if "logits" in logits_holder:
        _save(out_dir, "logits", logits_holder["logits"])

    torch.cuda.synchronize()
    print(f"prompt_token_ids ({n_prompt}): {ids[0]}")
    print(f"wrote {len(sorted(out_dir.glob('s0_*.npy')))} tensors to {out_dir}")
    for f in sorted(out_dir.glob("s0_*.npy")):
        print("  ", f.name)


if __name__ == "__main__":
    main()
