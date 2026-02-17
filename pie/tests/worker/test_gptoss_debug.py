"""
GPT-OSS Weight & Forward Pass Diagnostic Script.

Loads the gpt_oss model using Engine.load and inspects:
  1. All weight names, shapes, and dtypes
  2. Detailed stats for layer 0 weights
  3. Per-layer hidden state norms during a forward pass
  4. Top-5 predicted tokens and probabilities

Run:
    cd /Users/ingim/Workspace/pie-mac/pie && \
    $HOME/.local/bin/uv run python tests/worker/test_gptoss_debug.py
"""

from __future__ import annotations

import sys
sys.path.insert(0, "src")

import torch
from transformers import AutoTokenizer

from pie_backend.engine import Engine
from pie_backend.config import RuntimeConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stat_line(name: str, t: torch.Tensor, show_values: bool = False, num_values: int = 8) -> None:
    """Print shape, dtype, and basic statistics for a tensor."""
    t_float = t.float()
    parts = [
        f"  {name}",
        f"shape={tuple(t.shape)}",
        f"dtype={t.dtype}",
    ]
    # Norm/min/max only make sense for numeric (non-quantized) tensors
    if t.dtype not in (torch.uint8,):
        parts.append(f"norm={t_float.norm().item():.6f}")
        parts.append(f"min={t_float.min().item():.6f}")
        parts.append(f"max={t_float.max().item():.6f}")
    if show_values:
        flat = t_float.flatten()
        vals = flat[:num_values].tolist()
        parts.append(f"values={[f'{v:.6f}' for v in vals]}")
    print("  |  ".join(parts))


def _shape_dtype(name: str, t: torch.Tensor) -> None:
    """Print only shape and dtype (for quantized weights where stats are meaningless)."""
    print(f"  {name}  |  shape={tuple(t.shape)}  |  dtype={t.dtype}")


# ---------------------------------------------------------------------------
# Section 1 & 2: Weight inspection
# ---------------------------------------------------------------------------

def inspect_weights(engine: Engine) -> None:
    weights = engine.forward_pass.weights

    print("\n" + "=" * 80)
    print("SECTION 1: ALL WEIGHT NAMES / SHAPES / DTYPES")
    print("=" * 80)
    for name in sorted(weights.keys()):
        t = weights.get(name)
        print(f"  {name:55s}  shape={str(tuple(t.shape)):30s}  dtype={t.dtype}")

    print("\n" + "=" * 80)
    print("SECTION 2: LAYER 0 WEIGHT DETAILS")
    print("=" * 80)

    layer = 0

    # norm_attn
    print("\n-- norm_attn --")
    _stat_line("norm_attn", weights.get(f"layers.{layer}.norm_attn"))

    # proj_qkv.weight
    print("\n-- proj_qkv.weight --")
    w = weights.get(f"layers.{layer}.proj_qkv.weight")
    t_float = w.float()
    print(f"  shape={tuple(w.shape)}  dtype={w.dtype}")
    print(f"  norm={t_float.norm().item():.6f}  min={t_float.min().item():.6f}  max={t_float.max().item():.6f}")
    print(f"  first 8 values: {t_float.flatten()[:8].tolist()}")

    # proj_qkv.bias
    print("\n-- proj_qkv.bias --")
    _stat_line("proj_qkv.bias", weights.get(f"layers.{layer}.proj_qkv.bias"))

    # proj_o
    print("\n-- proj_o --")
    _stat_line("proj_o", weights.get(f"layers.{layer}.proj_o"))

    # norm_mlp
    print("\n-- norm_mlp --")
    _stat_line("norm_mlp", weights.get(f"layers.{layer}.norm_mlp"))

    # router.weight
    print("\n-- router.weight --")
    _stat_line("router.weight", weights.get(f"layers.{layer}.router.weight"))

    # router.bias
    print("\n-- router.bias --")
    _stat_line("router.bias", weights.get(f"layers.{layer}.router.bias"))

    # moe.gemm1_weights (likely quantized uint8)
    print("\n-- moe.gemm1_weights --")
    _shape_dtype("moe.gemm1_weights", weights.get(f"layers.{layer}.moe.gemm1_weights"))

    # moe.gemm1_scales
    print("\n-- moe.gemm1_scales --")
    _shape_dtype("moe.gemm1_scales", weights.get(f"layers.{layer}.moe.gemm1_scales"))

    # moe.gemm1_bias
    print("\n-- moe.gemm1_bias --")
    _stat_line("moe.gemm1_bias", weights.get(f"layers.{layer}.moe.gemm1_bias"))

    # moe.gemm2_weights
    print("\n-- moe.gemm2_weights --")
    _shape_dtype("moe.gemm2_weights", weights.get(f"layers.{layer}.moe.gemm2_weights"))

    # moe.gemm2_scales
    print("\n-- moe.gemm2_scales --")
    _shape_dtype("moe.gemm2_scales", weights.get(f"layers.{layer}.moe.gemm2_scales"))

    # moe.gemm2_bias
    print("\n-- moe.gemm2_bias --")
    _stat_line("moe.gemm2_bias", weights.get(f"layers.{layer}.moe.gemm2_bias"))

    # attn_sinks
    print("\n-- attn_sinks --")
    sinks = weights.get(f"layers.{layer}.attn_sinks")
    _stat_line("attn_sinks", sinks, show_values=True, num_values=16)


# ---------------------------------------------------------------------------
# Section 3: Instrumented forward pass
# ---------------------------------------------------------------------------

def instrumented_forward(engine: Engine, token_ids: torch.Tensor) -> torch.Tensor:
    """
    Run a single prefill forward pass while printing per-layer norms.

    Mirrors the logic in ForwardPass.transform() but adds diagnostic prints
    after the attention and MoE sub-blocks of each layer.
    """
    import torch.nn.functional as fun

    fp = engine.forward_pass
    cfg = fp.model_config
    config = fp.runtime_config
    device = config.device

    print("\n" + "=" * 80)
    print("SECTION 3: INSTRUMENTED FORWARD PASS")
    print("=" * 80)

    # Embed
    embeddings = fp.embed_tokens(token_ids)
    print(f"\n  Embeddings: shape={tuple(embeddings.shape)}, norm={embeddings.float().norm().item():.6f}")

    # Prepare paged KV cache inputs (single sequence, prefill)
    seq_len = token_ids.shape[0]
    page_size = config.kv_page_size
    num_pages_needed = (seq_len + page_size - 1) // page_size

    kv_page_indices = torch.arange(num_pages_needed, dtype=torch.int32, device=device)
    kv_page_indptr = torch.tensor([0, num_pages_needed], dtype=torch.int32, device=device)
    kv_last_page_lens = torch.tensor(
        [seq_len % page_size or page_size], dtype=torch.int32, device=device
    )
    qo_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device)

    hidden_states = embeddings
    n = hidden_states.size(0)

    # Compute batch_indices and batch_positions (needed by attention)
    try:
        import flashinfer_metal as ops
    except ImportError:
        import flashinfer as ops

    seq_lens = ops.get_seq_lens(kv_page_indptr, kv_last_page_lens, page_size)
    batch_indices, batch_positions = ops.get_batch_indices_positions(
        append_indptr=qo_indptr,
        seq_lens=seq_lens,
        nnz=n,
    )

    # Plan attention wrappers
    local_num_q_heads = cfg.num_q_heads // fp.tp_size
    local_num_kv_heads = cfg.num_kv_heads // fp.tp_size

    fp.wrapper_window.plan(
        qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_lens,
        local_num_q_heads, local_num_kv_heads, cfg.dim_head, page_size,
        causal=True, window_left=cfg.sliding_window - 1,
        q_data_type=config.activation_dtype,
        kv_data_type=config.activation_dtype,
        non_blocking=True,
    )
    fp.wrapper_full.plan(
        qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_lens,
        local_num_q_heads, local_num_kv_heads, cfg.dim_head, page_size,
        causal=True, window_left=-1,
        q_data_type=config.activation_dtype,
        kv_data_type=config.activation_dtype,
        non_blocking=True,
    )

    kv_cache = engine.kv_cache_at_layer

    for layer_idx in range(cfg.num_layers):
        h_in = hidden_states.float().norm().item()

        wrapper = fp.wrapper_window if layer_idx % 2 == 0 else fp.wrapper_full

        # --- Attention ---
        residual_before_attn = hidden_states.clone()
        hidden_states = fp.attention(
            hidden_states=hidden_states,
            layer_idx=layer_idx,
            position_ids=position_ids,
            kv_cache_layer=kv_cache[layer_idx],
            kv_page_indices=kv_page_indices,
            kv_page_indptr=kv_page_indptr,
            kv_last_page_lens=kv_last_page_lens,
            batch_indices=batch_indices,
            batch_positions=batch_positions,
            adapter_subpass=None,
            wrapper=wrapper,
        )
        h_after_attn = hidden_states.float().norm().item()
        attn_delta = (hidden_states - residual_before_attn).float().norm().item()

        # --- MoE ---
        residual_before_moe = hidden_states.clone()
        hidden_states = fp.moe(hidden_states, layer_idx)
        h_after_moe = hidden_states.float().norm().item()
        moe_delta = (hidden_states - residual_before_moe).float().norm().item()

        print(
            f"  Layer {layer_idx:3d}  |  "
            f"in={h_in:12.4f}  |  "
            f"post_attn={h_after_attn:12.4f}  attn_delta={attn_delta:12.4f}  |  "
            f"post_moe={h_after_moe:12.4f}  moe_delta={moe_delta:12.4f}"
        )

    return hidden_states


# ---------------------------------------------------------------------------
# Section 4: Top-5 predictions
# ---------------------------------------------------------------------------

def show_top_predictions(engine: Engine, hidden_states: torch.Tensor, tokenizer) -> None:
    print("\n" + "=" * 80)
    print("SECTION 4: TOP-5 PREDICTED TOKENS")
    print("=" * 80)

    logits = engine.forward_pass.lm_head(hidden_states)
    # Take the last position
    last_logits = logits[-1, :]
    probs = torch.softmax(last_logits.float(), dim=-1)
    top5_probs, top5_ids = torch.topk(probs, k=5)

    for rank, (tid, prob) in enumerate(zip(top5_ids.tolist(), top5_probs.tolist())):
        token_str = tokenizer.decode([tid])
        print(f"  #{rank+1}  token_id={tid:8d}  prob={prob:.6f}  decoded={repr(token_str)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    model_name = "openai/gpt-oss-20b"
    prompt = "Hello, my name is"

    print("=" * 80)
    print("GPT-OSS DIAGNOSTIC SCRIPT")
    print(f"Model: {model_name}")
    print(f"Prompt: {repr(prompt)}")
    print("=" * 80)

    # Load tokenizer
    print("\n[*] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load engine
    print("[*] Loading engine ...")
    config = RuntimeConfig.from_args(hf_repo=model_name)
    engine = Engine.load(config)
    print(f"    Architecture: {engine.arch_type}")
    print(f"    Layers: {engine.model_config.num_layers}")
    print(f"    Device: {config.device}")
    print(f"    Activation dtype: {config.activation_dtype}")

    # Section 1 & 2: Weight inspection
    inspect_weights(engine)

    # Tokenize
    token_ids = tokenizer.encode(prompt, return_tensors="pt").squeeze(0).to(config.device)
    print(f"\n[*] Token IDs ({len(token_ids)} tokens): {token_ids.tolist()}")

    # Section 3: Instrumented forward pass
    with torch.inference_mode():
        hidden_states = instrumented_forward(engine, token_ids)

    # Section 4: Top-5 predictions
    with torch.inference_mode():
        show_top_predictions(engine, hidden_states, tokenizer)

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
