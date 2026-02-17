"""
Layer-0 step-by-step comparison: model vs manual PyTorch.
Pinpoints exactly where the divergence occurs.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "src")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from pie_backend.engine import Engine
from pie_backend.config import RuntimeConfig


def manual_rms_norm(x, weight, eps):
    """RMSNorm matching HuggingFace GPT-OSS implementation."""
    input_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight.float() * x).to(input_dtype)


def manual_rope_neox(q, k, cos_sin_cache, positions, head_dim):
    """Apply RoPE with cos/sin cache, NeoX-style (split halves)."""
    half = head_dim // 2
    for i in range(q.shape[0]):
        pos = positions[i].item()
        cos = cos_sin_cache[pos, :half]
        sin = cos_sin_cache[pos, half:]
        for h in range(q.shape[1]):
            x = q[i, h, :half].float()
            y = q[i, h, half:].float()
            q[i, h, :half] = (x * cos - y * sin).to(q.dtype)
            q[i, h, half:] = (x * sin + y * cos).to(q.dtype)
    for i in range(k.shape[0]):
        pos = positions[i].item()
        cos = cos_sin_cache[pos, :half]
        sin = cos_sin_cache[pos, half:]
        for h in range(k.shape[1]):
            x = k[i, h, :half].float()
            y = k[i, h, half:].float()
            k[i, h, :half] = (x * cos - y * sin).to(k.dtype)
            k[i, h, half:] = (x * sin + y * cos).to(k.dtype)
    return q, k


_FP4_VALUES = (
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def _dequantize_expert_from_fp4(blocks, scales, dtype):
    """Dequantize a single expert's FP4 packed weights (test helper)."""
    device = blocks.device
    rows, b = blocks.shape
    lut = torch.tensor(_FP4_VALUES, dtype=dtype, device=device)
    idx_lo = (blocks & 0x0F).to(torch.long)
    idx_hi = (blocks >> 4).to(torch.long)
    out = torch.empty(rows, b * 2, dtype=dtype, device=device)
    out[:, 0::2] = lut[idx_lo]
    out[:, 1::2] = lut[idx_hi]
    scale_exp = scales.to(torch.int32) - 127
    scale_exp = scale_exp.repeat_interleave(32, dim=1)
    torch.ldexp(out, scale_exp, out=out)
    return out


def manual_moe_layer(normed, router_w, router_b, gemm1_blocks, gemm1_scales,
                     gemm1_bias, gemm2_blocks, gemm2_scales, gemm2_bias,
                     intermediate_size, num_experts, top_k,
                     alpha, clamp_limit, dtype):
    """Manual MoE computation matching HF reference."""

    n, hidden_dim = normed.shape
    padded_hidden = gemm1_blocks.shape[-1] * 2  # unpack FP4

    # Pad input
    if hidden_dim != padded_hidden:
        padded = torch.zeros(n, padded_hidden, dtype=normed.dtype, device=normed.device)
        padded[:, :hidden_dim] = normed
        x = padded
    else:
        x = normed

    # Router
    logits = F.linear(normed, router_w, router_b).float()
    scores = torch.softmax(logits, dim=-1)
    topk_w, topk_idx = torch.topk(scores, top_k, dim=-1)
    topk_w = (topk_w / topk_w.sum(dim=-1, keepdim=True)).to(dtype)

    output = torch.zeros(n, padded_hidden, dtype=torch.float32, device=normed.device)

    for exp_id in range(num_experts):
        mask = (topk_idx == exp_id).any(dim=-1)
        if not mask.any():
            continue

        # Get routing weights for this expert
        exp_weights = torch.zeros(n, dtype=dtype, device=normed.device)
        for k in range(top_k):
            m = topk_idx[:, k] == exp_id
            exp_weights[m] += topk_w[:, k][m]

        tokens = mask.nonzero(as_tuple=True)[0]
        x_exp = x[tokens]

        # Dequant expert weights
        w1 = _dequantize_expert_from_fp4(gemm1_blocks[exp_id], gemm1_scales[exp_id], dtype)
        w2 = _dequantize_expert_from_fp4(gemm2_blocks[exp_id], gemm2_scales[exp_id], dtype)

        # GEMM1 + bias
        g1 = F.linear(x_exp, w1)
        if gemm1_bias is not None:
            g1 = g1 + gemm1_bias[exp_id]

        # GPT-OSS activation
        up = g1[:, :intermediate_size]
        gate = g1[:, intermediate_size:]
        gate = gate.clamp(max=clamp_limit)
        up = up.clamp(-clamp_limit, clamp_limit)
        glu = gate * torch.sigmoid(gate * alpha)
        activated = (up + 1) * glu

        # GEMM2 + bias
        g2 = F.linear(activated, w2)
        if gemm2_bias is not None:
            g2 = g2 + gemm2_bias[exp_id]

        # Weighted accumulate
        output[tokens] += (g2 * exp_weights[tokens].unsqueeze(-1)).float()

    if hidden_dim != padded_hidden:
        output = output[:, :hidden_dim]
    return output.to(dtype)


def main():
    model_name = "openai/gpt-oss-20b"
    prompt = "Hello, my name is"

    print("=" * 80)
    print("LAYER 0 STEP-BY-STEP COMPARISON")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = RuntimeConfig.from_args(hf_repo=model_name)
    engine = Engine.load(config)
    fp = engine.forward_pass
    cfg = fp.model_config
    device = config.device

    token_ids = tokenizer.encode(prompt, return_tensors="pt").squeeze(0).to(device)
    print(f"Tokens: {token_ids.tolist()}")

    with torch.inference_mode():
        # Step 1: Embedding
        embed = fp.embed_tokens(token_ids)
        print(f"\n[Embedding] shape={tuple(embed.shape)}, norm={embed.float().norm():.4f}")

        # Step 2: RMSNorm (layer 0)
        norm_w = fp.weights.get("layers.0.norm_attn")
        manual_normed = manual_rms_norm(embed, norm_w, cfg.rms_norm_eps)

        # Step 3: QKV projection
        qkv_w = fp.weights.get("layers.0.proj_qkv.weight")
        qkv_b = fp.weights.get("layers.0.proj_qkv.bias")
        manual_qkv = F.linear(manual_normed, qkv_w, qkv_b)

        # Split Q, K, V
        q_size = cfg.num_q_heads * cfg.dim_head
        kv_size = cfg.num_kv_heads * cfg.dim_head
        q, k, v = torch.split(manual_qkv, [q_size, kv_size, kv_size], dim=-1)
        n = q.shape[0]
        q = q.view(n, cfg.num_q_heads, cfg.dim_head)
        k = k.view(n, cfg.num_kv_heads, cfg.dim_head)
        v = v.view(n, cfg.num_kv_heads, cfg.dim_head)

        # Step 4: RoPE
        positions = torch.arange(n, dtype=torch.int32, device=device)
        cos_sin_cache = fp._rope_cos_sin_cache
        q_manual = q.clone()
        k_manual = k.clone()
        q_manual, k_manual = manual_rope_neox(q_manual, k_manual, cos_sin_cache, positions, cfg.dim_head)

        # Step 5-7: Manual attention + O proj + residual
        from flashinfer_metal._wrappers import trtllm_fp4_block_scale_moe

        sinks = fp.weights.get("layers.0.attn_sinks")
        scaling = cfg.dim_head ** -0.5

        # Simple attention (causal, GQA, with sinks)
        gqa_ratio = cfg.num_q_heads // cfg.num_kv_heads
        k_expanded = k_manual.repeat_interleave(gqa_ratio, dim=1)
        v_expanded = v.repeat_interleave(gqa_ratio, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q_manual.float(), k_expanded.float()) * scaling
        causal_mask = torch.triu(torch.ones(n, n, device=device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
        if sinks is not None and sinks.numel() > 0:
            sink_logits = sinks.unsqueeze(-1).unsqueeze(-1).expand(-1, n, 1)
            scores = torch.cat([sink_logits.float(), scores], dim=-1)
            probs = torch.softmax(scores, dim=-1)
            probs = probs[:, :, 1:]
        else:
            probs = torch.softmax(scores, dim=-1)
        manual_attn = torch.einsum("hqk,khd->qhd", probs, v_expanded.float()).to(q.dtype)
        manual_attn_flat = manual_attn.reshape(n, -1)

        proj_o = fp.weights.get("layers.0.proj_o")
        manual_o = F.linear(manual_attn_flat, proj_o)
        after_attn = embed + manual_o
        print(f"[After attn residual] norm={after_attn.float().norm():.4f}")

        # Step 8: MoE
        norm_mlp_w = fp.weights.get("layers.0.norm_mlp")
        moe_normed = manual_rms_norm(after_attn, norm_mlp_w, cfg.rms_norm_eps)

        router_w = fp.weights.get("layers.0.router.weight")
        router_b = fp.weights.get("layers.0.router.bias")
        gemm1_w = fp.weights.get("layers.0.moe.gemm1_weights")
        gemm1_s = fp.weights.get("layers.0.moe.gemm1_scales")
        gemm1_b = fp.weights.get("layers.0.moe.gemm1_bias")
        gemm2_w = fp.weights.get("layers.0.moe.gemm2_weights")
        gemm2_s = fp.weights.get("layers.0.moe.gemm2_scales")
        gemm2_b = fp.weights.get("layers.0.moe.gemm2_bias")

        padded_I = fp.padded_intermediate_size
        padded_H = fp.padded_hidden_size
        manual_moe_out = manual_moe_layer(
            moe_normed, router_w, router_b,
            gemm1_w, gemm1_s, gemm1_b,
            gemm2_w, gemm2_s, gemm2_b,
            padded_I, cfg.num_experts, cfg.experts_per_token,
            cfg.swiglu_alpha, cfg.swiglu_limit, config.activation_dtype,
        )
        after_moe = after_attn + manual_moe_out
        print(f"[After MoE residual] manual norm={after_moe.float().norm():.4f}")

        # Compare Metal MoE vs PyTorch fallback MoE vs manual MoE
        moe_input = moe_normed.to(torch.bfloat16)
        if cfg.dim_hidden != padded_H:
            padded = torch.zeros(n, padded_H, dtype=moe_input.dtype, device=device)
            padded[:, :cfg.dim_hidden] = moe_input
            moe_input = padded

        router_logits = F.linear(moe_normed, router_w, router_b).to(torch.bfloat16)

        # Run Metal path
        metal_out = trtllm_fp4_block_scale_moe(
            routing_logits=router_logits,
            routing_bias=None,
            hidden_states=moe_input.clone(),
            hidden_states_scale=None,
            gemm1_weights=gemm1_w,
            gemm1_weights_scale=gemm1_s,
            gemm1_bias=gemm1_b,
            gemm1_alpha=fp._gemm1_alpha,
            gemm1_beta=fp._gemm1_beta,
            gemm1_clamp_limit=fp._gemm1_clamp_limit,
            gemm2_weights=gemm2_w,
            gemm2_weights_scale=gemm2_s,
            gemm2_bias=gemm2_b,
            output1_scale_scalar=fp._output1_scale,
            output1_scale_gate_scalar=fp._output1_scale_gate,
            output2_scale_scalar=fp._output2_scale,
            num_experts=cfg.num_experts,
            top_k=cfg.experts_per_token,
            n_group=None,
            topk_group=None,
            intermediate_size=padded_I,
            local_expert_offset=0,
            local_num_experts=cfg.num_experts,
            routed_scaling_factor=None,
            routing_method_type=1,
            gated_act_type=0,
            do_finalize=True,
            tune_max_num_tokens=64,
        )[0]
        if cfg.dim_hidden != padded_H:
            metal_out = metal_out[:, :cfg.dim_hidden]
        metal_out = metal_out.to(moe_normed.dtype)

        # Compare
        mf_diff = (metal_out - manual_moe_out).float().norm()
        mf_rel = mf_diff / manual_moe_out.float().norm()
        print(f"\n[Metal MoE vs Manual] Metal_norm={metal_out.float().norm():.4f}, "
              f"Manual_norm={manual_moe_out.float().norm():.4f}")
        print(f"  abs_diff={mf_diff:.4f}, rel_diff={mf_rel:.6f}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
