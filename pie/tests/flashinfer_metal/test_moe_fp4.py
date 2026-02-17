"""Tests for trtllm_fp4_block_scale_moe (pure PyTorch MoE fallback)."""

import pytest
import torch

from .conftest import BenchmarkTimer


def make_moe_inputs(
    num_tokens: int = 8,
    hidden_dim: int = 64,
    intermediate_size: int = 128,
    num_experts: int = 4,
    top_k: int = 2,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = torch.device("cpu"),
):
    """Create realistic MoE input tensors."""
    return dict(
        routing_logits=torch.randn(num_tokens, num_experts, dtype=torch.bfloat16, device=device),
        routing_bias=None,
        hidden_states=torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device),
        hidden_states_scale=None,
        gemm1_weights=torch.randn(num_experts, 2 * intermediate_size, hidden_dim, dtype=dtype, device=device),
        gemm1_weights_scale=torch.ones(num_experts, dtype=dtype, device=device),
        gemm1_bias=None,
        gemm1_alpha=1.0,
        gemm1_beta=0.0,
        gemm1_clamp_limit=100.0,
        gemm2_weights=torch.randn(num_experts, hidden_dim, intermediate_size, dtype=dtype, device=device),
        gemm2_weights_scale=torch.ones(num_experts, dtype=dtype, device=device),
        gemm2_bias=None,
        output1_scale_scalar=1.0,
        output1_scale_gate_scalar=1.0,
        output2_scale_scalar=1.0,
        num_experts=num_experts,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate_size,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=None,
        routing_method_type=1,
        gated_act_type=0,
        do_finalize=True,
        tune_max_num_tokens=4096,
    )


def reference_moe(
    hidden_states, routing_logits, w1, w2, intermediate_size, top_k,
):
    """Simple reference MoE: route, project, SwiGLU, project back."""
    dtype = hidden_states.dtype
    num_tokens, hidden_dim = hidden_states.shape
    num_experts = routing_logits.shape[1]

    scores = torch.softmax(routing_logits.float(), dim=-1)
    topk_weights, topk_indices = torch.topk(scores, top_k, dim=-1)
    topk_weights = (topk_weights / topk_weights.sum(dim=-1, keepdim=True)).to(dtype)

    output = torch.zeros_like(hidden_states)
    for k_idx in range(top_k):
        for expert_id in range(num_experts):
            mask = topk_indices[:, k_idx] == expert_id
            if not mask.any():
                continue
            x = hidden_states[mask]
            g1 = x @ w1[expert_id].T
            gate = torch.nn.functional.silu(g1[:, :intermediate_size])
            up = g1[:, intermediate_size:]
            activated = gate * up
            g2 = activated @ w2[expert_id].T
            output[mask] += g2 * topk_weights[mask, k_idx].unsqueeze(-1)
    return output


class TestMoeFp4:
    """Accuracy and benchmark tests for trtllm_fp4_block_scale_moe."""

    def test_output_shape(self, device):
        """Output should be tuple with [0] shaped [num_tokens, hidden_dim]."""
        from flashinfer_metal import trtllm_fp4_block_scale_moe

        inputs = make_moe_inputs(num_tokens=8, hidden_dim=64, device=device)
        result = trtllm_fp4_block_scale_moe(**inputs)

        assert isinstance(result, tuple)
        assert result[0].shape == (8, 64)
        assert result[0].dtype == torch.bfloat16

    def test_accuracy_vs_reference(self, device):
        """MoE output should match manual reference (unit scales, no bias)."""
        from flashinfer_metal import trtllm_fp4_block_scale_moe

        num_tokens, hidden_dim, intermediate_size = 8, 64, 128
        num_experts, top_k = 4, 2

        inputs = make_moe_inputs(
            num_tokens=num_tokens, hidden_dim=hidden_dim,
            intermediate_size=intermediate_size, num_experts=num_experts,
            top_k=top_k, device=device,
        )
        result = trtllm_fp4_block_scale_moe(**inputs)[0]

        ref = reference_moe(
            inputs["hidden_states"], inputs["routing_logits"],
            inputs["gemm1_weights"], inputs["gemm2_weights"],
            intermediate_size, top_k,
        )

        torch.testing.assert_close(result, ref, atol=1e-3, rtol=1e-3)

    def test_routing_selects_top_k(self, device):
        """With deterministic routing, only top-k experts should contribute."""
        from flashinfer_metal import trtllm_fp4_block_scale_moe

        num_tokens, hidden_dim, intermediate_size = 4, 32, 64
        num_experts, top_k = 4, 1

        inputs = make_moe_inputs(
            num_tokens=num_tokens, hidden_dim=hidden_dim,
            intermediate_size=intermediate_size, num_experts=num_experts,
            top_k=top_k, device=device,
        )
        # Make routing logits deterministic: all tokens go to expert 0
        inputs["routing_logits"] = torch.zeros(num_tokens, num_experts, dtype=torch.bfloat16, device=device)
        inputs["routing_logits"][:, 0] = 100.0

        result = trtllm_fp4_block_scale_moe(**inputs)[0]

        # With top_k=1 and all tokens routed to expert 0, weights are 1.0
        # Verify output is non-zero (expert 0 contributes)
        assert result.abs().sum() > 0

    def test_routing_bias(self, device):
        """Routing bias should shift expert selection."""
        from flashinfer_metal import trtllm_fp4_block_scale_moe

        inputs = make_moe_inputs(num_tokens=4, hidden_dim=32, device=device)

        # Without bias
        inputs["routing_bias"] = None
        r1 = trtllm_fp4_block_scale_moe(**inputs)[0]

        # With large bias favoring expert 0
        bias = torch.zeros(inputs["num_experts"], dtype=torch.bfloat16, device=device)
        bias[0] = 100.0
        inputs["routing_bias"] = bias
        r2 = trtllm_fp4_block_scale_moe(**inputs)[0]

        # Results should differ
        assert not torch.allclose(r1, r2)

    def test_routed_scaling_factor(self, device):
        """routed_scaling_factor should scale the output."""
        from flashinfer_metal import trtllm_fp4_block_scale_moe

        inputs = make_moe_inputs(num_tokens=4, hidden_dim=32, device=device)

        inputs["routed_scaling_factor"] = None
        r1 = trtllm_fp4_block_scale_moe(**inputs)[0]

        inputs["routed_scaling_factor"] = 2.0
        r2 = trtllm_fp4_block_scale_moe(**inputs)[0]

        torch.testing.assert_close(r2, r1 * 2.0, atol=1e-3, rtol=1e-3)

    def test_benchmark(self, device):
        """Benchmark MoE fallback."""
        from flashinfer_metal import trtllm_fp4_block_scale_moe

        inputs = make_moe_inputs(
            num_tokens=64, hidden_dim=256, intermediate_size=512,
            num_experts=8, top_k=2, device=device,
        )

        timer = BenchmarkTimer("moe_fp4", device)

        def run():
            return trtllm_fp4_block_scale_moe(**inputs)

        _, ms = timer.run(run)
        print(f"\n  moe_fp4 [64 tokens, 256 dim, 8 experts, top-2]: {ms:.3f} ms")
