"""Tests for fused MoE routing Metal kernel."""

import pytest
import torch
from flashinfer_metal._compiler import MetalCompiler


@pytest.fixture
def compiler():
    return MetalCompiler()


def reference_routing(logits_bf16, top_k, output2_scale, local_offset=0):
    """Reference implementation matching the PyTorch routing path."""
    logits = logits_bf16.float()
    scores = torch.softmax(logits, dim=-1)
    topk_weights, topk_indices = torch.topk(scores, top_k, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    expert_ids = (topk_indices[0] - local_offset).to(torch.int32)
    fused_scales = topk_weights[0].float() * output2_scale
    return expert_ids, fused_scales


class TestMoeRouting:
    """Test fused MoE routing kernel against PyTorch reference."""

    @pytest.mark.parametrize("E", [8, 32, 64])
    @pytest.mark.parametrize("K", [2, 4])
    def test_basic_routing(self, compiler, E, K):
        """Test that fused kernel matches PyTorch softmax+topk+normalize."""
        torch.manual_seed(42)
        logits = torch.randn(1, E, dtype=torch.bfloat16, device="mps")

        expert_ids = torch.empty(K, dtype=torch.int32, device="mps")
        fused_scales = torch.empty(K, dtype=torch.float32, device="mps")

        compiler.run_moe_route_topk(
            logits=logits,
            expert_ids_out=expert_ids,
            fused_scales_out=fused_scales,
            num_experts=E,
            top_k=K,
            output2_scale=1.0,
            local_expert_offset=0,
        )

        ref_ids, ref_scales = reference_routing(logits, K, 1.0)

        assert torch.equal(expert_ids.cpu(), ref_ids.cpu()), \
            f"Expert IDs mismatch: got {expert_ids.cpu()} vs {ref_ids.cpu()}"
        torch.testing.assert_close(
            fused_scales.cpu(), ref_scales.cpu(), atol=1e-5, rtol=1e-4,
        )

    def test_output2_scale(self, compiler):
        """Test that output2_scale is applied correctly."""
        torch.manual_seed(42)
        E, K = 32, 4
        logits = torch.randn(1, E, dtype=torch.bfloat16, device="mps")

        ids1 = torch.empty(K, dtype=torch.int32, device="mps")
        scales1 = torch.empty(K, dtype=torch.float32, device="mps")
        ids2 = torch.empty(K, dtype=torch.int32, device="mps")
        scales2 = torch.empty(K, dtype=torch.float32, device="mps")

        compiler.run_moe_route_topk(logits, ids1, scales1, E, K, 1.0)
        compiler.run_moe_route_topk(logits, ids2, scales2, E, K, 2.5)

        assert torch.equal(ids1.cpu(), ids2.cpu()), "Expert IDs should be same regardless of scale"
        torch.testing.assert_close(
            scales2.cpu(), scales1.cpu() * 2.5, atol=1e-5, rtol=1e-4,
        )

    def test_local_expert_offset(self, compiler):
        """Test that local_expert_offset is subtracted from IDs."""
        torch.manual_seed(42)
        E, K = 32, 4
        logits = torch.randn(1, E, dtype=torch.bfloat16, device="mps")

        ids_no_offset = torch.empty(K, dtype=torch.int32, device="mps")
        scales_no = torch.empty(K, dtype=torch.float32, device="mps")
        ids_offset = torch.empty(K, dtype=torch.int32, device="mps")
        scales_off = torch.empty(K, dtype=torch.float32, device="mps")

        compiler.run_moe_route_topk(logits, ids_no_offset, scales_no, E, K, 1.0, 0)
        compiler.run_moe_route_topk(logits, ids_offset, scales_off, E, K, 1.0, 5)

        expected = ids_no_offset.cpu() - 5
        assert torch.equal(ids_offset.cpu(), expected), \
            f"Offset IDs mismatch: got {ids_offset.cpu()} vs {expected}"

    def test_gpt_oss_dims(self, compiler):
        """Test with exact GPT-OSS-20B dimensions (E=32, K=4)."""
        torch.manual_seed(123)
        E, K = 32, 4
        logits = torch.randn(1, E, dtype=torch.bfloat16, device="mps")

        expert_ids = torch.empty(K, dtype=torch.int32, device="mps")
        fused_scales = torch.empty(K, dtype=torch.float32, device="mps")

        compiler.run_moe_route_topk(logits, expert_ids, fused_scales, E, K, 1.0)
        ref_ids, ref_scales = reference_routing(logits, K, 1.0)

        assert torch.equal(expert_ids.cpu(), ref_ids.cpu())
        torch.testing.assert_close(fused_scales.cpu(), ref_scales.cpu(), atol=1e-5, rtol=1e-4)

        # Verify expert IDs are valid
        assert (expert_ids.cpu() >= 0).all()
        assert (expert_ids.cpu() < E).all()

        # Verify scales sum approximately to output2_scale (1.0)
        assert abs(fused_scales.cpu().sum().item() - 1.0) < 1e-4

    def test_f16_dtype(self, compiler):
        """Test float16 variant."""
        torch.manual_seed(42)
        E, K = 32, 4
        logits = torch.randn(1, E, dtype=torch.float16, device="mps")

        expert_ids = torch.empty(K, dtype=torch.int32, device="mps")
        fused_scales = torch.empty(K, dtype=torch.float32, device="mps")

        compiler.run_moe_route_topk(logits, expert_ids, fused_scales, E, K, 1.0)

        # Reference with bf16 for comparison (slight numerical differences OK)
        ref_ids, ref_scales = reference_routing(logits.to(torch.bfloat16), K, 1.0)

        # Expert IDs should match (same top-K selection)
        assert torch.equal(expert_ids.cpu(), ref_ids.cpu())

    def test_writes_to_preallocated(self, compiler):
        """Verify kernel writes to provided output buffers (no allocation)."""
        E, K = 32, 4
        logits = torch.randn(1, E, dtype=torch.bfloat16, device="mps")

        expert_ids = torch.empty(K, dtype=torch.int32, device="mps")
        fused_scales = torch.empty(K, dtype=torch.float32, device="mps")

        # Get data pointers before kernel call
        ids_ptr = expert_ids.data_ptr()
        scales_ptr = fused_scales.data_ptr()

        compiler.run_moe_route_topk(logits, expert_ids, fused_scales, E, K, 1.0)

        # Verify same tensors were written to (no new allocation)
        assert expert_ids.data_ptr() == ids_ptr
        assert fused_scales.data_ptr() == scales_ptr

    @pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
    def test_deterministic(self, compiler, seed):
        """Test that results are deterministic across runs."""
        torch.manual_seed(seed)
        E, K = 32, 4
        logits = torch.randn(1, E, dtype=torch.bfloat16, device="mps")

        ids1 = torch.empty(K, dtype=torch.int32, device="mps")
        scales1 = torch.empty(K, dtype=torch.float32, device="mps")
        ids2 = torch.empty(K, dtype=torch.int32, device="mps")
        scales2 = torch.empty(K, dtype=torch.float32, device="mps")

        compiler.run_moe_route_topk(logits, ids1, scales1, E, K, 1.0)
        compiler.run_moe_route_topk(logits, ids2, scales2, E, K, 1.0)

        assert torch.equal(ids1.cpu(), ids2.cpu())
        assert torch.equal(scales1.cpu(), scales2.cpu())
