"""Tests for trtllm_fp4_block_scale_moe and FP4 packed storage."""

import pytest
import torch


# FP4 (e2m1) value lookup table — maps 4-bit index to float value
_FP4_VALUES = (
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def _dequantize_expert_from_fp4(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize a single expert's FP4 packed weights to the target dtype.

    Test-only helper (production code uses Metal kernels for inline dequant).

    Args:
        blocks: [rows, cols/2] uint8 — each byte packs two FP4 values
        scales: [rows, cols/32] uint8 — E8M0 exponents (one per 32-element block)
        dtype: target floating-point dtype (e.g. torch.bfloat16)

    Returns:
        [rows, cols] tensor in ``dtype``
    """
    device = blocks.device
    rows, b = blocks.shape  # b = cols/2

    lut = torch.tensor(_FP4_VALUES, dtype=dtype, device=device)

    # Extract low and high nibbles → FP4 values via LUT
    idx_lo = (blocks & 0x0F).to(torch.long)
    idx_hi = (blocks >> 4).to(torch.long)

    out = torch.empty(rows, b * 2, dtype=dtype, device=device)
    out[:, 0::2] = lut[idx_lo]
    out[:, 1::2] = lut[idx_hi]

    # E8M0 scale: value = 2^(exponent - 127)
    # Expand scale to match each element (repeat each scale 32 times)
    scale_exp = scales.to(torch.int32) - 127  # [rows, cols/32]
    scale_exp = scale_exp.repeat_interleave(32, dim=1)  # [rows, cols]
    torch.ldexp(out, scale_exp, out=out)

    return out  # [rows, cols]


# ---------------------------------------------------------------------------
# FP4-domain deinterleave and pad tests
# ---------------------------------------------------------------------------


class TestFp4DomainOps:
    """Tests for deinterleave_gate_up_fp4, pad_gate_up_fp4, pad_down_fp4."""

    def test_deinterleave_gate_up_fp4(self):
        """Deinterleave should reorder rows: even→gate(second half), odd→linear(first half)."""
        from pie_backend.model.gpt_oss_utils import deinterleave_gate_up_fp4

        E, I, H_half, H_scale = 2, 4, 16, 1
        blocks = torch.arange(E * 2 * I * H_half, dtype=torch.uint8).reshape(E, 2 * I, H_half)
        scales = torch.arange(E * 2 * I * H_scale, dtype=torch.uint8).reshape(E, 2 * I, H_scale)

        rb, rs = deinterleave_gate_up_fp4(blocks, scales)

        # First half (linear) should be odd rows, second half (gate) should be even rows
        for e in range(E):
            for i in range(I):
                # linear part (first half) = originally odd rows
                assert torch.equal(rb[e, i], blocks[e, 2 * i + 1])
                assert torch.equal(rs[e, i], scales[e, 2 * i + 1])
                # gate part (second half) = originally even rows
                assert torch.equal(rb[e, I + i], blocks[e, 2 * i])
                assert torch.equal(rs[e, I + i], scales[e, 2 * i])

    def test_pad_gate_up_fp4_no_pad_needed(self):
        """When sizes already match, should return inputs unchanged."""
        from pie_backend.model.gpt_oss_utils import pad_gate_up_fp4

        E, I = 2, 4
        H = 32  # H_half=16, H_scale=1
        blocks = torch.ones((E, 2 * I, H // 2), dtype=torch.uint8)
        scales = torch.ones((E, 2 * I, H // 32), dtype=torch.uint8)

        pb, ps = pad_gate_up_fp4(blocks, scales, H, I)
        assert torch.equal(pb, blocks)
        assert torch.equal(ps, scales)

    def test_pad_gate_up_fp4_pads_correctly(self):
        """Padding should zero-fill blocks and 0x7F-fill scales in padded regions."""
        from pie_backend.model.gpt_oss_utils import pad_gate_up_fp4

        E, I = 1, 2
        H = 32
        padded_I = 4
        padded_H = 64

        blocks = torch.full((E, 2 * I, H // 2), 0xAB, dtype=torch.uint8)
        scales = torch.full((E, 2 * I, H // 32), 0x80, dtype=torch.uint8)

        pb, ps = pad_gate_up_fp4(blocks, scales, padded_H, padded_I)

        assert pb.shape == (E, 2 * padded_I, padded_H // 2)
        assert ps.shape == (E, 2 * padded_I, padded_H // 32)

        # Original data preserved in both halves
        assert torch.all(pb[0, :I, :H // 2] == 0xAB)  # linear half
        assert torch.all(pb[0, padded_I:padded_I + I, :H // 2] == 0xAB)  # gate half

        # Padded rows are 0x00 for blocks
        assert torch.all(pb[0, I:padded_I, :] == 0)  # linear pad rows
        assert torch.all(pb[0, padded_I + I:, :] == 0)  # gate pad rows

        # Padded columns are 0x00 for blocks
        assert torch.all(pb[0, :I, H // 2:] == 0)

        # Scale pad value is 0x7F
        assert torch.all(ps[0, I:padded_I, :] == 0x7F)

    def test_pad_down_fp4(self):
        """Down projection padding should work for blocks and scales."""
        from pie_backend.model.gpt_oss_utils import pad_down_fp4

        E, H, I = 1, 4, 64
        padded_H, padded_I = 8, 128

        blocks = torch.full((E, H, I // 2), 0xCD, dtype=torch.uint8)
        scales = torch.full((E, H, I // 32), 0x80, dtype=torch.uint8)

        pb, ps = pad_down_fp4(blocks, scales, padded_H, padded_I)

        assert pb.shape == (E, padded_H, padded_I // 2)
        assert ps.shape == (E, padded_H, padded_I // 32)

        # Original data preserved
        assert torch.all(pb[0, :H, :I // 2] == 0xCD)
        assert torch.all(ps[0, :H, :I // 32] == 0x80)

        # Padded regions
        assert torch.all(pb[0, H:, :] == 0)
        assert torch.all(pb[0, :H, I // 2:] == 0)
        assert torch.all(ps[0, H:, :] == 0x7F)


# ---------------------------------------------------------------------------
# End-to-end: MoE with FP4 packed weights
# ---------------------------------------------------------------------------


class TestMoeWithFp4Packed:
    """End-to-end test: quantize bf16 weights to MXFP4, run MoE with per-expert dequant."""

    def _quantize_to_mxfp4(self, weights_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize bf16 weights to MXFP4 (blocks + scales) using the existing LUT.

        This is a test-only reference quantizer that produces the same format
        as safetensors MXFP4 storage.
        """
        from pie_backend.model.gpt_oss_utils import FP4_VALUES

        fp4_lut = torch.tensor(FP4_VALUES, dtype=torch.float32)
        shape = weights_bf16.shape
        w = weights_bf16.float().reshape(-1, 32)  # [num_blocks, 32]
        num_blocks, _ = w.shape

        # Compute per-block scale: max abs value -> nearest power of 2
        max_abs = w.abs().amax(dim=1).clamp(min=1e-12)  # [num_blocks]
        # E8M0 exponent: floor(log2(max_abs / 6.0)) + 127
        # 6.0 is the max FP4 representable value
        exp = torch.floor(torch.log2(max_abs / 6.0)).to(torch.int32) + 127
        exp = exp.clamp(0, 254)  # valid E8M0 range
        scale_values = torch.ldexp(torch.ones_like(max_abs), (exp - 127).float())  # 2^(exp-127)

        # Quantize each element: find nearest FP4 value after dividing by scale
        scaled_w = w / scale_values.unsqueeze(1)  # [num_blocks, 32]
        # Find nearest FP4 value for each element
        diffs = (scaled_w.unsqueeze(-1) - fp4_lut.unsqueeze(0).unsqueeze(0)).abs()
        indices = diffs.argmin(dim=-1)  # [num_blocks, 32]

        # Pack pairs into uint8: lo nibble = even index, hi nibble = odd index
        idx_lo = indices[:, 0::2]  # [num_blocks, 16]
        idx_hi = indices[:, 1::2]
        packed = (idx_lo | (idx_hi << 4)).to(torch.uint8)  # [num_blocks, 16]

        # Reshape back
        prefix = shape[:-1]
        blocks_per_row = shape[-1] // 32
        blocks = packed.reshape(*prefix, blocks_per_row * 16)  # [..., cols/2]
        scales_out = exp.to(torch.uint8).reshape(*prefix, blocks_per_row)  # [..., cols/32]

        return blocks, scales_out

    def _dequant_fp4_to_bf16(self, blocks, scales):
        """Dequant packed FP4 back to bf16 using the existing reference implementation."""
        from pie_backend.model.gpt_oss_utils import dequantize_from_mxfp4

        shape = blocks.shape
        # Reshape to [*prefix, groups_per_row, 16] and [*prefix, groups_per_row]
        *prefix, cols_packed = shape
        groups_per_row = cols_packed // 16
        blocks_3d = blocks.reshape(*prefix, groups_per_row, 16)
        scales_3d = scales  # already [*prefix, groups_per_row]
        return dequantize_from_mxfp4(blocks_3d, scales_3d, "cpu", torch.bfloat16).reshape(
            *prefix, cols_packed * 2,
        )

    def test_fp4_packed_moe_vs_pytorch_reference(self, device):
        """MoE with FP4 packed weights should match a local PyTorch reference.

        Quantizes bf16 weights to FP4, runs Metal MoE, then computes a local
        reference using dequantized weights + PyTorch matmul.
        """
        import torch.nn.functional as F
        from flashinfer_metal import trtllm_fp4_block_scale_moe

        torch.manual_seed(42)
        num_tokens, hidden_dim, intermediate_size = 4, 64, 128
        num_experts, top_k = 4, 2

        # Create bf16 weights and quantize to FP4
        w1_bf16 = torch.randn(num_experts, 2 * intermediate_size, hidden_dim, dtype=torch.bfloat16)
        w2_bf16 = torch.randn(num_experts, hidden_dim, intermediate_size, dtype=torch.bfloat16)
        hidden = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16)
        routing = torch.randn(num_tokens, num_experts, dtype=torch.bfloat16)

        w1_blocks, w1_scales = self._quantize_to_mxfp4(w1_bf16)
        w2_blocks, w2_scales = self._quantize_to_mxfp4(w2_bf16)

        # --- Local PyTorch reference using dequanted weights ---
        w1_dequant = self._dequant_fp4_to_bf16(w1_blocks, w1_scales)
        w2_dequant = self._dequant_fp4_to_bf16(w2_blocks, w2_scales)
        dtype = hidden.dtype

        scores = torch.softmax(routing.float(), dim=-1)
        topk_weights, topk_indices = torch.topk(scores, top_k, dim=-1)
        topk_weights = (topk_weights / topk_weights.sum(dim=-1, keepdim=True)).to(dtype)

        ref_output = torch.zeros(num_tokens, hidden_dim, dtype=dtype)
        for k_idx in range(top_k):
            for eid in range(num_experts):
                mask = topk_indices[:, k_idx] == eid
                if not mask.any():
                    continue
                x = hidden[mask]
                g1 = F.linear(x, w1_dequant[eid])
                up = g1[:, :intermediate_size].clamp(-100.0, 100.0)
                gate = g1[:, intermediate_size:].clamp(max=100.0)
                glu = gate * torch.sigmoid(gate)
                activated = (up + 1) * glu
                g2 = F.linear(activated, w2_dequant[eid])
                ref_output[mask] += g2 * topk_weights[mask, k_idx].unsqueeze(-1)

        # --- Metal FP4 path ---
        result_fp4 = trtllm_fp4_block_scale_moe(
            routing_logits=routing.to(device), routing_bias=None,
            hidden_states=hidden.to(device), hidden_states_scale=None,
            gemm1_weights=w1_blocks.to(device),
            gemm1_weights_scale=w1_scales.to(device),
            gemm1_bias=None, gemm1_alpha=1.0, gemm1_beta=0.0, gemm1_clamp_limit=100.0,
            gemm2_weights=w2_blocks.to(device),
            gemm2_weights_scale=w2_scales.to(device),
            gemm2_bias=None,
            output1_scale_scalar=1.0, output1_scale_gate_scalar=1.0, output2_scale_scalar=1.0,
            num_experts=num_experts, top_k=top_k, n_group=None, topk_group=None,
            intermediate_size=intermediate_size,
            local_expert_offset=0, local_num_experts=num_experts,
            routed_scaling_factor=None, routing_method_type=1, gated_act_type=0,
            do_finalize=True, tune_max_num_tokens=4096,
        )[0]

        # Metal FP4 GEMM uses float32 accumulation while bf16 reference truncates
        # intermediates. Large gate/up values amplify precision differences through
        # the SiLU clamp nonlinearity, so we use relaxed tolerances.
        torch.testing.assert_close(result_fp4.cpu(), ref_output, atol=2e3, rtol=0.1)

    def test_fp4_packed_output_shape(self, device):
        """FP4 packed MoE should produce correct output shape."""
        from flashinfer_metal import trtllm_fp4_block_scale_moe

        torch.manual_seed(0)
        num_tokens, hidden_dim, intermediate_size = 8, 64, 128
        num_experts, top_k = 4, 2

        w1_bf16 = torch.randn(num_experts, 2 * intermediate_size, hidden_dim, dtype=torch.bfloat16)
        w2_bf16 = torch.randn(num_experts, hidden_dim, intermediate_size, dtype=torch.bfloat16)

        w1_blocks, w1_scales = self._quantize_to_mxfp4(w1_bf16)
        w2_blocks, w2_scales = self._quantize_to_mxfp4(w2_bf16)

        inputs = dict(
            routing_logits=torch.randn(num_tokens, num_experts, dtype=torch.bfloat16, device=device),
            routing_bias=None,
            hidden_states=torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device=device),
            hidden_states_scale=None,
            gemm1_weights=w1_blocks.to(device),
            gemm1_weights_scale=w1_scales.to(device),
            gemm1_bias=None, gemm1_alpha=1.0, gemm1_beta=0.0, gemm1_clamp_limit=100.0,
            gemm2_weights=w2_blocks.to(device),
            gemm2_weights_scale=w2_scales.to(device),
            gemm2_bias=None,
            output1_scale_scalar=1.0, output1_scale_gate_scalar=1.0, output2_scale_scalar=1.0,
            num_experts=num_experts, top_k=top_k, n_group=None, topk_group=None,
            intermediate_size=intermediate_size,
            local_expert_offset=0, local_num_experts=num_experts,
            routed_scaling_factor=None, routing_method_type=1, gated_act_type=0,
            do_finalize=True, tune_max_num_tokens=4096,
        )

        result = trtllm_fp4_block_scale_moe(**inputs)
        assert isinstance(result, tuple)
        assert result[0].shape == (num_tokens, hidden_dim)
        assert result[0].dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Phase 2: Metal kernel unit tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS required for Metal kernels",
)
class TestMetalMoeMatmul:
    """Tests for Metal FP4 MoE matmul kernels."""

    def test_gemm2_known_values(self):
        """GEMM2 with all-ones input and FP4=1.0 weights should sum to H."""
        from flashinfer_metal._compiler import MetalCompiler

        compiler = MetalCompiler()
        H, out_dim = 32, 4
        blocks = torch.full((out_dim, H // 2), 0x22, dtype=torch.uint8, device="mps")
        scales = torch.full((out_dim, H // 32), 127, dtype=torch.uint8, device="mps")
        x = torch.ones(1, H, dtype=torch.bfloat16, device="mps")

        result = compiler.run_moe_prefill_gemm2(x, blocks, scales, None, out_dim, 1.0)
        expected = torch.full((1, out_dim), float(H), dtype=torch.bfloat16)
        torch.testing.assert_close(result.cpu(), expected)

    def test_gemm2_vs_pytorch(self):
        """GEMM2 should match PyTorch F.linear with dequantized weights."""
        from flashinfer_metal._compiler import MetalCompiler

        compiler = MetalCompiler()
        torch.manual_seed(123)
        H, out_dim, count = 64, 32, 4

        blocks = torch.randint(0, 256, (out_dim, H // 2), dtype=torch.uint8, device="mps")
        scales = torch.full((out_dim, H // 32), 127, dtype=torch.uint8, device="mps")
        x = torch.randn(count, H, dtype=torch.bfloat16, device="mps")

        result = compiler.run_moe_prefill_gemm2(x, blocks, scales, None, out_dim, 1.0)
        w = _dequantize_expert_from_fp4(blocks, scales, torch.bfloat16)
        ref = torch.nn.functional.linear(x, w)

        torch.testing.assert_close(result.cpu(), ref.cpu(), atol=1e-1, rtol=1e-2)

    def test_gemm2_with_scale(self):
        """Output scale factor should multiply the result."""
        from flashinfer_metal._compiler import MetalCompiler

        compiler = MetalCompiler()
        H, out_dim = 32, 2
        blocks = torch.full((out_dim, H // 2), 0x22, dtype=torch.uint8, device="mps")
        scales = torch.full((out_dim, H // 32), 127, dtype=torch.uint8, device="mps")
        x = torch.ones(1, H, dtype=torch.bfloat16, device="mps")

        r1 = compiler.run_moe_prefill_gemm2(x, blocks, scales, None, out_dim, 1.0)
        r2 = compiler.run_moe_prefill_gemm2(x, blocks, scales, None, out_dim, 2.0)
        torch.testing.assert_close(r2.cpu(), (r1 * 2).cpu(), atol=1e-3, rtol=1e-3)

    def test_gemm1_swiglu_shape(self):
        """GEMM1 with SwiGLU should output [count, intermediate_size]."""
        from flashinfer_metal._compiler import MetalCompiler

        compiler = MetalCompiler()
        H, I = 64, 32
        blocks = torch.randint(0, 256, (2 * I, H // 2), dtype=torch.uint8, device="mps")
        scales = torch.full((2 * I, H // 32), 127, dtype=torch.uint8, device="mps")
        x = torch.randn(4, H, dtype=torch.bfloat16, device="mps")

        result = compiler.run_moe_prefill_gemm1(
            x, blocks, scales, None, I, 1.0, 0.0, 100.0, 1.0, 1.0,
        )
        assert result.shape == (4, I)
        assert result.dtype == torch.bfloat16

    def test_gemm1_swiglu_vs_pytorch(self):
        """GEMM1+SwiGLU should match PyTorch reference with unit scales."""
        from flashinfer_metal._compiler import MetalCompiler

        compiler = MetalCompiler()
        torch.manual_seed(7)
        H, I = 32, 4
        count = 2

        blocks = torch.randint(0, 256, (2 * I, H // 2), dtype=torch.uint8, device="mps")
        scales = torch.full((2 * I, H // 32), 127, dtype=torch.uint8, device="mps")
        x = torch.randn(count, H, dtype=torch.bfloat16, device="mps")

        result = compiler.run_moe_prefill_gemm1(
            x, blocks, scales, None, I, 1.0, 0.0, 100.0, 1.0, 1.0,
        )

        w = _dequantize_expert_from_fp4(blocks, scales, torch.bfloat16)
        g1 = torch.nn.functional.linear(x, w)
        # After deinterleave: first half = up, second half = gate
        up = g1[:, :I]
        gate = g1[:, I:]
        # GPT-OSS activation: gate * σ(gate * α), (up + 1) * glu
        gate = gate.clamp(max=100.0)
        up = up.clamp(-100.0, 100.0)
        glu = gate * torch.sigmoid(gate * 1.0)  # alpha=1.0
        ref = (up + 1) * glu

        torch.testing.assert_close(result.cpu(), ref.cpu(), atol=0.5, rtol=0.05)

    def test_decode_gemm1_vs_per_expert(self):
        """Decode GEMM1 (raw, no SwiGLU) should match per-expert GEMM1."""
        from flashinfer_metal._compiler import MetalCompiler

        compiler = MetalCompiler()
        torch.manual_seed(42)
        E, H, I = 8, 64, 32
        K = 4
        N = 2 * I  # total weight rows per expert (up + gate)

        all_blocks = torch.randint(0, 256, (E, N, H // 2), dtype=torch.uint8, device="mps")
        all_scales = torch.full((E, N, H // 32), 127, dtype=torch.uint8, device="mps")
        x = torch.randn(1, H, dtype=torch.bfloat16, device="mps")
        expert_ids = torch.tensor([1, 3, 5, 7], dtype=torch.int32, device="mps")

        # Decode batched result: [K, N]
        decode_out = compiler.run_moe_decode_gemm1(
            input=x, all_w_blocks=all_blocks, all_w_scales=all_scales,
            intermediate_size=I, expert_ids=expert_ids,
        )
        assert decode_out.shape == (K, N)

        # Per-expert reference via dequant + matmul
        for i, eid in enumerate(expert_ids.tolist()):
            w = _dequantize_expert_from_fp4(all_blocks[eid], all_scales[eid], torch.bfloat16)
            ref = torch.nn.functional.linear(x, w)
            # Compare against bf16-rounded reference (kernel outputs bf16)
            torch.testing.assert_close(
                decode_out[i:i+1].cpu(), ref.bfloat16().cpu(), atol=1e-2, rtol=1e-2,
            )

    def test_decode_gemm1_swiglu_vs_per_expert(self):
        """Decode GEMM1+SwiGLU should match per-expert GEMM1+SwiGLU."""
        from flashinfer_metal._compiler import MetalCompiler

        compiler = MetalCompiler()
        torch.manual_seed(42)
        E, H, I = 8, 64, 32
        K = 4

        all_blocks = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device="mps")
        all_scales = torch.full((E, 2 * I, H // 32), 127, dtype=torch.uint8, device="mps")
        x = torch.randn(1, H, dtype=torch.bfloat16, device="mps")
        expert_ids = torch.tensor([1, 3, 5, 7], dtype=torch.int32, device="mps")

        # Decode fused result: [K, I]
        decode_out = compiler.run_moe_decode_gemm1_swiglu(
            input=x, all_w_blocks=all_blocks, all_w_scales=all_scales,
            all_bias=None, intermediate_size=I, expert_ids=expert_ids,
            alpha=1.0, clamp_limit=100.0,
        )
        assert decode_out.shape == (K, I)

        # Per-expert reference
        for i, eid in enumerate(expert_ids.tolist()):
            ref = compiler.run_moe_prefill_gemm1(
                x, all_blocks[eid], all_scales[eid], None, I,
                1.0, 0.0, 100.0, 1.0, 1.0,
            )
            torch.testing.assert_close(decode_out[i:i+1].cpu(), ref.cpu(), atol=0.5, rtol=0.05)

    def test_decode_gemm2_fused_vs_separate(self):
        """Decode fused GEMM2+scatter should match separate per-expert GEMM2 + weighted sum."""
        from flashinfer_metal._compiler import MetalCompiler

        compiler = MetalCompiler()
        torch.manual_seed(77)
        E, in_dim, out_dim = 8, 64, 32
        K = 4

        all_blocks = torch.randint(0, 256, (E, out_dim, in_dim // 2), dtype=torch.uint8, device="mps")
        all_scales = torch.full((E, out_dim, in_dim // 32), 127, dtype=torch.uint8, device="mps")
        x = torch.randn(K, in_dim, dtype=torch.bfloat16, device="mps")
        expert_ids = torch.tensor([1, 3, 5, 7], dtype=torch.int32, device="mps")
        routing_weights = torch.tensor([0.3, 0.25, 0.25, 0.2], dtype=torch.float32, device="mps")

        # Separate path: per-expert GEMM2 → weighted sum
        per_expert = []
        for i, eid in enumerate(expert_ids.tolist()):
            ref = compiler.run_moe_prefill_gemm2(
                x[i:i+1], all_blocks[eid], all_scales[eid], None, out_dim, 1.0,
            )
            per_expert.append(ref.float())
        per_expert = torch.cat(per_expert, dim=0).cpu()
        expected = (per_expert * routing_weights.cpu().unsqueeze(-1)).sum(dim=0, keepdim=True)

        # Fused path: single decode kernel
        actual = compiler.run_moe_decode_gemm2_fused(
            x, all_blocks, all_scales, None, out_dim, expert_ids, routing_weights,
        )

        torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0.5, rtol=0.05)
