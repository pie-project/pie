#!/usr/bin/env python3
"""Regression checks for dense Qwen3.5 linear-attention FP8 metadata wiring."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "src" / "model"


def read_model_file(name: str) -> str:
    return (MODEL_DIR / name).read_text()


class Qwen35LinearAttnQuantMetaTests(unittest.TestCase):
    def test_qwen35_linear_attention_weights_keep_quantmeta_companions(self) -> None:
        header = read_model_file("qwen3_5.hpp")

        for field in (
            "la_in_proj_qkv_quant",
            "la_in_proj_z_quant",
            "la_in_proj_a_quant",
            "la_in_proj_b_quant",
            "la_out_proj_quant",
        ):
            self.assertIn(f"std::optional<QuantMeta> {field};", header)

    def test_qwen35_bind_populates_linear_attention_quantmeta_from_weight_store(
        self,
    ) -> None:
        binder = read_model_file("qwen3_5.cpp")

        self.assertRegex(
            binder,
            r'const auto qkv_quant\s*=\s*engine\.quant_meta\(la \+ "in_proj_qkv.weight"\);',
        )
        self.assertIn("Lw.la_in_proj_qkv_quant = qkv_quant;", binder)

        for field, weight_name in (
            ("la_in_proj_z_quant", "in_proj_z.weight"),
            ("la_in_proj_a_quant", "in_proj_a.weight"),
            ("la_in_proj_b_quant", "in_proj_b.weight"),
            ("la_out_proj_quant", "out_proj.weight"),
        ):
            self.assertRegex(
                binder,
                rf'Lw\.{field}\s*=\s*engine\.quant_meta\(la \+ "{weight_name}"\);',
            )

        self.assertIn("const bool can_fuse_gdn_projection_weights =", binder)
        for field in (
            "la_in_proj_qkv_quant",
            "la_in_proj_z_quant",
            "la_in_proj_a_quant",
            "la_in_proj_b_quant",
        ):
            self.assertIn(f"!Lw.{field}.has_value()", binder)

    def test_qwen35_bind_rejects_fp8_qkv_when_tp_slice_would_drop_scale_slice(
        self,
    ) -> None:
        binder = read_model_file("qwen3_5.cpp")

        self.assertIn("T > 1 && qkv_quant.has_value()", binder)
        self.assertIn("TP-sliced FP8 linear_attn.in_proj_qkv", binder)
        self.assertLess(
            binder.index("T > 1 && qkv_quant.has_value()"),
            binder.index("slice_la_kkv_blocked(*full_qkv"),
        )

    def test_qwen35_linear_attention_forward_passes_weight_views_to_gemm(
        self,
    ) -> None:
        forward = read_model_file("qwen3_5_forward.cpp")

        for tensor, quant in (
            ("la_in_proj_qkv", "la_in_proj_qkv_quant"),
            ("la_in_proj_z", "la_in_proj_z_quant"),
            ("la_in_proj_a", "la_in_proj_a_quant"),
            ("la_in_proj_b", "la_in_proj_b_quant"),
            ("la_out_proj", "la_out_proj_quant"),
        ):
            self.assertIn(
                f"make_weight_view(Lw.{tensor}, Lw.{quant})",
                forward,
            )


if __name__ == "__main__":
    unittest.main()
