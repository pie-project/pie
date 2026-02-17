"""Metal shader compilation and kernel dispatch.

Single compiler class that lazily compiles and caches Metal kernels for
attention, RoPE, and KV cache append operations.
"""

import os
import platform
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_KERNEL_DIR = Path(__file__).parent / "kernels"


def _check_environment() -> None:
    """Validate Apple Silicon + MPS availability. Raises on failure."""
    if platform.system() != "Darwin" or platform.processor() != "arm":
        raise RuntimeError("flashinfer_metal requires macOS with Apple Silicon")
    if not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS backend is not available")
    if not hasattr(torch.mps, "compile_shader"):
        raise RuntimeError(
            "torch.mps.compile_shader not available; "
            "upgrade PyTorch to a version with MPS shader compilation support"
        )


def _validate_mps_device(tensor: torch.Tensor, name: str) -> None:
    """Ensure tensor is on MPS device."""
    if tensor.device.type != "mps":
        raise RuntimeError(
            f"flashinfer_metal requires MPS tensors. "
            f"'{name}' is on {tensor.device}"
        )


def _validate_page_size(page_size: int) -> None:
    """Validate page_size is a power of 2 and within Metal threadgroup memory."""
    if page_size <= 0 or (page_size & (page_size - 1)) != 0:
        raise ValueError(f"page_size must be a power of 2, got {page_size}")
    if page_size > 16:
        raise ValueError(
            f"page_size={page_size} exceeds Metal threadgroup memory limit. "
            f"Maximum supported: 16"
        )


# ---------------------------------------------------------------------------
# Attention param struct patching
# ---------------------------------------------------------------------------

_PARAM_STRUCT_OLD = "constant Params& params [[buffer(7)]]"
_PARAM_STRUCT_NEW = "device const float* params_raw [[buffer(7)]]"

_PARAM_REPLACEMENTS = [
    ("const int num_qo = params.num_qo;", "const int num_qo = (int)params_raw[0];"),
    ("const int head_dim = params.head_dim;", "const int head_dim = (int)params_raw[1];"),
    ("const int kv_head_dim = params.kv_head_dim;", "const int kv_head_dim = (int)params_raw[2];"),
    ("const int head_size = params.head_size;", "const int head_size = (int)params_raw[3];"),
    ("const int page_size = params.page_size;", "const int page_size = (int)params_raw[4];"),
    ("const int num_query_heads = params.num_query_heads;", "const int num_query_heads = (int)params_raw[5];"),
    ("const int num_kv_heads = params.num_kv_heads;", "const int num_kv_heads = (int)params_raw[6];"),
    ("const float scale = params.scale;", "const float scale = params_raw[7];"),
]


def _patch_attention_params(source: str) -> str:
    """Replace Params struct access with raw buffer for torch.mps compatibility."""
    source = source.replace(_PARAM_STRUCT_OLD, _PARAM_STRUCT_NEW)
    for old, new in _PARAM_REPLACEMENTS:
        source = source.replace(old, new)
    return source


# ---------------------------------------------------------------------------
# MetalCompiler — singleton, lazy per-kernel compilation
# ---------------------------------------------------------------------------


class MetalCompiler:
    """Compiles and dispatches all Metal kernels. Singleton with lazy init."""

    _instance: "MetalCompiler | None" = None

    def __new__(cls) -> "MetalCompiler":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "_inited"):
            return
        self._inited = True
        self._env_checked = False
        self._page_size = int(os.environ.get("PIE_METAL_PAGE_SIZE", "16"))
        _validate_page_size(self._page_size)
        self._libs: dict[str, object] = {}

    def _ensure_env(self) -> None:
        if not self._env_checked:
            _check_environment()
            self._env_checked = True

    def _read_metal(self, filename: str) -> str:
        return (_KERNEL_DIR / filename).read_text()

    def _compile(self, source: str, name: str) -> object:
        lib = torch.mps.compile_shader(source)
        self._libs[name] = lib
        return lib

    # -------------------------------------------------------------------
    # Attention
    # -------------------------------------------------------------------

    def _ensure_attention(self) -> object:
        if "attention" in self._libs:
            return self._libs["attention"]
        self._ensure_env()
        source = self._read_metal("metal_attention_simdgroup_opt.metal")
        source = _patch_attention_params(source)
        source = f"#define BLOCK_SIZE {self._page_size}\n\n{source}"
        return self._compile(source, "attention")

    def run_attention(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        qo_indptr: torch.Tensor,
        sinks: "torch.Tensor | None" = None,
        scaling: "float | None" = None,
    ) -> torch.Tensor:
        """Run paged attention. Returns output shaped [num_tokens, num_heads * head_dim].

        Args:
            sinks: Optional [num_qo_heads] float32 tensor of per-head sink logits.
                   When provided, injects a virtual sink token into the softmax
                   denominator so sum(attention_weights) < 1.0.
            scaling: Optional attention scale. If None, uses 1/sqrt(head_dim).
        """
        lib = self._ensure_attention()

        original_dtype = query.dtype
        if original_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"Unsupported dtype {original_dtype}. "
                f"Supported: float16, bfloat16."
            )

        num_tokens, num_heads, head_dim = query.shape
        _num_pages, _, page_size, num_kv_heads, _ = kv_cache.shape

        if head_dim > 128:
            raise ValueError(
                f"Head dimension {head_dim} exceeds Metal kernel limit of 128."
            )

        # Build params tensor
        scale = scaling if scaling is not None else 1.0 / (head_dim**0.5)

        # Cast index tensors once
        kv_flat = kv_cache.contiguous().view(-1)
        qo_indptr_i32 = qo_indptr.to(torch.int32)
        kv_page_indptr_i32 = kv_page_indptr.to(torch.int32)
        kv_page_indices_i32 = kv_page_indices.to(torch.int32)
        kv_last_page_lens_i32 = kv_last_page_lens.to(torch.int32)

        if num_tokens == 1:
            # Decode path — v2/v3 kernels use head_dim (head_size) directly
            # for Q/output addressing, so no padding needed.
            params_list = [
                num_tokens,
                num_heads * head_dim,
                num_kv_heads * head_dim,
                head_dim,
                page_size,
                num_heads,
                num_kv_heads,
                scale,
            ]
            if sinks is not None:
                params_list.append(1.0)
                params_list.extend(sinks.to(torch.float32).cpu().tolist())
            else:
                params_list.append(0.0)
            params = torch.tensor(params_list, dtype=torch.float32, device="mps")

            q_flat = query.contiguous().view(-1)
            output = torch.empty(
                num_tokens * num_heads * head_dim,
                device="mps", dtype=original_dtype,
            )

            dtype_prefix = {
                torch.bfloat16: "bf16",
                torch.float16: "fp16",
            }[original_dtype]
            if head_dim == 128:
                kernel_name = f"attention_decode_v3_{dtype_prefix}_{head_dim}"
            else:
                kernel_name = f"attention_decode_v2_{dtype_prefix}_{head_dim}"

            getattr(lib, kernel_name)(
                q_flat, kv_flat,
                qo_indptr_i32, kv_page_indptr_i32,
                kv_page_indices_i32, kv_last_page_lens_i32,
                output, params,
                threads=(num_heads * 1024, 1, 1),
                group_size=(1024, 1, 1),
            )
            return output.view(num_tokens, num_heads * head_dim)
        else:
            # Prefill path — kernel uses MAX_HEAD_DIM=128 as per-head stride
            # for Q loading and output writing.  When the real head_dim < 128,
            # pad Q to stride-128 per head and unpad the output afterwards.
            MAX_HD = 128
            need_pad = head_dim < MAX_HD

            if need_pad:
                q_padded = torch.zeros(
                    num_tokens, num_heads, MAX_HD,
                    dtype=original_dtype, device="mps",
                )
                q_padded[:, :, :head_dim] = query
                q_flat = q_padded.contiguous().view(-1)
                padded_total_dim = num_heads * MAX_HD
            else:
                q_flat = query.contiguous().view(-1)
                padded_total_dim = num_heads * head_dim

            params_list = [
                num_tokens,
                padded_total_dim,          # token stride in Q/output
                num_kv_heads * head_dim,   # kv_head_dim (real, for KV cache)
                head_dim,                  # head_size (real, for KV cache)
                page_size,
                num_heads,
                num_kv_heads,
                scale,
            ]
            if sinks is not None:
                params_list.append(1.0)
                params_list.extend(sinks.to(torch.float32).cpu().tolist())
            else:
                params_list.append(0.0)
            params = torch.tensor(params_list, dtype=torch.float32, device="mps")

            output = torch.empty(
                num_tokens * padded_total_dim,
                device="mps", dtype=original_dtype,
            )

            if original_dtype == torch.bfloat16:
                kernel_name = "batch_prefill_attention_unified_bfloat16_simdgroup_kernel"
                bq, tpg = 32, 128
            else:
                kernel_name = "batch_prefill_attention_unified_fp16_simdgroup_kernel"
                bq, tpg = 32, 128

            num_q_blocks = (num_tokens + bq - 1) // bq

            getattr(lib, kernel_name)(
                q_flat, kv_flat,
                qo_indptr_i32, kv_page_indptr_i32,
                kv_page_indices_i32, kv_last_page_lens_i32,
                output, params,
                threads=(num_q_blocks * tpg, num_heads, 1),
                group_size=(tpg, 1, 1),
            )

            if need_pad:
                # Unpad: extract first head_dim elements per head
                output = output.view(num_tokens, num_heads, MAX_HD)[
                    :, :, :head_dim
                ].contiguous()
            return output.view(num_tokens, num_heads * head_dim)

    # -------------------------------------------------------------------
    # RoPE
    # -------------------------------------------------------------------

    def _ensure_rope(self) -> object:
        if "rope" in self._libs:
            return self._libs["rope"]
        self._ensure_env()
        source = self._read_metal("metal_rope.metal")
        return self._compile(source, "rope")

    def run_rope(
        self,
        input_qk: torch.Tensor,
        position_ids: torch.Tensor,
        rope_theta: float = 10000.0,
        rope_factor: float = 1.0,
        interleaved: bool = False,
    ) -> None:
        """Apply RoPE in-place to input_qk [num_tokens, num_heads, head_size]."""
        lib = self._ensure_rope()

        num_tokens, num_heads, head_size = input_qk.shape

        # Handle non-contiguous tensors
        was_contiguous = input_qk.is_contiguous()
        if was_contiguous:
            flat = input_qk.view(-1)
        else:
            contiguous_copy = input_qk.contiguous()
            flat = contiguous_copy.view(-1)

        params = torch.tensor(
            [num_tokens, num_heads, head_size, rope_theta, rope_factor,
             1 if interleaved else 0],
            dtype=torch.float32,
            device="mps",
        )

        # Select kernel by dtype
        _ROPE_KERNELS = {
            torch.float16: "metal_rope_float16",
            torch.bfloat16: "metal_rope_bfloat16",
            torch.float32: "metal_rope_float32",
        }
        kernel_name = _ROPE_KERNELS.get(input_qk.dtype)
        if kernel_name is None:
            raise ValueError(
                f"Unsupported dtype {input_qk.dtype}. "
                f"Supported: float32, float16, bfloat16."
            )

        getattr(lib, kernel_name)(
            flat,
            position_ids.to(torch.int32),
            params,
            threads=(num_tokens, num_heads, head_size // 2),
            group_size=(8, 8, 4),
        )

        if not was_contiguous:
            input_qk.copy_(contiguous_copy)

    # -------------------------------------------------------------------
    # RoPE with cos/sin cache
    # -------------------------------------------------------------------

    def _ensure_rope_cos_sin(self) -> object:
        if "rope_cos_sin" in self._libs:
            return self._libs["rope_cos_sin"]
        self._ensure_env()
        source = self._read_metal("metal_rope_cos_sin_cache.metal")
        return self._compile(source, "rope_cos_sin")

    def run_rope_cos_sin(
        self,
        input_qk: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        head_size: int,
        is_neox: bool = True,
    ) -> None:
        """Apply RoPE using precomputed cos/sin cache, in-place.

        Args:
            input_qk: [num_tokens, num_heads, head_size] — modified in-place
            positions: [num_tokens] int32 position indices
            cos_sin_cache: [max_pos, head_dim] float32 — first half cos, second half sin
            head_size: head dimension
            is_neox: True for NeoX-style (split halves), False for interleaved
        """
        lib = self._ensure_rope_cos_sin()

        num_tokens, num_heads = input_qk.shape[0], input_qk.shape[1]

        was_contiguous = input_qk.is_contiguous()
        if was_contiguous:
            flat = input_qk.view(-1)
        else:
            contiguous_copy = input_qk.contiguous()
            flat = contiguous_copy.view(-1)

        params = torch.tensor(
            [num_tokens, num_heads, head_size, 1 if is_neox else 0],
            dtype=torch.float32,
            device="mps",
        )

        cache_f32 = cos_sin_cache.to(device="mps", dtype=torch.float32).contiguous()

        _KERNELS = {
            torch.float16: "metal_rope_cos_sin_cache_float16",
            torch.bfloat16: "metal_rope_cos_sin_cache_bfloat16",
            torch.float32: "metal_rope_cos_sin_cache_float32",
        }
        kernel_name = _KERNELS.get(input_qk.dtype)
        if kernel_name is None:
            raise ValueError(f"Unsupported dtype {input_qk.dtype}")

        getattr(lib, kernel_name)(
            flat,
            positions.to(device="mps", dtype=torch.int32),
            cache_f32,
            params,
            threads=(num_tokens, num_heads, head_size // 2),
            group_size=(8, 8, 4),
        )

        if not was_contiguous:
            input_qk.copy_(contiguous_copy)

    # -------------------------------------------------------------------
    # Append KV cache
    # -------------------------------------------------------------------

    def _ensure_append_kv(self) -> object:
        if "append_kv" in self._libs:
            return self._libs["append_kv"]
        self._ensure_env()
        source = self._read_metal("metal_append_paged_kv_cache.metal")
        return self._compile(source, "append_kv")

    def run_append_kv(
        self,
        k_input: torch.Tensor,
        v_input: torch.Tensor,
        paged_kv_cache: torch.Tensor,
        kv_batch_indices: torch.Tensor,
        kv_positions: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
        page_size: int,
    ) -> None:
        """Append KV states to paged cache in-place.

        Args:
            k_input: [num_tokens, num_kv_heads * head_size]
            v_input: [num_tokens, num_kv_heads * head_size]
            paged_kv_cache: flattened 1D unified buffer
        """
        lib = self._ensure_append_kv()

        num_tokens = k_input.shape[0]
        batch_size = kv_page_indptr.shape[0] - 1
        max_num_pages = paged_kv_cache.numel() // (
            2 * page_size * num_kv_heads * head_size
        )

        params = torch.tensor(
            [
                num_tokens,
                num_kv_heads,
                head_size,
                page_size,
                max_num_pages,
                batch_size,
                num_kv_heads * head_size,  # k_stride_token
                head_size,                  # k_stride_head
                num_kv_heads * head_size,  # v_stride_token
                head_size,                  # v_stride_head
            ],
            dtype=torch.int32,
            device="mps",
        )

        _APPEND_KV_KERNELS = {
            torch.bfloat16: "metal_append_paged_kv_cache_bfloat16",
            torch.float32: "metal_append_paged_kv_cache_float32",
            torch.float16: "metal_append_paged_kv_cache_float16",
        }
        kernel_name = _APPEND_KV_KERNELS.get(k_input.dtype)
        if kernel_name is None:
            raise ValueError(
                f"Unsupported dtype {k_input.dtype}. "
                f"Supported: float32, float16, bfloat16."
            )

        getattr(lib, kernel_name)(
            k_input,
            v_input,
            paged_kv_cache,
            kv_batch_indices.to(torch.int32),
            kv_positions.to(torch.int32),
            kv_page_indices.to(torch.int32),
            kv_page_indptr.to(torch.int32),
            kv_last_page_lens.to(torch.int32),
            params.float(),
            threads=(num_tokens, num_kv_heads, head_size),
            group_size=(8, 8, 8),
        )

    # -------------------------------------------------------------------
    # MoE FP4 matmul
    # -------------------------------------------------------------------

    def _ensure_moe_matmul(self) -> object:
        if "moe_matmul" in self._libs:
            return self._libs["moe_matmul"]
        self._ensure_env()
        source = self._read_metal("metal_moe_matmul.metal")
        return self._compile(source, "moe_matmul")

    def run_moe_gemm1_fp4(
        self,
        input: torch.Tensor,
        w_blocks: torch.Tensor,
        w_scales: torch.Tensor,
        bias: "torch.Tensor | None",
        intermediate_size: int,
        alpha: float,
        beta: float,
        clamp_limit: float,
        scale_gate: float,
        scale_up: float,
    ) -> torch.Tensor:
        """GEMM1 with fused SwiGLU on FP4 packed weights (single expert).

        Args:
            input: [count, hidden_dim] bfloat16
            w_blocks: [2*I, H/2] uint8 — single expert weights
            w_scales: [2*I, H/32] uint8 — single expert scales
            bias: [2*I] bfloat16 or None
            intermediate_size: I
            alpha, beta, clamp_limit: SwiGLU parameters
            scale_gate, scale_up: output scaling factors

        Returns:
            [count, I] bfloat16
        """
        lib = self._ensure_moe_matmul()
        count = input.shape[0]
        hidden_dim = input.shape[1]

        output = torch.empty(count, intermediate_size, dtype=input.dtype, device="mps")

        params = torch.tensor([
            count, hidden_dim, intermediate_size,
            alpha, beta, clamp_limit,
            scale_gate, scale_up,
            1.0 if bias is not None else 0.0,
        ], dtype=torch.float32, device="mps")

        bias_buf = bias.to(input.dtype) if bias is not None else torch.zeros(
            1, dtype=input.dtype, device="mps",
        )

        kernel = "moe_matmul_swiglu" if count == 1 else "moe_dense_matmul_swiglu"

        if count == 1:
            getattr(lib, kernel)(
                input.contiguous().view(-1),
                w_blocks.contiguous().view(-1),
                w_scales.contiguous().view(-1),
                bias_buf.contiguous().view(-1),
                output.view(-1),
                params,
                threads=(count, intermediate_size, 1),
                group_size=(1, min(intermediate_size, 256), 1),
            )
        else:
            tile_m, tile_n = 16, 16
            getattr(lib, kernel)(
                input.contiguous().view(-1),
                w_blocks.contiguous().view(-1),
                w_scales.contiguous().view(-1),
                bias_buf.contiguous().view(-1),
                output.view(-1),
                params,
                threads=(
                    ((count + tile_m - 1) // tile_m) * tile_m,
                    ((intermediate_size + tile_n - 1) // tile_n) * tile_n,
                    1,
                ),
                group_size=(tile_m, tile_n, 1),
            )

        return output

    def run_moe_gemm2_fp4(
        self,
        input: torch.Tensor,
        w_blocks: torch.Tensor,
        w_scales: torch.Tensor,
        bias: "torch.Tensor | None",
        out_dim: int,
        scale: float,
    ) -> torch.Tensor:
        """GEMM2 on FP4 packed weights (single expert, plain matmul).

        Args:
            input: [count, in_dim] bfloat16
            w_blocks: [out_dim, in_dim/2] uint8
            w_scales: [out_dim, in_dim/32] uint8
            bias: [out_dim] bfloat16 or None
            out_dim: output dimension
            scale: output scaling factor

        Returns:
            [count, out_dim] bfloat16
        """
        lib = self._ensure_moe_matmul()
        count = input.shape[0]
        in_dim = input.shape[1]

        output = torch.empty(count, out_dim, dtype=input.dtype, device="mps")

        params = torch.tensor([
            count, in_dim, out_dim, scale,
            1.0 if bias is not None else 0.0,
        ], dtype=torch.float32, device="mps")

        bias_buf = bias.to(input.dtype) if bias is not None else torch.zeros(
            1, dtype=input.dtype, device="mps",
        )

        kernel = "moe_matmul" if count == 1 else "moe_dense_matmul"

        if count == 1:
            getattr(lib, kernel)(
                input.contiguous().view(-1),
                w_blocks.contiguous().view(-1),
                w_scales.contiguous().view(-1),
                bias_buf.contiguous().view(-1),
                output.view(-1),
                params,
                threads=(count, out_dim, 1),
                group_size=(1, min(out_dim, 256), 1),
            )
        else:
            tile_m, tile_n = 16, 16
            getattr(lib, kernel)(
                input.contiguous().view(-1),
                w_blocks.contiguous().view(-1),
                w_scales.contiguous().view(-1),
                bias_buf.contiguous().view(-1),
                output.view(-1),
                params,
                threads=(
                    ((count + tile_m - 1) // tile_m) * tile_m,
                    ((out_dim + tile_n - 1) // tile_n) * tile_n,
                    1,
                ),
                group_size=(tile_m, tile_n, 1),
            )

        return output
