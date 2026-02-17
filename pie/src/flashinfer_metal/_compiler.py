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
    ) -> torch.Tensor:
        """Run paged attention. Returns output shaped [num_tokens, num_heads * head_dim]."""
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
        scale = 1.0 / (head_dim**0.5)
        params = torch.tensor(
            [
                num_tokens,
                num_heads * head_dim,
                num_kv_heads * head_dim,
                head_dim,
                page_size,
                num_heads,
                num_kv_heads,
                scale,
            ],
            dtype=torch.float32,
            device="mps",
        )

        q_flat = query.contiguous().view(-1)
        kv_flat = kv_cache.contiguous().view(-1)
        output = torch.empty(
            num_tokens * num_heads * head_dim, device="mps", dtype=original_dtype
        )

        # Cast index tensors once
        qo_indptr_i32 = qo_indptr.to(torch.int32)
        kv_page_indptr_i32 = kv_page_indptr.to(torch.int32)
        kv_page_indices_i32 = kv_page_indices.to(torch.int32)
        kv_last_page_lens_i32 = kv_last_page_lens.to(torch.int32)

        if num_tokens == 1:
            # Decode path — use v3 for fp16/bf16 head_dim=128, v2 otherwise
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
        else:
            # Prefill path
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
