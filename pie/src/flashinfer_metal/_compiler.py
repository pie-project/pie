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
        # Pre-allocated buffers for MoE dispatch (avoids per-call tensor allocs)
        self._moe_params9: torch.Tensor | None = None  # gemm1 (9 floats)
        self._moe_params5: torch.Tensor | None = None  # gemm2 (5 floats)
        self._moe_dummy_bias: dict[torch.dtype, torch.Tensor] = {}  # keyed by dtype
        # Pre-allocated buffer for expert IDs (used by decode kernels)
        self._moe_batched_expert_ids: torch.Tensor | None = None  # [K] int32
        # Pre-allocated buffer for KV append params (avoids per-call tensor alloc)
        self._append_kv_params: torch.Tensor | None = None  # 10 floats
        # Cached attention decode params tensors keyed by
        # (num_heads, head_dim, num_kv_heads, page_size, id(sinks))
        self._attn_decode_params: dict[tuple, torch.Tensor] = {}
        # Pre-allocated RoPE params buffers (avoids per-call tensor alloc)
        self._rope_params: torch.Tensor | None = None  # 4 floats
        self._rope_params_fused: torch.Tensor | None = None  # 5 floats
        # Cached cos_sin_cache f32 conversion
        self._rope_cache_id: int | None = None
        self._rope_cache_f32: torch.Tensor | None = None
        # Pre-allocated buffer for MoE routing params (4 floats)
        self._moe_routing_params: torch.Tensor | None = None
        # Pre-allocated buffer for RMS norm params (2 floats: H, eps)
        self._rms_norm_params: torch.Tensor | None = None
        # Cached params tensors: keyed by (method_name, value_tuple) → GPU tensor.
        # Eliminates ~63 μs/write per-element GPU tensor writes for constant params.
        self._params_cache: dict[tuple, torch.Tensor] = {}
        # Cached kernel function references to avoid repeated getattr()
        self._kernel_fn_cache: dict[str, object] = {}

    def _ensure_env(self) -> None:
        if not self._env_checked:
            _check_environment()
            self._env_checked = True

    def _get_params(self, key: tuple, values: list[float]) -> torch.Tensor:
        """Get or create a cached params tensor on MPS.

        Using element-by-element GPU tensor writes (tensor[i] = float) costs
        ~63 μs PER WRITE due to CPU→GPU command encoding.  For constant params
        (model dimensions, eps, etc. that never change during decode), caching
        the tensor eliminates this overhead entirely.
        """
        t = self._params_cache.get(key)
        if t is None:
            t = torch.tensor(values, dtype=torch.float32, device="mps")
            self._params_cache[key] = t
        return t

    def _get_kernel_fn(self, lib: object, name: str) -> object:
        """Cache kernel function reference to avoid repeated getattr()."""
        fn = self._kernel_fn_cache.get(name)
        if fn is None:
            fn = getattr(lib, name)
            self._kernel_fn_cache[name] = fn
        return fn

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
            # Cache params tensor to avoid GPU→CPU sync (.cpu().tolist()) every call.
            cache_key = (num_heads, head_dim, num_kv_heads, page_size,
                         id(sinks) if sinks is not None else None)
            if cache_key not in self._attn_decode_params:
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
                self._attn_decode_params[cache_key] = torch.tensor(
                    params_list, dtype=torch.float32, device="mps"
                )
            params = self._attn_decode_params[cache_key]

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

        params = self._get_params(
            ("rope_cos_sin", num_tokens, num_heads, head_size, is_neox),
            [float(num_tokens), float(num_heads), float(head_size),
             1.0 if is_neox else 0.0],
        )

        cid = id(cos_sin_cache)
        if cid != self._rope_cache_id:
            self._rope_cache_f32 = cos_sin_cache.to(device="mps", dtype=torch.float32).contiguous()
            self._rope_cache_id = cid
        cache_f32 = self._rope_cache_f32

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
    # Fused Q+K RoPE with cos/sin cache
    # -------------------------------------------------------------------

    def run_rope_cos_sin_fused(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        head_size: int,
        is_neox: bool = True,
    ) -> None:
        """Apply RoPE to both Q and K in a single kernel dispatch.

        Args:
            query: [num_tokens, num_q_heads, head_size] — modified in-place
            key: [num_tokens, num_kv_heads, head_size] — modified in-place
            positions: [num_tokens] int32 position indices
            cos_sin_cache: [max_pos, head_dim] float32
            head_size: head dimension
            is_neox: True for NeoX-style (split halves)
        """
        lib = self._ensure_rope_cos_sin()

        num_tokens = query.shape[0]
        num_q_heads = query.shape[1]
        num_kv_heads = key.shape[1]

        q_was_contiguous = query.is_contiguous()
        k_was_contiguous = key.is_contiguous()
        if q_was_contiguous:
            q_flat = query.view(-1)
        else:
            q_copy = query.contiguous()
            q_flat = q_copy.view(-1)
        if k_was_contiguous:
            k_flat = key.view(-1)
        else:
            k_copy = key.contiguous()
            k_flat = k_copy.view(-1)

        params = self._get_params(
            ("rope_fused", num_tokens, num_q_heads, num_kv_heads, head_size, is_neox),
            [float(num_tokens), float(num_q_heads), float(num_kv_heads),
             float(head_size), 1.0 if is_neox else 0.0],
        )

        cid = id(cos_sin_cache)
        if cid != self._rope_cache_id:
            self._rope_cache_f32 = cos_sin_cache.to(device="mps", dtype=torch.float32).contiguous()
            self._rope_cache_id = cid
        cache_f32 = self._rope_cache_f32

        _FUSED_KERNELS = {
            torch.float16: "metal_rope_cos_sin_cache_fused_float16",
            torch.bfloat16: "metal_rope_cos_sin_cache_fused_bfloat16",
            torch.float32: "metal_rope_cos_sin_cache_fused_float32",
        }
        kernel_name = _FUSED_KERNELS.get(query.dtype)
        if kernel_name is None:
            raise ValueError(f"Unsupported dtype {query.dtype}")

        max_heads = max(num_q_heads, num_kv_heads)

        self._get_kernel_fn(lib, kernel_name)(
            q_flat,
            k_flat,
            positions.to(device="mps", dtype=torch.int32),
            cache_f32,
            params,
            threads=(num_tokens, max_heads, head_size // 2),
            group_size=(8, 8, 4),
        )

        if not q_was_contiguous:
            query.copy_(q_copy)
        if not k_was_contiguous:
            key.copy_(k_copy)

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

        kv_stride = num_kv_heads * head_size
        p = self._get_params(
            ("append_kv", num_tokens, num_kv_heads, head_size, page_size,
             max_num_pages, batch_size),
            [float(num_tokens), float(num_kv_heads), float(head_size),
             float(page_size), float(max_num_pages), float(batch_size),
             float(kv_stride), float(head_size),
             float(kv_stride), float(head_size)],
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

        def _i32(t: torch.Tensor) -> torch.Tensor:
            return t if t.dtype == torch.int32 else t.to(torch.int32)

        self._get_kernel_fn(lib, kernel_name)(
            k_input,
            v_input,
            paged_kv_cache,
            _i32(kv_batch_indices),
            _i32(kv_positions),
            _i32(kv_page_indices),
            _i32(kv_page_indptr),
            _i32(kv_last_page_lens),
            p,
            threads=(num_tokens, num_kv_heads, head_size),
            group_size=(8, 8, 8),
        )

    # -------------------------------------------------------------------
    # MoE routing (fused softmax + topk + normalize)
    # -------------------------------------------------------------------

    def _ensure_moe_routing(self) -> object:
        if "moe_routing" in self._libs:
            return self._libs["moe_routing"]
        self._ensure_env()
        source = self._read_metal("metal_moe_routing.metal")
        return self._compile(source, "moe_routing")

    def run_moe_route_topk(
        self,
        logits: torch.Tensor,
        expert_ids_out: torch.Tensor,
        fused_scales_out: torch.Tensor,
        num_experts: int,
        top_k: int,
        output2_scale: float,
        local_expert_offset: int = 0,
    ) -> None:
        """Fused MoE routing: softmax + top-K + normalize + scale.

        Writes results to pre-allocated output tensors (no allocation).

        Args:
            logits: [num_tokens, E] bf16/f16 routing logits (num_tokens must be 1)
            expert_ids_out: [K] int32 pre-allocated output for expert IDs
            fused_scales_out: [K] float32 pre-allocated output for fused scales
            num_experts: E (number of experts)
            top_k: K (experts per token)
            output2_scale: scalar to multiply routing weights by
            local_expert_offset: offset to subtract from expert IDs (for TP)
        """
        lib = self._ensure_moe_routing()

        params = self._get_params(
            ("moe_routing", num_experts, top_k, output2_scale, local_expert_offset),
            [float(num_experts), float(top_k), float(output2_scale),
             float(local_expert_offset)],
        )

        # Flatten logits to 1D for kernel
        logits_flat = logits.contiguous().view(-1)

        # Select kernel based on dtype
        if logits.dtype == torch.bfloat16:
            kernel_fn = self._get_kernel_fn(lib, "moe_route_topk_bf16")
        elif logits.dtype == torch.float16:
            kernel_fn = self._get_kernel_fn(lib, "moe_route_topk_f16")
        else:
            raise ValueError(f"Unsupported routing logits dtype: {logits.dtype}")

        kernel_fn(
            logits_flat,
            expert_ids_out,
            fused_scales_out,
            params,
            threads=(1, 1, 1),
            group_size=(1, 1, 1),
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

    def run_moe_prefill_gemm1(
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

        has_bias = 1.0 if bias is not None else 0.0
        params = self._get_params(
            ("prefill_gemm1", count, hidden_dim, intermediate_size, alpha, beta,
             clamp_limit, scale_gate, scale_up, has_bias),
            [float(count), float(hidden_dim), float(intermediate_size),
             float(alpha), float(beta), float(clamp_limit),
             float(scale_gate), float(scale_up), float(has_bias)],
        )

        if bias is not None:
            bias_buf = bias if bias.dtype == input.dtype else bias.to(input.dtype)
        else:
            if input.dtype not in self._moe_dummy_bias:
                self._moe_dummy_bias[input.dtype] = torch.zeros(
                    1, dtype=input.dtype, device="mps",
                )
            bias_buf = self._moe_dummy_bias[input.dtype]

        kernel = "moe_prefill_gemm1_swiglu" if count == 1 else "moe_prefill_gemm1_swiglu_tiled"

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

    def run_moe_prefill_gemm2(
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

        has_bias = 1.0 if bias is not None else 0.0
        params = self._get_params(
            ("prefill_gemm2", count, in_dim, out_dim, scale, has_bias),
            [float(count), float(in_dim), float(out_dim),
             float(scale), float(has_bias)],
        )

        if bias is not None:
            bias_buf = bias if bias.dtype == input.dtype else bias.to(input.dtype)
        else:
            if input.dtype not in self._moe_dummy_bias:
                self._moe_dummy_bias[input.dtype] = torch.zeros(
                    1, dtype=input.dtype, device="mps",
                )
            bias_buf = self._moe_dummy_bias[input.dtype]

        kernel = "moe_prefill_gemm2" if count == 1 else "moe_prefill_gemm2_tiled"

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

    # -------------------------------------------------------------------
    # Decode MoE kernels (SIMD K-split, all K experts per dispatch)
    # -------------------------------------------------------------------

    def _ensure_expert_ids_buf(self, K: int) -> None:
        """Ensure pre-allocated expert_ids buffer is large enough for K experts."""
        if self._moe_batched_expert_ids is None or self._moe_batched_expert_ids.shape[0] < K:
            self._moe_batched_expert_ids = torch.empty(K, dtype=torch.int32, device="mps")

    def run_moe_decode_gemm1_swiglu(
        self,
        input: torch.Tensor,
        all_w_blocks: torch.Tensor,
        all_w_scales: torch.Tensor,
        all_bias: "torch.Tensor | None",
        intermediate_size: int,
        expert_ids: torch.Tensor,
        alpha: float,
        clamp_limit: float,
    ) -> torch.Tensor:
        """Decode GEMM1 with fused SwiGLU: reads up+gate rows, outputs [K,I]."""
        lib = self._ensure_moe_matmul()
        hidden_dim = input.shape[1]
        K = expert_ids.shape[0]
        I = intermediate_size

        self._ensure_expert_ids_buf(K)

        if not hasattr(self, '_decode_gemm1_swiglu_out') or self._decode_gemm1_swiglu_out is None \
                or self._decode_gemm1_swiglu_out.shape != (K, I) \
                or self._decode_gemm1_swiglu_out.dtype != input.dtype:
            self._decode_gemm1_swiglu_out = torch.empty(K, I, dtype=input.dtype, device="mps")
        output = self._decode_gemm1_swiglu_out

        has_bias = 1.0 if all_bias is not None else 0.0

        params = self._get_params(
            ("decode_gemm1_swiglu", hidden_dim, I, alpha, clamp_limit, has_bias),
            [float(hidden_dim), float(I), float(alpha),
             float(clamp_limit), float(has_bias)],
        )

        if all_bias is not None:
            bias_buf = all_bias if all_bias.dtype == input.dtype else all_bias.to(input.dtype)
        else:
            if input.dtype not in self._moe_dummy_bias:
                self._moe_dummy_bias[input.dtype] = torch.zeros(
                    1, dtype=input.dtype, device="mps",
                )
            bias_buf = self._moe_dummy_bias[input.dtype]

        # Grid: (ceil(I/2), K) threadgroups, each (32, 2) threads
        num_col_tiles = (I + 1) // 2
        self._get_kernel_fn(lib, "moe_decode_gemm1_swiglu")(
            input.contiguous().view(-1),
            all_w_blocks.contiguous().view(-1),
            all_w_scales.contiguous().view(-1),
            bias_buf.contiguous().view(-1),
            output.view(-1),
            expert_ids,
            params,
            threads=(num_col_tiles * 32, K * 2, 1),
            group_size=(32, 2, 1),
        )
        return output

    def run_moe_decode_gemm1(
        self,
        input: torch.Tensor,
        all_w_blocks: torch.Tensor,
        all_w_scales: torch.Tensor,
        intermediate_size: int,
        expert_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Decode GEMM1 (raw): SIMD K-split, 32 lanes split K dimension."""
        lib = self._ensure_moe_matmul()
        hidden_dim = input.shape[1]
        K = expert_ids.shape[0]
        N = 2 * intermediate_size  # output dim per expert

        self._ensure_expert_ids_buf(K)

        if not hasattr(self, '_decode_gemm1_out') or self._decode_gemm1_out is None \
                or self._decode_gemm1_out.shape != (K, N) \
                or self._decode_gemm1_out.dtype != input.dtype:
            self._decode_gemm1_out = torch.empty(K, N, dtype=input.dtype, device="mps")
        output = self._decode_gemm1_out

        params = self._get_params(
            ("decode_gemm1", hidden_dim, N),
            [float(hidden_dim), float(N)],
        )

        # Grid: (ceil(N/4), K) threadgroups, each (32, 2) threads
        num_row_tiles = (N + 3) // 4
        self._get_kernel_fn(lib, "moe_decode_gemm1")(
            input.contiguous().view(-1),
            all_w_blocks.contiguous().view(-1),
            all_w_scales.contiguous().view(-1),
            output.view(-1),
            expert_ids,
            params,
            threads=(num_row_tiles * 32, K * 2, 1),
            group_size=(32, 2, 1),
        )
        return output

    def run_moe_decode_gemm2_fused(
        self,
        input: torch.Tensor,
        all_w_blocks: torch.Tensor,
        all_w_scales: torch.Tensor,
        all_bias: "torch.Tensor | None",
        out_dim: int,
        expert_ids: torch.Tensor,
        fused_scales: torch.Tensor,
    ) -> torch.Tensor:
        """Decode GEMM2 fused across experts. Returns float32."""
        lib = self._ensure_moe_matmul()
        K = input.shape[0]
        in_dim = input.shape[1]

        self._ensure_expert_ids_buf(K)

        if not hasattr(self, '_decode_gemm2_out') or self._decode_gemm2_out is None \
                or self._decode_gemm2_out.shape != (1, out_dim):
            self._decode_gemm2_out = torch.zeros(1, out_dim, dtype=torch.float32, device="mps")
        output = self._decode_gemm2_out
        output.zero_()

        has_bias = 1.0 if all_bias is not None else 0.0

        params = self._get_params(
            ("decode_gemm2_fused", out_dim, in_dim, K, has_bias),
            [float(out_dim), float(in_dim), float(K), float(has_bias)],
        )

        if all_bias is not None:
            bias_buf = all_bias if all_bias.dtype == input.dtype else all_bias.to(input.dtype)
        else:
            if input.dtype not in self._moe_dummy_bias:
                self._moe_dummy_bias[input.dtype] = torch.zeros(
                    1, dtype=input.dtype, device="mps",
                )
            bias_buf = self._moe_dummy_bias[input.dtype]

        # Grid: (ceil(out_dim/4), 1) threadgroups, each (32, 2) threads
        num_row_tiles = (out_dim + 3) // 4
        self._get_kernel_fn(lib, "moe_decode_gemm2_fused")(
            input.contiguous().view(-1),
            all_w_blocks.contiguous().view(-1),
            all_w_scales.contiguous().view(-1),
            bias_buf.contiguous().view(-1),
            output.view(-1),
            expert_ids,
            fused_scales.contiguous(),
            params,
            threads=(num_row_tiles * 32, 2, 1),
            group_size=(32, 2, 1),
        )
        return output

    # -------------------------------------------------------------------
    # RMS Norm
    # -------------------------------------------------------------------

    def _ensure_rms_norm(self) -> object:
        if "rms_norm" in self._libs:
            return self._libs["rms_norm"]
        self._ensure_env()
        source = self._read_metal("metal_rms_norm.metal")
        return self._compile(source, "rms_norm")

    def run_rms_norm(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
        eps: float,
    ) -> None:
        """RMS norm + weight multiply. Writes to pre-allocated output."""
        lib = self._ensure_rms_norm()
        H = input.shape[-1]
        params = self._get_params(("rms_norm", H, eps), [float(H), eps])
        kernel = "rms_norm_bf16" if input.dtype == torch.bfloat16 else "rms_norm_f16"
        self._get_kernel_fn(lib, kernel)(
            input.contiguous().view(-1),
            weight.contiguous().view(-1),
            output.view(-1),
            params,
            threads=(256, 1, 1),
            group_size=(256, 1, 1),
        )

    def run_residual_rms_norm(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        weight: torch.Tensor,
        res_out: torch.Tensor,
        norm_out: torch.Tensor,
        eps: float,
    ) -> None:
        """Fused residual add + RMS norm + weight multiply."""
        lib = self._ensure_rms_norm()
        H = a.shape[-1]
        params = self._get_params(("residual_rms_norm", H, eps), [float(H), eps])
        kernel = "residual_rms_norm_bf16" if a.dtype == torch.bfloat16 else "residual_rms_norm_f16"
        self._get_kernel_fn(lib, kernel)(
            a.contiguous().view(-1),
            b.contiguous().view(-1),
            weight.contiguous().view(-1),
            res_out.view(-1),
            norm_out.view(-1),
            params,
            threads=(256, 1, 1),
            group_size=(256, 1, 1),
        )

