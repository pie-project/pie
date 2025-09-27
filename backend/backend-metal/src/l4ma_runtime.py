"""Metal-backed runtime implementation skeleton for the L4MA architecture."""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional

import torch

# Ensure backend-python and common_python modules are importable so we can reuse shared interfaces
BACKEND_PYTHON_PATH = Path(__file__).resolve().parents[2] / "backend-python"
COMMON_PYTHON_PATH = Path(__file__).resolve().parents[2] / "common_python"
if str(BACKEND_PYTHON_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PYTHON_PATH))
if str(COMMON_PYTHON_PATH) not in sys.path:
    sys.path.insert(0, str(COMMON_PYTHON_PATH))

from common_model.l4ma_runtime import L4maBackend, L4maForwardContext, RuntimeInputs
from config.l4ma import L4maArch
from debug_utils import is_tensor_debug_enabled, is_capture_debug_enabled

try:  # pragma: no cover - optional dependency guard
    from metal_backend import MetalBackend
except ImportError:  # pragma: no cover - optional dependency guard
    MetalBackend = None  # type: ignore[assignment]


def _get_debug_logger():
    """Get or create a debug logger that writes to a file."""
    logger = logging.getLogger('metal_runtime_debug')
    if not logger.handlers:
        # Set up file logging
        log_file = Path(os.environ.get("METAL_DEBUG_LOG_FILE", "/tmp/metal_debug.log"))
        log_file.parent.mkdir(parents=True, exist_ok=True)

        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        # Also log the start of a new session
        logger.info("=== NEW METAL DEBUG SESSION ===")

    return logger


@dataclass(frozen=True)
class MetalRuntimeMetadata:
    """Metadata describing the Metal runtime configuration."""

    page_size: int
    device: str


class _MetalForwardContext(L4maForwardContext):
    """Forward context that currently falls back to Torch operations.

    The intent is to progressively replace these implementations with true Metal
    kernels as they become available. The structure mirrors the FlashInfer
    context but keeps the surface torch-based so the model can execute while the
    Metal backend matures.
    """

    def __init__(
        self,
        *,
        config: L4maArch,
        inputs: RuntimeInputs,
    backend: Any | None,
        metadata: MetalRuntimeMetadata,
    ) -> None:
        self._config = config
        self._inputs = inputs
        self._backend = backend
        self._metadata = metadata

        capture_env = os.environ.get("METAL_CAPTURE_BOTH_PATHS", "0").lower()
        self._capture_both_paths = capture_env in {"1", "true", "yes", "on"}
        force_env = os.environ.get("METAL_FORCE_REFERENCE_OUTPUT", "0").lower()
        self._force_reference_output = force_env in {"1", "true", "yes", "on"}


        self._capture_output_dir: Path | None = None
        self._capture_counter = 0
        if self._capture_both_paths:
            capture_dir_env = os.environ.get("METAL_CAPTURE_OUTPUT_DIR")
            if capture_dir_env:
                self._capture_output_dir = Path(capture_dir_env).expanduser()
                self._capture_output_dir.mkdir(parents=True, exist_ok=True)

        # Derive batch indices/positions in a backend-agnostic way.
        self._batch_indices = self._compute_batch_indices(inputs.qo_indptr, inputs.num_tokens, device=config.device)
        self._batch_positions = self._compute_batch_positions(inputs.qo_indptr, device=config.device)

    @property
    def batch_indices(self) -> torch.Tensor:
        return self._batch_indices

    @property
    def batch_positions(self) -> torch.Tensor:
        return self._batch_positions

    @property
    def metadata(self) -> MetalRuntimeMetadata:
        return self._metadata

    @staticmethod
    def _compute_batch_indices(qo_indptr: torch.Tensor, num_tokens: int, device: str) -> torch.Tensor:
        batch_indices = torch.empty(num_tokens, dtype=torch.int32, device=device)
        for batch_idx in range(qo_indptr.numel() - 1):
            start = int(qo_indptr[batch_idx].item())
            end = int(qo_indptr[batch_idx + 1].item())
            batch_indices[start:end] = batch_idx
        return batch_indices

    @staticmethod
    def _compute_batch_positions(qo_indptr: torch.Tensor, device: str) -> torch.Tensor:
        positions = []
        for batch_idx in range(qo_indptr.numel() - 1):
            start = int(qo_indptr[batch_idx].item())
            end = int(qo_indptr[batch_idx + 1].item())
            seq_len = end - start
            positions.append(torch.arange(seq_len, dtype=torch.int32, device=device))
        if positions:
            return torch.cat(positions)
        return torch.empty(0, dtype=torch.int32, device=device)

    def apply_rope(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> None:
        # TODO: Replace with Metal RoPE kernel invocation. For now, use torch implementation.
        self._apply_rope_torch(query_states, position_ids, self._config)
        self._apply_rope_torch(key_states, position_ids, self._config)

    @staticmethod
    def _apply_rope_torch(tensor: torch.Tensor, position_ids: torch.Tensor, config: L4maArch) -> None:
        if tensor.ndim != 3:
            raise ValueError("Expected tensor shape [tokens, heads, head_size] for RoPE application")

        head_size = config.head_size
        half = head_size // 2
        sinusoids = _build_rope_sinusoids(position_ids, half, config.rope_theta, device=tensor.device, dtype=tensor.dtype)
        cos, sin = sinusoids

        # Expand sinusoids to match tensor shape [batch, heads, half_head_size]
        cos = cos.unsqueeze(1)  # [batch, 1, half_head_size]
        sin = sin.unsqueeze(1)  # [batch, 1, half_head_size]

        tensor_left = tensor[..., :half]
        tensor_right = tensor[..., half:]

        rotated_left = tensor_left * cos - tensor_right * sin
        rotated_right = tensor_left * sin + tensor_right * cos

        tensor[..., :half] = rotated_left
        tensor[..., half:] = rotated_right

    def append_kv_cache(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        kv_cache_layer: torch.Tensor,
    ) -> None:
        # TODO: Implement Metal-aware KV cache writes. Currently falls back to torch copy.
        positions = self._batch_positions
        page_size = self._metadata.page_size
        kv_indices = self._inputs.kv_page_indices

        if key_states.numel() and is_tensor_debug_enabled():
            k_min, k_max = key_states.aminmax()
            k_nan = torch.isnan(key_states).any().item()
            k_inf = torch.isinf(key_states).any().item()
            print(
                "[MetalTensorDebug]",
                f"layer={layer_idx}",
                "stage=kv_append_keys",
                "dtype=",
                key_states.dtype,
                "min=",
                float(k_min),
                "max=",
                float(k_max),
                "has_nan=",
                bool(k_nan),
                "has_inf=",
                bool(k_inf),
            )

        if value_states.numel() and is_tensor_debug_enabled():
            v_min, v_max = value_states.aminmax()
            v_nan = torch.isnan(value_states).any().item()
            v_inf = torch.isinf(value_states).any().item()
            print(
                "[MetalTensorDebug]",
                f"layer={layer_idx}",
                "stage=kv_append_values",
                "dtype=",
                value_states.dtype,
                "min=",
                float(v_min),
                "max=",
                float(v_max),
                "has_nan=",
                bool(v_nan),
                "has_inf=",
                bool(v_inf),
            )

        kv_page_indptr = self._inputs.kv_page_indptr

        for token_idx in range(key_states.size(0)):
            batch_idx = int(self._batch_indices[token_idx].item())
            seq_pos = int(positions[token_idx].item())

            # seq_pos is position within this sequence, so page_slot is logical page within sequence
            page_slot = seq_pos // page_size

            # Map logical page_slot to physical page index using indptr
            page_start = int(kv_page_indptr[batch_idx].item())
            page_end = int(kv_page_indptr[batch_idx + 1].item())
            physical_page_idx = page_start + page_slot

            # Handle edge case where sequence needs more pages than allocated
            if physical_page_idx >= page_end:
                physical_page_idx = page_end - 1

            offset = seq_pos % page_size
            cache_page = int(kv_indices[physical_page_idx].item())

            kv_cache_layer[cache_page, 0, offset].copy_(key_states[token_idx])
            kv_cache_layer[cache_page, 1, offset].copy_(value_states[token_idx])

    def run_attention(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        kv_cache_layer: torch.Tensor,
    ) -> torch.Tensor:
        if self._backend is None:
            raise RuntimeError("Metal backend is not available")

        # Add debug logging for Metal runtime inputs
        debug_runtime = os.environ.get("METAL_DEBUG_RUNTIME", "0") == "1"
        logger = None
        if debug_runtime:
            logger = _get_debug_logger()
            logger.info(f"=== METAL RUNTIME DEBUG layer={layer_idx} ===")
            logger.info(f"query_states: shape={query_states.shape}, dtype={query_states.dtype}")
            logger.info(f"kv_cache_layer: shape={kv_cache_layer.shape}, dtype={kv_cache_layer.dtype}")
            logger.info(f"qo_indptr: {self._inputs.qo_indptr.tolist()}")
            logger.info(f"kv_page_indices: {self._inputs.kv_page_indices.tolist()}")
            logger.info(f"kv_page_indptr: {self._inputs.kv_page_indptr.tolist()}")
            logger.info(f"kv_last_page_lens: {self._inputs.kv_last_page_lens.tolist()}")
            logger.info(f"custom_mask: {self._inputs.custom_mask is not None}")
            if query_states.numel() > 0:
                q_sample = query_states.flatten()[:10].tolist()
                logger.info(f"query sample (first 10): {q_sample}")

        try:
            original_dtype = query_states.dtype

            query_cpu = query_states.detach().to(device="cpu", dtype=torch.float32)
            cache_cpu = kv_cache_layer.detach().to(device="cpu", dtype=torch.float32)

            query_np = query_cpu.numpy()
            cache_np = cache_cpu.numpy()
            kv_indices_np = self._inputs.kv_page_indices.detach().cpu().numpy()
            kv_indptr_np = self._inputs.kv_page_indptr.detach().cpu().numpy()
            kv_last_len_np = self._inputs.kv_last_page_lens.detach().cpu().numpy()

            # Prepare optional custom mask for Metal backend (uint8 flat array)
            custom_mask_np = None
            if self._inputs.custom_mask is not None:
                # If masked kernel is unavailable, fall back to Torch for correctness
                masked_capable = False
                try:
                    caps = self._backend.get_capabilities()
                    masked_capable = bool(caps.get('masked_attention', False))
                except Exception:
                    masked_capable = False

                if not masked_capable:
                    import warnings
                    warnings.warn(
                        "Metal backend does not expose a masked attention kernel; using segmented Metal execution per request.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    # Segmented execution: run Metal attention separately per request
                    original_dtype = query_states.dtype
                    device = query_states.device
                    query_cpu_full = query_states.detach().to(device="cpu", dtype=torch.float32)
                    cache_cpu = kv_cache_layer.detach().to(device="cpu", dtype=torch.float32)

                    query_np_full = query_cpu_full.numpy()
                    cache_np = cache_cpu.numpy()
                    kv_indices_np = self._inputs.kv_page_indices.detach().cpu().numpy()
                    kv_indptr_np = self._inputs.kv_page_indptr.detach().cpu().numpy()
                    kv_last_len_np = self._inputs.kv_last_page_lens.detach().cpu().numpy()
                    qo_indptr_np_full = self._inputs.qo_indptr.detach().cpu().numpy()

                    import numpy as np
                    nh = query_np_full.shape[1]
                    hs = query_np_full.shape[2] if query_np_full.ndim == 3 else self._config.head_size
                    outputs = np.empty((query_np_full.shape[0], nh * hs), dtype=np.float32)

                    for b in range(len(qo_indptr_np_full) - 1):
                        q_start = int(qo_indptr_np_full[b])
                        q_end = int(qo_indptr_np_full[b + 1])
                        if q_end <= q_start:
                            continue

                        p_start = int(kv_indptr_np[b])
                        p_end = int(kv_indptr_np[b + 1])
                        kv_idx_sub = kv_indices_np[p_start:p_end]
                        kv_indptr_sub = np.array([0, len(kv_idx_sub)], dtype=np.int32)
                        kv_last_len_sub = np.array([int(kv_last_len_np[b])], dtype=np.int32)
                        qo_indptr_sub = np.array([0, q_end - q_start], dtype=np.int32)

                        query_sub = query_np_full[q_start:q_end]

                        sub_result = self._backend.run_attention_with_kv_cache(
                            query_sub,
                            cache_np,
                            kv_page_indices=kv_idx_sub,
                            kv_page_indptr=kv_indptr_sub,
                            kv_last_page_lens=kv_last_len_sub,
                            qo_indptr=qo_indptr_sub,
                        )
                        if not hasattr(sub_result, "output") or sub_result.output is None:
                            raise RuntimeError("Metal attention execution returned no output for a segment")
                        outputs[q_start:q_end] = sub_result.output

                    output = torch.from_numpy(outputs).to(device=device, dtype=torch.float32)
                    if output.dtype != original_dtype:
                        output = output.to(original_dtype)
                    return output

                # Otherwise, inform that masked Metal path will be used
                import warnings
                warnings.warn(
                    "Metal backend will apply custom_mask using masked attention kernel (may impact performance).",
                    RuntimeWarning,
                    stacklevel=2,
                )

                cm = self._inputs.custom_mask.detach().to(device="cpu")
                try:
                    import numpy as np  # local import to avoid hard dep in environments without numpy
                    custom_mask_np = cm.numpy().astype(np.uint8, copy=False)
                except Exception:
                    custom_mask_np = None

            qo_indptr_np = self._inputs.qo_indptr.detach().cpu().numpy()

            if debug_runtime:
                logger.info(f"Calling Metal backend with:")
                logger.info(f"  query_np.shape: {query_np.shape}")
                logger.info(f"  cache_np.shape: {cache_np.shape}")
                logger.info(f"  kv_indices_np: {kv_indices_np}")
                logger.info(f"  kv_indptr_np: {kv_indptr_np}")
                logger.info(f"  kv_last_len_np: {kv_last_len_np}")
                logger.info(f"  qo_indptr_np: {qo_indptr_np}")
                logger.info(f"  custom_mask_np: {custom_mask_np is not None}")

            result = self._backend.run_attention_with_kv_cache(
                query_np,
                cache_np,
                kv_page_indices=kv_indices_np,
                kv_page_indptr=kv_indptr_np,
                kv_last_page_lens=kv_last_len_np,
                qo_indptr=qo_indptr_np,
                custom_mask=custom_mask_np,
            )
        except Exception as exc:
            raise RuntimeError(f"Metal attention execution failed: {exc}") from exc

        if not hasattr(result, "output") or result.output is None:
            raise RuntimeError("Metal attention execution returned no output")

        output = torch.from_numpy(result.output).to(device=query_states.device, dtype=torch.float32)
        if output.dtype != original_dtype:
            output = output.to(original_dtype)
        if not torch.isfinite(output).all():
            raise RuntimeError("Metal attention execution produced non-finite values")

        if debug_runtime and logger:
            logger.info(f"Metal backend returned:")
            logger.info(f"  result.output.shape: {result.output.shape}")
            logger.info(f"  output range: [{float(output.min()):.6f}, {float(output.max()):.6f}]")
            if output.numel() > 0:
                out_sample = output.flatten()[:10].tolist()
                logger.info(f"  output sample (first 10): {out_sample}")
            logger.info("=== END METAL RUNTIME DEBUG ===\n")

        debug_compare = os.environ.get("METAL_DEBUG_COMPARE", "0") == "1"
        compute_reference = self._capture_both_paths or debug_compare or self._force_reference_output
        ref_output: torch.Tensor | None = None

        if compute_reference:
            ref_output = self._compute_torch_reference(layer_idx, query_states, kv_cache_layer)

        if self._capture_both_paths and ref_output is not None:
            self._record_attention_snapshot(layer_idx, output, ref_output)

        if debug_compare and ref_output is not None:
            diff = torch.max(torch.abs(output.to(torch.float32) - ref_output.to(torch.float32))).item()
            print("[MetalCompareDebug] max |metal - torch| =", diff)

        if self._force_reference_output and ref_output is not None:
            if ref_output.dtype != original_dtype:
                ref_output = ref_output.to(original_dtype)
            output = ref_output.to(device=query_states.device)
            if is_capture_debug_enabled():
                print(
                    "[MetalCaptureDebug]",
                    f"layer={layer_idx}",
                    "using_reference_output=1",
                )

        return output.view(query_states.size(0), -1)

    def _record_attention_snapshot(
        self,
        layer_idx: int,
        metal_output: torch.Tensor,
        reference_output: torch.Tensor,
    ) -> None:
        metal_cpu = metal_output.detach().to(device="cpu", dtype=torch.float32)
        reference_cpu = reference_output.detach().to(device="cpu", dtype=torch.float32)

        if metal_cpu.numel() == 0:
            return

        diff_cpu = metal_cpu - reference_cpu
        abs_diff_cpu = diff_cpu.abs()
        token_norms = abs_diff_cpu.max(dim=1).values
        metal_min, metal_max = float(metal_cpu.min().item()), float(metal_cpu.max().item())
        ref_min, ref_max = float(reference_cpu.min().item()), float(reference_cpu.max().item())
        diff_max = float(abs_diff_cpu.max().item())
        diff_mean = float(abs_diff_cpu.mean().item()) if abs_diff_cpu.numel() else 0.0

        worst_token_idx = int(token_norms.argmax().item()) if token_norms.numel() else -1
        batch_token_indices = getattr(self._inputs, "batch_token_indices", None)
        if worst_token_idx >= 0 and batch_token_indices is not None and len(batch_token_indices) > worst_token_idx:
            worst_token = int(batch_token_indices[worst_token_idx].item())
        else:
            worst_token = worst_token_idx

        if is_capture_debug_enabled():
            print(
                "[MetalCaptureDebug]",
                f"layer={layer_idx}",
                f"tokens={metal_cpu.size(0)}",
                f"metal_min={metal_min:.6f}",
                f"metal_max={metal_max:.6f}",
                f"ref_min={ref_min:.6f}",
                f"ref_max={ref_max:.6f}",
                f"diff_max={diff_max:.6f}",
                f"diff_mean={diff_mean:.6f}",
                f"worst_token={worst_token}",
            )

        if self._capture_output_dir:
            capture_path = self._capture_output_dir / (
                f"layer_{layer_idx:02d}_capture_{self._capture_counter:03d}.pt"
            )
            torch.save(
                {
                    "layer_idx": layer_idx,
                    "metal": metal_cpu,
                    "reference": reference_cpu,
                    "diff_max": diff_max,
                    "diff_mean": diff_mean,
                    "worst_token_idx": worst_token_idx,
                    "token_norms": token_norms,
                },
                capture_path,
            )
        self._capture_counter += 1

    def _compute_torch_reference(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        kv_cache_layer: torch.Tensor,
    ) -> torch.Tensor:
        import torch.nn.functional as F

        device = query_states.device
        dtype = query_states.dtype

        kv_indices = self._inputs.kv_page_indices
        kv_page_indptr = self._inputs.kv_page_indptr
        kv_last_page_lens = self._inputs.kv_last_page_lens
        page_size = self._metadata.page_size
        qo_indptr = self._inputs.qo_indptr

        # Build KV tensors per request ("batch"), keeping per-request grouping
        batch_keys: list[torch.Tensor] = []
        batch_values: list[torch.Tensor] = []

        for batch_idx in range(qo_indptr.numel() - 1):
            page_start = int(kv_page_indptr[batch_idx].item())
            page_end = int(kv_page_indptr[batch_idx + 1].item())

            keys: list[torch.Tensor] = []
            values: list[torch.Tensor] = []

            for page_idx in range(page_start, page_end):
                page_ptr = int(kv_indices[page_idx].item())
                page_tensor = kv_cache_layer[page_ptr]

                # Determine length for this page
                if page_idx == page_end - 1 and kv_last_page_lens.numel() > batch_idx:
                    length = int(kv_last_page_lens[batch_idx].item()) or page_size
                else:
                    length = page_size

                keys.append(page_tensor[0, :length])
                values.append(page_tensor[1, :length])

            if keys:
                batch_keys.append(torch.cat(keys, dim=0))
                batch_values.append(torch.cat(values, dim=0))
            else:
                batch_keys.append(torch.empty(0, device=device, dtype=dtype))
                batch_values.append(torch.empty(0, device=device, dtype=dtype))

        num_q_heads = query_states.size(1)
        out = torch.empty(query_states.size(0), num_q_heads * self._config.head_size, device=device, dtype=dtype)

        # Compute attention per request to prevent cross-request leakage
        q_f32 = query_states.to(torch.float32)
        for batch_idx in range(qo_indptr.numel() - 1):
            q_start = int(qo_indptr[batch_idx].item())
            q_end = int(qo_indptr[batch_idx + 1].item())
            if q_end <= q_start:
                continue

            k_b = batch_keys[batch_idx]
            v_b = batch_values[batch_idx]

            # If no KV for this request yet, output zeros
            if k_b.numel() == 0 or v_b.numel() == 0:
                out[q_start:q_end].zero_()
                continue

            num_kv_heads = k_b.size(1)
            if num_kv_heads != num_q_heads:
                repeat_factor = max(1, num_q_heads // max(1, num_kv_heads))
                k_b = k_b.repeat_interleave(repeat_factor, dim=1)
                v_b = v_b.repeat_interleave(repeat_factor, dim=1)

            q_b = q_f32[q_start:q_end]
            attn_b = F.scaled_dot_product_attention(
                q_b.permute(1, 0, 2),
                k_b.to(torch.float32).permute(1, 0, 2),
                v_b.to(torch.float32).permute(1, 0, 2),
                is_causal=True,
            )

            if attn_b.numel() and is_tensor_debug_enabled():
                ref_min, ref_max = attn_b.aminmax()
                ref_nan = torch.isnan(attn_b).any().item()
                ref_inf = torch.isinf(attn_b).any().item()
                print(
                    "[MetalTensorDebug]",
                    f"layer={layer_idx}",
                    "stage=torch_ref_output_batch",
                    "batch=", batch_idx,
                    "dtype=",
                    attn_b.dtype,
                    "min=",
                    float(ref_min),
                    "max=",
                    float(ref_max),
                    "has_nan=",
                    bool(ref_nan),
                    "has_inf=",
                    bool(ref_inf),
                )

            out[q_start:q_end] = attn_b.permute(1, 0, 2).reshape(q_b.size(0), -1).to(device=device, dtype=dtype)

        return out


def _build_rope_sinusoids(position_ids: torch.Tensor, half_head: int, rope_theta: float, *, device: torch.device | str, dtype: torch.dtype):
    float_dtype = torch.float32
    positions = position_ids.to(device=device, dtype=float_dtype)
    indices = torch.arange(half_head, device=device, dtype=float_dtype)
    frequencies = 1.0 / (rope_theta ** (indices / half_head))
    angles = torch.einsum("b,h->bh", positions, frequencies)
    cos = torch.cos(angles).to(dtype)
    sin = torch.sin(angles).to(dtype)
    return cos, sin


class MetalL4maBackend(L4maBackend):
    """Metal runtime backend that mirrors the FlashInfer interface."""

    @staticmethod
    def is_available() -> bool:
        return MetalBackend is not None

    def __init__(self, metal_backend: Optional[Any] = None) -> None:
        mb: Any | None = metal_backend
        if mb is None and MetalBackend is not None:
            mb = MetalBackend()
            ok = False
            try:
                ok = bool(getattr(mb, "initialize", lambda: False)())
            except Exception:
                ok = False
            if not ok:
                mb = None

        self._backend = mb

        if self._backend is None:
            raise RuntimeError(
                "Metal backend is not available; install the Metal debug framework to use MetalL4maBackend."
            )

    def create_forward_context(
        self,
        *,
        config: L4maArch,
        inputs: RuntimeInputs,
    ) -> L4maForwardContext:
        page_size = int(inputs.kv_cache_at_layer[0].shape[2]) if inputs.kv_cache_at_layer else config.head_size
        metadata = MetalRuntimeMetadata(
            page_size=page_size,
            device=str(config.device),
        )

        return _MetalForwardContext(
            config=config,
            inputs=inputs,
            backend=self._backend,
            metadata=metadata,
        )


__all__ = [
    "MetalL4maBackend",
    "MetalRuntimeMetadata",
]
