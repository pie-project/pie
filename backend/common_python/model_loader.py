"""Shared model loading utilities for PIE backends."""

from __future__ import annotations

import sys
import math
from pathlib import Path
from typing import Callable, Iterable, Tuple

import torch
import ztensor
from tqdm import tqdm

from config.common import ModelInfo


CreateModelFn = Callable[[ModelInfo], Tuple[torch.nn.Module, dict]]


def load_model(
    config: dict, create_model_fn: CreateModelFn
) -> tuple[torch.nn.Module, ModelInfo]:
    """Load a model using the provided factory function and fusion metadata."""

    model_name = config["model"]
    cache_dir = config["cache_dir"]
    model_path = Path(cache_dir) / "models"
    metadata_path = model_path / f"{model_name}.toml"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")

    model_device = config["device"]
    model_dtype = getattr(torch, config["dtype"])
    model_info = ModelInfo.load_from_file(str(metadata_path), model_device, model_dtype)

    try:
        model, fusion_map = create_model_fn(model_info)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to instantiate model for architecture {model_info.architecture.type}: {exc}"
        ) from exc

    source_to_fusion_target = {
        source: target
        for target, details in fusion_map.items()
        for source in details["sources"]
    }

    pending_fusion_tensors = {}
    model_state_keys = set(model.state_dict().keys())
    loaded_keys = set()

    try:
        for param_file in model_info.parameters:
            weights_path = model_path / model_name / param_file
            with ztensor.Reader(str(weights_path)) as reader:
                tensor_names = reader.get_tensor_names()
                pbar_desc = (
                    f"Loading {param_file[:30]}..."
                    if len(param_file) > 30
                    else f"Loading {param_file}"
                )
                for name in tqdm(tensor_names, desc=pbar_desc, unit="tensors"):
                    if name in source_to_fusion_target:
                        pending_fusion_tensors[name] = reader.read_tensor(
                            name, to="torch"
                        )
                        continue

                    if name in model_state_keys and name not in loaded_keys:
                        tensor_data = reader.read_tensor(name, to="torch")
                        if tensor_data is None:
                            print(
                                f"    Warning: Could not read tensor '{name}'. Skipping."
                            )
                            continue
                        param = model.state_dict()[name]

                        if tensor_data.shape != param.shape:
                            print(
                                f"    Warning: Shape mismatch for tensor '{name}'. Skipping."
                            )
                            continue
                        # Ensure dtype/device compatibility using a single conversion
                        if hasattr(tensor_data, "to"):
                            needs_conversion = False
                            conversion_kwargs = {}

                            if hasattr(tensor_data, "dtype"):
                                tensor_dtype = getattr(tensor_data, "dtype")
                                if tensor_dtype != param.dtype:
                                    conversion_kwargs["dtype"] = param.dtype
                                    needs_conversion = True

                            if hasattr(tensor_data, "device"):
                                tensor_device = getattr(tensor_data, "device")
                                if tensor_device != param.device:
                                    conversion_kwargs["device"] = param.device
                                    needs_conversion = True

                            if needs_conversion:
                                to_method = getattr(tensor_data, "to")
                                tensor_data = to_method(**conversion_kwargs)
                        with torch.no_grad():
                            param.copy_(tensor_data, non_blocking=True)
                        loaded_keys.add(name)

        for target_name, details in tqdm(
            fusion_map.items(), desc="Fusing tensors", unit="tensors"
        ):
            source_names = details["sources"]
            if all(s in pending_fusion_tensors for s in source_names):
                param = model.state_dict()[target_name]
                if details["op"] == "fusion":
                    tensors_to_fuse = [pending_fusion_tensors[s] for s in source_names]
                    fused_tensor = torch.cat(tensors_to_fuse, dim=details["dim"])
                elif details["op"] == "dequantize_mxfp4":
                    blocks, scales = [pending_fusion_tensors[s] for s in source_names]
                    fused_tensor = dequantize_from_mxfp4(
                        blocks,
                        scales,
                        fp4_values=details["fp4_values"],
                        dtype=param.dtype,
                        device=param.device,
                    )
                else:
                    raise ValueError(f"Unknown fusion operation: {details['op']}")
                if fused_tensor.shape != param.shape:
                    print(
                        f"    Warning: Shape mismatch for fused tensor '{target_name}'. Skipping."
                    )
                    continue
                with torch.no_grad():
                    param.copy_(fused_tensor, non_blocking=True)
                loaded_keys.add(target_name)

        if "lm_head.weight" in model_state_keys and "lm_head.weight" not in loaded_keys:
            if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
                embed_tokens = getattr(model.model, "embed_tokens")
                if hasattr(embed_tokens, "weight"):
                    target_tensor = model.state_dict()["lm_head.weight"]
                    source_tensor = getattr(embed_tokens, "weight").to(
                        dtype=target_tensor.dtype, device=target_tensor.device
                    )
                    target_tensor.copy_(source_tensor, non_blocking=True)
                    loaded_keys.add("lm_head.weight")

        missing_keys = model_state_keys - loaded_keys
        if missing_keys:
            print("\nWarning: Some model weights were not found in any parameter file:")
            for key in sorted(list(missing_keys)):
                print(f"  - {key}")
        else:
            print("\nSuccessfully loaded all expected model weights.")

        model.eval()
        return model, model_info

    except ztensor.ZTensorError as exc:
        print(
            f"Fatal Error: Failed to read a ztensor file. Error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    except Exception as exc:
        print(
            f"An unexpected fatal error occurred during model loading: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def dequantize_from_mxfp4(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    fp4_values: Iterable[float],
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Convert MXFP4 format tensors (blocks and scales) to bfloat16 format.

    Args:
        blocks: The packed FP4 values tensor (uint8)
        scales: The block scales tensor
        dtype: Target dtype for conversion (default: torch.bfloat16)

    Returns:
        Converted tensor in the target dtype
    """
    scales = scales.to(torch.int32) - 127

    assert (
        blocks.shape[:-1] == scales.shape
    ), f"{blocks.shape=} does not match {scales.shape=}"

    lut = torch.tensor(fp4_values, dtype=dtype, device=device)

    *prefix_shape, g, b = blocks.shape
    rows_total = math.prod(prefix_shape) * g

    blocks = blocks.reshape(rows_total, b).to(device)
    scales = scales.reshape(rows_total, 1).to(device)

    # Extract low and high 4-bit indices
    idx_lo = (blocks & 0x0F).to(torch.long)
    idx_hi = (blocks >> 4).to(torch.long)

    # Create output tensor and populate
    out = torch.empty(rows_total, b * 2, dtype=dtype, device=device)
    out[:, 0::2] = lut[idx_lo]  # Low 4-bit values at even indices
    out[:, 1::2] = lut[idx_hi]  # High 4-bit values at odd indices

    torch.ldexp(out, scales, out=out)

    return out.reshape(*prefix_shape, g, b * 2).view(*prefix_shape, g * b * 2)


__all__ = ["load_model"]
