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


class MetadataNotFoundError(FileNotFoundError):
    """Exception raised when a metadata file is not found."""

    model_name: str

    def __init__(self, model_name: str, error: str):
        self.model_name = model_name
        super().__init__(error)


def load_model_info(config: dict) -> ModelInfo:
    """Load the model information from the metadata file."""

    # Locate the metadata file from the backend config.
    model_name = config["model"]
    cache_dir = config["cache_dir"]
    model_path = Path(cache_dir) / "models"
    metadata_path = model_path / f"{model_name}.toml"

    if not metadata_path.exists():
        raise MetadataNotFoundError(
            model_name, f"Metadata file not found at: {metadata_path}"
        )

    # Load the model information from the metadata file.
    model_device = config["device"]
    model_dtype = getattr(torch, config["dtype"])
    model_info = ModelInfo.load_from_file(str(metadata_path), model_device, model_dtype)

    return model_info


def load_model(
    config: dict, model_info: ModelInfo, create_model_fn: CreateModelFn
) -> torch.nn.Module:
    """Load a model using the provided factory function and fusion metadata."""
    model_name = config["model"]
    cache_dir = config["cache_dir"]
    model_path = Path(cache_dir) / "models"
    # Use blocking copy for MPS device to avoid race conditions
    # MPS async operations may not complete before function returns
    use_non_blocking = config["device"] != "mps"
    # Instantiate the model and its fusion map.
    try:
        model, fusion_map = create_model_fn(model_info)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to instantiate model for architecture {model_info.architecture.type}: {exc}"
        ) from exc

    tensor_to_file_map = {}
    file_readers = {}

    try:
        # Scan the tensor files and build the mapping of tensor names to the corresponding files.
        for param_file in tqdm(
            model_info.parameters, desc="Scanning tensor files", unit="files"
        ):
            weights_path = model_path / model_name / param_file
            reader = ztensor.Reader(str(weights_path))
            file_readers[param_file] = reader

            tensor_names = reader.get_tensor_names()
            for name in tensor_names:
                tensor_to_file_map[name] = param_file

        model_state_keys = set(model.state_dict().keys())
        loaded_keys = set()
        model_state = model.state_dict()

        # Load the parameters from the files to the model.
        for param_name in tqdm(
            model_state_keys, desc="Loading model parameters", unit="tensors"
        ):
            param = model_state[param_name]

            if param_name in fusion_map:
                success = _load_fused_parameter(
                    param_name,
                    param,
                    fusion_map[param_name],
                    tensor_to_file_map,
                    file_readers,
                    use_non_blocking,
                )
            else:
                success = _load_regular_parameter(
                    param_name,
                    param,
                    tensor_to_file_map,
                    file_readers,
                    use_non_blocking,
                )

            if success:
                loaded_keys.add(param_name)

        # Handle weight tying for the LM head.
        if "lm_head.weight" in model_state_keys and "lm_head.weight" not in loaded_keys:
            if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
                embed_tokens = getattr(model.model, "embed_tokens")
                if hasattr(embed_tokens, "weight"):
                    target_tensor = model.state_dict()["lm_head.weight"]
                    source_tensor = getattr(embed_tokens, "weight").to(
                        dtype=target_tensor.dtype, device=target_tensor.device
                    )
                    param = model_state["lm_head.weight"]
                    target_tensor.copy_(source_tensor, non_blocking=use_non_blocking)
                    loaded_keys.add("lm_head.weight")

        missing_keys = model_state_keys - loaded_keys
        if missing_keys:
            print("\nWarning: Some model weights were not found in any parameter file:")
            for key in sorted(list(missing_keys)):
                print(f"  - {key}")
        else:
            print("\nSuccessfully loaded all expected model weights.")

        model.eval()
        return model

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


def _load_fused_parameter(
    param_name: str,
    param: torch.Tensor,
    fusion_details: dict,
    tensor_to_file_map: dict,
    file_readers: dict,
    non_blocking: bool = True,
) -> bool:
    """Load and process a fusion parameter (concatenation or dequantization).

    Returns True if successful, False if the parameter loading is skipped due to errors.
    """
    source_names = fusion_details["sources"]

    source_tensors = []
    for source_name in source_names:
        if source_name not in tensor_to_file_map:
            print(
                f"    Warning: Could not load fusion source tensor '{source_name}'. "
                f"Skipping fusion for '{param_name}'."
            )
            return False

        param_file = tensor_to_file_map[source_name]
        reader = file_readers[param_file]
        tensor_data = reader.read_tensor(source_name, to="torch")
        source_tensors.append(tensor_data)

    with torch.no_grad():
        if fusion_details["op"] == "fusion":
            dim = fusion_details["dim"]

            if param.shape[dim] != sum(
                source_tensor.shape[dim] for source_tensor in source_tensors
            ):
                print(
                    f"    Warning: Shape mismatch for fused tensor '{param_name}'. "
                    "Skipping."
                )
                return False

            current_offset = 0
            slice_indices = [slice(None)] * param.ndim

            for source_tensor in source_tensors:
                slice_size = source_tensor.shape[dim]
                slice_indices[dim] = slice(current_offset, current_offset + slice_size)

                param[tuple(slice_indices)].copy_(
                    source_tensor, non_blocking=non_blocking
                )
                current_offset += slice_size

        elif fusion_details["op"] == "dequantize_mxfp4":
            blocks, scales = source_tensors[0], source_tensors[1]
            fused_tensor = _dequantize_from_mxfp4(
                blocks,
                scales,
                fp4_values=fusion_details["fp4_values"],
                dtype=param.dtype,
                device=str(param.device),
            )
            if fused_tensor.shape != param.shape:
                print(
                    f"    Warning: Shape mismatch for fused tensor '{param_name}'. "
                    "Skipping."
                )
                return False

            param.copy_(fused_tensor, non_blocking=non_blocking)
        else:
            raise ValueError(f"Unknown fusion operation: {fusion_details['op']}")

    return True


def _load_regular_parameter(
    param_name: str,
    param: torch.Tensor,
    tensor_to_file_map: dict,
    file_readers: dict,
    non_blocking: bool = True,
) -> bool:
    """Load and process a regular (non-fused) parameter.

    Returns True if successful, False if the parameter loading is skipped due to errors.
    """
    if param_name not in tensor_to_file_map:
        print(f"    Warning: Could not read tensor '{param_name}'. Skipping.")
        return False

    param_file = tensor_to_file_map[param_name]
    reader = file_readers[param_file]
    tensor_data = reader.read_tensor(param_name, to="torch")

    if tensor_data.shape != param.shape:
        print(f"    Warning: Shape mismatch for tensor '{param_name}'. Skipping.")
        return False

    with torch.no_grad():
        param.copy_(tensor_data, non_blocking=non_blocking)

    return True


def _dequantize_from_mxfp4(
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


__all__ = ["load_model", "load_model_info"]
