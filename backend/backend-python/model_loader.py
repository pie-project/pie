"""Shared model loading utilities for PIE backends."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Tuple

import torch
import ztensor
from tqdm import tqdm

from model.config import ModelInfo
from model.gptoss_utils import (
    prepare_gptoss_moe_gate_up,
    prepare_gptoss_moe_down,
)


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
    op = fusion_details["op"]

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
        if op == "fusion":
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

        elif op == "prepare_gptoss_moe_gate_up":
            result = prepare_gptoss_moe_gate_up(
                source_tensors[0],  # blocks
                source_tensors[1],  # scales
                source_tensors[2],  # bias
                fusion_details,
                str(param.device),
            )
            output_type = fusion_details["output_type"]
            param.copy_(result[output_type], non_blocking=non_blocking)

        elif op == "prepare_gptoss_moe_down":
            result = prepare_gptoss_moe_down(
                source_tensors[0],  # blocks
                source_tensors[1],  # scales
                source_tensors[2],  # bias
                fusion_details,
                str(param.device),
            )
            output_type = fusion_details["output_type"]
            param.copy_(result[output_type], non_blocking=non_blocking)

        elif op == "to_float32":
            converted = source_tensors[0].to(torch.float32)
            param.copy_(converted, non_blocking=non_blocking)

        else:
            raise ValueError(f"Unknown fusion operation: {op}")

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


__all__ = ["load_model", "load_model_info"]
