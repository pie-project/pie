"""
Model loading for PIE backend.

This module handles the messy file I/O, ztensor vs safetensors,
TOML parsing, and weight schema loading.

See schema.py for the declarative weight schema API (Schema, Source, WeightStore).
"""

from __future__ import annotations

import tomllib
from contextlib import ExitStack
from pathlib import Path

import torch
from tqdm import tqdm
import safetensors

from .config import RuntimeConfig
from .schema import Schema, Source, Definition, WeightStore, ReaderFn
from . import hf_utils


# =============================================================================
# MODEL LOADER
# =============================================================================


class ModelLoader:
    """
    Handles model loading from HuggingFace cache.

    This separates the loading concerns from the runtime orchestration.
    The loader returns a WeightStore and model config - the runtime
    is responsible for creating the ForwardPass and KV cache.
    """

    def __init__(self, config: "RuntimeConfig", log_queue: object):
        """
        Initialize the model loader.

        Args:
            config: Runtime configuration with repo_id and arch
            log_queue: Optional queue for sending progress updates to CLI
        """
        self.config = config
        self.info: dict = {}
        self.snapshot_dir: Path | None = None
        self.log_queue = log_queue

    def load(self) -> tuple[WeightStore, dict, dict]:
        """
        Load the model weights and return components.

        Returns:
            Tuple of (weights, arch, model_info)
        """
        # Get HuggingFace snapshot directory
        self.snapshot_dir = hf_utils.get_hf_snapshot_dir(self.config.hf_repo)

        # Load config from HuggingFace config.json
        hf_config = hf_utils.load_hf_config(self.snapshot_dir)

        # Derive architecture from HF model_type
        hf_model_type = hf_config.get("model_type", "")
        # Handle case where model_type might be mapped (e.g. llama -> llama3)
        # We need hf_utils.HF_TO_PIE_ARCH to map it.
        # If it's not in the map, check if it's already a valid PIE type (less likely but possible)
        arch_type = hf_utils.HF_TO_PIE_ARCH.get(hf_model_type)

        if arch_type is None:
            # Basic fallback or error
            if hf_model_type in hf_utils.HF_TO_PIE_ARCH.values():
                arch_type = hf_model_type
            else:
                raise ValueError(
                    f"Unsupported HuggingFace model_type: '{hf_model_type}'. "
                    f"Supported types: {list(hf_utils.HF_TO_PIE_ARCH.keys())}"
                )

        # Inject PIE type into config for runtime to use
        hf_config["type"] = arch_type

        # Store info for later (tokenizer, template, etc.)
        self.info = {
            "architecture": hf_config,
            "hf_config": hf_config,
        }

        # Get schema for architecture type
        match arch_type:
            case "llama3":
                from .model import llama3

                # from_dict now expects raw HF config
                model_config = llama3.ModelConfig.from_dict(hf_config)
                schema = llama3.create_schema(model_config)
                num_layers = model_config.num_layers

            case "qwen2":
                from .model import qwen2

                model_config = qwen2.ModelConfig.from_dict(hf_config)
                schema = qwen2.create_schema(model_config)
                num_layers = model_config.num_layers

            case "qwen3":
                from .model import qwen3

                model_config = qwen3.ModelConfig.from_dict(hf_config)
                schema = qwen3.QWEN3_SCHEMA
                num_layers = model_config.num_layers

            case "gptoss":
                from .model import gpt_oss

                model_config = gpt_oss.ModelConfig.from_dict(hf_config)
                schema = gpt_oss.create_gpt_oss_schema(model_config)
                num_layers = model_config.num_layers

            case "gemma2":
                from .model import gemma2

                model_config = gemma2.ModelConfig.from_dict(hf_config)
                schema = gemma2.create_schema(model_config)
                num_layers = model_config.num_layers

            case "gemma3":
                from .model import gemma3

                model_config = gemma3.ModelConfig.from_dict(hf_config)
                schema = gemma3.create_schema(model_config)
                num_layers = model_config.num_layers

            case "mistral3":
                from .model import mistral3

                model_config = mistral3.ModelConfig.from_dict(hf_config)
                schema = mistral3.create_schema(model_config)
                num_layers = model_config.num_layers

            case "olmo3":
                from .model import olmo3

                model_config = olmo3.ModelConfig.from_dict(hf_config)
                schema = olmo3.OLMO3_SCHEMA
                num_layers = model_config.num_layers

            case _:
                raise ValueError(f"Unsupported architecture type: {arch_type}")

        # Load weights using schema
        weights = self.load_weights(schema, num_layers)

        return weights, hf_config, self.info

    def load_weights(self, schema: Schema, num_layers: int) -> WeightStore:
        """
        Load weights using the provided schema from HuggingFace cache.

        Args:
            schema: Weight schema defining the tensor mapping
            num_layers: Number of layers for expanding '*' patterns

        Returns:
            WeightStore with all loaded weights
        """
        if self.snapshot_dir is None:
            raise ValueError("snapshot_dir not set - call load() first")

        # Find all safetensor files in the snapshot
        safetensor_files = hf_utils.get_safetensor_files(self.snapshot_dir)
        if not safetensor_files:
            raise ValueError(f"No safetensor files found in {self.snapshot_dir}")

        # Load weights
        with ExitStack() as stack:
            readers: dict[str, object] = {}

            # Build tensor name -> reader mapping
            for param_file in tqdm(
                safetensor_files,
                desc="Scanning tensor files",
                unit="files",
                disable=True,
            ):
                param_path = self.snapshot_dir / param_file
                f = stack.enter_context(
                    safetensors.safe_open(str(param_path), framework="pt", device="cpu")
                )
                names = list(f.keys())

                for n in names:
                    readers[n] = f

            def reader(
                name: str, *, expected_shape: tuple[int, ...] | None = None
            ) -> torch.Tensor:
                f = readers.get(name)
                if f is None:
                    raise KeyError(f"Tensor '{name}' not found")

                t = f.get_tensor(name)

                if expected_shape is not None and tuple(t.shape) != tuple(
                    expected_shape
                ):
                    raise ValueError(
                        f"{name} has shape {tuple(t.shape)}, expected {tuple(expected_shape)}"
                    )
                return t

            # Load weights using schema
            weights = schema.load(
                reader=reader,
                config=self.config,
                num_layers=num_layers,
                log_queue=self.log_queue,
            )

        return weights
