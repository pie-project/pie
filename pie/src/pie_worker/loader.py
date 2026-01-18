"""
Model loading for PIE backend.

This module handles the messy file I/O, ztensor vs safetensors,
TOML parsing, and weight schema loading.

WeightSchema provides a declarative API for defining how model weights
are loaded, fused, sharded, and transformed.

Example:
    schema = (
        Schema("llama3")
        .define("token_embeds",
            Source("model.embed_tokens.weight")
            .shard("row"))
        .define("layers.*.attn.qkv",
            Source.fuse([
                "model.layers.*.self_attn.q_proj.weight",
                "model.layers.*.self_attn.k_proj.weight",
                "model.layers.*.self_attn.v_proj.weight",
            ], dim=0)
            .shard("column")
            .quantize())
    )

    weights = schema.load(reader, config, num_layers=32)
"""

from __future__ import annotations

import tomllib
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

import torch
from tqdm import tqdm
import safetensors

from .quantization import quantize
from .config import RuntimeConfig
from . import hf_utils


# =============================================================================
# WEIGHT SCHEMA CLASSES
# =============================================================================

ReaderFn = Callable[[str], torch.Tensor]


@dataclass
class Source:
    """
    Defines a source tensor or fusion of multiple source tensors.

    Use method chaining to apply transforms:
        Source("weight.name").shard("row").quantize()
        Source.fuse([...], dim=0).transform(my_fn, arg1=val1)
        Source("sinks").dtype(torch.float32)
    """

    _patterns: list[str]
    _fuse_dim: int | None = None
    _sharding: str | None = None  # None, 'column', or 'row'
    _should_quantize: bool = False
    _expected_shapes: list[tuple[int, ...] | None] | None = None
    _transform_fn: (
        Callable[[list[torch.Tensor], dict[str, Any]], torch.Tensor | dict] | None
    ) = None
    _transform_kwargs: dict[str, Any] | None = None
    _dtype: torch.dtype | None = None
    _gather_only: bool = (
        False  # For transforms that need multiple tensors without concatenation
    )

    def __init__(self, pattern: str):
        """Create a source from a single tensor pattern."""
        self._patterns = [pattern]
        self._fuse_dim = None
        self._sharding = None
        self._should_quantize = False
        self._expected_shapes = None
        self._transform_fn = None
        self._transform_kwargs = None
        self._dtype = None
        self._gather_only = False

    @classmethod
    def fuse(cls, patterns: list[str], dim: int = 0) -> Source:
        """
        Create a fused source from multiple tensor patterns.

        Args:
            patterns: List of tensor name patterns to fuse
            dim: Dimension along which to concatenate (default: 0)
                 Use dim=None with gather_only=True to gather without concat.

        Returns:
            A new Source configured for fusion
        """
        source = cls.__new__(cls)
        source._patterns = patterns
        source._fuse_dim = dim
        source._sharding = None
        source._should_quantize = False
        source._expected_shapes = None
        source._transform_fn = None
        source._transform_kwargs = None
        source._dtype = None
        source._gather_only = False
        return source

    @classmethod
    def gather(cls, patterns: list[str]) -> Source:
        """
        Gather multiple tensors without fusing. Used with .transform() for
        operations that need multiple input tensors.

        Args:
            patterns: List of tensor name patterns to gather

        Returns:
            A new Source configured for gathering (no concatenation)
        """
        source = cls.__new__(cls)
        source._patterns = patterns
        source._fuse_dim = None
        source._sharding = None
        source._should_quantize = False
        source._expected_shapes = None
        source._transform_fn = None
        source._transform_kwargs = None
        source._dtype = None
        source._gather_only = True
        return source

    def shard(self, strategy: str) -> Source:
        """
        Set sharding strategy: 'column', 'row', or 'interleaved_column'.

        'interleaved_column': Shards input tensors individually along dim 0 BEFORE fusion.
                              Useful for fused weights like QKV where head alignment matters.
        """
        if strategy not in ("column", "row", "interleaved_column"):
            raise ValueError(
                f"Invalid sharding strategy: {strategy}. Use 'column', 'row', or 'interleaved_column'."
            )
        self._sharding = strategy
        return self

    def quantize(self) -> Source:
        """Enable quantization for this weight."""
        self._should_quantize = True
        return self

    def transform(
        self,
        fn: Callable[[list[torch.Tensor], dict[str, Any]], torch.Tensor | dict],
        **kwargs: Any,
    ) -> Source:
        """
        Apply a custom transformation function to loaded tensors.

        The function receives:
            - tensors: List of loaded tensors (from patterns)
            - kwargs: Additional keyword arguments passed here + 'device'

        It should return:
            - A single tensor, OR
            - A dict with 'output_type' key selecting which tensor to return

        Example:
            Source.gather(["blocks", "scales", "bias"])
                .transform(prepare_moe_weights, output_type="weights")
        """
        self._transform_fn = fn
        self._transform_kwargs = kwargs
        return self

    def dtype(self, dt: torch.dtype) -> Source:
        """Convert tensor to specified dtype."""
        self._dtype = dt
        return self

    def expect_shapes(self, shapes: list[tuple[int, ...] | None]) -> Source:
        """Set expected shapes for validation (optional)."""
        self._expected_shapes = shapes
        return self

    @property
    def is_fused(self) -> bool:
        return self._fuse_dim is not None and not self._gather_only

    @property
    def is_gathered(self) -> bool:
        return self._gather_only

    @property
    def has_transform(self) -> bool:
        return self._transform_fn is not None

    @property
    def patterns(self) -> list[str]:
        return self._patterns

    @property
    def fuse_dim(self) -> int | None:
        return self._fuse_dim

    @property
    def sharding(self) -> str | None:
        return self._sharding

    @property
    def should_quantize(self) -> bool:
        return self._should_quantize

    @property
    def target_dtype(self) -> torch.dtype | None:
        return self._dtype


@dataclass
class Definition:
    """A named weight definition: logical name -> Source with transforms."""

    name: str  # Logical name, may contain '*' for layer patterns
    source: Source

    def has_layer_pattern(self) -> bool:
        """Check if this definition uses layer patterns (*)."""
        return "*" in self.name

    def expand_for_layer(self, layer_idx: int) -> str:
        """Expand the logical name for a specific layer."""
        return self.name.replace("*", str(layer_idx))

    def expand_source_for_layer(self, layer_idx: int) -> list[str]:
        """Expand source patterns for a specific layer."""
        return [p.replace("*", str(layer_idx)) for p in self.source.patterns]


class WeightStore:
    """Container for loaded weights, accessible by logical name."""

    def __init__(self):
        self._weights: dict[str, torch.Tensor] = {}

    def put(self, name: str, tensor: torch.Tensor) -> None:
        """Store a tensor by logical name."""
        self._weights[name] = tensor

    def get(self, name: str) -> torch.Tensor:
        """Retrieve a tensor by logical name."""
        if name not in self._weights:
            raise KeyError(f"Weight '{name}' not found in store")
        return self._weights[name]

    def get_list(self, pattern: str, count: int) -> list[torch.Tensor]:
        """
        Retrieve a list of tensors matching a pattern.

        Args:
            pattern: Pattern with '*' placeholder (e.g., "layers.*.proj_qkv")
            count: Number of items to retrieve

        Returns:
            List of tensors for indices 0..count-1
        """
        return [self.get(pattern.replace("*", str(i))) for i in range(count)]

    def keys(self) -> list[str]:
        """List all stored weight names."""
        return list(self._weights.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._weights

    def __len__(self) -> int:
        return len(self._weights)


class Schema:
    """
    Schema definition for weight loading.

    Use method chaining to define weights:
        schema = Schema("model_name").define("name", Source(...))
    """

    def __init__(self, name: str):
        self.name = name
        self._definitions: list[Definition] = []

    def define(self, name: str, source: Source) -> Schema:
        """
        Define a logical weight with its source.

        Args:
            name: Logical name (may contain '*' for per-layer weights)
            source: Source configuration

        Returns:
            self for method chaining
        """
        self._definitions.append(Definition(name=name, source=source))
        return self

    def load(
        self,
        reader: ReaderFn,
        config: "RuntimeConfig",
        num_layers: int = 0,
        log_queue: object | None = None,
    ) -> WeightStore:
        """
        Load all weights according to the schema.

        Args:
            reader: Function to read tensors by name
            config: Runtime configuration (device, sharding, quantization)
            num_layers: Number of layers (for expanding '*' patterns)
            log_queue: Optional queue to send progress updates to CLI

        Returns:
            WeightStore with all loaded weights
        """
        store = WeightStore()

        # Calculate total number of weight operations for progress tracking
        total_ops = 0
        for defn in self._definitions:
            if defn.has_layer_pattern():
                total_ops += num_layers
            else:
                total_ops += 1

        # Only show progress on rank 0
        show_progress = config.rank == 0

        # If log_queue is provided, send progress to CLI instead of local tqdm
        use_queue_progress = log_queue is not None and show_progress

        def send_progress(current: int, total: int, desc: str):
            """Send progress update to CLI via log queue."""
            if use_queue_progress:
                log_queue.put(
                    {
                        "level": "PROGRESS",
                        "current": current,
                        "total": total,
                        "description": desc,
                    }
                )

        # Send initial progress
        if use_queue_progress:
            send_progress(0, total_ops, "Starting weight loading...")

        # Use local tqdm only if not using queue progress
        with tqdm(
            total=total_ops,
            desc="\033[1;36m Loading weights\033[0m",
            unit="tensors",
            disable=use_queue_progress or not show_progress,
            bar_format="{desc} │{bar:40}│ {percentage:3.0f}% • {n_fmt}/{total_fmt} • {rate_fmt} • ETA: {remaining}",
            colour="cyan",
            dynamic_ncols=True,
        ) as pbar:
            current_op = 0
            for defn in self._definitions:
                if defn.has_layer_pattern():
                    # Expand for each layer
                    for layer_idx in range(num_layers):
                        logical_name = defn.expand_for_layer(layer_idx)
                        physical_names = defn.expand_source_for_layer(layer_idx)
                        desc = f"layer {layer_idx + 1}/{num_layers}"
                        pbar.set_postfix_str(desc, refresh=False)
                        tensor = self._load_single(
                            reader, config, defn.source, physical_names
                        )
                        store.put(logical_name, tensor)
                        current_op += 1
                        pbar.update(1)
                        send_progress(current_op, total_ops, desc)
                else:
                    # Single tensor (not per-layer)
                    desc = defn.name
                    pbar.set_postfix_str(desc, refresh=False)
                    tensor = self._load_single(
                        reader, config, defn.source, defn.source.patterns
                    )
                    store.put(defn.name, tensor)
                    current_op += 1
                    pbar.update(1)
                    send_progress(current_op, total_ops, desc)

        # Send completion
        if use_queue_progress:
            log_queue.put(
                {"level": "PROGRESS_DONE", "message": "Weight loading complete"}
            )

        return store

    def _load_single(
        self,
        reader: ReaderFn,
        config: "RuntimeConfig",
        source: Source,
        physical_names: list[str],
    ) -> torch.Tensor:
        # print("Loading", physical_names)
        """Load a single (possibly fused/gathered) tensor with transforms applied."""
        # Read tensors
        tensors = [reader(name) for name in physical_names]

        # Apply custom transform if specified
        if source.has_transform:
            # Build kwargs for transform function
            transform_kwargs = dict(source._transform_kwargs or {})
            transform_kwargs["device"] = str(config.device)
            # Inject distributed info for transforms that need custom sharding (like MoE)
            transform_kwargs["rank"] = config.rank % config.tensor_parallel_size
            transform_kwargs["world_size"] = config.tensor_parallel_size

            # Call the transform function
            result = source._transform_fn(tensors, transform_kwargs)  # type: ignore[misc]

            # Handle dict result (select output by type)
            if isinstance(result, dict):
                output_type = transform_kwargs.get("output_type")
                if output_type is None:
                    raise ValueError(
                        "Transform returned dict but no 'output_type' specified"
                    )
                tensor = result[output_type]
            else:
                tensor = result
        # Apply interleaved sharding BEFORE fusion (if requested)
        # This is CRITICAL for fused QKV weights where naive chunking breaks head alignment
        elif (
            config.tensor_parallel_size > 1 and source.sharding == "interleaved_column"
        ):
            # Shard each source source tensor individually along dim=0 (Column Parallel)
            sharded_tensors = []
            # print(f"DEBUG: Interleaved sharding for {physical_names}", flush=True)
            for i, t in enumerate(tensors):
                if t.shape[0] % config.tensor_parallel_size != 0:
                    raise ValueError(
                        f"Cannot interleaved-shard tensor {t.shape}: dim 0 not divisible by tp_size={config.tensor_parallel_size}"
                    )
                # Ensure we have a clean copy in memory (avoid ztensor/mmap issues with views)
                # chunk = torch.chunk(t, config.tensor_parallel_size, dim=0)[
                #    config.rank % config.tensor_parallel_size
                # ]
                # Actually, simply remove t.clone() and use t.
                # Use t directly (even if it's mmapped, slicing is fine)
                chunk = torch.chunk(t, config.tensor_parallel_size, dim=0)[
                    config.rank % config.tensor_parallel_size
                ]
                # Check chunk validity
                # print(f"  Shard {i}: {t.shape} -> {chunk.shape}", flush=True)
                sharded_tensors.append(chunk)
            tensors = sharded_tensors

            # After interleaved sharding, we might still need to fuse or it might be single
            if source.is_fused:
                tensor = torch.cat(tensors, dim=source.fuse_dim)
            else:
                tensor = tensors[0]

        # Fuse if needed (concatenation)
        elif source.is_fused:
            tensor = torch.cat(tensors, dim=source.fuse_dim)
        # Gathered but no transform - just take first tensor (unusual case)
        elif source.is_gathered:
            raise ValueError(
                "Source.gather() should be used with .transform(). "
                "For single tensor, use Source(pattern) instead."
            )
        else:
            tensor = tensors[0]

        # Apply dtype conversion
        if source._dtype is not None:
            tensor = tensor.to(source._dtype)

        # Apply sharding (only if tp_size > 1)
        # interleaved_column is handled above, so we skip it here
        if (
            config.tensor_parallel_size > 1
            and source.sharding is not None
            and source.sharding != "interleaved_column"
        ):
            dim = 1 if source.sharding == "row" else 0

            # Validate dimension is divisible by tp_size
            if tensor.shape[dim] % config.tensor_parallel_size != 0:
                raise ValueError(
                    f"Cannot shard tensor of shape {tuple(tensor.shape)} along dim={dim}: "
                    f"dimension size {tensor.shape[dim]} is not divisible by tp_size={config.tensor_parallel_size}"
                )

            tensor = torch.chunk(
                tensor.contiguous(), config.tensor_parallel_size, dim=dim
            )[config.rank % config.tensor_parallel_size]

        # Determine final dtype and device, then apply in single fused call
        # This avoids multiple intermediate tensor allocations
        final_dtype = None
        
        if source.should_quantize and config.quantization is not None:
            # Quantization is applied separately (returns different tensor structure)
            tensor = quantize(tensor, config.quantization)
            # Quantized tensors go directly to device without dtype change
            return tensor.to(config.device)
        elif source.target_dtype is not None:
            # Explicit dtype override from source definition
            final_dtype = source.target_dtype
        elif tensor.dtype not in (torch.uint8, torch.float8_e4m3fn):
            # Apply dtype casting for float weight types
            final_dtype = config.compute_dtype
        
        # Fused device + dtype transfer
        if final_dtype is not None:
            tensor = tensor.to(device=config.device, dtype=final_dtype)
        else:
            tensor = tensor.to(config.device)

        return tensor


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
