"""
Base configuration module containing common classes and utilities for model configurations.
This module provides shared functionality for different model architectures.
"""

import os
import tomllib
import base64
from dataclasses import dataclass
from typing import Any, Optional
import torch


@dataclass
class Tokenizer:
    """Struct for tokenizer configuration."""

    type: str
    num_vocab: int
    merge_table: dict[int, bytes]
    split_regex: str
    special_tokens: dict[str, int]
    escape_non_printable: bool


@dataclass
class CommonArch:
    """Base architecture configuration with common fields."""

    # Fields that will be set by the model config file
    type: str
    num_layers: int
    num_query_heads: int
    num_key_value_heads: int
    head_size: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    use_qkv_bias: bool
    rms_norm_eps: float

    # Fields that will be set by the backend config file
    device: str
    dtype: torch.dtype


@dataclass
class ModelInfo:
    """Top-level struct to hold the entire parsed model information."""

    name: str
    description: str
    version: str
    parameters: list[str]
    architecture: CommonArch
    tokenizer: Tokenizer
    template_type: str
    template_content: str
    stop_tokens: list[str]

    @staticmethod
    def load_from_file(
        cfg_file_path: str, device: str, dtype: torch.dtype
    ) -> "ModelInfo":
        """
        Parses a dictionary (from loaded TOML data) into the ModelMetadata struct,
        with improved and informative error handling.
        """

        # Load and parse TOML file
        cfg = ModelConfig.load_from_file(cfg_file_path)

        # Set the device and dtype fields we got from the backend config file
        # so that they will be populated in the architecture object
        arch_dict = cfg.get_required_key(cfg.root, "architecture")
        arch_dict["device"] = device
        arch_dict["dtype"] = dtype

        # Get the architecture object
        match arch_dict["type"]:
            # Disable pylint import-outside-toplevel check for this block
            # because we need to do lazy import for the architecture classes
            # to avoid circular imports
            case "l4ma":
                from .l4ma import L4maArch  # pylint: disable=import-outside-toplevel

                arch = L4maArch.from_config(cfg)
            case "qwen2":
                from .qwen2 import (  # pylint: disable=import-outside-toplevel
                    Qwen2Arch,
                )

                arch = Qwen2Arch.from_config(cfg)
            case "qwen3":
                from .qwen3 import (  # pylint: disable=import-outside-toplevel
                    Qwen3Arch,
                )

                arch = Qwen3Arch.from_config(cfg)
            case "gptoss":
                from .gptoss import (  # pylint: disable=import-outside-toplevel
                    GptOssArch,
                )

                arch = GptOssArch.from_config(cfg)
            case _:
                raise ValueError(f"Unsupported architecture type: {arch_dict['type']}")

        # Get other common information
        tokenizer = cfg.get_tokenizer()
        metadata_dict = cfg.get_metadata_fields()
        template_dict = cfg.get_chat_template_dict()

        # Construct final metadata object
        return ModelInfo(
            architecture=arch,
            tokenizer=tokenizer,
            **metadata_dict,
            **template_dict,
        )


class ModelConfig:
    """
    Class to hold the path and data of a model configuration file.
    Data stored in the object has been parsed into a tree of Python dictionaries,
    but the required fields have not been validated. This is done in the ModelInfo class.
    """

    path: str
    root: dict

    def __init__(self, path: str, root: dict):
        self.path = path
        self.root = root

    @staticmethod
    def load_from_file(cfg_file_path: str) -> "ModelConfig":
        """Load a TOML file and return a ModelConfig object."""
        try:
            with open(cfg_file_path, "rb") as f:
                root = tomllib.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Configuration file not found at path: {cfg_file_path}"
            ) from None
        except tomllib.TOMLDecodeError as e:
            raise tomllib.TOMLDecodeError(
                f"Error parsing TOML file at {cfg_file_path}: {e}"
            ) from e

        return ModelConfig(cfg_file_path, root)

    def get_required_key(self, node: dict, key: str) -> Any:
        """Gets a key from the data dictionary, raising a contextual KeyError if it's missing."""
        try:
            return node[key]
        except KeyError:
            node_name = self._get_node_name(node)

            raise KeyError(
                f"Missing required key '{key}' in the '[{node_name}]' section "
                f"of the TOML file: {self.path}"
            ) from None

    def get_common_arch_dict(self) -> dict:
        """Get common architecture fields from the configuration file."""

        arch_dict = self.get_required_key(self.root, "architecture")

        def get_required_arch_key(key: str):
            return self.get_required_key(arch_dict, key)

        return {
            "type": get_required_arch_key("type"),
            "num_layers": get_required_arch_key("num_layers"),
            "num_query_heads": get_required_arch_key("num_query_heads"),
            "num_key_value_heads": get_required_arch_key("num_key_value_heads"),
            "head_size": get_required_arch_key("head_size"),
            "hidden_size": get_required_arch_key("hidden_size"),
            "intermediate_size": get_required_arch_key("intermediate_size"),
            "vocab_size": get_required_arch_key("vocab_size"),
            "use_qkv_bias": get_required_arch_key("use_qkv_bias"),
            "rms_norm_eps": get_required_arch_key("rms_norm_eps"),
            "device": get_required_arch_key("device"),
            "dtype": get_required_arch_key("dtype"),
        }

    def get_chat_template_dict(self) -> dict:
        """Get the template section of the configuration."""
        template_dict = self.get_required_key(self.root, "template")

        def get_required_template_key(key: str):
            return self.get_required_key(template_dict, key)

        return {
            "template_type": get_required_template_key("type"),
            "template_content": get_required_template_key("content"),
            "stop_tokens": get_required_template_key("stop_tokens"),
        }

    def get_metadata_fields(self) -> dict:
        """Get common metadata fields."""

        def get_required_top_level_key(key: str):
            return self.get_required_key(self.root, key)

        return {
            "name": get_required_top_level_key("name"),
            "description": get_required_top_level_key("description"),
            "version": get_required_top_level_key("version"),
            "parameters": get_required_top_level_key("parameters"),
        }

    def get_tokenizer(self) -> Tokenizer:
        """Get the tokenizer from the configuration."""
        tokenizer_data = self.get_required_key(self.root, "tokenizer")
        architecture_data = self.get_required_key(self.root, "architecture")

        def get_required_tokenizer_key(key: str):
            return self.get_required_key(tokenizer_data, key)

        def get_required_architecture_key(key: str):
            return self.get_required_key(architecture_data, key)

        vocab_file_name = get_required_tokenizer_key("vocabulary_file")
        model_name = self.get_required_key(self.root, "name")

        metadata_dir = os.path.dirname(os.path.abspath(self.path))
        vocab_file_path = os.path.join(metadata_dir, model_name, vocab_file_name)

        try:
            merge_rules = ModelConfig._load_merge_rules(vocab_file_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load vocabulary file '{vocab_file_name}' "
                f"at path '{vocab_file_path}'"
            ) from e

        special_tokens = get_required_tokenizer_key("special_tokens")

        return Tokenizer(
            type=get_required_tokenizer_key("type"),
            num_vocab=get_required_architecture_key("vocab_size"),
            merge_table=merge_rules,
            split_regex=get_required_tokenizer_key("split_regex"),
            special_tokens=special_tokens,
            escape_non_printable=get_required_tokenizer_key("escape_non_printable"),
        )

    def _get_node_name(self, node: dict) -> str:
        node_name = ModelConfig._recur_get_node_name(node, self.root, "top-level")
        return node_name if node_name is not None else "unknown"

    @staticmethod
    def _recur_get_node_name(node: dict, cur: dict, cur_name: str) -> Optional[str]:
        if cur is node:
            return cur_name
        for key, value in cur.items():
            if isinstance(key, str) and isinstance(value, dict):
                result = ModelConfig._recur_get_node_name(node, value, key)
                if result is not None:
                    return result
        return None

    @staticmethod
    def _load_merge_rules(vocab_file_path: str) -> dict[int, bytes]:
        """
        Loads merge rules from a file.

        The file is expected to contain lines where each line has a
        base64-encoded token and an integer rank, separated by whitespace.
        Empty lines are skipped.

        Args:
            path: The path to the file to read.

        Returns:
            A dictionary mapping each rank (int) to its corresponding
            decoded token (bytes).

        Raises:
            FileNotFoundError: If the file at the specified path does not exist.
            ValueError: If a line in the file is malformed (e.g., incorrect
                        number of parts, invalid base64, or non-integer rank).
        """
        merge_rules: dict[int, bytes] = {}
        with open(vocab_file_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                # Skip empty or blank lines
                if not line:
                    continue

                # Expect two parts: base64-encoded token and rank
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(
                        f"Error on line {line_number}: expected 2 parts, "
                        f"but found {len(parts)} (line: '{line}')"
                    )

                b64_token, rank_str = parts

                # 1. Decode base64 token
                try:
                    decoded_token = base64.b64decode(b64_token)
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Error on line {line_number}: failed to decode base64 token."
                    ) from e

                # 2. Parse rank into an integer
                try:
                    rank = int(rank_str)
                except ValueError as e:
                    raise ValueError(
                        f"Error on line {line_number}: failed to parse "
                        f"rank '{rank_str}' as an integer."
                    ) from e

                # Insert into the dictionary
                merge_rules[rank] = decoded_token

        return merge_rules
