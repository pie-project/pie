"""
Qwen2 model configuration module.
"""

from dataclasses import dataclass

from .common import (
    CommonArch,
    ModelConfig,
)


@dataclass
class Qwen2Arch(CommonArch):
    """Qwen2 specific architecture configuration."""

    rope_theta: float

    @staticmethod
    def from_config(cfg: ModelConfig) -> "Qwen2Arch":
        """Parse Qwen2-specific architecture configuration."""
        # Get common architecture fields
        common_arch_dict = cfg.get_common_arch_dict()

        # Get all the fields for the architecture section to grab other
        # architecture-specific fields
        arch_dict = cfg.get_required_key(cfg.root, "architecture")

        # Get RoPE configuration
        rope_dict = cfg.get_required_key(arch_dict, "rope")
        rope_theta = cfg.get_required_key(rope_dict, "theta")

        return Qwen2Arch(
            # Common fields
            **common_arch_dict,
            # Qwen2-specific fields
            rope_theta=rope_theta,
        )
