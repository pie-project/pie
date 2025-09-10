"""
GPT OSS model configuration module.
"""

from dataclasses import dataclass

from .common import (
    CommonArch,
    ModelConfig,
)


@dataclass
class GptOssArch(CommonArch):
    """GPT OSS specific architecture configuration."""

    # MoE configuration
    num_experts: int
    experts_per_token: int

    # RoPE configuration
    rope_theta: float
    rope_scaling_factor: float
    rope_ntk_alpha: float
    rope_ntk_beta: float

    # Model specific parameters
    initial_context_length: int
    sliding_window: int
    swiglu_limit: float

    @staticmethod
    def from_config(cfg: ModelConfig) -> "GptOssArch":
        """Parse GPT OSS-specific architecture configuration."""
        # Get common architecture fields
        common_arch_dict = cfg.get_common_arch_dict()

        # Get all the fields for the architecture section to grab other
        # architecture-specific fields
        arch_dict = cfg.get_required_key(cfg.root, "architecture")

        # Get MoE configuration
        moe_dict = cfg.get_required_key(arch_dict, "moe")
        num_experts = cfg.get_required_key(moe_dict, "num_experts")
        experts_per_token = cfg.get_required_key(moe_dict, "experts_per_token")

        # Get RoPE configuration (GPT OSS uses YaRN-style RoPE)
        rope_dict = cfg.get_required_key(arch_dict, "rope")
        rope_theta = cfg.get_required_key(rope_dict, "theta")
        rope_scaling_factor = cfg.get_required_key(rope_dict, "scaling_factor")
        rope_ntk_alpha = cfg.get_required_key(rope_dict, "ntk_alpha")
        rope_ntk_beta = cfg.get_required_key(rope_dict, "ntk_beta")

        # Get model specific parameters
        initial_context_length = cfg.get_required_key(
            arch_dict, "initial_context_length"
        )
        sliding_window = cfg.get_required_key(arch_dict, "sliding_window")
        swiglu_limit = cfg.get_required_key(arch_dict, "swiglu_limit")

        return GptOssArch(
            # Common fields
            **common_arch_dict,
            # GPT OSS-specific fields
            num_experts=num_experts,
            experts_per_token=experts_per_token,
            rope_theta=rope_theta,
            rope_scaling_factor=rope_scaling_factor,
            rope_ntk_alpha=rope_ntk_alpha,
            rope_ntk_beta=rope_ntk_beta,
            initial_context_length=initial_context_length,
            sliding_window=sliding_window,
            swiglu_limit=swiglu_limit,
        )
