#!/usr/bin/env python3
"""Writes corpus entries for the normalization branches no real config covers.

Most of `parse_hf_config` is exercised by the configs models actually ship, and
those are in `corpus/` verbatim -- they are the ground truth and no synthetic
config can replace them. But a local model cache only holds what someone
downloaded, and the branches it misses are not the easy ones: MLA `head_dim`
synthesis, the kimi_k3-vs-kimi_linear disambiguation, nemotron_h's
`hybrid_override_pattern`, CSM's nested depth-decoder and codec configs, the
YaRN mscale derivation. Those had no fixture anywhere in the repo before this
file.

Each entry below names the branch it exists to reach. They are minimal by
intent: enough to make the branch fire and to make its output distinguishable
from the default, not a faithful copy of any shipped model.

    python3 driver/cuda/tests/hf_config_dump/synthesize.py
"""
import json
import pathlib

OUT = pathlib.Path(__file__).with_name("corpus")

# The five fields `parse_hf_config` requires of every config.
BASE = {
    "hidden_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "vocab_size": 1000,
    "max_position_embeddings": 2048,
}


def cfg(model_type, arch, **rest):
    return {**BASE, "model_type": model_type, "architectures": [arch], **rest}


MLA = {
    "q_lora_rank": 64,
    "kv_lora_rank": 32,
    # head_dim is synthesized as qk_nope + qk_rope for these archs, and the
    # explicit head_dim below must lose to it -- that is the whole branch.
    "qk_nope_head_dim": 48,
    "qk_rope_head_dim": 16,
    "v_head_dim": 64,
    "head_dim": 16,
    "first_k_dense_replace": 1,
    "n_shared_experts": 2,
    "moe_intermediate_size": 96,
    "n_routed_experts": 16,
    "num_experts_per_tok": 4,
}

MAMBA = {
    "mamba_num_heads": 8,
    "mamba_head_dim": 16,
    "ssm_state_size": 32,
    "n_groups": 2,
    "conv_kernel": 4,
    "chunk_size": 64,
    "time_step_min": 0.001,
}

ENTRIES = {
    # -- MLA head_dim synthesis (qk_nope + qk_rope), per model_type ----------
    "synthetic--deepseek-v2": cfg("deepseek_v2", "DeepseekV2ForCausalLM", **MLA),
    "synthetic--deepseek-v3": cfg("deepseek_v3", "DeepseekV3ForCausalLM", **MLA),
    "synthetic--kimi-k2": cfg("kimi_k2", "KimiK2ForCausalLM", **MLA),
    # An MLA-shaped config whose model_type is *not* on the synthesis list:
    # head_dim stays as given. Pins the gate rather than the arithmetic.
    "synthetic--mla-shape-unlisted-arch": cfg("llama", "LlamaForCausalLM", **MLA),

    # -- glm_moe_dsa: MLA plus indexer_types synthesis from index_topk_freq --
    "synthetic--glm-moe-dsa": cfg(
        "glm_moe_dsa", "GlmMoeDsaForCausalLM",
        index_topk=8, index_head_dim=32, index_n_heads=4, index_topk_freq=2,
        **MLA,
    ),

    # -- kimi_k3: the outer model_type wins over the inner `kimi_linear`, ----
    # which is a different architecture that ships standalone.
    "synthetic--kimi-k3": cfg(
        "kimi_k3", "KimiK3ForCausalLM",
        text_config={
            **BASE, "model_type": "kimi_linear", **MLA,
            "linear_attn_config": {"kda_layers": [2, 4]},
            "moe_intermediate_size": 96,
        },
    ),
    # The same inner type standing alone stays kimi_linear.
    "synthetic--kimi-linear-standalone": cfg("kimi_linear", "KimiLinearForCausalLM", **MLA),
    # kimi_k25: a multimodal wrapper detected from the outer name plus a
    # text_config, with no vision_config to go on.
    "synthetic--kimi-k25": cfg(
        "kimi_k25", "KimiK25ForConditionalGeneration",
        text_config={**BASE, "model_type": "deepseek_v3", **MLA},
    ),

    # -- deepseek_v4: its own block, and attention_has_sinks forced on ------
    "synthetic--deepseek-v4": cfg(
        "deepseek_v4", "DeepseekV4ForCausalLM",
        o_lora_rank=32, o_groups=2, index_head_dim=32, index_n_heads=4,
        index_topk=8, hc_mult=2, num_hash_layers=2, hc_eps=1e-5,
        sliding_window=256, compress_rope_theta=10000.0,
        compress_ratios=[1, 2, 4], scoring_func="sigmoid", expert_dtype="fp8",
        **MLA,
    ),

    # -- nemotron_h: hybrid_override_pattern -> layer_types, mamba dims, ----
    # and its own multimodal skip-prefix list.
    "synthetic--nemotron-h": cfg(
        "nemotron_h", "NemotronHForCausalLM",
        hybrid_override_pattern="M*E-",
        **MAMBA,
    ),

    # -- gemma2: layer_types from a hardcoded i%2 pattern, tie default on ---
    "synthetic--gemma2": cfg(
        "gemma2", "Gemma2ForCausalLM",
        sliding_window=256, query_pre_attn_scalar=144.0,
        final_logit_softcapping=30.0, attn_logit_softcapping=50.0,
    ),
    # -- gemma3: layer_types from sliding_window_pattern -------------------
    "synthetic--gemma3": cfg(
        "gemma3_text", "Gemma3ForCausalLM",
        sliding_window=256, sliding_window_pattern=3, rope_local_base_freq=10000.0,
    ),

    # -- gemma4: the two keys only the Metal driver reads. Carried by the
    # descriptor because `pie.model/1` is the artifact's model config, not one
    # driver's -- and with values, so the differential says something about
    # them rather than comparing two defaults.
    "synthetic--gemma4-metal-keys": cfg(
        "gemma4_text", "Gemma4ForCausalLM",
        global_head_dim=256, use_double_wide_mlp=True,
        num_global_key_value_heads=2, sliding_window=512,
        num_kv_shared_layers=1, hidden_size_per_layer_input=64,
        final_logit_softcapping=30.0,
        layer_types=["sliding_attention", "sliding_attention",
                     "sliding_attention", "full_attention"],
        rope_parameters={
            "full_attention": {"rope_theta": 1000000.0,
                               "partial_rotary_factor": 0.25},
            "sliding_attention": {"rope_theta": 10000.0},
        },
    ),

    # -- CSM: depth decoder and Mimi codec as nested configs ---------------
    "synthetic--csm": cfg(
        "csm", "CsmForConditionalGeneration",
        text_vocab_size=1000, audio_vocab_size=2048, num_codebooks=8,
        codebook_eos_token_id=0, audio_eos_token_id=1, audio_token_id=2,
        depth_decoder_config={
            "hidden_size": 64, "backbone_hidden_size": 128, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "intermediate_size": 128, "num_codebooks": 8, "vocab_size": 2048,
            "max_position_embeddings": 64, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
        },
        codec_config={
            "hidden_size": 64, "codebook_dim": 32, "codebook_size": 1024,
            "num_quantizers": 8, "num_semantic_quantizers": 1, "num_filters": 16,
            "upsampling_ratios": [8, 6, 5, 4], "sampling_rate": 24000,
        },
    ),

    # -- RoPE variants -----------------------------------------------------
    "synthetic--rope-llama3": cfg(
        "llama", "LlamaForCausalLM",
        rope_scaling={
            "rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
            "high_freq_factor": 4.0, "original_max_position_embeddings": 512,
        },
    ),
    # YaRN, whose attention_factor and MLA softmax mscale are *derived* from
    # mscale/mscale_all_dim rather than read.
    "synthetic--rope-yarn": cfg(
        "deepseek_v3", "DeepseekV3ForCausalLM",
        rope_scaling={
            "rope_type": "yarn", "factor": 4.0, "beta_fast": 32.0, "beta_slow": 1.0,
            "mscale": 0.707, "mscale_all_dim": 0.707,
            "original_max_position_embeddings": 512,
        },
        **MLA,
    ),
    # `rope_parameters` flat, the newer spelling of the same thing.
    "synthetic--rope-parameters-flat": cfg(
        "llama", "LlamaForCausalLM",
        rope_parameters={"rope_type": "llama3", "factor": 2.0,
                         "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                         "original_max_position_embeddings": 512},
    ),

    # -- Defaulting rules, each isolated ------------------------------------
    # No num_key_value_heads (<- num_attention_heads) and no head_dim
    # (<- hidden_size / num_attention_heads).
    "synthetic--defaults-bare": cfg("llama", "LlamaForCausalLM"),
    # eps under its two alternate spellings.
    "synthetic--eps-layer-norm-epsilon": cfg(
        "llama", "LlamaForCausalLM", layer_norm_epsilon=1e-6),
    "synthetic--eps-norm-eps": cfg("llama", "LlamaForCausalLM", norm_eps=1e-4),
    # qwen2: attention_bias defaults true here and nowhere else.
    "synthetic--qwen2": cfg("qwen2", "Qwen2ForCausalLM"),
    # An explicit use_qk_norm beats the model_type inference.
    "synthetic--qk-norm-explicit-off": cfg("qwen3", "Qwen3ForCausalLM", use_qk_norm=False),
    "synthetic--qk-norm-explicit-on": cfg("llama", "LlamaForCausalLM", use_qk_norm=True),
    # The three MoE expert-count spellings.
    "synthetic--moe-num-local-experts": cfg(
        "mixtral", "MixtralForCausalLM", num_local_experts=8, num_experts_per_tok=2),
    "synthetic--moe-n-routed-experts": cfg(
        "qwen3_moe", "Qwen3MoeForCausalLM", n_routed_experts=8,
        num_experts_per_tok=2, moe_intermediate_size=96),
    # Explicit null softcapping is not the same as absent.
    "synthetic--gemma-null-softcap": cfg(
        "gemma2", "Gemma2ForCausalLM", sliding_window=256,
        final_logit_softcapping=None, attn_logit_softcapping=None),
    # A quantized config, both spellings of the zero point.
    "synthetic--quantized-fp8": cfg(
        "llama", "LlamaForCausalLM",
        quantization_config={"quant_method": "fp8", "bits": 8, "group_size": 128,
                             "sym": True, "desc_act": False}),
    # An unrecognized model_type: the parser still produces a config, and the
    # arch table is what refuses later.
    "synthetic--unknown-model-type": cfg("wobbegong", "WobbegongForCausalLM"),
}


def main():
    OUT.mkdir(exist_ok=True)
    for name, body in sorted(ENTRIES.items()):
        (OUT / f"{name}.json").write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(f"{len(ENTRIES)} synthetic configs -> {OUT}")


if __name__ == "__main__":
    main()
