"""Make vLLM recognise Kimi-K3.

vLLM main ships the whole K3 implementation under ``vllm/models/kimi_k3/``
(plus its config, processor, tokenizer and CUDA kernels) but the *registry*
rows that point at it -- the architecture and the ``kimi_k3`` config class --
are not in the tree at the commit we build from. Registering them from the
outside is what vLLM's out-of-tree model API is for, and it keeps our checkout
of vLLM unpatched, which matters because the point of this harness is to
compare pie against stock vLLM.

Two architectures are registered:

``KimiK3ForConditionalGeneration``
    The shipped multimodal wrapper, read with vLLM's own ``KimiK3Config``.

``KimiK3ForCausalLM``
    The text-only tower, for a checkpoint whose ``text_config`` has been
    flattened to the top level (``shrink_checkpoint.py --text-only``). Its
    config *is* a ``KimiLinearConfig``, which is why ``kimi_k3`` maps to a
    different class in each case -- the flattened file has no ``text_config``
    for ``KimiK3Config`` to find, and ``KimiK3Config`` would silently fall
    back to a default 4096-wide text tower rather than fail.

    Text-only is not a shortcut around the vision tower: vLLM main's K3 builds
    its ViT out of ``kimi_k25_vit``, which derives the 2-D rope width from
    ``vt_hidden_size // vt_num_attention_heads`` (1024 // 12 = 85) and asserts
    it is a multiple of four. K3's ViT carries its head width in
    ``qkv_hidden_size`` instead, so the vision half of this snapshot cannot
    construct at all. Text parity does not need it.

Import and call ``register()`` before constructing an ``LLM``; it is
idempotent. Installed as a ``vllm.general_plugins`` entry point so it also
runs inside the TP worker processes, where the model is actually built.
"""

from __future__ import annotations

import os


def _text_only_enabled() -> bool:
    return os.environ.get("PIE_K3_MULTIMODAL", "") in ("", "0")


def register() -> None:
    from transformers import AutoConfig

    from vllm import ModelRegistry
    from vllm.transformers_utils.config import _CONFIG_REGISTRY
    from vllm.transformers_utils.configs.kimi_k3 import (
        KimiK3Config,
        KimiK3VisionConfig,
    )
    from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

    # A flattened text-only K3 has `model_type: kimi_k3` at the top level with
    # the text fields beside it, so the class that reads it is the *text*
    # config. The multimodal file keeps its nested `text_config` and wants
    # `KimiK3Config`.
    text_only = _text_only_enabled()
    root_cls = KimiLinearConfig if text_only else KimiK3Config

    for model_type, cls in (
        ("kimi_k3", root_cls),
        ("kimi_k3_vision", KimiK3VisionConfig),
    ):
        cls.model_type = model_type
        AutoConfig.register(model_type, cls, exist_ok=True)
        _CONFIG_REGISTRY[model_type] = cls
    # `KimiLinearConfig.model_type` is class state, and leaving it reassigned
    # above would strand the stock `kimi_linear` row.
    if text_only:
        KimiLinearConfig.model_type = "kimi_linear"

    supported = ModelRegistry.get_supported_archs()
    for arch, target in (
        ("KimiK3ForConditionalGeneration",
         "vllm.models.kimi_k3:KimiK3ForConditionalGeneration"),
        ("KimiK3ForCausalLM", "vllm.models.kimi_k3:KimiLinearForCausalLM"),
    ):
        if arch not in supported:
            ModelRegistry.register_model(arch, target)

    _drop_unknown_mla_spec_fields()
    _backfill_missing_env_knobs()
    _teach_mla_about_kimi_k3()
    _default_mla_prefill_supports_out()
    _run_kimi_k3_mla_post_load()


def _run_kimi_k3_mla_post_load() -> None:
    """Call K3's MLA post-load hook, which the loader's isinstance test misses.

    ``MultiHeadLatentAttention.process_weights_after_loading`` absorbs
    ``kv_b_proj`` into the decode-time ``W_UK_T`` / ``W_UV`` bmm weights, and
    the loader only invokes that hook for ``(Attention, MLAAttention,
    MMEncoderAttention)``. K3's MLA derives from ``AttentionLayerBase``
    instead, so on this tree the absorption never happens and the *decode*
    step -- not prefill, which does not use the absorbed form -- dies on a
    missing ``W_UK_T``.
    """
    from vllm.model_executor.model_loader import base_loader

    original = base_loader.process_weights_after_loading
    if getattr(original, "_pie_k3_mla_post_load", False):
        return

    def wrapper(model, model_config, target_device):
        original(model, model_config, target_device)
        from vllm.models.kimi_k3.nvidia.mla import MultiHeadLatentAttention

        for module in model.modules():
            if isinstance(module, MultiHeadLatentAttention) and not hasattr(
                module, "W_UK_T"
            ):
                module.process_weights_after_loading(model_config.dtype)

    wrapper._pie_k3_mla_post_load = True
    base_loader.process_weights_after_loading = wrapper


def _default_mla_prefill_supports_out() -> None:
    """Give the MLA prefill backends the `supports_out()` K3 asks them for.

    ``MultiHeadLatentAttention._forward_prefill`` skips a slice-flatten-copy
    when the prefill backend can write straight into ``out``; the predicate it
    calls does not exist on ``MLAPrefillBackend`` at this commit. Answering
    ``False`` takes the copy path, which the same function already implements
    (`elif not writes_out: out.copy_(...)`) and which produces the same
    numbers -- the right default for a backend that has not declared itself.
    """
    from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

    if hasattr(MLAPrefillBackend, "supports_out"):
        return
    MLAPrefillBackend.supports_out = lambda self: False


def _teach_mla_about_kimi_k3() -> None:
    """Let `model_type: kimi_k3` be recognised as an MLA model.

    ``ModelArchConfigConvertorBase.is_deepseek_mla`` matches the text config's
    ``model_type`` against a fixed list that has ``kimi_k2`` and
    ``kimi_linear`` but not ``kimi_k3`` -- the same missing-row state as the
    architecture registry. It only matters for a *flattened* text-only
    checkpoint, where ``kimi_k3`` is the text config's own model_type instead
    of the wrapper's.

    Getting this wrong is not a load failure, which is the point of fixing it
    here: ``get_head_size`` silently falls through to
    ``hidden_size // num_attention_heads`` = 7168 // 96 = 74, and MLA only
    rejects it later because 74 happens not to be a supported head size.
    """
    from vllm.transformers_utils.model_arch_config_convertor import (
        ModelArchConfigConvertorBase,
    )

    original = ModelArchConfigConvertorBase.is_deepseek_mla
    if getattr(original, "_pie_k3_mla", False):
        return

    def is_deepseek_mla(self) -> bool:
        model_type = getattr(self.hf_text_config, "model_type", None)
        if model_type == "kimi_k3":
            return getattr(self.hf_text_config, "kv_lora_rank", None) is not None
        return original(self)

    is_deepseek_mla._pie_k3_mla = True
    ModelArchConfigConvertorBase.is_deepseek_mla = is_deepseek_mla


def _backfill_missing_env_knobs() -> None:
    """Define the K3 env knobs this tree's ``vllm.envs`` does not declare.

    ``KimiMoE._maybe_overlap_router_and_down_proj`` reads
    ``VLLM_ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD`` to decide whether to run
    the router and the latent down-projection on separate streams. It is a
    scheduling choice with no effect on the numbers, and it is the last of the
    K3 knobs missing from ``envs.py`` on this commit. Zero means "never use the
    side stream", which is also the most deterministic answer for a parity run.

    Setting the attribute on the module shadows ``envs.__getattr__``, which is
    what raises for names it does not know.
    """
    from vllm import envs

    defaults = {"VLLM_ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD": 0}
    for name, value in defaults.items():
        try:
            getattr(envs, name)
        except AttributeError:
            setattr(envs, name, value)


def _drop_unknown_mla_spec_fields() -> None:
    """Let K3's MLA build a KV-cache spec on a tree where its field is missing.

    ``MultiHeadLatentAttention.get_kv_cache_spec`` passes
    ``non_causal_multi_token_decode`` to ``MLAAttentionSpec``, which does not
    declare it at this commit -- the same half-landed state as the missing
    registry rows, and vLLM's own source says so ("TODO: Remove this mypy
    workaround once the K3 PR is fully merged"). Nothing on this tree *reads*
    the field, so filtering it out is behaviour-neutral, and doing it here
    rather than by editing vLLM keeps the reference engine stock.
    """
    import dataclasses
    import functools

    from vllm.models.kimi_k3.nvidia.mla import MultiHeadLatentAttention
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    known = {f.name for f in dataclasses.fields(MLAAttentionSpec)}
    original = MultiHeadLatentAttention.get_kv_cache_spec
    if getattr(original, "_pie_k3_filtered", False):
        return

    def patched(self, vllm_config):
        real_init = MLAAttentionSpec.__init__

        @functools.wraps(real_init)
        def filtered_init(spec_self, *args, **kwargs):
            unknown = [k for k in kwargs if k not in known]
            for key in unknown:
                kwargs.pop(key)
            return real_init(spec_self, *args, **kwargs)

        MLAAttentionSpec.__init__ = filtered_init
        try:
            return original(self, vllm_config)
        finally:
            MLAAttentionSpec.__init__ = real_init

    patched._pie_k3_filtered = True
    MultiHeadLatentAttention.get_kv_cache_spec = patched


if __name__ == "__main__":
    register()
    from vllm import ModelRegistry

    archs = ModelRegistry.get_supported_archs()
    for name in ("KimiK3ForConditionalGeneration", "KimiK3ForCausalLM"):
        print(f"{name}: {name in archs}")
