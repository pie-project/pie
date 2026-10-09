# Qwen 3.8 Flash Next: hyper-connected streams over gated delta-net layers
# with full attention every fourth, routed experts, an n-gram embedding,
# a multi-token draft head and a vision tower; its four-layer, sixteen-expert
# miniature; and a micro model of toy widths for engine tests, which lists no
# deployment.

def flash(id, **kwargs):
    return model(id, template = "qwen_3_chatml_interleaved", tokenizer = "qwen_3.38", **kwargs)

MODELS = [
    flash("qwen38-flash-next", parts = ["vision"], drafters = ["mtp"], arch = "qwen4_exp", layers = 48, vocab = 248320),
    flash("qwen38-flash-next-mini", mini = True, arch = "qwen4_exp", layers = 4, vocab = 248320),
    flash("qwen38-flash-next-micro", mini = True),
]

MIXED = [dtype.u4g64, dtype.u2g128]

DEPLOYMENTS = [
    deployment("qwen38-flash-next", weights = dtype.u4g64, kv = dtype.bf16),
    deployment("qwen38-flash-next", weights = MIXED, kv = dtype.bf16, drafter = "mtp"),
    deployment("qwen38-flash-next", weights = MIXED, kv = dtype.bf16),
    deployment("qwen38-flash-next-mini", weights = MIXED, kv = dtype.bf16),
    deployment("qwen38-flash-next", weights = dtype.bf16, kv = dtype.bf16),
    deployment("qwen38-flash-next", weights = MIXED, kv = dtype.bf16, parts = ["vision"], drafter = "mtp"),
    deployment("qwen38-flash-next", weights = MIXED, kv = dtype.bf16, parts = ["vision"]),
]
