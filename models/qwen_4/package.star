# Qwen 3.8 Flash Next: hyper-connected streams over gated delta-net layers
# with full attention every fourth, routed experts, an n-gram embedding,
# a multi-token draft head and a vision tower; its four-layer, sixteen-expert
# miniature; and a micro model of toy widths for engine tests, which lists no
# deployment.

MODELS = [
    model(
        "qwen38-flash-next",
        template = "qwen_3_chatml_interleaved",
        tokenizer = "qwen_3",
        parts = ["vision"],
        drafters = ["mtp"],
    ),
    model(
        "qwen38-flash-next-mini",
        mini = True,
        template = "qwen_3_chatml_interleaved",
        tokenizer = "qwen_3",
    ),
    model(
        "qwen38-flash-next-micro",
        mini = True,
        template = "qwen_3_chatml_interleaved",
        tokenizer = "qwen_3",
    ),
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
