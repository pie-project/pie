# GLM-5.3-Flash, with its vision tower and multi-token draft head or
# without, and its eight-layer, 32-expert miniature.

MODELS = [
    model(
        "glm53-flash",
        template = "glm_5_next",
        tokenizer = "glm_5_next",
        parts = ["vision"],
        drafters = ["mtp"],
    ),
    model("glm53-flash-mini", mini = True, template = "glm_5_next", tokenizer = "glm_5_next"),
]

DEPLOYMENTS = [
    deployment("glm53-flash", weights = [dtype.u8g64, dtype.u2g64, dtype.u4g64], kv = dtype.bf16, drafter = "mtp"),
    deployment("glm53-flash-mini", weights = [dtype.u4g64, dtype.u4g64], kv = dtype.bf16),
    deployment("glm53-flash", weights = [dtype.u8g64, dtype.u2g64], kv = dtype.bf16),
    deployment(
        "glm53-flash",
        weights = [dtype.u8g64, dtype.u2g64, dtype.u4g64],
        kv = dtype.bf16,
        parts = ["vision"],
        drafter = "mtp",
    ),
    deployment("glm53-flash", weights = [dtype.u8g64, dtype.u2g64], kv = dtype.bf16, parts = ["vision"]),
    deployment(
        "glm53-flash",
        weights = [dtype.u4g64, dtype.u2g64, dtype.u4g64],
        kv = dtype.bf16,
        parts = ["vision"],
        drafter = "mtp",
    ),
]
