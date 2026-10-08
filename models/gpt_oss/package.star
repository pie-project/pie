# gpt-oss 20B (with a DFlash drafter or without), its five-layer,
# sixteen-expert miniature, and 120B.

MODELS = [
    model("gptoss-20b", template = "gpt_oss", tokenizer = "gpt_oss", drafters = ["dflash"]),
    model("gptoss-20b-mini", mini = True, template = "gpt_oss", tokenizer = "gpt_oss"),
    model("gptoss-120b", template = "gpt_oss", tokenizer = "gpt_oss"),
]

DEPLOYMENTS = [
    deployment("gptoss-20b", weights = [dtype.u4g64, dtype.mxfp4], kv = dtype.bf16, drafter = "dflash"),
    deployment("gptoss-20b", weights = [dtype.u4g64, dtype.mxfp4], kv = dtype.bf16),
    deployment("gptoss-20b", weights = [dtype.bf16, dtype.mxfp4], kv = dtype.bf16),
    deployment("gptoss-20b-mini", weights = [dtype.bf16, dtype.mxfp4], kv = dtype.bf16),
    deployment("gptoss-120b", weights = [dtype.bf16, dtype.mxfp4], kv = dtype.bf16),
]
