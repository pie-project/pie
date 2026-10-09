# gpt-oss 20B (with a DFlash drafter or without), its five-layer,
# sixteen-expert miniature, and 120B.

MODELS = [
    model("gptoss-20b", template = "gpt_oss", tokenizer = "gpt_oss", drafters = ["dflash"], arch = "gptoss", layers = 24, vocab = 201088),
    model("gptoss-20b-mini", mini = True, template = "gpt_oss", tokenizer = "gpt_oss", arch = "gptoss", layers = 5, vocab = 201088),
    model("gptoss-120b", template = "gpt_oss", tokenizer = "gpt_oss", arch = "gptoss", layers = 36, vocab = 201088),
]

DEPLOYMENTS = [
    deployment("gptoss-20b", weights = [dtype.u4g64, dtype.mxfp4], kv = dtype.bf16, drafter = "dflash"),
    deployment("gptoss-20b", weights = [dtype.u4g64, dtype.mxfp4], kv = dtype.bf16),
    deployment("gptoss-20b", weights = [dtype.bf16, dtype.mxfp4], kv = dtype.bf16),
    deployment("gptoss-20b-mini", weights = [dtype.bf16, dtype.mxfp4], kv = dtype.bf16),
    deployment("gptoss-120b", weights = [dtype.bf16, dtype.mxfp4], kv = dtype.bf16),
]

# The drafter published apart from the model it drafts for.
PUBLISHED = [
    published(target = "mlx-community/gpt-oss-20b-MXFP4-Q4", head = "z-lab/gpt-oss-20b-DFlash", drafter = "dflash", deployment = "gptoss-20b-dflash-u4g64-mxfp4-kv-bf16"),
]
