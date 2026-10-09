# DeepSeek-V4: V4.1-Flash (CSA2, Single-Pass mHC, Engram) and its
# eight-layer miniature; V4-Flash (with its MTP head or not) and its
# miniature; and the small V4 base.

MODELS = [
    model("dsv41-flash", template = "deepseek_v4", tokenizer = "deepseek_v4", arch = "deepseek_v4", layers = 40, vocab = 129280),
    model("dsv41-flash-mini", mini = True, template = "deepseek_v4", tokenizer = "deepseek_v4", arch = "deepseek_v4", layers = 8, vocab = 129280),
    model("dsv4-flash", template = "deepseek_v4", tokenizer = "deepseek_v4", drafters = ["mtp"], arch = "deepseek_v4", layers = 43, vocab = 129280),
    model("dsv4-flash-mini", mini = True, template = "deepseek_v4", tokenizer = "deepseek_v4", drafters = ["mtp"], arch = "deepseek_v4", layers = 5, vocab = 129280),
    model("dsv4-base", mini = True, template = "deepseek_v4", tokenizer = "deepseek_v4", arch = "deepseek_v4", layers = 6, vocab = 129280),
]

BF = dtype.bf16
U4 = dtype.u4g64
U2 = dtype.u2g64
FP4 = dtype.mxfp4

DEPLOYMENTS = [
    deployment("dsv41-flash", weights = U4, kv = BF),
    deployment("dsv41-flash", weights = [BF, FP4], kv = BF),
    deployment("dsv41-flash-mini", weights = [BF, FP4], kv = BF),
    deployment("dsv4-flash", weights = [U4, U2, FP4], kv = BF, drafter = "mtp"),
    deployment("dsv4-flash-mini", weights = [U4, U2, FP4], kv = BF, drafter = "mtp"),
    deployment("dsv4-flash", weights = [U4, U2], kv = BF),
    deployment("dsv4-flash-mini", weights = [U4, U2], kv = BF),
    deployment("dsv4-flash-mini", weights = BF, kv = BF),
    deployment("dsv4-flash-mini", weights = [BF, FP4], kv = BF),
    deployment("dsv4-base", weights = BF, kv = BF),
    deployment("dsv4-flash", weights = BF, kv = BF),
]
