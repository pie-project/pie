# Kimi-K3: its eight-layer, 32-expert miniature cut from the released
# checkpoint, and the fixture's shape it was first brought up at.

MODELS = [
    model("kimik3-mini", mini = True, template = "kimi_k3.instruct3", tokenizer = "kimi_k3.instruct3", arch = "kimi_k3", layers = 8, vocab = 163840),
    model("kimik3", template = "kimi_k3", tokenizer = "kimi_k3", arch = "kimi_k3", layers = 8, vocab = 163840),
]

DEPLOYMENTS = [
    deployment("kimik3-mini", weights = [dtype.bf16, dtype.mxfp4], kv = dtype.bf16),
    deployment("kimik3", weights = [dtype.bf16, dtype.mxfp4], kv = dtype.bf16),
]
