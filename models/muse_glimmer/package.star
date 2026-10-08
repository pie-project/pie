# Muse Glimmer 30B, and its eight-layer miniature, which reads the first
# eight layers of the whole model's checkpoint.

MODELS = [
    model("muse-glimmer-30b", template = "muse_glimmer", tokenizer = "muse_glimmer"),
    model("muse-glimmer-30b-mini-l8", mini = True, template = "muse_glimmer", tokenizer = "muse_glimmer"),
]

DEPLOYMENTS = [
    deployment("muse-glimmer-30b", weights = dtype.bf16, kv = dtype.bf16),
    deployment("muse-glimmer-30b", weights = dtype.u4g64, kv = dtype.bf16),
    deployment("muse-glimmer-30b-mini-l8", weights = dtype.bf16, kv = dtype.bf16),
]
