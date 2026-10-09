# Wan 2.2 TI2V 5B (its UMT5 text encoder and its VAE with it), and two
# miniatures of its denoiser alone.

MODELS = [
    model("wan22-ti2v-5b", template = "wan_2", tokenizer = "wan_2", arch = "wan_2", layers = 24, vocab = 256384),
    model("wan22-mini-d128", mini = True, template = "wan_2", tokenizer = "wan_2", arch = "wan_2", layers = 2, vocab = 0),
    model("wan22-mini-nano", mini = True, template = "wan_2", tokenizer = "wan_2", arch = "wan_2", layers = 2, vocab = 0),
]

DEPLOYMENTS = [
    deployment("wan22-ti2v-5b", weights = dtype.bf16, kv = dtype.bf16),
    deployment("wan22-ti2v-5b", weights = dtype.u4g64, kv = dtype.bf16),
    deployment("wan22-mini-d128", weights = dtype.bf16, kv = dtype.bf16),
    deployment("wan22-mini-nano", weights = dtype.bf16, kv = dtype.bf16),
]
