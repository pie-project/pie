# LTX-2.5, an audio-video diffusion transformer, with its VAE decoder; and
# a miniature of its transformer alone.

MODELS = [
    model("ltx25", template = "qwen_3", tokenizer = "qwen_3", arch = "ltx_2", layers = 48, vocab = 0),
    model("ltx25-mini", mini = True, template = "qwen_3", tokenizer = "qwen_3", arch = "ltx_2", layers = 2, vocab = 0),
]

DEPLOYMENTS = [
    deployment("ltx25", weights = dtype.bf16, kv = dtype.bf16),
    deployment("ltx25", weights = dtype.u4g64, kv = dtype.bf16),
    deployment("ltx25-mini", weights = dtype.bf16, kv = dtype.bf16),
]
