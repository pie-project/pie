# FLUX.2 [klein] 4B, with its Qwen3 text encoder and its VAE, and a miniature
# of its transformer alone.

MODELS = [
    model("flux2-klein-4b", template = "flux_2", tokenizer = "qwen_3", arch = "flux_2", layers = 27, vocab = 151936),
    model("flux2-mini", mini = True, template = "flux_2", tokenizer = "qwen_3", arch = "flux_2", layers = 4, vocab = 0),
]

DEPLOYMENTS = [
    deployment("flux2-klein-4b", weights = dtype.bf16, kv = dtype.bf16),
    deployment("flux2-klein-4b", weights = dtype.u4g64, kv = dtype.bf16),
    deployment("flux2-mini", weights = dtype.bf16, kv = dtype.bf16),
]
