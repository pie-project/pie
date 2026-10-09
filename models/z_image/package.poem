# Z-Image Turbo, with its Qwen3 text encoder and its VAE, and a miniature of
# its diffusion transformer alone.

MODELS = [
    model("z-image-turbo", template = "qwen_3", tokenizer = "qwen_3", arch = "z_image", layers = 35, vocab = 151936),
    model("z-image-mini", mini = True, template = "qwen_3", tokenizer = "qwen_3", arch = "z_image", layers = 6, vocab = 0),
]

DEPLOYMENTS = [
    deployment("z-image-turbo", weights = dtype.bf16, kv = dtype.bf16),
    deployment("z-image-turbo", weights = dtype.u4g64, kv = dtype.bf16),
    deployment("z-image-mini", weights = dtype.bf16, kv = dtype.bf16),
]
