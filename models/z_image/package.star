# Z-Image Turbo, with its Qwen3 text encoder and its VAE, and a miniature of
# its diffusion transformer alone.

MODELS = [
    model("z-image-turbo", template = "z_image", tokenizer = "z_image"),
    model("z-image-mini", mini = True, template = "z_image", tokenizer = "z_image"),
]

DEPLOYMENTS = [
    deployment("z-image-turbo", weights = dtype.bf16, kv = dtype.bf16),
    deployment("z-image-turbo", weights = dtype.u4g64, kv = dtype.bf16),
    deployment("z-image-mini", weights = dtype.bf16, kv = dtype.bf16),
]
