# HunyuanImage 3: an autoregressive mixture of experts that also denoises an
# image canvas through the same trunk, a small U-Net at each end of it; and
# its two-layer miniature.

MODELS = [
    model("hunyuanimage3-80b-a13b", template = "hunyuan_image_3", tokenizer = "hunyuan_image_3"),
    model("hunyuanimage3-mini", mini = True, template = "hunyuan_image_3", tokenizer = "hunyuan_image_3"),
]

DEPLOYMENTS = [
    deployment("hunyuanimage3-80b-a13b", weights = [dtype.bf16, dtype.u8g64], kv = dtype.bf16),
    deployment("hunyuanimage3-80b-a13b", weights = [dtype.bf16, dtype.u4g64], kv = dtype.bf16),
    deployment("hunyuanimage3-mini", weights = dtype.bf16, kv = dtype.bf16),
]
