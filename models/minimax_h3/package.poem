# MiniMax H3 FL2VA: a joint video-and-audio diffusion transformer with the
# Qwen3-VL text encoder it reads captions through; and its miniature, which
# carries no text encoder.

MODELS = [
    model("minimax-h3-fl2va", template = "qwen_3", tokenizer = "qwen_3", arch = "minimax_h3", layers = 50, vocab = 151936),
    model("minimax-h3-mini", mini = True, template = "qwen_3", tokenizer = "qwen_3", arch = "minimax_h3", layers = 3, vocab = 0),
]

DEPLOYMENTS = [
    deployment("minimax-h3-fl2va", weights = dtype.bf16, kv = dtype.bf16),
    deployment("minimax-h3-mini", weights = dtype.bf16, kv = dtype.bf16),
]
