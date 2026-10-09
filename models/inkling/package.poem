# Inkling, and its miniature of seven layers and eight experts, which reads a
# prefix of the whole model's checkpoint.

MODELS = [
    model("inkling", template = "inkling", tokenizer = "inkling", arch = "inkling", layers = 66, vocab = 200058),
    model("inkling-mini-l7-e8", mini = True, template = "inkling", tokenizer = "inkling", arch = "inkling", layers = 7, vocab = 200058),
]

DEPLOYMENTS = [
    deployment("inkling", weights = dtype.bf16, kv = dtype.bf16),
    deployment("inkling-mini-l7-e8", weights = dtype.bf16, kv = dtype.bf16),
]
