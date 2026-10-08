# Inkling, and its miniature of seven layers and eight experts, which reads a
# prefix of the whole model's checkpoint.

MODELS = [
    model("inkling", template = "inkling", tokenizer = "inkling"),
    model("inkling-mini-l7-e8", mini = True, template = "inkling", tokenizer = "inkling"),
]

DEPLOYMENTS = [
    deployment("inkling", weights = dtype.bf16, kv = dtype.bf16),
    deployment("inkling-mini-l7-e8", weights = dtype.bf16, kv = dtype.bf16),
]
