# mini-dit: a miniature diffusion transformer of one single-stream, one
# double-stream and one cross-attention block, for kernel bring-up.

MODELS = [model("mini-dit", mini = True, template = "mini_dit", tokenizer = "mini_dit", arch = "mini_dit", layers = 3, vocab = 0)]

DEPLOYMENTS = [deployment("mini-dit", weights = dtype.bf16, kv = dtype.bf16)]
