# mini-dit: a miniature diffusion transformer of one single-stream, one
# double-stream and one cross-attention block, for kernel bring-up.

MODELS = [model("mini-dit", mini = True, template = "mini_dit", tokenizer = "mini_dit")]

DEPLOYMENTS = [deployment("mini-dit", weights = dtype.bf16, kv = dtype.bf16)]
