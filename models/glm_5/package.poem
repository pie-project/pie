# GLM-5: a mixture of experts behind multi-head latent attention, its keys
# chosen per query by a lightning indexer.

MODELS = [model("glm5-a12b", template = "glm_5", tokenizer = "glm_5", arch = "glm_moe_dsa", layers = 46, vocab = 151552)]

DEPLOYMENTS = [deployment("glm5-a12b", weights = dtype.bf16, kv = dtype.bf16)]
