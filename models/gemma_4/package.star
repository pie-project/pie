# Gemma 4: 26B-A4B (experts; with a DFlash drafter, its MTP assistant or its
# vision tower), 31B (with its assistant or vision tower), E4B (with an EAGLE
# head or vision tower) and E4B's miniatures; and DiffusionGemma, 26B-A4B
# as a block-diffusion text model.

MODELS = [
    model("gemma4-26b-a4b", template = "gemma_4", tokenizer = "gemma_4", parts = ["vision"], drafters = ["mtp", "dflash"]),
    model("gemma4-31b", template = "gemma_4", tokenizer = "gemma_4", parts = ["vision"], drafters = ["mtp"]),
    model("gemma4-e4b", template = "gemma_4", tokenizer = "gemma_4", parts = ["vision"], drafters = ["eagle"]),
    model("gemma4-e4b-mini-l1", mini = True, template = "gemma_4", tokenizer = "gemma_4"),
    model("gemma4-e4b-mini-l6", mini = True, template = "gemma_4", tokenizer = "gemma_4"),
    model("gemma4-e4b-mini-l24", mini = True, template = "gemma_4", tokenizer = "gemma_4"),
    model("gemma4-e4b-mini-l30", mini = True, template = "gemma_4", tokenizer = "gemma_4"),
    model("gemma4-e4b-mini-l36", mini = True, template = "gemma_4", tokenizer = "gemma_4"),
    model("diffusiongemma-26b-a4b", template = "gemma_4", tokenizer = "gemma_4", parts = ["selfcond"]),
]

U4 = dtype.u4g64
BF = dtype.bf16

DEPLOYMENTS = [
    deployment("gemma4-26b-a4b", weights = U4, kv = BF, drafter = "dflash"),
    deployment("gemma4-26b-a4b", weights = U4, kv = BF, drafter = "mtp"),
    deployment("gemma4-26b-a4b", weights = U4, kv = BF),
    deployment("gemma4-31b", weights = U4, kv = BF, drafter = "mtp"),
    deployment("gemma4-31b", weights = U4, kv = BF),
    deployment("gemma4-e4b", weights = BF, kv = BF, drafter = "eagle"),
    deployment("gemma4-e4b", weights = BF, kv = BF),
    deployment("gemma4-31b", weights = BF, kv = BF),
    deployment("gemma4-e4b-mini-l1", weights = BF, kv = BF),
    deployment("gemma4-e4b-mini-l6", weights = BF, kv = BF),
    deployment("gemma4-e4b-mini-l24", weights = BF, kv = BF),
    deployment("gemma4-e4b-mini-l30", weights = BF, kv = BF),
    deployment("gemma4-e4b-mini-l36", weights = BF, kv = BF),
    deployment("gemma4-e4b", weights = BF, kv = BF, parts = ["vision"]),
    deployment("gemma4-26b-a4b", weights = U4, kv = BF, parts = ["vision"]),
    deployment("gemma4-31b", weights = U4, kv = BF, parts = ["vision"]),
    deployment("diffusiongemma-26b-a4b", weights = U4, kv = BF),
    deployment("diffusiongemma-26b-a4b", weights = dtype.u8g64, kv = BF),
    deployment("diffusiongemma-26b-a4b", weights = [dtype.u8g64, U4], kv = BF),
    deployment("diffusiongemma-26b-a4b", weights = [U4, dtype.u8g64], kv = BF),
    deployment("diffusiongemma-26b-a4b", weights = [dtype.u8g64, U4, U4], kv = BF, parts = ["selfcond"]),
    deployment("diffusiongemma-26b-a4b", weights = [BF, U4], kv = BF),
]
