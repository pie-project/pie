# Qwen 3.5 / 3.6 / 3.8: gated-delta (GDN) layers with a gated attention one
# every few, dense or routed MLPs, a vision tower, and the drafters each
# pairs with (MTP, EAGLE, DFlash, DFlash2, DSpark). The Ternary-Bonsai 27B and
# the test geometries are models of it too, listed nowhere: a test traces
# and imports them by id.

def qwen(id, template = "qwen_3", **kwargs):
    return model(id, template = template, tokenizer = "qwen_3", **kwargs)

MODELS = [
    qwen("qwen36-27b", parts = ["vision"], drafters = ["mtp", "dflash"]),
    qwen("qwen38-27b", template = "qwen_3_chatml_interleaved", parts = ["vision"], drafters = ["mtp", "dflash2", "dspark"]),
    qwen("qwen35-d0.8b", parts = ["vision"], drafters = ["eagle"]),
    qwen("qwen35-d2b"),
    qwen("qwen35-d3b"),
    qwen("qwen35-d4b"),
    qwen("qwen35-a3b"),
    qwen("qwen35-d9b", drafters = ["dflash"]),
    qwen("qwen36-35b-a3b", drafters = ["mtp", "dflash"]),
    qwen("qwen35-tiny", mini = True),
    qwen("qwen36-35b-a3b-mini", mini = True),
    qwen("qwen36-35b-a3b-mini64", mini = True),
    # Listed nowhere.
    qwen("qwen36-27b-bonsai", mini = True),
    qwen("qwen3-a3b-micro", mini = True),
    qwen("qwen3-a3b-uncached-bank", mini = True),
    qwen("qwen3-micro-text", mini = True),
    qwen("qwen3-micro-text-rotated", mini = True),
    qwen("qwen3-micro-text-hd128", mini = True),
    qwen("qwen3-micro-text-hd128-rotated", mini = True),
    qwen("qwen3-micro-text-hd256", mini = True),
    qwen("qwen3-micro-text-hd256-rotated", mini = True),
]

U4 = dtype.u4g64
BF = dtype.bf16

def row(id, weights, drafter = None, parts = []):
    return deployment(id, weights = weights, kv = BF, drafter = drafter, parts = parts)

DEPLOYMENTS = [
    row("qwen36-27b", U4, "mtp"),
    row("qwen36-27b", U4, "dflash"),
    row("qwen36-27b", U4),
    row("qwen35-tiny", U4),
    row("qwen35-d0.8b", U4),
    row("qwen35-d2b", U4),
    row("qwen35-d4b", U4),
    row("qwen35-d9b", U4, "dflash"),
    row("qwen35-d9b", U4),
    row("qwen36-35b-a3b", U4, "dflash"),
    row("qwen36-35b-a3b", U4, "mtp"),
    row("qwen36-35b-a3b", U4),
    row("qwen36-35b-a3b-mini", U4),
    row("qwen36-35b-a3b-mini64", U4),
    row("qwen36-27b", BF, "mtp"),
    row("qwen38-27b", BF, "mtp"),
    row("qwen38-27b", U4, "dflash2"),
    row("qwen38-27b", U4, "dspark"),
    row("qwen38-27b", U4, "mtp"),
    row("qwen38-27b", U4),
    row("qwen35-a3b", BF),
    row("qwen35-d3b", BF),
    row("qwen35-d0.8b", BF, "eagle"),
    row("qwen35-d0.8b", BF),
    row("qwen35-d0.8b", BF, "eagle", ["vision"]),
    row("qwen36-27b", U4, None, ["vision"]),
    row("qwen36-27b", BF, "mtp", ["vision"]),
    row("qwen38-27b", U4, None, ["vision"]),
    row("qwen38-27b", BF, "mtp", ["vision"]),
    row("qwen35-d0.8b", U4, None, ["vision"]),
    row("qwen35-d0.8b", BF, None, ["vision"]),
]
